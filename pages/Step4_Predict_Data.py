import streamlit as st



from src.constants import (project_dir)
import sys, os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import rankdata

from dateutil.relativedelta import relativedelta
import datetime
import holidays

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import statsmodels.api as sm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from itertools import cycle
from pages.Step1_Fetch_Data import (fred_metrics_info_df, fred_metrics_df, stock_price_df)


# generate window
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

WindowGenerator.split_window = split_window

def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Date [d]')

WindowGenerator.plot = plot

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

    ds = ds.map(self.split_window)

    return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

def make_pred_dataset(self, data):
#     data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.input_width,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)
    return ds

WindowGenerator.make_pred_dataset = make_pred_dataset

MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history

@st.cache_data
def process_data(fred_metrics_info_df, fred_metrics_df, stock_price_df):

    leading_df = fred_metrics_df.merge(fred_metrics_info_df, on='metric', how='left')
    etf_df = stock_price_df.copy()
    # pivots the leading indicators and etf to predict
    leading_metric_df = leading_df.groupby(['metric', 'frequency', 'title']
                                           ).count().index.to_frame(index=False, name=['metric', 'frequency', 'title'])
    leading_pivoted = pd.pivot_table(leading_df, index='activity_date', columns='metric', values='value',
                                     aggfunc='sum').reset_index()

    etf_pivoted = pd.pivot_table(etf_df, index='activity_date', columns='ticker', values='close',
                                 aggfunc='sum').reset_index()
    # create data_df including both leading metrics value and etf value
    data_df = etf_pivoted.merge(leading_pivoted, on=['activity_date'], how='outer') \
        .sort_values(by=['activity_date']) \
        .reset_index(drop=True)
    # preprocess data_df
    # use activity date date type as index
    date_index = pd.date_range(start=data_df['activity_date'].min(), end=data_df['activity_date'].max(), freq='D')
    data_df.index = pd.to_datetime(data_df['activity_date'])
    data_df = data_df.drop('activity_date', axis=1)
    data_df = data_df.reindex(date_index)
    data_df.index = data_df.index.date
    etf_list = etf_df.ticker.unique().tolist()
    # create date_features
    data_df['weekday'] = data_df.index.map(lambda x: x.weekday())
    data_df['is_weekend'] = np.where(data_df['weekday'].isin([0, 6]), 1, 0)
    us_holidays = holidays.UnitedStates()
    data_df['is_holiday'] = data_df.index.map(lambda x: 1 if x in us_holidays else 0)
    date_feature_list = ['weekday', 'is_weekend', 'is_holiday']
    # TODO:
    # etf_list=['VFH']
    leading_metric_list = leading_metric_df.metric.unique().tolist()
    # replace the fred metric with 0 value to null
    data_df = data_df.replace(0, np.nan)
    # forwardfill then backwardfill null value to fill all null value
    data_df = data_df.ffill()
    return data_df, etf_list, leading_metric_list, date_feature_list


def on_click_preprocess(data_df, etf_list, leading_metric_list, date_feature_list,
                            train_start_date_input, train_end_date_input, n_forecast,
                            leading_cols, target_col):
    # start_date = '2014-01-01'
    n_input_width = n_forecast
    # one of 'VOX', 'VCR', 'VDC', 'VDE', 'VFH', 'VHT', 'VIS', 'VGT', 'VAW', 'VNQ', 'VPU', 'QQQ'
    target_etf = 'VOO'
    # use related data for model
    data_start_date = data_df.index.min()
    data_end_date = data_df.index.max()
    train_start_date = max(data_start_date, train_start_date_input)
    train_end_date = min(data_end_date, train_end_date_input)
    # for deep learning model in this Notebook, the forecast period is decided by n_input_width rather than n_forecast
    forecast_start_date, forecast_end_date = train_end_date + relativedelta(days=1), train_end_date + relativedelta(
        days=1 + n_forecast)

    plot_cols =  etf_list+leading_cols + date_feature_list
    df = data_df.loc[(data_df.index >= train_start_date) & (data_df.index <= train_end_date),
    plot_cols].copy()
                     # etf_list + leading_metric_list + date_feature_list].copy()

    # split data
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n * 0.7)].copy()
    val_df = df[int(n * 0.7):int(n * 0.9)].copy()
    test_df = df[int(n * 0.9):].copy()

    num_features = df.shape[1]

    #  normalize data
    # use one degree difference rather than normalization to improve the accuracy of prediction
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = train_df.diff(1).dropna()
    val_df = val_df.diff(1).dropna()
    test_df = test_df.diff(1).dropna()

    # # check distribution
    # df_std = df.diff(1).dropna()
    # df_std = df_std.melt(var_name='Column', value_name='Normalized')
    # plt.figure(figsize=(12, 6))
    # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    # _ = ax.set_xticklabels(df.keys(), rotation=90)

    OUT_STEPS = n_input_width
    multi_window = WindowGenerator(input_width=n_input_width,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   train_df=train_df, val_df=val_df, test_df=test_df)

    # multi_window.plot(plot_col=target_etf)
    if retrain_input=='Yes':
        # RNN
        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(OUT_STEPS * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        history = compile_and_fit(multi_lstm_model, multi_window)
        multi_lstm_model.save('models/multi_lstm_model.keras')
    else:
        multi_lstm_model = tf.keras.models.load_model('models/multi_lstm_model.keras')
    # IPython.display.clear_output()

    multi_val_performance = {}
    multi_performance = {}

    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_lstm_model, plot_col=target_etf)


    # make prediction


    diff_order = 1

    final_window = multi_window
    final_model = multi_lstm_model


    final_window_size = final_window.input_width

    # actual_df_std = (df - train_mean) / train_std
    actual_df_std = df.diff(diff_order).dropna()
    actual_df_std = actual_df_std[-final_window_size:]

    predictions = final_model.predict(final_window.make_pred_dataset(actual_df_std))
    # predictions = final_model.predict([np.array(actual_df_std, dtype=np.float32)])
    print("predictions shape:", predictions.shape)

    # reverse the prediction
    predictions_reversed = pd.DataFrame(data=predictions[0], columns=plot_cols)
    # method to reverse prediction to the original scale as we used 1 degree difference to preprocess data
    predictions_reversed = np.cumsum(predictions_reversed, axis=0) + df[-1:].values
    # method to reverse prediction to the original scale if we used normalization to preprocess data
    # predictions_reversed = predictions_reversed*train_std+train_mean

    predictions_reversed.index = pd.date_range(forecast_start_date, periods=predictions_reversed.shape[0],
                                               freq='D').date

    # compare current with predicted future
    current_reversed = df.copy()
    # current_reversed = actual_df_std.copy()
    current = current_reversed[etf_list].tail(1).values
    # current = current_reversed[etf_list].tail(3).mean()
    future1 = np.array(predictions_reversed[etf_list].mean())
    future2 = predictions_reversed[etf_list].tail(1).values
    future_tomorrow = predictions_reversed[etf_list].head(1).values

    print("The higher the percentage of increase, the lower the number of rank:")
    st.dataframe(
        pd.DataFrame(data={'etf': etf_list,
                           '% change by mean': (future1 / current - 1)[0],
                           'change by mean': (future1 - current)[0],
                           'rank1': len(etf_list) + 1 - rankdata((future1 / current - 1)[0], method='min'),
                           '% change by last value': (future2 / current - 1)[0],
                           'change by last value': (future2 - current)[0],
                           'rank2': len(etf_list) + 1 - rankdata((future2 / current - 1)[0], method='min'),
                           '% change by tomorrow': (future_tomorrow / current - 1)[0],
                           'rank3': len(etf_list) + 1 - rankdata((future_tomorrow / current - 1)[0], method='min')
                           })
        , column_config={})

    # plot the current and predicted future
    final_actual = current_reversed[etf_list + leading_metric_list].copy()
    final_actual['Type'] = 'Actual'
    final_pred = predictions_reversed[etf_list + leading_metric_list].copy()
    final_pred['Type'] = 'Prediction'

    final_df = pd.concat([final_actual,final_pred],axis=0, ignore_index=True)
    final_df['Created Date'] = train_end_date

    # store the actual and prediction
    final_df.to_csv( f'data/processed/ts_leading_metric_impact_on_sectors_pred_{train_end_date}.csv',
        encoding='utf-8', index=False)
    st.dataframe(final_df, column_config={})

    # visualize ETFs
    # colors
    palette = cycle(px.colors.qualitative.Bold)

    max_y = final_df[etf_list].max().max()
    n_digit = 10 ** (len(str(int(max_y))) - 1)
    data = []
    for etf in etf_list:
        trace_color = next(palette)
        # TODO: add ETF name
        trace_actual = go.Scatter(x=final_actual.index, y=final_actual[etf],
                                  mode='lines',
                                  name=f'{etf}-actual',
                                  line=go.Line(color=trace_color,
                                               width=1.5, ))
        trace_pred = go.Scatter(x=final_pred.index, y=final_pred[etf],
                                mode='lines',
                                name=f'{etf}-prediction',
                                line=go.Line(color=trace_color,
                                             width=1.5,
                                             dash='dashdot'))
        data.append(trace_actual)
        data.append(trace_pred)

    layout = go.Layout(title='ETF',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Value'))
    layout.update(dict(shapes=[
        {
            'type': 'line',
            'x0': train_end_date,
            'y0': 0,
            'x1': train_end_date,
            'y1': np.ceil(max_y / n_digit) * n_digit,
            'line': {
                'color': '#909090',
                'width': 1
            }

        }
    ]))
    layout.update(dict(annotations=[
        go.Annotation(text='End of training',
                      x=train_end_date,
                      y=np.floor(max_y / n_digit) * n_digit
                      )
    ]))
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, theme="streamlit")

    # visualize leading metrics

    # colors
    palette = cycle(px.colors.qualitative.Vivid)

    max_y = final_df[leading_metric_list].max().max()
    n_digit = 10 ** (len(str(int(max_y))) - 1)
    data = []
    for leading_metric in leading_metric_list:
        trace_color = next(palette)
        trace_actual = go.Scatter(x=final_actual.index, y=final_actual[leading_metric],
                                  mode='lines',
                                  name=f'{leading_metric}-actual',
                                  line=go.Line(color=trace_color,
                                               width=1.5, ))
        trace_pred = go.Scatter(x=final_pred.index, y=final_pred[leading_metric],
                                mode='lines',
                                name=f'{leading_metric}-prediction',
                                line=go.Line(color=trace_color,
                                             width=1.5,
                                             dash='dashdot'))
        data.append(trace_actual)
        data.append(trace_pred)

    layout = go.Layout(title='Leading Metrics',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Value'))
    layout.update(dict(shapes=[
        {
            'type': 'line',
            'x0': train_end_date,
            'y0': 0,
            'x1': train_end_date,
            'y1': np.ceil(max_y / n_digit) * n_digit,
            'line': {
                'color': '#909090',
                'width': 1
            }

        }
    ]))
    layout.update(dict(annotations=[
        go.Annotation(text='End of training',
                      x=train_end_date,
                      y=np.floor(max_y / n_digit) * n_digit
                      )
    ]))
    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig, theme="streamlit")

    return

data_df, etf_list, leading_metric_list, date_feature_list = process_data(fred_metrics_info_df, fred_metrics_df, stock_price_df)


with st.container():
    st.subheader("Predict")
    st.divider()
    retrain_input = st.selectbox("Do you want to retrain the model", options=['Yes','No'], index=1)
    train_start_date_input = st.date_input("Training data start date", datetime.date(2014, 1, 1))
    train_end_date_input = st.date_input("Training data end date", datetime.datetime.today().date(),
                                         # max_value=data_df.index.max()
                                         )
    n_forecast = st.slider("N days to predict", 30, 120, 90)
    leading_cols = st.multiselect(
        "Which metrics to use as feature?",
        options = leading_metric_list,
        default = leading_metric_list
    )
    target_col = st.selectbox("Which stock to use as target?", options=etf_list, index=etf_list.index("VOO"))
    st.divider()
    st.button('Pre-Process Data', type="primary", on_click=on_click_preprocess(
        data_df, etf_list, leading_metric_list, date_feature_list,
        train_start_date_input, train_end_date_input, n_forecast,
        leading_cols, target_col)
              )
