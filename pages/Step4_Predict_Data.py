import streamlit as st
st.set_page_config(
    page_title="Market Analysis",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:yilin.space@gmail.com",
        "Report a bug": "https://github.com/lemonsong/streamlit-apps",
        "About": "About my application **Hello World!**"
    }
)
from pages.Step1_Fetch_Data import (fred_metrics_info_df, fred_metrics_df, stock_price_df)

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

    df = data_df.loc[(data_df.index >= train_start_date) & (data_df.index <= train_end_date),
    etf_list+leading_cols + date_feature_list].copy()
                     # etf_list + leading_metric_list + date_feature_list].copy()

    return

data_df, etf_list, leading_metric_list, date_feature_list = process_data(fred_metrics_info_df, fred_metrics_df, stock_price_df)


with st.container():
    st.subheader("Feature Selection:")
    st.divider()

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