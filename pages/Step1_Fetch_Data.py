import streamlit as st
st.set_page_config(
    page_title="Market Analysis",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items = {
        "Get Help": "mailto:yilin.space@gmail.com",
        "Report a bug": "https://github.com/lemonsong/streamlit-apps",
        "About": "About my application **Hello World!**"
    }
)

from src.constants import (project_dir, FRED_API)
import sys, os
import pandas as pd
from src.utils.fred import FredClass
from src.utils.yahoofinance import YfClass

@st.cache_data
def collect_data(from_api=False):

    if from_api:
        # collect fred data

        # leading indicators of economics: 'M2', 'T10YFF', 'DGS10', 'SP500', 'UMCSENT', 'PERMIT','ANDENO','AWHAEMAN','DTCDISA066MSFRBNY','ICSA'
        # long-term leading indicators of economics: 'M2', 'T10YFF', 'PERMIT', 价格对单位劳动成本对比率
        # lag indicators of economics:  'MDSP', 工商业的贷款余额，平均银行贷款基本率， 服务业消费者物价指数的变动，单位劳动产量的劳动成本变动，制造与贸易库存对销售额的比率，反向的平均就业持续时间
        # leading indicators of residential real estate: ·支付能力，指每月支付按揭占可支配收入的部分, 房价与雇员收入之比, 房价与GDP的比率
        fred_metric_list = ['M2', 'T10YFF', 'DGS10', 'SP500', 'UMCSENT',
                            'PERMIT', 'NEWY636BPPRIV', 'NYBPPRIVSA',
                            'ANDENO', 'ACOGNO', 'AMTMNO', 'ACDGNO', 'DGORDER',
                            'AWHMAN', 'AWHAETP', 'AWHAEMAN',
                            'DTCDISA066MSFRBNY',
                            'ICSA', 'NYICLAIMS', 'NJICLAIMS',
                            'MDSP', 'DRSFRMACBS', 'M0264AUSM500NNBR', 'BOGZ1FL153165106Q', 'DCOILWTICO']
        # fred_metric_list = ['GDP','MDSP']
        start_date_dt = pd.to_datetime('1980-01-01')

        fred = FredClass(FRED_API)
        # get FRED metric info
        fred_metrics_info_df = fred.get_metrics_info(fred_metric_list)
        fred_metrics_info_df.to_csv(os.path.join(project_dir, 'data/raw/fred_metrics_info_df.csv'), encoding='utf-8', index=False)
        # get FRED metric value
        fred_metrics_df = fred.fetch(metrics=fred_metric_list,start_date=start_date_dt)
        fred_metrics_df.to_csv(os.path.join(project_dir, 'data/raw/fred_metrics_value.csv'), encoding='utf-8', index=False)

        # collect stock value
        stock_price_list = [
            # 'JPM','GS', 'C', 'NIO', 'TSM', 'NVDA', 'EPD', 'AMD','BA','BABA','DAL','LMND','MA','MSFT','SNOW',
            'VOX', 'VCR', 'VDC', 'VDE', 'VFH', 'VHT', 'VIS', 'VGT', 'VAW', 'VNQ', 'VPU',
            'QQQ', 'VOO', 'VTV', 'VIGAX', 'VO', 'VB', 'VGK']
        # get ticker prices from yahoo finance api
        yahoofin = YfClass(stock_price_list)
        stock_price_df = yahoofin.get_tickers_history(period='max')
        stock_price_df.to_csv(os.path.join(project_dir, 'data/raw/yahoofinance_stock_price_df.csv'), encoding='utf-8',
                              index=False)
    else:
        # get FRED metric info
        fred_metrics_info_df = pd.read_csv(os.path.join(project_dir, 'data/raw/fred_metrics_info_df.csv'))
        # get FRED metric value
        fred_metrics_df = pd.read_csv(os.path.join(project_dir, 'data/raw/fred_metrics_value.csv'))
        # get ticker prices
        stock_price_df = pd.read_csv(os.path.join(project_dir, 'data/raw/yahoofinance_stock_price_df.csv'))

    return fred_metrics_info_df, fred_metrics_df, stock_price_df
def change_data_source_callback():
    # st.cache_data.clear()
    collect_data.clear()

# radio button to select data source
st.radio(
        "Choose which data to use 👉",
        key="data_source",
        options=["Static saved data", "Fetch new data"],
        on_change=change_data_source_callback
    )
if st.session_state.data_source == "Static saved data":
    from_api = False
elif st.session_state.data_source == "Fetch new data":
    from_api = True

fred_metrics_info_df, fred_metrics_df, stock_price_df = collect_data(from_api)

# Display data
# column layout for data output
col1, col2 = st.columns([2,3])
with col1:
    config = {
        "activity_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "value": st.column_config.NumberColumn("Value ($)", format='$%d'),
    }
    st.subheader("FRED Metric Detail")
    st.dataframe(fred_metrics_df, column_config=config)
    # TODO: add dataset start and end date info
with col2:
    st.subheader("FRED Metric Value")
    st.dataframe(fred_metrics_info_df, column_config={})
    # TODO: number of metrics info
st.subheader("Stock Price")
stock_price_config = {
        "activity_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "open": st.column_config.NumberColumn("open", format='$%d'),
        "high": st.column_config.NumberColumn("high", format='$%d'),
        "low": st.column_config.NumberColumn("low", format='$%d'),
        "close": st.column_config.NumberColumn("close", format='$%d'),
        "dividens": st.column_config.NumberColumn("dividends", format='$%d'),
    }
with st.container():
    st.dataframe(stock_price_df, column_config=stock_price_config)




