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
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from pages.Step1_Fetch_Data import (fred_metrics_info_df, fred_metrics_df, stock_price_df)
profile = ProfileReport(fred_metrics_info_df, title="Profiling Report")
profile.to_notebook_iframe()