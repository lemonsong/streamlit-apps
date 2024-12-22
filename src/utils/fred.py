from fredapi import Fred
import pandas as pd
from src.constants import FRED_API
import sys


class FredClass:
    def __init__(self, FRED_API):
        self.fred = Fred(api_key=FRED_API)

    def search(self, search_text):
        self.search_df = self.fred.search(search_text)
        self.search_df['popularity'] = self.search_df['popularity'].astype(int)
        self.search_df = self.search_df.sort_values(by=['popularity'], ascending=False)
        return self.search_df

    def fetch(self, metrics, start_date):
        lst = []
        for metric in metrics:
            print(f'Fetch metric: {metric}')
            try:
                # TODO: confirm usage of realtime_start parameter
                metrics_df_sub = self.fred.get_series_latest_release(metric, realtime_start= start_date.strftime("%Y-%m-%d")).to_frame()
                metrics_df_sub['metric'] = metric
                lst.append(metrics_df_sub)
            except ValueError:
                print(f'Skip metric: {metric}')
                continue
        self.metrics_df = pd.concat(lst, axis=0)
        self.metrics_df = self.metrics_df.reset_index(drop=False)
        self.metrics_df = self.metrics_df.rename(columns={'index':'activity_date', 0:'value'})
        print(self.metrics_df.dtypes)
        self.metrics_df = self.metrics_df.loc[self.metrics_df.activity_date >= start_date]
        return self.metrics_df

    def get_metrics_info(self,metrics):
        lst = []
        for metric in metrics:
            info_df_sub = self.fred.get_series_info(metric).to_frame()
            lst.append(info_df_sub)
        info_df= pd.concat(lst, axis=1, ignore_index=True).T
        print(info_df)
        info_df = info_df[[
                'id','title','frequency','units','seasonal_adjustment','popularity','notes'
            ]]
        info_df = info_df.rename(columns={'id':'metric'})
        info_df['source'] = 'FRED'
        info_df['deprecated'] = False
        return info_df
