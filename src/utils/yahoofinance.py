import yfinance as yf
import pandas as pd

class YfClass:
    def __init__(self, tickers_list):
        self.tickers_list = tickers_list
        self.tickers_str = ' '.join(self.tickers_list)
        self.tickers = yf.Tickers(self.tickers_str)

    def get_tickers_history(self, period):
        lst = []
        for ticker in self.tickers_list:
            print(f'Get history of {ticker}')
            # hist_df_sub = getattr(self.tickers.tickers, ticker).history(period=period)
            print(self.tickers.tickers)
            hist_df_sub = self.tickers.tickers[ticker].history(period=period)
            hist_df_sub['ticker'] = ticker
            # hist_df = hist_df.append(hist_df_sub)
            lst.append(hist_df_sub)
        hist_df = pd.concat(lst, axis=0)
        hist_df.columns = [i.lower().replace(' ', '_') for i in hist_df.columns]
        hist_df.reset_index(drop=False, inplace=True)
        hist_df = hist_df.rename(columns={'Date': 'activity_date'})
        hist_df['activity_date'] = hist_df.activity_date.dt.date
        hist_df = hist_df[['ticker', 'activity_date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']]
        return hist_df