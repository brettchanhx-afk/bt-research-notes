import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

token = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = "http://jiaoch.site"


ASSET_CONFIG = {
    'CSI300': {'code': '000300.SH', 'source': 'tushare', 'type': 'index'},
    'HSI': {'code': 'HSI.HI', 'source': 'yfinance', 'type': 'index'},
    'Nikkei225': {'code': 'N225.GI', 'source': 'yfinance', 'type': 'index'},
    'SP500': {'code': 'SPX.GI', 'source': 'yfinance', 'type': 'index'},
    'Gold': {'code': 'GC.CMX', 'source': 'yfinance', 'type': 'future'},
    'CrudeOil': {'code': 'B.IPE', 'source': 'yfinance', 'type': 'future'},
    'Copper': {'code': 'CU.SHF', 'source': 'tushare', 'type': 'future'},
    'USTBond': {'code': 'IEF.O', 'source': 'yfinance', 'type': 'etf'},
    'CNBond5Y': {'code': 'CBA00641.CS', 'source': 'tushare', 'type': 'bond'},
    'CNCorpAAA': {'code': 'CBA04201.CS', 'source': 'tushare', 'type': 'bond'},
}


class DataLoader:
    def __init__(self, start_date='20070101', end_date='20240930'):
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = {}
        self.returns_data = None

    def fetch_tushare_index(self, code, name):
        try:
            df = pro.index_daily(ts_code=code, start_date=self.start_date, end_date=self.end_date)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                self.price_data[name] = df['close']
                return True
        except Exception as e:
            print(f"Failed to get {name}: {e}")
        return False

    def fetch_tushare_bond(self, code, name):
        try:
            df = pro.index_daily(ts_code=code, start_date=self.start_date, end_date=self.end_date)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                self.price_data[name] = df['close']
                return True
        except Exception as e:
            print(f"Failed to get {name}: {e}")
        return False

    def fetch_yfinance_data(self, code, name):
        try:
            import yfinance as yf
            ticker = yf.Ticker(code)
            df = ticker.history(start=self.start_date[:4] + '-' + self.start_date[4:6] + '-' + self.start_date[6:],
                                end=self.end_date[:4] + '-' + self.end_date[4:6] + '-' + self.end_date[6:])
            if df is not None and len(df) > 0:
                self.price_data[name] = df['Close']
                return True
        except Exception as e:
            print(f"Failed to get {name} (yfinance): {e}")
        return False

    def fetch_tushare_future(self, code, name):
        try:
            df = pro.index_daily(ts_code=code, start_date=self.start_date, end_date=self.end_date)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                self.price_data[name] = df['close']
                return True
        except Exception as e:
            print(f"Failed to get {name}: {e}")
        return False

    def load_all_data(self):
        for name, config in ASSET_CONFIG.items():
            code = config['code']
            source = config['source']
            asset_type = config['type']

            if source == 'tushare':
                if asset_type == 'bond':
                    self.fetch_tushare_bond(code, name)
                else:
                    self.fetch_tushare_index(code, name)
            elif source == 'yfinance':
                self.fetch_yfinance_data(code, name)

        if len(self.price_data) == 0:
            print("Warning: No data retrieved!")
            return False
        return True

    def get_returns(self, freq='monthly'):
        prices_df = pd.DataFrame(self.price_data)
        prices_df = prices_df.dropna()

        if freq == 'daily':
            returns = prices_df.pct_change().dropna()
        elif freq == 'monthly':
            returns = prices_df.resample('M').last().pct_change().dropna()
        else:
            returns = prices_df.pct_change().dropna()

        self.returns_data = returns
        return returns

    def get_prices(self, freq='monthly'):
        prices_df = pd.DataFrame(self.price_data)
        prices_df = prices_df.dropna()

        if freq == 'monthly':
            prices_df = prices_df.resample('M').last()

        return prices_df

    def get_rolling_covariance(self, window=6):
        if self.returns_data is None:
            self.get_returns(freq='daily')

        cov_matrices = {}
        dates = self.returns_data.resample('M').last().index

        for i, date in enumerate(dates[window:]):
            start_idx = i
            end_idx = i + window
            cov_data = self.returns_data.iloc[start_idx * 22:(end_idx * 22 + 22)]
            cov_matrices[date] = cov_data.cov()

        return cov_matrices

    def get_rolling_correlation(self, window=6):
        if self.returns_data is None:
            self.get_returns(freq='daily')

        corr_matrices = {}
        dates = self.returns_data.resample('M').last().index

        for i, date in enumerate(dates[window:]):
            start_idx = i
            end_idx = i + window
            corr_data = self.returns_data.iloc[start_idx * 22:(end_idx * 22 + 22)]
            corr_matrices[date] = corr_data.corr()

        return corr_matrices

    def save_data(self, filepath='output/price_data.csv'):
        prices_df = pd.DataFrame(self.price_data)
        prices_df.to_csv(filepath)
        print(f"Data saved to: {filepath}")


if __name__ == '__main__':
    loader = DataLoader(start_date='20070101', end_date='20240930')
    loader.load_all_data()
    print(f"\nSuccessfully loaded {len(loader.price_data)} assets")
    print("\nAsset list:")
    for name in loader.price_data.keys():
        print(f"  - {name}")
