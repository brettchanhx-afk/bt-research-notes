import os
import pickle
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.config import (
    TUSHARE_TOKEN, TUSHARE_API_URL, ASSETS, DATA_DIR,
    BACKTEST_PARAMS
)

class DataFetcher:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.cache_dir = DATA_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.token = TUSHARE_TOKEN
        self.api_url = TUSHARE_API_URL
        self._init_tushare()

    def _init_tushare(self):
        self.pro = ts.pro_api(self.token)
        self.pro._DataApi__token = self.token
        self.pro._DataApi__http_url = self.api_url

    def _get_cache_path(self, asset_key):
        return os.path.join(self.cache_dir, f'{asset_key}_data.pkl')

    def _load_from_cache(self, asset_key):
        cache_path = self._get_cache_path(asset_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, asset_key, data):
        cache_path = self._get_cache_path(asset_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def fetch_index_data(self, ts_code, start_date, end_date):
        try:
            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['returns'] = df['close'].pct_change()
                return df[['close', 'returns']]
            return None
        except Exception as e:
            print(f"Error fetching index {ts_code}: {e}")
            return None

    def fetch_etf_data(self, ts_code, start_date, end_date):
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                api=self.pro,
                start_date=start_date,
                end_date=end_date,
                asset='O'
            )
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['returns'] = df['close'].pct_change()
                return df[['close', 'returns']]
            return None
        except Exception as e:
            print(f"Error fetching ETF {ts_code}: {e}")
            return None

    def fetch_future_data(self, ts_code, start_date, end_date):
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                api=self.pro,
                start_date=start_date,
                end_date=end_date,
                asset='FT'
            )
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['returns'] = df['close'].pct_change()
                return df[['close', 'returns']]
            return None
        except Exception as e:
            print(f"Error fetching future {ts_code}: {e}")
            return None

    def fetch_bond_index_data(self, ts_code, start_date, end_date):
        try:
            df = self.pro.bond_指数(
                index_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['returns'] = df['close'].pct_change()
                return df[['close', 'returns']]
            return None
        except Exception as e:
            print(f"Error fetching bond index {ts_code}: {e}")
            return self._fetch_bond_index_from_wind(ts_code, start_date, end_date)

    def _fetch_bond_index_from_wind(self, ts_code, start_date, end_date):
        print(f"Using alternative method for bond index {ts_code}")
        return None

    def fetch_asset_data(self, asset_key, start_date, end_date):
        if self.use_cache:
            cached_data = self._load_from_cache(asset_key)
            if cached_data is not None:
                return cached_data

        asset_info = ASSETS.get(asset_key)
        if not asset_info:
            print(f"Unknown asset key: {asset_key}")
            return None

        ts_code = asset_info['ts_code']
        asset_type = asset_info['type']

        if asset_type == 'index':
            data = self.fetch_index_data(ts_code, start_date, end_date)
        elif asset_type == 'etf':
            data = self.fetch_etf_data(ts_code, start_date, end_date)
        elif asset_type == 'future':
            data = self.fetch_future_data(ts_code, start_date, end_date)
        elif asset_type == 'bond_index':
            data = self.fetch_bond_index_data(ts_code, start_date, end_date)
        else:
            print(f"Unsupported asset type: {asset_type}")
            return None

        if data is not None and self.use_cache:
            self._save_to_cache(asset_key, data)

        return data

    def fetch_all_assets(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = BACKTEST_PARAMS['start_date']
        if end_date is None:
            end_date = BACKTEST_PARAMS['end_date']

        all_data = {}
        failed_assets = []

        for asset_key in ASSETS.keys():
            print(f"Fetching data for {asset_key}...")
            data = self.fetch_asset_data(asset_key, start_date, end_date)
            if data is not None:
                all_data[asset_key] = data
                print(f"  {asset_key}: {len(data)} records")
            else:
                failed_assets.append(asset_key)
                print(f"  {asset_key}: FAILED")

        if failed_assets:
            print(f"\nWarning: Failed to fetch data for: {failed_assets}")

        return all_data, failed_assets

    def get_monthly_returns(self, data_dict, start_date=None, end_date=None):
        monthly_returns = {}

        for asset_key, df in data_dict.items():
            if df is None or 'returns' not in df.columns:
                continue

            df_monthly = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns[asset_key] = df_monthly

        if not monthly_returns:
            return pd.DataFrame()

        result = pd.DataFrame(monthly_returns)
        result.index = result.index.to_period('M')

        if start_date:
            start_period = pd.Period(start_date, freq='M')
            result = result[result.index >= start_period]
        if end_date:
            end_period = pd.Period(end_date, freq='M')
            result = result[result.index <= end_period]

        return result

    def get_daily_returns(self, data_dict, start_date=None, end_date=None):
        daily_returns = {}

        for asset_key, df in data_dict.items():
            if df is None or 'returns' not in df.columns:
                continue
            daily_returns[asset_key] = df['returns']

        if not daily_returns:
            return pd.DataFrame()

        result = pd.DataFrame(daily_returns)

        if start_date:
            result = result[result.index >= start_date]
        if end_date:
            result = result[result.index <= end_date]

        return result


def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2007-01-01', end='2024-09-30', freq='B')
    n = len(dates)

    csi300 = pd.Series(np.cumsum(np.random.randn(n) * 0.02 + 0.0003), index=dates)
    hsi = pd.Series(np.cumsum(np.random.randn(n) * 0.015 + 0.0002), index=dates)
    nikkei = pd.Series(np.cumsum(np.random.randn(n) * 0.012 + 0.0002), index=dates)
    sp500 = pd.Series(np.cumsum(np.random.randn(n) * 0.013 + 0.0003), index=dates)
    gold = pd.Series(np.cumsum(np.random.randn(n) * 0.008 + 0.0004), index=dates)
    brent = pd.Series(np.cumsum(np.random.randn(n) * 0.02 + 0.0003), index=dates)
    copper = pd.Series(np.cumsum(np.random.randn(n) * 0.018 + 0.0002), index=dates)
    usbond = pd.Series(np.cumsum(np.random.randn(n) * 0.004 + 0.0001), index=dates)
    cnbond = pd.Series(np.cumsum(np.random.randn(n) * 0.003 + 0.0001), index=dates)
    cncorp = pd.Series(np.cumsum(np.random.randn(n) * 0.004 + 0.0001), index=dates)

    data_dict = {
        'CSI300': pd.DataFrame({'close': np.exp(csi300), 'returns': np.exp(csi300).pct_change()}),
        'HSI': pd.DataFrame({'close': np.exp(hsi), 'returns': np.exp(hsi).pct_change()}),
        'Nikkei225': pd.DataFrame({'close': np.exp(nikkei), 'returns': np.exp(nikkei).pct_change()}),
        'SP500': pd.DataFrame({'close': np.exp(sp500), 'returns': np.exp(sp500).pct_change()}),
        'Gold': pd.DataFrame({'close': np.exp(gold), 'returns': np.exp(gold).pct_change()}),
        'BrentOil': pd.DataFrame({'close': np.exp(brent), 'returns': np.exp(brent).pct_change()}),
        'Copper': pd.DataFrame({'close': np.exp(copper), 'returns': np.exp(copper).pct_change()}),
        'USTBond': pd.DataFrame({'close': np.exp(usbond), 'returns': np.exp(usbond).pct_change()}),
        'CNTBond': pd.DataFrame({'close': np.exp(cnbond), 'returns': np.exp(cnbond).pct_change()}),
        'CNCorpBond': pd.DataFrame({'close': np.exp(cncorp), 'returns': np.exp(cncorp).pct_change()}),
    }

    for key in data_dict:
        data_dict[key] = data_dict[key].iloc[1:]

    return data_dict


CSV_ASSET_MAPPING = {
    '沪深300': 'CSI300',
    '恒生指数': 'HSI',
    '标普500': 'SP500',
    '日经225': 'Nikkei225',
    '布伦特原油连续': 'BrentOil',
    '沪铜连续': 'Copper',
    '中债企业债总指数(总值)财富指数': 'CNCorpBond',
    '中债国债总指数(总值)全价指数': 'CNTBond',
    '黄金': 'Gold',
    '美国国债0-5年期债券': 'USTBond',
}


def load_csv_data(file_path):
    df = pd.read_csv(file_path, header=1)

    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        df = df.rename(columns={unnamed_cols[0]: 'trade_date'})

    existing_cols = [c for c in CSV_ASSET_MAPPING.keys() if c in df.columns]

    df = df[['trade_date'] + existing_cols].copy()
    df.columns = ['trade_date'] + [CSV_ASSET_MAPPING[c] for c in existing_cols]

    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date'])
    df = df.set_index('trade_date')

    data_dict = {}
    for col in df.columns:
        prices = pd.to_numeric(df[col], errors='coerce')
        valid_mask = prices.notna() & (prices != 0) & (prices > 0)
        prices = prices[valid_mask]
        if len(prices) > 0:
            returns = prices.pct_change()
            data_dict[col] = pd.DataFrame({
                'close': prices,
                'returns': returns
            })

    return data_dict


if __name__ == "__main__":
    print("Initializing DataFetcher...")
    fetcher = DataFetcher(use_cache=True)

    print("\nFetching real market data...")
    all_data, failed = fetcher.fetch_all_assets()

    if failed:
        print(f"\nSome assets failed. Using sample data for: {failed}")
        sample_data = create_sample_data()
        for key in failed:
            if key in sample_data:
                all_data[key] = sample_data[key]

    print(f"\nSuccessfully fetched data for {len(all_data)} assets")

    monthly_returns = fetcher.get_monthly_returns(all_data)
    print(f"\nMonthly returns shape: {monthly_returns.shape}")
    print(monthly_returns.tail())
