import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
pro = ts.pro_api(TOKEN)
pro._DataApi__token = TOKEN
pro._DataApi__http_url = "http://jiaoch.site"


class DataFetcher:
    def __init__(self):
        self.token = TOKEN
        self.pro = pro
        self.cache = {}
        self.multi_asset_data = None

    def load_multi_asset_data(self, file_path='data/多资产行情序列.csv'):
        if self.multi_asset_data is not None:
            return self.multi_asset_data

        try:
            df = pd.read_csv(file_path, index_col=0, encoding='gbk', header=0)
            df = df.iloc[2:]
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.replace('--', np.nan)
            df = df.astype(float)
            returns = df.pct_change()
            returns = returns.dropna(how='all')
            self.multi_asset_data = returns
            return returns
        except Exception as e:
            print(f"Error loading multi-asset data: {e}")
            return pd.DataFrame()

    def get_index_daily(self, ts_code, start_date=None, end_date=None):
        cache_key = f"index_daily_{ts_code}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            if start_date is None:
                start_date = '20100101'
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')

            df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['returns'] = df['close'].pct_change()
                self.cache[cache_key] = df
                return df
        except Exception as e:
            print(f"Error fetching {ts_code}: {e}")
        return pd.DataFrame()

    def get_bond_index(self, index_code, start_date=None, end_date=None):
        cache_key = f"bond_index_{index_code}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            if start_date is None:
                start_date = '20100101'
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')

            df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['returns'] = df['close'].pct_change()
                self.cache[cache_key] = df
                return df
        except Exception as e:
            print(f"Error fetching bond index {index_code}: {e}")
        return pd.DataFrame()

    def get_commodity_index(self, index_code, start_date=None, end_date=None):
        try:
            import akshare as ak
            if start_date is None:
                start_date = '2010-01-01'
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            df = ak.index_zh_a_hist(symbol=index_code, period='daily',
                                    start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df.columns = ['trade_date', 'open', 'close', 'high', 'low', 'volume', 'amount']
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()
                df['returns'] = df['close'].pct_change()
                return df
        except Exception as e:
            print(f"Error fetching commodity index {index_code}: {e}")
        return pd.DataFrame()

    def get_futures_index(self, index_code, start_date=None, end_date=None):
        try:
            import akshare as ak
            if start_date is None:
                start_date = '2010-01-01'
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            df = ak.get_reits(index_code)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()
                df['returns'] = df['close'].pct_change()
                return df
        except:
            pass

        try:
            df = self.get_index_daily(index_code, start_date.replace('-', ''), end_date.replace('-', ''))
            return df
        except Exception as e:
            print(f"Error fetching futures index {index_code}: {e}")
        return pd.DataFrame()

    def get_industry_index(self, industry_code, start_date=None, end_date=None):
        try:
            if start_date is None:
                start_date = '20100101'
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')

            df = self.pro.index_daily(ts_code=industry_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['returns'] = df['close'].pct_change()
        except Exception as e:
            print(f"Error fetching industry index {industry_code}: {e}")
        return pd.DataFrame()

    def get_macro_data(self, indicator_name, start_date=None, end_date=None):
        try:
            macro_map = {
                'gdp': 'M00803000000000003',
                'cpi': 'M00194000000000003',
                'ppi': 'M00631000000000003',
            }
            indicator_code = macro_map.get(indicator_name, indicator_name)

            df = self.pro.cn_gdp(indicator=indicator_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['period'] = pd.to_datetime(df['period'])
                df = df.set_index('period').sort_index()
                return df
        except Exception as e:
            print(f"Error fetching macro data {indicator_name}: {e}")
        return pd.DataFrame()

    def fetch_asset_returns(self, asset_config, start_date=None, end_date=None):
        results = {}
        asset_map = {
            '沪深300': '000300.SH',
            '中证1000': '000852.SH',
            '恒生指数': 'HSI.HK',
            '标普500': 'SPX.GI',
            '中债-国债总财富': 'CBA00101.CI',
            '中债-企业债总财富': 'CBA00201.CI',
            '南华商品指数': 'NH0100.NH',
            'COMEX黄金': 'GC00Y.GC',
            'ICE布油': 'CL00Y.NYM',
            '美元指数': 'DX0001.DXY',
        }

        for asset_name, ts_code in asset_config.items():
            if '国债' in asset_name or '企业债' in asset_name:
                df = self.get_bond_index(ts_code, start_date, end_date)
            elif '南华' in asset_name or '黄金' in asset_name or '布油' in asset_name or '美元' in asset_name:
                if '黄金' in asset_name:
                    df = self.get_futures_index('GC00Y.GC', start_date, end_date)
                elif '布油' in asset_name:
                    df = self.get_futures_index('CL00Y.NYM', start_date, end_date)
                elif '美元' in asset_name:
                    df = self.get_futures_index('DX0001.DXY', start_date, end_date)
                else:
                    df = self.get_commodity_index(ts_code, start_date, end_date)
            else:
                df = self.get_index_daily(ts_code, start_date, end_date)

            if len(df) > 0:
                results[asset_name] = df['returns'].dropna()

        if results:
            returns_df = pd.DataFrame(results)
            returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
            returns_df = returns_df.dropna()
            return returns_df

        return pd.DataFrame()

    def fetch_industry_returns(self, industry_codes, start_date=None, end_date=None):
        results = {}
        for industry_name, ts_code in industry_codes.items():
            df = self.get_industry_index(ts_code, start_date, end_date)
            if len(df) > 0:
                results[industry_name] = df['returns'].dropna()

        if results:
            returns_df = pd.DataFrame(results)
            returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
            returns_df = returns_df.dropna()
            return returns_df

        return pd.DataFrame()

    def get_trading_days(self, start_date, end_date):
        try:
            df = self.pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
            if df is not None:
                trading_days = df[df['is_open'] == 1]['trade_date'].tolist()
                return [pd.to_datetime(d) for d in trading_days]
        except Exception as e:
            print(f"Error getting trading days: {e}")
        return []

    def resample_to_monthly(self, daily_returns):
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly_returns

    def resample_to_weekly(self, daily_returns):
        weekly_returns = daily_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
        return weekly_returns

    def calculate_rolling_covariance(self, returns, lookback):
        return returns.rolling(window=lookback).cov()

    def clear_cache(self):
        self.cache = {}