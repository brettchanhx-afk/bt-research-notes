import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("Warning: akshare not installed. Some data may not be available.")

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("Warning: tushare not installed.")

DATA_DIR = "data"
OUTPUT_DIR = "output"
START_DATE = "20160430"
END_DATE = "20230531"

def init_tushare():
    if not TUSHARE_AVAILABLE:
        return None
    token = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
    pro = ts.pro_api(token)
    pro._DataApi__token = token
    pro._DataApi__http_url = "http://jiaoch.site"
    return pro

PRO = init_tushare()

class DataFetcher:
    def __init__(self, pro=None):
        self.pro = pro or PRO
        self.data_dir = DATA_DIR
        self.use_akshare = AKSHARE_AVAILABLE
        os.makedirs(self.data_dir, exist_ok=True)

    def get_index_daily(self, ts_code, start_date=None, end_date=None):
        if self.pro is None:
            return pd.DataFrame()
        try:
            code = ts_code.replace(".SH", "").replace(".SZ", "").replace(".SI", "")
            df = self.pro.index_daily(ts_code=code)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df = df.set_index('trade_date')
                start = pd.to_datetime(start_date) if start_date else None
                end = pd.to_datetime(end_date) if end_date else None
                if start:
                    df = df[df.index >= start]
                if end:
                    df = df[df.index <= end]
                return df
        except Exception as e:
            print(f"Error fetching {ts_code}: {e}")
        return pd.DataFrame()

    def get_sw_industry_daily(self, start_date=None, end_date=None):
        if self.pro is None:
            return pd.DataFrame()
        try:
            df = self.pro.sw_index_daily(start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                return df
        except Exception as e:
            print(f"Error fetching sw industry data: {e}")
        return pd.DataFrame()

    def get_akshare_sw_industry(self):
        if not self.use_akshare:
            print("akshare not available for sw industry data")
            return pd.DataFrame()
        try:
            df = ak.sw_index_daily(symbol="L1")
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(['index_code', 'date'])
                return df
        except Exception as e:
            print(f"Error fetching akshare sw industry: {e}")
        return pd.DataFrame()

    def get_akshare_bond_zh(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.bond_zh_name()
            return df
        except Exception as e:
            print(f"Error fetching bond data: {e}")
        return pd.DataFrame()

    def get_akshare_macro_cnzb(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.macro_cnzb()
            return df
        except Exception as e:
            print(f"Error fetching macro data: {e}")
        return pd.DataFrame()

    def get_akshare_cn_gdp(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.cn_gdp()
            return df
        except Exception as e:
            print(f"Error fetching GDP data: {e}")
        return pd.DataFrame()

    def get_akshare_cpi(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.cpi()
            return df
        except Exception as e:
            print(f"Error fetching CPI data: {e}")
        return pd.DataFrame()

    def get_akshare_ppi(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.ppi()
            return df
        except Exception as e:
            print(f"Error fetching PPI data: {e}")
        return pd.DataFrame()

    def get_akshare_stock_zh_a(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.stock_zh_a_spot_em()
            return df
        except Exception as e:
            print(f"Error fetching A-share data: {e}")
        return pd.DataFrame()

    def get_akshare_gold(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.spot_gold()
            return df
        except Exception as e:
            print(f"Error fetching gold data: {e}")
        return pd.DataFrame()

    def get_akshare_oil(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.spot_oil()
            return df
        except Exception as e:
            print(f"Error fetching oil data: {e}")
        return pd.DataFrame()

    def get_akshare_interest_rate(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.interest_rate_china()
            return df
        except Exception as e:
            print(f"Error fetching interest rate: {e}")
        return pd.DataFrame()

    def get_akshare_cn_curency(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.currency()
            return df
        except Exception as e:
            print(f"Error fetching currency data: {e}")
        return pd.DataFrame()

    def get_stock_basic(self):
        if self.pro is None:
            return pd.DataFrame()
        try:
            df = self.pro.stock_basic(exchange='', list_status='L')
            return df
        except Exception as e:
            print(f"Error fetching stock basic: {e}")
        return pd.DataFrame()

    def get_daily_basic(self, ts_code, start_date=None, end_date=None):
        if self.pro is None:
            return pd.DataFrame()
        try:
            df = self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df = df.set_index('trade_date')
                return df
        except Exception as e:
            print(f"Error fetching daily basic for {ts_code}: {e}")
        return pd.DataFrame()

    def get_financial_data(self, ts_code, start_date=None, end_date=None):
        if self.pro is None:
            return pd.DataFrame()
        try:
            df = self.pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['ann_date'] = pd.to_datetime(df['ann_date'])
                df = df.sort_values('ann_date')
                return df
        except Exception as e:
            print(f"Error fetching financial data for {ts_code}: {e}")
        return pd.DataFrame()

    def get_industry_indices_tushare(self, start_date=None, end_date=None):
        sw_codes = [
            "801010.SI", "801020.SI", "801030.SI", "801040.SI", "801050.SI",
            "801060.SI", "801080.SI", "801110.SI", "801120.SI", "801130.SI",
            "801140.SI", "801150.SI", "801160.SI", "801170.SI", "801180.SI",
            "801200.SI", "801210.SI", "801230.SI", "801710.SI", "801720.SI",
            "801730.SI", "801740.SI", "801750.SI", "801760.SI", "801770.SI",
            "801780.SI", "801790.SI", "801880.SI", "801890.SI"
        ]

        all_data = []
        for code in sw_codes:
            if self.pro is None:
                break
            try:
                df = self.pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
                if df is not None and len(df) > 0:
                    df['ts_code'] = code
                    all_data.append(df)
            except Exception as e:
                print(f"Error fetching {code}: {e}")

        if all_data:
            result = pd.concat(all_data, ignore_index=False)
            result['trade_date'] = pd.to_datetime(result['trade_date'])
            result = result.sort_values(['ts_code', 'trade_date'])
            return result
        return pd.DataFrame()

    def get_industry_indices_akshare(self):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.sw_index_daily(symbol="L1")
            return df
        except Exception as e:
            print(f"Error fetching sw industry via akshare: {e}")
        return pd.DataFrame()

    def get_akshare_index_daily(self, symbol, period="daily", start_date=None, end_date=None):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df = df.set_index('date')
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                return df
        except Exception as e:
            print(f"Error fetching index {symbol}: {e}")
        return pd.DataFrame()

    def get_market_cap(self, ts_code, start_date=None, end_date=None):
        if self.pro is None:
            return pd.DataFrame()
        try:
            df = self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                return df[['trade_date', 'total_mv', 'cir_mv']].set_index('trade_date')
        except Exception as e:
            print(f"Error fetching market cap for {ts_code}: {e}")
        return pd.DataFrame()

    def save_data(self, df, filename):
        if df is not None and len(df) > 0:
            filepath = os.path.join(self.data_dir, f"{filename}.csv")
            df.to_csv(filepath)
            print(f"Data saved to {filepath}")
            return filepath
        return None

    def load_data(self, filename):
        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                return df
            except:
                df = pd.read_csv(filepath)
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df = df.set_index('trade_date')
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                return df
        return pd.DataFrame()

def fetch_all_data(start_date=None, end_date=None):
    fetcher = DataFetcher(pro=PRO)
    start = start_date or START_DATE
    end = end_date or END_DATE

    print("Fetching industry index data via Tushare...")
    industry_df = fetcher.get_industry_indices_tushare(start, end)
    if len(industry_df) > 0:
        fetcher.save_data(industry_df, "industry_indices_tushare")

    print("Fetching industry index data via Akshare...")
    industry_ak_df = fetcher.get_industry_indices_akshare()
    if len(industry_ak_df) > 0:
        fetcher.save_data(industry_ak_df, "industry_indices_akshare")

    print("Fetching macro indicators...")
    try:
        cpi_df = fetcher.get_akshare_cpi()
        if len(cpi_df) > 0:
            fetcher.save_data(cpi_df, "macro_cpi")
    except Exception as e:
        print(f"CPI data fetch failed: {e}")

    try:
        ppi_df = fetcher.get_akshare_ppi()
        if len(ppi_df) > 0:
            fetcher.save_data(ppi_df, "macro_ppi")
    except Exception as e:
        print(f"PPI data fetch failed: {e}")

    try:
        gdp_df = fetcher.get_akshare_cn_gdp()
        if len(gdp_df) > 0:
            fetcher.save_data(gdp_df, "macro_gdp")
    except Exception as e:
        print(f"GDP data fetch failed: {e}")

    try:
        interest_df = fetcher.get_akshare_interest_rate()
        if len(interest_df) > 0:
            fetcher.save_data(interest_df, "macro_interest_rate")
    except Exception as e:
        print(f"Interest rate data fetch failed: {e}")

    return fetcher

if __name__ == "__main__":
    print("Testing data fetcher...")
    fetcher = DataFetcher(pro=PRO)
    print(f"Tushare available: {TUSHARE_AVAILABLE}")
    print(f"Akshare available: {AKSHARE_AVAILABLE}")
    print(f"Data directory: {fetcher.data_dir}")
