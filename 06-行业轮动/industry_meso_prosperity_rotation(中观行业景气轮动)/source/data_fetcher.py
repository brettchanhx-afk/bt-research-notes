import pandas as pd
import numpy as np
import tushare as ts
from typing import Dict, List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.config import (
    TUSHARE_TOKEN, TUSHARE_API_URL, CITIC_CODES,
    CITIC_INDUSTRIES, OUTPUT_DIR, DATA_DIR
)
from source.utils import ensure_dir


class DataFetcher:
    def __init__(self):
        self.token = TUSHARE_TOKEN
        self.api = ts.pro_api(self.token)
        self.api._DataApi__token = self.token
        self.api._DataApi__http_url = TUSHARE_API_URL
        ensure_dir(DATA_DIR)
        ensure_dir(OUTPUT_DIR)

    def get_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.api.trade_cal(start_date=start_date, end_date=end_date)
        df = df[df['is_open'] == 1]
        return df

    def get_month_end_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        df = self.get_trade_cal(start_date, end_date)
        df['cal_date'] = pd.to_datetime(df['cal_date'])
        month_ends = df.groupby(df['cal_date'].dt.to_period('M'))['cal_date'].last()
        return pd.DatetimeIndex(month_ends.values)

    def get_index_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        df = self.api.index_daily(
            ts_code=ts_code,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', '')
        )
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df = df.set_index('trade_date')
        return df

    def get_index_monthly(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        df = self.api.index_monthly(
            ts_code=ts_code,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', '')
        )
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df = df.set_index('trade_date')
        return df

    def get_industry_daily(
        self,
        industry_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        try:
            df = self.api.index_daily(
                ts_code=industry_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df = df.set_index('trade_date')
                return df
        except Exception as e:
            print(f"Error fetching industry {industry_code}: {e}")
        return pd.DataFrame()

    def get_sw_industry_daily(
        self,
        level: str = "L1",
        start_date: str = "20160401",
        end_date: str = "20220630"
    ) -> Dict[str, pd.DataFrame]:
        industry_data = {}
        try:
            df = self.api.sw_daily(
                start_date=start_date,
                end_date=end_date,
                level=level
            )
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                for name, group in df.groupby('name'):
                    group = group.sort_values('trade_date')
                    industry_data[name] = group.set_index('trade_date')
        except Exception as e:
            print(f"Error fetching SW industry data: {e}")
        return industry_data

    def get_sw_industry_list(self, level: str = "L1") -> pd.DataFrame:
        try:
            df = self.api.sw_daily(start_date='', end_date='', level=level)
            if df is not None:
                return df[['code', 'name', 'level', 'list_date']].drop_duplicates()
        except Exception as e:
            print(f"Error fetching SW industry list: {e}")
        return pd.DataFrame()

    def get_citici_industry_index(
        self,
        industry_name: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        citic_code = CITIC_CODES.get(industry_name)
        if not citic_code:
            print(f"Unknown industry: {industry_name}")
            return pd.DataFrame()

        return self.get_index_monthly(
            ts_code=citic_code,
            start_date=start_date,
            end_date=end_date
        )

    def get_financial_data(
        self,
        ts_code: str = None,
        start_date: str = None,
        end_date: str = None,
        indicator: str = None
    ) -> pd.DataFrame:
        try:
            if indicator:
                df = self.api.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    indicator=indicator
                )
            else:
                df = self.api.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            if df is not None and len(df) > 0:
                df['ann_date'] = pd.to_datetime(df['ann_date'])
                df = df.sort_values(['ts_code', 'ann_date'])
            return df
        except Exception as e:
            print(f"Error fetching financial data: {e}")
        return pd.DataFrame()

    def get_industry_financial(
        self,
        industry_name: str,
        start_year: int = 2015,
        end_year: int = 2022
    ) -> pd.DataFrame:
        results = []
        for year in range(start_year, end_year + 1):
            for quarter in [1, 2, 3, 4]:
                try:
                    df = self.api.cndawork(
                        type='行业财务指标',
                        year=str(year),
                        quarter=str(quarter)
                    )
                    if df is not None and len(df) > 0:
                        results.append(df)
                except:
                    continue
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_macro_data(
        self,
        indicator_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        try:
            df = self.api.cn_data(
                indicator_code=indicator_code,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 0:
                df['period'] = pd.to_datetime(df['period'])
                df = df.sort_values('period')
                df = df.set_index('period')
            return df
        except Exception as e:
            print(f"Error fetching macro data {indicator_code}: {e}")
        return pd.DataFrame()

    def get_gdp_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        try:
            df = self.api.cn_gdp(start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['period'] = pd.to_datetime(df['period'])
                df = df.sort_values('period')
                df = df.set_index('period')
            return df
        except Exception as e:
            print(f"Error fetching GDP data: {e}")
        return pd.DataFrame()

    def get_ppi_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        try:
            df = self.api.ppi(start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['month'] = pd.to_datetime(df['month'])
                df = df.sort_values('month')
                df = df.set_index('month')
            return df
        except Exception as e:
            print(f"Error fetching PPI data: {e}")
        return pd.DataFrame()

    def get_pmi_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        try:
            df = self.api.cn_pmi(start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                if 'month' in df.columns:
                    df['month'] = pd.to_datetime(df['month'])
                    df = df.sort_values('month')
                    df = df.set_index('month')
            return df
        except Exception as e:
            print(f"Error fetching PMI data: {e}")
        return pd.DataFrame()

    def calculate_industry_metrics(
        self,
        industry_name: str,
        stock_list: List[str]
    ) -> pd.DataFrame:
        all_metrics = []
        for ts_code in stock_list:
            try:
                df = self.get_financial_data(
                    ts_code=ts_code,
                    start_date='20150101',
                    end_date='20220630'
                )
                if df is not None and len(df) > 0:
                    df['industry'] = industry_name
                    all_metrics.append(df)
            except Exception as e:
                continue

        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        return pd.DataFrame()

    def fetch_all_industry_index_data(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        industry_data = {}
        for industry_name in CITIC_INDUSTRIES:
            print(f"Fetching data for {industry_name}...")
            df = self.get_citici_industry_index(industry_name, start_date, end_date)
            if df is not None and len(df) > 0:
                industry_data[industry_name] = df
        return industry_data

    def save_industry_data(self, industry_data: Dict[str, pd.DataFrame]) -> None:
        for industry_name, df in industry_data.items():
            filepath = os.path.join(DATA_DIR, f"{industry_name}.parquet")
            df.to_parquet(filepath)
            print(f"Saved {industry_name} data to {filepath}")

    def load_industry_data(self, industry_name: str) -> pd.DataFrame:
        filepath = os.path.join(DATA_DIR, f"{industry_name}.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return pd.DataFrame()


def main():
    fetcher = DataFetcher()
    print("Testing data fetcher...")

    dates = fetcher.get_month_end_dates("2016-04-01", "2022-06-30")
    print(f"Got {len(dates)} month-end dates")

    test_df = fetcher.get_pmi_data("2016-01-01", "2022-06-30")
    print(f"PMI data shape: {test_df.shape}")


if __name__ == "__main__":
    main()
