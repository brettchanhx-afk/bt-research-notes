import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class ETFFunds:
    def __init__(self, data_loader):
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE
        self.dl = data_loader

    def get_etf_list(self):
        try:
            df = self.pro.fund_basic(
                market="E",
                status="L",
                fields="ts_code, name, management, found_date, issue_date"
            )
            return df
        except Exception as e:
            print(f"Error fetching ETF list: {e}")
            return pd.DataFrame()

    def get_etf_daily(self, ts_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.fund_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching ETF daily for {ts_code}: {e}")
            return pd.DataFrame()

    def get_etf_nav(self, ts_code):
        try:
            df = self.pro.fund_nav(
                ts_code=ts_code
            )
            return df
        except Exception as e:
            print(f"Error fetching ETF NAV for {ts_code}: {e}")
            return pd.DataFrame()

    def get_etf_owners(self, ts_code):
        try:
            df = self.pro.fund_owners(
                ts_code=ts_code
            )
            return df
        except Exception as e:
            print(f"Error fetching ETF owners for {ts_code}: {e}")
            return pd.DataFrame()

    def calculate_etf_flow(self, ts_code, start_date=None, end_date=None):
        df = self.get_etf_daily(ts_code, start_date, end_date)

        if len(df) == 0:
            return pd.DataFrame()

        df = df.sort_values("trade_date")

        if "nav" not in df.columns or "nav" not in df.columns:
            return pd.DataFrame()

        df["share_change"] = df["share"].diff()
        df["flow"] = df["share_change"] * df["nav"]

        return df

    def aggregate_market_etf_flow(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        etf_list = self.get_etf_list()
        if len(etf_list) == 0:
            return pd.DataFrame()

        result = []

        for _, row in etf_list.iterrows():
            ts_code = row["ts_code"]
            df = self.calculate_etf_flow(ts_code, start_date, end_date)

            if len(df) > 0:
                df["etf_code"] = ts_code
                df["etf_name"] = row.get("name", "")
                result.append(df)

        if len(result) > 0:
            return pd.concat(result, ignore_index=True)
        return pd.DataFrame()

    def calculate_etf_flow_by_period(self, df, freq="W"):
        if len(df) == 0:
            return pd.DataFrame()

        df = df.sort_values("trade_date")

        if freq == "W":
            df["period"] = df["trade_date"].dt.to_period("W")
        elif freq == "M":
            df["period"] = df["trade_date"].dt.to_period("M")
        else:
            df["period"] = df["trade_date"]

        agg_df = df.groupby("period").agg({
            "flow": "sum",
            "share": "last",
            "nav": "last",
            "close": "last"
        }).reset_index()

        return agg_df

    def filter_trend_etf_flow(self, df, lookback=5):
        if len(df) == 0 or len(df) < lookback:
            return df

        df = df.sort_values("trade_date")
        df["price_change"] = df["close"].pct_change()
        df["share_change_pct"] = df["share"].pct_change()

        df["trend_flow"] = np.where(
            (df["share_change_pct"] > 0) & (df["price_change"] > 0),
            df["flow"],
            0
        )

        df["trend_flow"] = df["trend_flow"].rolling(window=lookback).sum()

        return df

    def normalize_etf_flow(self, df, market_turnover_df):
        if len(df) == 0 or len(market_turnover_df) == 0:
            return df

        df = df.merge(market_turnover_df, on="period", how="left")
        df["flow_norm"] = df["flow"] / df["turnover"]
        return df

    def normalize_etf_flow_by_market_cap(self, df, market_cap_df):
        if len(df) == 0 or len(market_cap_df) == 0:
            return df

        df = df.merge(market_cap_df, on="period", how="left")
        df["flow_norm_mc"] = df["flow"] / df["market_cap"]
        return df

    def calculate_yoy_change(self, df, value_col="flow"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_yoy"] = df[value_col].pct_change(periods=12)
        return df

    def calculate_qoq_change(self, df, value_col="flow"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_qoq"] = df[value_col].pct_change(periods=1)
        return df

    def build_etf_indicators(self, freq="W", trend_only=False):
        indicators = []

        df = self.aggregate_market_etf_flow(self.start_date, self.end_date)

        if len(df) == 0:
            return indicators

        df_period = self.calculate_etf_flow_by_period(df, freq=freq)

        if trend_only:
            df_period = self.filter_trend_etf_flow(df_period)

        if len(df_period) > 0:
            df_orig = df_period.copy()
            df_orig["indicator_name"] = f"allETF_orig_{freq}_orig"

            df_yoy = self.calculate_yoy_change(df_period.copy(), "flow")
            df_yoy["indicator_name"] = f"allETF_orig_{freq}_yoy"

            df_qoq = self.calculate_qoq_change(df_period.copy(), "flow")
            df_qoq["indicator_name"] = f"allETF_orig_{freq}_qoq"

            indicators.extend([df_orig, df_yoy, df_qoq])

        return indicators

    def get_sector_etf_list(self):
        try:
            df = self.pro.fund_basic(
                market="E",
                status="L",
                fields="ts_code, name, management, found_date"
            )
            sector_etfs = df[df["name"].str.contains("行业|主题|指数", na=False)]
            return sector_etfs
        except Exception as e:
            print(f"Error fetching sector ETF list: {e}")
            return pd.DataFrame()

    def get_industry_etf_flow(self, industry_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        etf_list = self.get_sector_etf_list()
        if len(etf_list) == 0:
            return pd.DataFrame()

        result = []

        for _, row in etf_list.iterrows():
            ts_code = row["ts_code"]

            if industry_code.lower() in row["name"].lower():
                df = self.calculate_etf_flow(ts_code, start_date, end_date)
                if len(df) > 0:
                    df["industry_code"] = industry_code
                    result.append(df)

        if len(result) > 0:
            return pd.concat(result, ignore_index=True)
        return pd.DataFrame()

    def calculate_etf_flow_statistics(self, df, freq="W"):
        if len(df) == 0:
            return pd.DataFrame()

        df = df.sort_values("trade_date")

        if freq == "W":
            df["period"] = df["trade_date"].dt.to_period("W")
        elif freq == "M":
            df["period"] = df["trade_date"].dt.to_period("M")
        else:
            df["period"] = df["trade_date"]

        stats_df = df.groupby("period").agg({
            "flow": ["sum", "mean", "std", "count"],
            "share": "last"
        }).reset_index()

        stats_df.columns = ["period", "flow_sum", "flow_mean", "flow_std", "flow_count", "share_last"]

        return stats_df