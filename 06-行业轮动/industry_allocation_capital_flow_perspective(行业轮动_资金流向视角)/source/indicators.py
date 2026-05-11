import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class IndicatorCalculator:
    def __init__(self, data_loader):
        self.dl = data_loader
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE

    def get_industry_list(self, level="L1"):
        try:
            df = self.pro.index_classify(src="SW2021", level=level)
            return df
        except Exception as e:
            print(f"Error fetching industry list: {e}")
            return pd.DataFrame()

    def get_industry_members(self, industry_code):
        try:
            df = self.pro.index_member(ts_code=industry_code)
            return df
        except Exception as e:
            print(f"Error fetching industry members for {industry_code}: {e}")
            return pd.DataFrame()

    def get_industry_daily_value(self, industry_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.index_daily(
                ts_code=industry_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching industry daily for {industry_code}: {e}")
            return pd.DataFrame()

    def calculate_industry_returns(self, industry_df):
        if len(industry_df) == 0:
            return pd.DataFrame()

        industry_df = industry_df.sort_values("trade_date")
        industry_df["return"] = industry_df["close"].pct_change()
        return industry_df

    def aggregate_to_period(self, df, freq="W"):
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
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "vol": "sum",
            "amount": "sum"
        }).reset_index()

        agg_df["return"] = agg_df["close"].pct_change()
        return agg_df

    def normalize_indicator(self, df, value_col, method="zscore"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        if method == "zscore":
            mean = df[value_col].mean()
            std = df[value_col].std()
            df[f"{value_col}_norm"] = (df[value_col] - mean) / std
        elif method == "minmax":
            min_val = df[value_col].min()
            max_val = df[value_col].max()
            df[f"{value_col}_norm"] = (df[value_col] - min_val) / (max_val - min_val)
        elif method == "rank":
            df[f"{value_col}_norm"] = df[value_col].rank(pct=True)

        return df

    def calculate_quantile(self, df, value_col, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df[f"{value_col}_quantile"] = pd.qcut(
            df[value_col],
            q=[0] + quantiles + [1],
            labels=[1, 2, 3, 4, 5],
            duplicates="drop"
        )
        return df

    def merge_industry_indicator(self, industry_df, indicator_df, on_col="period"):
        if len(industry_df) == 0 or len(indicator_df) == 0:
            return pd.DataFrame()

        merged_df = industry_df.merge(indicator_df, on=on_col, how="left")
        return merged_df

    def calculate_period_return(self, df, periods=[1, 5, 20]):
        if len(df) == 0:
            return df

        df = df.sort_values("trade_date")

        for period in periods:
            df[f"return_{period}d"] = df["close"].pct_change(periods=period)

        return df

    def calculate_rolling_stats(self, df, value_col, windows=[5, 10, 20]):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("trade_date")

        for window in windows:
            df[f"{value_col}_ma{window}"] = df[value_col].rolling(window=window).mean()
            df[f"{value_col}_std{window}"] = df[value_col].rolling(window=window).std()

        return df

    def calculate_cumulative_return(self, df, period_col="period"):
        if len(df) == 0 or "return" not in df.columns:
            return df

        df = df.sort_values(period_col)
        df["cumulative_return"] = (1 + df["return"]).cumprod() - 1
        return df

    def build_industry_matrix(self, industry_list, date_col="trade_date"):
        result = []

        for code in industry_list:
            df = self.get_industry_daily_value(code)
            if len(df) > 0:
                df["industry_code"] = code
                result.append(df)

        if len(result) > 0:
            matrix_df = pd.concat(result, ignore_index=True)
            return matrix_df.pivot_table(
                index=date_col,
                columns="industry_code",
                values="return"
            )
        return pd.DataFrame()

    def calculate_turnover(self, industry_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.index_daily(
                ts_code=industry_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
                df["turnover"] = df["amount"] / df["close"]
            return df
        except Exception as e:
            print(f"Error calculating turnover for {industry_code}: {e}")
            return pd.DataFrame()

    def get_market_cap(self, ts_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields="trade_date, ts_code, total_mv, circ_mv"
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching market cap for {ts_code}: {e}")
            return pd.DataFrame()

    def calculate_industry_market_cap(self, industry_members_df, start_date=None, end_date=None):
        if len(industry_members_df) == 0:
            return pd.DataFrame()

        result = []

        for _, row in industry_members_df.iterrows():
            ts_code = row.get("con_code", "")
            if not ts_code:
                continue

            df = self.get_market_cap(ts_code, start_date, end_date)
            if len(df) > 0:
                df["industry_code"] = row.get("index_code", "")
                result.append(df)

        if len(result) > 0:
            result_df = pd.concat(result, ignore_index=True)
            agg_df = result_df.groupby(["trade_date", "industry_code"]).agg({
                "total_mv": "sum",
                "circ_mv": "sum"
            }).reset_index()
            return agg_df
        return pd.DataFrame()

    def filter_by_threshold(self, df, value_col, threshold, direction="above"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        if direction == "above":
            return df[df[value_col] >= threshold]
        elif direction == "below":
            return df[df[value_col] <= threshold]
        return df

    def rank_industries(self, df, value_col, ascending=False):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df[f"{value_col}_rank"] = df[value_col].rank(ascending=ascending, pct=True)
        return df

    def calculate_quantile_groups(self, df, value_col, n_groups=5):
        if len(df) == 0 or value_col not in df.columns:
            return df

        try:
            df[f"{value_col}_group"] = pd.qcut(
                df[value_col],
                q=n_groups,
                labels=range(1, n_groups + 1),
                duplicates="drop"
            )
        except Exception as e:
            print(f"Error calculating quantile groups: {e}")
            df[f"{value_col}_group"] = 3

        return df

    def merge_all_industries_indicators(self, indicator_dict):
        if not indicator_dict:
            return pd.DataFrame()

        result = None

        for indicator_name, df in indicator_dict.items():
            if len(df) == 0:
                continue

            df_copy = df.copy()
            df_copy["indicator"] = indicator_name

            if result is None:
                result = df_copy
            else:
                result = pd.concat([result, df_copy], ignore_index=True)

        return result

    def calculate_coverage_rate(self, df, group_col, industry_col, threshold=0.1):
        if len(df) == 0 or group_col not in df.columns or industry_col not in df.columns:
            return 0.0

        group1_df = df[df[group_col] == 1]

        if len(group1_df) == 0:
            return 0.0

        industry_counts = group1_df[industry_col].value_counts(normalize=True)
        biased_industries = industry_counts[industry_counts > threshold]

        coverage_rate = 1 - len(biased_industries) / len(industry_counts)
        return coverage_rate

    def save_indicators(self, df, filename):
        output_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
        os.makedirs(settings.DATA_OUTPUT_DIR, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Indicators saved to {output_path}")