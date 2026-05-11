import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import INDUSTRY_INDEX_MAPPING


class ETFFlowCalculator:
    def __init__(
        self,
        industry_etf_flow: Dict[str, pd.DataFrame],
        industry_list: Optional[List[str]] = None,
    ):
        self.industry_etf_flow = industry_etf_flow
        self.industry_list = (
            industry_list
            if industry_list is not None
            else list(INDUSTRY_INDEX_MAPPING.keys())
        )

    def calculate_weekly_net_flow(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        method: str = "natural_week",
    ) -> pd.DataFrame:
        if method == "natural_week":
            return self._calculate_natural_week_flow(start_date, end_date)
        elif method == "rolling_trading_day":
            return self._calculate_rolling_5day_flow(start_date, end_date)
        else:
            raise ValueError(f"未知的计算方法: {method}")

    def _calculate_natural_week_flow(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        weekly_flows = {}

        for industry in self.industry_list:
            if industry not in self.industry_etf_flow:
                continue

            df = self.industry_etf_flow[industry].copy()
            if df is None or len(df) == 0:
                continue

            if "trade_date" not in df.columns or "net_flow" not in df.columns:
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")

            if start_date:
                df = df[df["trade_date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["trade_date"] <= pd.to_datetime(end_date)]

            df = df[df["trade_date"].dt.dayofweek < 5]

            df["week_start"] = df["trade_date"] - pd.to_timedelta(df["trade_date"].dt.dayofweek, unit="D")
            df["week_start"] = df["week_start"].dt.normalize()

            weekly_sum = (
                df.groupby("week_start")
                .agg({"net_flow": "sum", "trade_date": "max"})
                .reset_index()
            )
            weekly_sum.columns = ["week_start", "net_flow", "week_end"]
            weekly_sum["week_end"] = weekly_sum["week_end"].dt.normalize()
            weekly_sum["industry"] = industry
            weekly_sum["year"] = weekly_sum["week_start"].dt.year
            weekly_sum["week"] = weekly_sum["week_start"].dt.isocalendar().week

            weekly_flows[industry] = weekly_sum

        if not weekly_flows:
            return pd.DataFrame()

        result = pd.concat(weekly_flows.values(), ignore_index=True)
        result = result.sort_values(["week_start", "industry"]).reset_index(drop=True)

        return result

    def _calculate_rolling_5day_flow(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        rolling_flows = {}

        for industry in self.industry_list:
            if industry not in self.industry_etf_flow:
                continue

            df = self.industry_etf_flow[industry].copy()
            if df is None or len(df) == 0:
                continue

            if "trade_date" not in df.columns or "net_flow" not in df.columns:
                continue

            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date").reset_index(drop=True)

            if start_date:
                df = df[df["trade_date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["trade_date"] <= pd.to_datetime(end_date)]

            df["rolling_net_flow"] = df["net_flow"].rolling(window=5, min_periods=5).sum()

            df_valid = df.dropna(subset=["rolling_net_flow"]).copy()
            df_valid["industry"] = industry

            rolling_flows[industry] = df_valid[
                ["trade_date", "rolling_net_flow", "industry"]
            ]

        if not rolling_flows:
            return pd.DataFrame()

        result = pd.concat(rolling_flows.values(), ignore_index=True)
        result = result.sort_values(["trade_date", "industry"]).reset_index(drop=True)

        return result

    def calculate_rolling_percentile(
        self,
        net_flow_data: pd.DataFrame,
        window_years: int = 2,
        percentile_col: str = "net_flow",
        date_col: str = "week_start",
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        if len(net_flow_data) == 0:
            return pd.DataFrame()

        df = net_flow_data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        df = df.sort_values([industry_col, date_col]).reset_index(drop=True)

        result_list = []

        for industry in df[industry_col].unique():
            industry_df = df[df[industry_col] == industry].copy()
            industry_df = industry_df.sort_values(date_col).reset_index(drop=True)

            rolling_percentile = []

            for idx in range(len(industry_df)):
                current_date = industry_df.loc[idx, date_col]

                lookback_start = current_date - pd.DateOffset(years=window_years)
                lookback_data = industry_df[
                    (industry_df[date_col] >= lookback_start)
                    & (industry_df[date_col] < current_date)
                ][percentile_col]

                if len(lookback_data) > 0:
                    current_value = industry_df.loc[idx, percentile_col]
                    percentile = (lookback_data < current_value).sum() / len(lookback_data)
                    rolling_percentile.append(percentile)
                else:
                    rolling_percentile.append(np.nan)

            industry_df["rolling_percentile"] = rolling_percentile
            result_list.append(industry_df)

        result = pd.concat(result_list, ignore_index=True)

        return result

    def calculate_future_returns(
        self,
        net_flow_data: pd.DataFrame,
        index_returns: pd.DataFrame,
        date_col: str = "week_start",
        industry_col: str = "industry",
        forward_col: str = "week_end",
    ) -> pd.DataFrame:
        if len(net_flow_data) == 0 or len(index_returns) == 0:
            return pd.DataFrame()

        df = net_flow_data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df[forward_col] = pd.to_datetime(df[forward_col])

        index_returns = index_returns.copy()
        if "trade_date" in index_returns.columns:
            index_returns["trade_date"] = pd.to_datetime(index_returns["trade_date"])

        result_list = []

        for industry in df[industry_col].unique():
            industry_df = df[df[industry_col] == industry].copy()
            if len(industry_df) == 0:
                continue

            ind_index_data = index_returns[index_returns["industry"] == industry].copy()
            if len(ind_index_data) == 0:
                continue

            ind_index_data = ind_index_data.sort_values("trade_date")

            future_returns = []

            for _, row in industry_df.iterrows():
                start_date = row[forward_col]
                next_week_start = start_date + pd.Timedelta(days=1)

                next_week_end = next_week_start + pd.Timedelta(days=6)

                period_returns = ind_index_data[
                    (ind_index_data["trade_date"] >= next_week_start)
                    & (ind_index_data["trade_date"] <= next_week_end)
                ]["pct_change"]

                if len(period_returns) > 0:
                    cumulative_return = (1 + period_returns / 100).prod() - 1
                    future_returns.append(cumulative_return * 100)
                else:
                    forward_start = row.get("week_end", row.get("trade_date"))
                    if pd.isna(forward_start):
                        forward_start = row[date_col]

                    next_start = pd.to_datetime(forward_start) + pd.Timedelta(days=1)
                    next_end = next_start + pd.Timedelta(days=7)

                    period_returns = ind_index_data[
                        (ind_index_data["trade_date"] >= next_start)
                        & (ind_index_data["trade_date"] < next_end)
                    ]["pct_change"]

                    if len(period_returns) > 0:
                        cumulative_return = (1 + period_returns / 100).prod() - 1
                        future_returns.append(cumulative_return * 100)
                    else:
                        future_returns.append(np.nan)

            industry_df["future_return"] = future_returns
            result_list.append(industry_df)

        if not result_list:
            return pd.DataFrame()

        result = pd.concat(result_list, ignore_index=True)

        return result

    def get_signals(
        self,
        percentile_data: pd.DataFrame,
        long_threshold: float = 0.10,
        short_threshold: float = 0.90,
        industry_col: str = "industry",
        percentile_col: str = "rolling_percentile",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(percentile_data) == 0:
            return pd.DataFrame(), pd.DataFrame()

        df = percentile_data.copy()

        long_signals = df[df[percentile_col] <= long_threshold].copy()
        short_signals = df[df[percentile_col] >= short_threshold].copy()

        return long_signals, short_signals

    def analyze_flow规律(
        self,
        percentile_data: pd.DataFrame,
        return_data: pd.DataFrame,
        thresholds: List[float] = [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95],
        percentile_col: str = "rolling_percentile",
        return_col: str = "future_return",
        industry_col: str = "industry",
    ) -> pd.DataFrame:
        if len(percentile_data) == 0 or len(return_data) == 0:
            return pd.DataFrame()

        df = percentile_data.merge(
            return_data[["industry", "trade_date", "future_return"]],
            on=["industry", "trade_date"],
            how="left",
        )

        df["percentile_bin"] = pd.cut(
            df[percentile_col],
            bins=[0] + thresholds + [1],
            labels=[f"{thresholds[i]}%-{thresholds[i+1]}%" for i in range(len(thresholds)-1)] + ["90%-100%"],
            include_lowest=True,
        )

        grouped = df.groupby("percentile_bin", observed=False).agg(
            {
                return_col: ["mean", "median", "count"],
                "industry": lambda x: (x.notna()).sum(),
            }
        )

        result = pd.DataFrame({
            "percentile_range": grouped.index,
            "mean_return": grouped[(return_col, "mean")],
            "median_return": grouped[(return_col, "median")],
            "observation_count": grouped[(return_col, "count")],
        })

        return result.reset_index(drop=True)


def calculate_etf_flow_statistics(
    net_flow_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
    return_df: pd.DataFrame,
    groupby_col: str = "rolling_percentile",
) -> Dict:
    if len(percentile_df) == 0 or len(return_df) == 0:
        return {}

    merged = percentile_df.merge(
        return_df[["industry", "trade_date", "future_return"]],
        on=["industry", "trade_date"],
        how="left",
    )

    stats = {}

    percentile_ranges = [
        ("0-5%", lambda x: x <= 0.05),
        ("5-10%", lambda x: (x > 0.05) & (x <= 0.10)),
        ("10-20%", lambda x: (x > 0.10) & (x <= 0.20)),
        ("20-50%", lambda x: (x > 0.20) & (x <= 0.50)),
        ("50-80%", lambda x: (x > 0.50) & (x <= 0.80)),
        ("80-90%", lambda x: (x > 0.80) & (x <= 0.90)),
        ("90-95%", lambda x: (x > 0.90) & (x <= 0.95)),
        ("95-100%", lambda x: x > 0.95),
    ]

    for range_name, condition in percentile_ranges:
        subset = merged[condition(merged[groupby_col])]
        if len(subset) > 0:
            stats[range_name] = {
                "count": len(subset),
                "mean_return": subset["future_return"].mean(),
                "median_return": subset["future_return"].median(),
                "positive_ratio": (subset["future_return"] > 0).sum() / len(subset),
            }
        else:
            stats[range_name] = {
                "count": 0,
                "mean_return": np.nan,
                "median_return": np.nan,
                "positive_ratio": np.nan,
            }

    return stats


if __name__ == "__main__":
    print("ETF资金流计算模块测试...")

    sample_data = {
        "非银金融": pd.DataFrame({
            "trade_date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "net_flow": np.random.randn(20) * 1000000,
            "nav": np.random.uniform(0.8, 1.2, 20),
            "vol": np.random.randint(1000, 10000, 20),
        })
    }

    calculator = ETFFlowCalculator(sample_data)

    print("\n1. 测试周度资金流计算...")
    weekly_flow = calculator.calculate_weekly_net_flow(method="natural_week")
    print(f"周度资金流数据: {len(weekly_flow)} 条")

    print("\n2. 测试滚动分位数计算...")
    if len(weekly_flow) > 0:
        percentile_flow = calculator.calculate_rolling_percentile(
            weekly_flow, window_years=2
        )
        print(f"滚动分位数计算完成")

    print("\nETF资金流计算模块测试完成")
