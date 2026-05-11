import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class CompositeIndicator:
    def __init__(self, data_loader, indicator_calculator):
        self.dl = data_loader
        self.ic = indicator_calculator
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE

    def standardize_indicator(self, df, value_col, method="zscore"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.copy()

        if method == "zscore":
            mean = df[value_col].mean()
            std = df[value_col].std()
            if std > 0:
                df[f"{value_col}_std"] = (df[value_col] - mean) / std
            else:
                df[f"{value_col}_std"] = 0
        elif method == "minmax":
            min_val = df[value_col].min()
            max_val = df[value_col].max()
            if max_val > min_val:
                df[f"{value_col}_std"] = (df[value_col] - min_val) / (max_val - min_val)
            else:
                df[f"{value_col}_std"] = 0

        return df

    def normalize_to_score(self, df, value_col, higher_is_better=True):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.copy()

        values = df[value_col].dropna()
        if len(values) == 0:
            return df

        min_val = values.min()
        max_val = values.max()

        if max_val > min_val:
            if higher_is_better:
                df[f"{value_col}_score"] = (df[value_col] - min_val) / (max_val - min_val) * 100
            else:
                df[f"{value_col}_score"] = (max_val - df[value_col]) / (max_val - min_val) * 100
        else:
            df[f"{value_col}_score"] = 50

        return df

    def combine_indicators(self, indicator_list, method="equal_weight"):
        if not indicator_list:
            return pd.DataFrame()

        result = None

        for indicator_df in indicator_list:
            if len(indicator_df) == 0:
                continue

            if result is None:
                result = indicator_df.copy()
            else:
                result = result.merge(indicator_df, on=["period", "industry_code"], how="outer")

        if result is None or len(result) == 0:
            return pd.DataFrame()

        if method == "equal_weight":
            score_cols = [col for col in result.columns if col.endswith("_score")]
            if len(score_cols) > 0:
                result["composite_score"] = result[score_cols].mean(axis=1)
            else:
                value_cols = [col for col in result.columns if "_std" in col]
                if len(value_cols) > 0:
                    result["composite_value"] = result[value_cols].mean(axis=1)

        return result

    def build_weekly_composite_indicator(self, north_df, margin_df, etf_df, ic_df):
        indicators = []

        if len(north_df) > 0:
            indicators.append(north_df)

        if len(margin_df) > 0:
            indicators.append(margin_df)

        if len(etf_df) > 0:
            indicators.append(etf_df)

        if len(ic_df) > 0:
            indicators.append(ic_df)

        return self.combine_indicators(indicators, method="equal_weight")

    def build_monthly_composite_indicator(self, north_df, margin_df, etf_df, ic_df):
        indicators = []

        if len(north_df) > 0:
            indicators.append(north_df)

        if len(margin_df) > 0:
            indicators.append(margin_df)

        if len(etf_df) > 0:
            indicators.append(etf_df)

        if len(ic_df) > 0:
            indicators.append(ic_df)

        return self.combine_indicators(indicators, method="equal_weight")

    def rank_industries_by_score(self, df, score_col="composite_score", top_pct=0.2):
        if len(df) == 0 or score_col not in df.columns:
            return df

        df = df.copy()
        df = df.dropna(subset=[score_col])

        df["rank"] = df.groupby("period")[score_col].rank(ascending=False, pct=True)
        df["is_top"] = df["rank"] <= top_pct
        df["is_bottom"] = df["rank"] >= (1 - top_pct)

        return df

    def calculate_prosperity_score(self, ts_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.tmt_profit_sheet(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("trade_date")

                df["profit_growth"] = df["n_income"].pct_change()
                df["roe"] = df["roe"]
                df["prosperity_score"] = df["profit_growth"] * 0.5 + df["roe"] * 0.5

            return df
        except Exception as e:
            print(f"Error calculating prosperity score for {ts_code}: {e}")
            return pd.DataFrame()

    def build_prosperity_composite(self, industry_members_df, start_date=None, end_date=None):
        if len(industry_members_df) == 0:
            return pd.DataFrame()

        result = []

        for _, row in industry_members_df.iterrows():
            ts_code = row.get("con_code", "")
            if not ts_code:
                continue

            df = self.calculate_prosperity_score(ts_code, start_date, end_date)
            if len(df) > 0:
                df["industry_code"] = row.get("index_code", "")
                result.append(df)

        if len(result) > 0:
            result_df = pd.concat(result, ignore_index=True)
            agg_df = result_df.groupby(["trade_date", "industry_code"]).agg({
                "prosperity_score": "mean"
            }).reset_index()
            return agg_df
        return pd.DataFrame()

    def combine_prosperity_and_flow(self, prosperity_df, flow_df):
        if len(prosperity_df) == 0 or len(flow_df) == 0:
            return pd.DataFrame()

        merged = prosperity_df.merge(
            flow_df,
            on=["period", "industry_code"],
            how="inner"
        )

        if len(merged) == 0:
            return pd.DataFrame()

        merged = self.normalize_to_score(merged, "prosperity_score", higher_is_better=True)
        merged = self.normalize_to_score(merged, "composite_score", higher_is_better=True)

        merged["combined_score"] = merged["prosperity_score_score"] * 0.5 + merged["composite_score_score"] * 0.5

        return merged

    def select_top_industries(self, df, score_col="composite_score", n=5):
        if len(df) == 0 or score_col not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["rank"] = df.groupby("period")[score_col].rank(ascending=False)
        top_df = df[df["rank"] <= n]

        return top_df

    def select_bottom_industries(self, df, score_col="composite_score", n=5):
        if len(df) == 0 or score_col not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        max_rank = df.groupby("period")[score_col].transform("max")
        df["reverse_rank"] = max_rank - df[score_col]
        df["rank"] = df.groupby("period")["reverse_rank"].rank(ascending=False)
        bottom_df = df[df["rank"] <= n]

        return bottom_df

    def calculate_strategy_returns(self, selected_df, returns_df):
        if len(selected_df) == 0 or len(returns_df) == 0:
            return pd.DataFrame()

        merged = selected_df.merge(
            returns_df,
            on=["period", "industry_code"],
            how="inner"
        )

        if len(merged) == 0:
            return pd.DataFrame()

        strategy_returns = merged.groupby("period").agg({
            "return": "mean"
        }).reset_index()
        strategy_returns.columns = ["period", "strategy_return"]

        return strategy_returns

    def calculate_cumulative_returns(self, returns_df, return_col="return"):
        if len(returns_df) == 0 or return_col not in returns_df.columns:
            return pd.DataFrame()

        df = returns_df.sort_values("period")
        df["cumulative_return"] = (1 + df[return_col]).cumprod()
        df["cumulative_excess"] = df["cumulative_return"] / df["cumulative_return"].iloc[0] - 1

        return df

    def evaluate_composite_strategy(self, strategy_returns_df, benchmark_returns_df):
        if len(strategy_returns_df) == 0:
            return {}

        merged = strategy_returns_df.merge(
            benchmark_returns_df,
            on="period",
            suffixes=("_strategy", "_benchmark")
        )

        if len(merged) == 0:
            return {}

        merged["excess_return"] = merged["strategy_return"] - merged["return_benchmark"]
        merged["cumulative_strategy"] = (1 + merged["strategy_return"]).cumprod()
        merged["cumulative_benchmark"] = (1 + merged["return_benchmark"]).cumprod()
        merged["cumulative_excess"] = merged["cumulative_strategy"] / merged["cumulative_benchmark"] - 1

        total_return = merged["cumulative_strategy"].iloc[-1] - 1
        n_years = len(merged) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        annual_vol = merged["strategy_return"].std() * np.sqrt(252)
        sharpe = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0

        running_max = merged["cumulative_strategy"].expanding().max()
        drawdown = (merged["cumulative_strategy"] - running_max) / running_max
        max_drawdown = drawdown.min()

        wins = (merged["excess_return"] > 0).sum()
        total = len(merged)
        win_rate = wins / total if total > 0 else 0

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_return": total_return
        }

    def save_composite_results(self, df, filename):
        output_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
        os.makedirs(settings.DATA_OUTPUT_DIR, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Composite results saved to {output_path}")