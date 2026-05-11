import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class IndustryRotationStrategy:
    def __init__(self, data_loader, indicator_calculator):
        self.dl = data_loader
        self.ic = indicator_calculator
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE
        self.threshold_rules = settings.THRESHOLD_RULES
        self.coverage_threshold = settings.COVERAGE_THRESHOLD
        self.excess_return_threshold = settings.EXCESS_RETURN_THRESHOLD

    def get_industry_returns(self, industry_list, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        result = []

        for code in industry_list:
            df = self.ic.get_industry_daily_value(code, start_date, end_date)
            if len(df) > 0:
                df = self.ic.calculate_industry_returns(df)
                df["industry_code"] = code
                result.append(df)

        if len(result) > 0:
            return pd.concat(result, ignore_index=True)
        return pd.DataFrame()

    def get_benchmark_returns(self, benchmark_code=settings.BENCHMARK_CODE, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        df = self.ic.get_industry_daily_value(benchmark_code, start_date, end_date)
        if len(df) > 0:
            df = self.ic.calculate_industry_returns(df)
        return df

    def stratify_by_indicator(self, df, indicator_col, n_groups=5):
        if len(df) == 0 or indicator_col not in df.columns:
            return df

        df = df.copy()
        df = df.dropna(subset=[indicator_col])

        try:
            df[f"{indicator_col}_group"] = pd.qcut(
                df[indicator_col],
                q=n_groups,
                labels=range(1, n_groups + 1),
                duplicates="drop"
            )
        except Exception as e:
            print(f"Error stratifying: {e}")
            df[f"{indicator_col}_group"] = 3

        return df

    def calculate_group_returns(self, df, group_col, return_col, industry_col):
        if len(df) == 0 or group_col not in df.columns or return_col not in df.columns:
            return pd.DataFrame()

        group_returns = df.groupby([group_col, "period"]).apply(
            lambda x: self.calculate_equal_weight_return(x, return_col, industry_col)
        ).reset_index()
        group_returns.columns = [group_col, "period", "group_return"]

        return group_returns

    def calculate_equal_weight_return(self, group_df, return_col, industry_col):
        industry_returns = group_df.groupby("period").apply(
            lambda x: x[return_col].mean()
        )
        return industry_returns

    def calculate_portfolio_return(self, df, period_col, industry_col, return_col):
        if len(df) == 0:
            return pd.DataFrame()

        portfolio_returns = df.groupby(period_col).apply(
            lambda x: x.groupby(industry_col)[return_col].mean().mean()
        ).reset_index()
        portfolio_returns.columns = [period_col, "portfolio_return"]

        return portfolio_returns

    def calculate_cumulative_return(self, df, return_col):
        if len(df) == 0 or return_col not in df.columns:
            return df

        df = df.sort_values("period")
        df["cumulative_return"] = (1 + df[return_col]).cumprod()
        return df

    def calculate_excess_return(self, portfolio_df, benchmark_df, return_col="return"):
        if len(portfolio_df) == 0 or len(benchmark_df) == 0:
            return pd.DataFrame()

        merged = portfolio_df.merge(
            benchmark_df[["period", return_col]],
            on="period",
            suffixes=("", "_benchmark")
        )

        merged["excess_return"] = merged["portfolio_return"] - merged[f"{return_col}_benchmark"]
        return merged

    def calculate_annual_return(self, df, return_col):
        if len(df) == 0 or return_col not in df.columns:
            return 0.0

        total_return = (1 + df[return_col]).prod() - 1
        n_years = len(df) / 252
        if n_years > 0:
            annual_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annual_return = 0.0

        return annual_return

    def calculate_sharpe_ratio(self, df, return_col, risk_free_rate=0.03):
        if len(df) == 0 or return_col not in df.columns:
            return 0.0

        excess_returns = df[return_col] - risk_free_rate / 252
        if excess_returns.std() > 0:
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        return sharpe

    def calculate_max_drawdown(self, df, return_col):
        if len(df) == 0 or return_col not in df.columns:
            return 0.0

        cumulative = (1 + df[return_col]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return max_drawdown

    def calculate_win_rate(self, df, return_col):
        if len(df) == 0 or return_col not in df.columns:
            return 0.0

        wins = (df[return_col] > 0).sum()
        total = len(df)
        win_rate = wins / total if total > 0 else 0.0

        return win_rate

    def threshold_backtest(self, df, indicator_col, return_col, periods):
        if len(df) == 0 or indicator_col not in df.columns:
            return pd.DataFrame()

        results = []

        for period_name, period in periods.items():
            threshold_90 = df[indicator_col].quantile(0.9)
            threshold_70 = df[indicator_col].quantile(0.7)
            threshold_50 = df[indicator_col].quantile(0.5)
            threshold_30 = df[indicator_col].quantile(0.3)
            threshold_10 = df[indicator_col].quantile(0.1)

            long1_df = df[df[indicator_col] >= threshold_90]
            long2_df = df[df[indicator_col] >= threshold_70]
            long3_df = df[df[indicator_col] >= threshold_50]
            short1_df = df[df[indicator_col] <= threshold_10]
            short2_df = df[df[indicator_col] <= threshold_30]
            short3_df = df[df[indicator_col] <= threshold_50]

            result = {
                "period": period_name,
                "long_threshold1": long1_df[return_col].mean() if len(long1_df) > 0 else 0,
                "long_threshold2": long2_df[return_col].mean() if len(long2_df) > 0 else 0,
                "long_threshold3": long3_df[return_col].mean() if len(long3_df) > 0 else 0,
                "short_threshold1": short1_df[return_col].mean() if len(short1_df) > 0 else 0,
                "short_threshold2": short2_df[return_col].mean() if len(short2_df) > 0 else 0,
                "short_threshold3": short3_df[return_col].mean() if len(short3_df) > 0 else 0,
            }
            results.append(result)

        return pd.DataFrame(results)

    def evaluate_strategy(self, df, return_col):
        if len(df) == 0 or return_col not in df.columns:
            return {}

        annual_return = self.calculate_annual_return(df, return_col)
        sharpe = self.calculate_sharpe_ratio(df, return_col)
        max_dd = self.calculate_max_drawdown(df, return_col)
        win_rate = self.calculate_win_rate(df, return_col)

        return {
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate
        }

    def run_stratification_test(self, df, indicator_col, n_groups=5):
        if len(df) == 0 or indicator_col not in df.columns:
            return pd.DataFrame()

        df = self.stratify_by_indicator(df, indicator_col, n_groups)

        results = []

        for group in range(1, n_groups + 1):
            group_df = df[df[f"{indicator_col}_group"] == group]
            if len(group_df) > 0:
                result = {
                    "group": group,
                    "mean_return": group_df["return"].mean(),
                    "std_return": group_df["return"].std(),
                    "count": len(group_df)
                }
                results.append(result)

        return pd.DataFrame(results)

    def run_long_short_test(self, df, indicator_col):
        if len(df) == 0 or indicator_col not in df.columns:
            return pd.DataFrame()

        df = self.stratify_by_indicator(df, indicator_col, n_groups=5)

        long_df = df[df[f"{indicator_col}_group"] == 5]
        short_df = df[df[f"{indicator_col}_group"] == 1]

        long_returns = long_df["return"].mean() if len(long_df) > 0 else 0
        short_returns = short_df["return"].mean() if len(short_df) > 0 else 0
        long_short = long_returns - short_returns

        return {
            "long_return": long_returns,
            "short_return": short_returns,
            "long_short_return": long_short
        }

    def check_monotonicity(self, strat_df, n_groups=5):
        if len(strat_df) == 0:
            return False

        expected_returns = strat_df[strat_df["group"] <= n_groups]["mean_return"].tolist()
        is_monotonic = all(expected_returns[i] >= expected_returns[i+1] for i in range(len(expected_returns)-1))

        excess_return = expected_returns[0] - expected_returns[-1]
        return is_monotonic and excess_return > self.excess_return_threshold

    def check_industry_coverage(self, df, group_col, industry_col):
        if len(df) == 0:
            return 0.0

        top_group_df = df[df[group_col] == 1]
        if len(top_group_df) == 0:
            return 0.0

        industry_counts = top_group_df[industry_col].value_counts(normalize=True)
        biased_industries = industry_counts[industry_counts > 0.1]
        coverage = 1 - len(biased_industries) / len(industry_counts)

        return coverage >= self.coverage_threshold

    def run_full_backtest(self, indicator_df, industry_returns_df, indicator_name):
        results = {}

        merged = indicator_df.merge(
            industry_returns_df,
            on=["period", "industry_code"],
            how="inner"
        )

        if len(merged) == 0:
            return results

        strat_results = self.run_stratification_test(merged, f"{indicator_name}_value")
        results["stratification"] = strat_results

        long_short_results = self.run_long_short_test(merged, f"{indicator_name}_value")
        results["long_short"] = long_short_results

        coverage = self.check_industry_coverage(merged, f"{indicator_name}_group", "industry_code")
        results["coverage"] = coverage

        threshold_results = self.threshold_backtest(
            merged,
            f"{indicator_name}_value",
            "return",
            {"threshold1": 0.9, "threshold2": 0.7, "threshold3": 0.5}
        )
        results["threshold"] = threshold_results

        return results

    def save_backtest_results(self, results, filename):
        output_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
        os.makedirs(settings.DATA_OUTPUT_DIR, exist_ok=True)

        for key, df in results.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(output_path.replace(".csv", f"_{key}.csv"), index=False)
            else:
                print(f"Result {key}: {results}")

        print(f"Backtest results saved to {output_path}")