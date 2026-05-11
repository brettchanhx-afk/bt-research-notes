import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    INDUSTRY_INDEX_MAPPING,
    INDUSTRY_NAME_MAPPING,
    BACKTEST_END_DATE,
    BACKTEST_START_DATE,
    STRATEGY_PARAMS,
)


class IndustryRotationStrategy:
    def __init__(
        self,
        long_threshold: float = 0.10,
        short_threshold: float = 0.90,
        rolling_window_years: int = 2,
        rebalance_freq: str = "weekly",
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.rolling_window_years = rolling_window_years
        self.rebalance_freq = rebalance_freq

        self.strategy_name = f"ETF_Flow_Long{int(long_threshold*100)}_Short{int(short_threshold*100)}_W{rolling_window_years}y"

    def select_industries_long(
        self, percentile_data: pd.DataFrame, date: pd.Timestamp
    ) -> List[str]:
        if len(percentile_data) == 0:
            return []

        date_data = percentile_data[
            percentile_data["week_start"].dt.normalize() == date.normalize()
        ]

        if len(date_data) == 0:
            week_data = percentile_data[
                (percentile_data["week_start"] < date)
                & (percentile_data["week_end"] >= date)
            ]
            if len(week_data) == 0:
                return []
            date_data = week_data

        selected = date_data[
            date_data["rolling_percentile"] <= self.long_threshold
        ]["industry"].tolist()

        return selected

    def select_industries_short(
        self, percentile_data: pd.DataFrame, date: pd.Timestamp
    ) -> List[str]:
        if len(percentile_data) == 0:
            return []

        date_data = percentile_data[
            percentile_data["week_start"].dt.normalize() == date.normalize()
        ]

        if len(date_data) == 0:
            week_data = percentile_data[
                (percentile_data["week_start"] < date)
                & (percentile_data["week_end"] >= date)
            ]
            if len(week_data) == 0:
                return []
            date_data = week_data

        selected = date_data[
            date_data["rolling_percentile"] >= self.short_threshold
        ]["industry"].tolist()

        return selected

    def select_industries_long_df(
        self, percentile_data: pd.DataFrame, date: pd.Timestamp
    ) -> pd.DataFrame:
        if len(percentile_data) == 0:
            return pd.DataFrame()

        date_data = percentile_data[
            percentile_data["week_start"].dt.normalize() == date.normalize()
        ]

        if len(date_data) == 0:
            week_data = percentile_data[
                (percentile_data["week_start"] < date)
                & (percentile_data["week_end"] >= date)
            ]
            if len(week_data) == 0:
                return pd.DataFrame()
            date_data = week_data

        selected = date_data[
            date_data["rolling_percentile"] <= self.long_threshold
        ]

        return selected[["industry", "week_end"]]

    def select_industries_short_df(
        self, percentile_data: pd.DataFrame, date: pd.Timestamp
    ) -> pd.DataFrame:
        if len(percentile_data) == 0:
            return pd.DataFrame()

        date_data = percentile_data[
            percentile_data["week_start"].dt.normalize() == date.normalize()
        ]

        if len(date_data) == 0:
            week_data = percentile_data[
                (percentile_data["week_start"] < date)
                & (percentile_data["week_end"] >= date)
            ]
            if len(week_data) == 0:
                return pd.DataFrame()
            date_data = week_data

        selected = date_data[
            date_data["rolling_percentile"] >= self.short_threshold
        ]

        return selected[["industry", "week_end"]]

    def generate_signals(
        self,
        percentile_data: pd.DataFrame,
        rebalance_dates: Optional[List[pd.Timestamp]] = None,
        signal_type: str = "long",
    ) -> pd.DataFrame:
        if len(percentile_data) == 0:
            return pd.DataFrame()

        if rebalance_dates is None:
            if self.rebalance_freq == "weekly":
                rebalance_dates = sorted(percentile_data["week_start"].unique())
            else:
                rebalance_dates = sorted(percentile_data["trade_date"].unique())

        signals = []

        for date in rebalance_dates:
            if signal_type == "long":
                selected_df = self.select_industries_long_df(percentile_data, date)
            else:
                selected_df = self.select_industries_short_df(percentile_data, date)

            for _, row in selected_df.iterrows():
                signals.append(
                    {
                        "date": date,
                        "industry": row["industry"].astype(str).str.replace(" ", "", regex=False),
                        "week_end": row["week_end"],
                        "signal": 1 if signal_type == "long" else -1,
                        "signal_type": signal_type,
                    }
                )

        return pd.DataFrame(signals)

    def calculate_portfolio_returns(
        self,
        signals: pd.DataFrame,
        returns_data: pd.DataFrame,
        industry_col: str = "industry",
        date_col: str = "date",
        return_col: str = "future_return",
    ) -> pd.DataFrame:
        if len(signals) == 0 or len(returns_data) == 0:
            return pd.DataFrame()

        if "week_end" not in signals.columns:
            signals["week_end"] = signals["date"] + pd.Timedelta(days=6)

        signals = signals.sort_values(["date", "industry"])

        returns_data = returns_data.copy()
        returns_data["trade_date"] = pd.to_datetime(returns_data["trade_date"])
        returns_data["industry"] = returns_data["industry"].astype(str).str.replace(" ", "", regex=False)

        if "pct_change" in returns_data.columns:
            returns_data["return_col"] = returns_data["pct_change"]
        elif "pct_chg" in returns_data.columns:
            returns_data["return_col"] = returns_data["pct_chg"]
        elif "return_col" not in returns_data.columns:
            returns_data["return_col"] = returns_data["close"].pct_change() * 100

        portfolio_returns_list = []

        for (date, signal_type), group in signals.groupby(["date", "signal_type"]):
            week_start = pd.to_datetime(date)
            week_end = pd.to_datetime(group["week_end"].iloc[0])

            week_returns = returns_data[
                (returns_data["trade_date"] >= week_start) &
                (returns_data["trade_date"] <= week_end)
            ]

            if len(week_returns) == 0:
                continue

            industries_in_signal = group["industry"].tolist()

            industry_returns = week_returns[week_returns["industry"].isin(industries_in_signal)]

            if len(industry_returns) > 0:
                mean_return = industry_returns.groupby("trade_date")["return_col"].mean()
                if len(mean_return) > 0:
                    week_return = (1 + mean_return / 100).prod() - 1
                    week_return = week_return * 100
                else:
                    week_return = 0
            else:
                week_return = 0

            portfolio_returns_list.append({
                "date": date,
                "signal_type": signal_type,
                "mean_return": week_return,
                "total_return": week_return,
                "num_positions": len(industries_in_signal),
                "industries": industries_in_signal,
            })

        if not portfolio_returns_list:
            return pd.DataFrame()

        portfolio_returns = pd.DataFrame(portfolio_returns_list)

        return portfolio_returns

    def calculate_cumulative_returns(
        self, portfolio_returns: pd.DataFrame, initial_value: float = 1.0
    ) -> pd.DataFrame:
        if len(portfolio_returns) == 0:
            return pd.DataFrame()

        df = portfolio_returns.copy().sort_values("date")

        df["cumulative_return"] = (1 + df["mean_return"] / 100).cumprod() * initial_value

        df["benchmark_cumulative"] = (
            (1 + df["mean_return"].expanding().mean() / 100).cumprod() * initial_value
        )

        df["excess_return"] = df["cumulative_return"] - df["benchmark_cumulative"]

        return df

    def run_backtest(
        self,
        percentile_data: pd.DataFrame,
        returns_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict:
        if len(percentile_data) == 0:
            return {}

        returns_data = returns_data.copy()
        returns_data["trade_date"] = pd.to_datetime(returns_data["trade_date"])
        returns_data["industry"] = returns_data["industry"].str.strip()

        if "pct_change" in returns_data.columns:
            returns_data["return_col"] = returns_data["pct_change"]
        elif "pct_chg" in returns_data.columns:
            returns_data["return_col"] = returns_data["pct_chg"]
        else:
            returns_data["return_col"] = returns_data["close"].pct_change() * 100

        rebalance_dates = sorted(percentile_data["week_start"].unique())

        long_signals = self.generate_signals(
            percentile_data, rebalance_dates, signal_type="long"
        )
        short_signals = self.generate_signals(
            percentile_data, rebalance_dates, signal_type="short"
        )

        long_returns = self.calculate_portfolio_returns(long_signals, returns_data)
        short_returns = self.calculate_portfolio_returns(short_signals, returns_data)

        all_returns = pd.concat([long_returns, short_returns], ignore_index=True)

        metrics = self.calculate_performance_metrics(all_returns, benchmark_data)

        result = {
            "strategy_name": self.strategy_name,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "portfolio_returns": all_returns,
            "performance_metrics": metrics,
            "params": {
                "long_threshold": self.long_threshold,
                "short_threshold": self.short_threshold,
                "rolling_window_years": self.rolling_window_years,
                "rebalance_freq": self.rebalance_freq,
            },
        }

        return result

    def calculate_performance_metrics(
        self,
        portfolio_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.DataFrame] = None,
    ) -> Dict:
        if len(portfolio_returns) == 0:
            return {}

        long_returns = portfolio_returns[portfolio_returns["signal_type"] == "long"]
        short_returns = portfolio_returns[portfolio_returns["signal_type"] == "short"]

        metrics = {}

        for signal_type, returns_df in [("long", long_returns), ("short", short_returns)]:
            if len(returns_df) == 0:
                metrics[signal_type] = {}
                continue

            returns = returns_df["mean_return"].dropna() / 100

            if len(returns) == 0:
                metrics[signal_type] = {}
                continue

            cumulative = (1 + returns).cumprod()
            total_return = cumulative.iloc[-1] - 1
            n_periods = len(returns)
            n_years = n_periods / 52

            annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

            annual_vol = returns.std() * np.sqrt(52)

            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            sharpe = (
                annual_return / annual_vol if annual_vol > 0 else 0
            )

            calmar = (
                annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            )

            positive_returns = (returns > 0).sum()
            total_returns_count = len(returns)
            weekly_win_rate = positive_returns / total_returns_count if total_returns_count > 0 else 0

            avg_positive = returns[returns > 0].mean() if (returns > 0).any() else 0
            avg_negative = returns[returns < 0].mean() if (returns < 0).any() else 0
            profit_loss_ratio = (
                abs(avg_positive / avg_negative) if avg_negative != 0 else np.inf
            )

            monthly_returns = returns_df.set_index("date")["mean_return"].resample("M").apply(
                lambda x: (1 + x / 100).prod() - 1
            ) * 100
            monthly_positive = (monthly_returns > 0).sum()
            monthly_total = len(monthly_returns)
            monthly_win_rate = monthly_positive / monthly_total if monthly_total > 0 else 0

            metrics[signal_type] = {
                "total_return": total_return * 100,
                "annual_return": annual_return * 100,
                "annual_vol": annual_vol * 100,
                "max_drawdown": max_drawdown * 100,
                "sharpe": sharpe,
                "calmar": calmar,
                "weekly_win_rate": weekly_win_rate * 100,
                "profit_loss_ratio": profit_loss_ratio,
                "monthly_win_rate": monthly_win_rate * 100,
                "num_trades": len(returns_df),
            }

        return metrics

    def get_yearly_returns(
        self, portfolio_returns: pd.DataFrame
    ) -> pd.DataFrame:
        if len(portfolio_returns) == 0:
            return pd.DataFrame()

        df = portfolio_returns.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        yearly = []
        for (year, signal_type), group in df.groupby(["year", "signal_type"]):
            cumulative = (1 + group["mean_return"].dropna() / 100).prod() - 1
            yearly.append(
                {
                    "year": year,
                    "signal_type": signal_type,
                    "annual_return": cumulative * 100,
                }
            )

        return pd.DataFrame(yearly)


def run_parameter_search(
    percentile_data: pd.DataFrame,
    returns_data: pd.DataFrame,
    long_thresholds: List[float] = None,
    short_thresholds: List[float] = None,
    window_years: List[int] = None,
) -> pd.DataFrame:
    if long_thresholds is None:
        long_thresholds = [0.01, 0.05, 0.10, 0.20, 0.50]
    if short_thresholds is None:
        short_thresholds = [0.99, 0.95, 0.90, 0.80, 0.50]
    if window_years is None:
        window_years = [1, 2, 3]

    results = []

    for long_th in long_thresholds:
        for short_th in short_thresholds:
            if long_th >= short_th:
                continue

            for window in window_years:
                strategy = IndustryRotationStrategy(
                    long_threshold=long_th,
                    short_threshold=short_th,
                    rolling_window_years=window,
                )

                result = strategy.run_backtest(percentile_data, returns_data)

                if "performance_metrics" in result:
                    long_metrics = result["performance_metrics"].get("long", {})
                    if long_metrics:
                        results.append(
                            {
                                "long_threshold": long_th,
                                "short_threshold": short_th,
                                "window_years": window,
                                "annual_return": long_metrics.get("annual_return", 0),
                                "sharpe": long_metrics.get("sharpe", 0),
                                "calmar": long_metrics.get("calmar", 0),
                                "max_drawdown": long_metrics.get("max_drawdown", 0),
                                "weekly_win_rate": long_metrics.get("weekly_win_rate", 0),
                                "monthly_win_rate": long_metrics.get("monthly_win_rate", 0),
                            }
                        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("行业轮动策略模块测试...")

    strategy = IndustryRotationStrategy(
        long_threshold=0.10, short_threshold=0.90, rolling_window_years=2
    )

    print(f"策略名称: {strategy.strategy_name}")
    print(f"做多阈值: {strategy.long_threshold}")
    print(f"做空阈值: {strategy.short_threshold}")
    print(f"滚动窗口: {strategy.rolling_window_years}年")

    print("\n行业轮动策略模块测试完成")
