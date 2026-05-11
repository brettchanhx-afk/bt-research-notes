import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from copy import deepcopy

from .data_loader import DataLoader
from .factor_generator import FactorGenerator
from .factor_exposure import FactorExposure
from .portfolio_optimizer import PortfolioOptimizer
from .macro_scoring import MacroScoring
from .config import (
    BACKTEST_CONFIG,
    PORTFOLIO_CONFIG,
    ASSETS_CONFIG,
)

warnings.filterwarnings("ignore")


class Backtest:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000000.0,
        commission_rate: float = 0.0003,
        stamp_tax: float = 0.001,
        rebalance_freq: str = "monthly",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.rebalance_freq = rebalance_freq

        self.data_loader = DataLoader()
        self.factor_generator = FactorGenerator(self.data_loader)
        self.factor_exposure = FactorExposure(self.data_loader, self.factor_generator)
        self.portfolio_optimizer = PortfolioOptimizer()
        self.macro_scoring = MacroScoring()

        self.asset_returns = None
        self.factor_returns = None
        self.factor_exposure_matrix = None
        self.portfolio_weights = None
        self.portfolio_value = None
        self.backtest_results = None

    def prepare_data(self):
        print("Loading asset returns data...")
        asset_list = list(ASSETS_CONFIG.keys())
        self.asset_returns = self.data_loader.load_asset_returns(
            asset_list, self.start_date, self.end_date
        )

        print("Generating factor returns...")
        self.factor_returns = self.factor_generator.get_high_freq_factors(
            self.start_date, self.end_date
        )

        if len(self.asset_returns) > 0 and len(self.factor_returns) > 0:
            aligned_dates = self.asset_returns.index.intersection(self.factor_returns.index)
            self.asset_returns = self.asset_returns.loc[aligned_dates]
            self.factor_returns = self.factor_returns.loc[aligned_dates]

        print(f"Data prepared: {len(self.asset_returns)} days of asset returns, {len(self.factor_returns)} days of factor returns")

    def calculate_factor_exposures(self, date: datetime):
        window_start = date - pd.DateOffset(years=5)
        window_end = date

        window_returns = self.asset_returns[
            (self.asset_returns.index >= window_start) & (self.asset_returns.index <= window_end)
        ]
        window_factors = self.factor_returns[
            (self.factor_returns.index >= window_start) & (self.factor_returns.index <= window_end)
        ]

        if len(window_returns) < 250 or len(window_factors) < 250:
            return None

        self.factor_exposure_matrix = self.factor_exposure.get_exposure_matrix_with_r_squared(
            window_returns, window_factors, date
        )

        return self.factor_exposure_matrix

    def run_backtest(
        self,
        use_macro_view: bool = False,
        macro_views: Optional[List[Dict[str, int]]] = None,
    ) -> Dict:
        print("Starting backtest...")

        self.prepare_data()

        if len(self.asset_returns) == 0 or len(self.factor_returns) == 0:
            print("No data available for backtest")
            return {}

        trading_dates = self.asset_returns.index
        rebalance_dates = self._get_rebalance_dates(trading_dates)

        portfolio_values = []
        weights_history = []
        returns_history = []
        dates_history = []

        current_value = self.initial_capital
        current_weights = pd.Series(
            {asset: 1.0 / len(ASSETS_CONFIG) for asset in ASSETS_CONFIG.keys()}
        )

        for i, date in enumerate(rebalance_dates):
            if date not in self.asset_returns.index:
                continue

            if date > self.asset_returns.index.max():
                break

            exposure = self.calculate_factor_exposures(date)
            if exposure is None or exposure.empty:
                exposure = self.factor_exposure_matrix

            if use_macro_view and macro_views is not None and i < len(macro_views):
                factor_scores = macro_views[i]
            else:
                factor_scores = self.macro_scoring.generate_macro_view_scores()

            factor_volatility = self.macro_scoring.calculate_factor_volatility(
                self.factor_returns.loc[:date]
            )

            target_weights = self.portfolio_optimizer.risk_parity_weights(
                self.asset_returns.loc[:date]
            )

            if exposure is not None and not exposure.empty:
                target_weights = self.portfolio_optimizer.optimize_portfolio(
                    self.asset_returns.loc[:date],
                    method="risk_parity",
                    factor_exposure=exposure,
                    factor_target={},
                )

            if use_macro_view:
                target_weights = self.macro_scoring.apply_macro_views_to_portfolio(
                    target_weights,
                    exposure,
                    factor_scores,
                    factor_volatility,
                )

            new_weights = self.portfolio_optimizer.rebalance_portfolio(
                current_weights, target_weights
            )

            date_returns = self.asset_returns.loc[date]
            aligned_weights = new_weights.reindex(date_returns.index).fillna(0)
            portfolio_return = (date_returns * aligned_weights).sum()

            if portfolio_return != portfolio_return or np.isinf(portfolio_return):
                portfolio_return = 0

            turnover = (new_weights - current_weights).abs().sum()
            transaction_cost = turnover * self.commission_rate

            if date.month in [3, 6, 9, 12]:
                transaction_cost += turnover * self.stamp_tax * 0.25

            net_return = portfolio_return - transaction_cost
            current_value = current_value * (1 + net_return)

            portfolio_values.append(current_value)
            weights_history.append(new_weights)
            returns_history.append(net_return)
            dates_history.append(date)

            current_weights = new_weights

            if (i + 1) % 12 == 0:
                print(f"Progress: {i + 1}/{len(rebalance_dates)} rebalances completed")

        results = self._calculate_performance_metrics(
            dates_history, portfolio_values, returns_history
        )

        results["portfolio_values"] = pd.Series(portfolio_values, index=dates_history)
        results["weights_history"] = pd.DataFrame(weights_history, index=dates_history)
        results["returns_history"] = pd.Series(returns_history, index=dates_history)

        self.backtest_results = results

        print(f"Backtest completed. Final portfolio value: {current_value:,.2f}")

        return results

    def run_benchmark_backtest(self) -> pd.Series:
        print("Running benchmark backtest...")

        if self.asset_returns is None:
            self.prepare_data()

        trading_dates = self.asset_returns.index
        rebalance_dates = self._get_rebalance_dates(trading_dates)

        benchmark_values = []
        dates_history = []

        current_value = self.initial_capital
        equal_weights = pd.Series(
            {asset: 1.0 / len(ASSETS_CONFIG) for asset in ASSETS_CONFIG.keys()}
        )

        for date in rebalance_dates:
            if date not in self.asset_returns.index:
                continue

            if date > self.asset_returns.index.max():
                break

            date_returns = self.asset_returns.loc[date]
            aligned_weights = equal_weights.reindex(date_returns.index).fillna(0)
            portfolio_return = (date_returns * aligned_weights).sum()

            if portfolio_return != portfolio_return or np.isinf(portfolio_return):
                portfolio_return = 0

            current_value = current_value * (1 + portfolio_return)

            benchmark_values.append(current_value)
            dates_history.append(date)

        benchmark_series = pd.Series(benchmark_values, index=dates_history)
        return benchmark_series

    def _get_rebalance_dates(self, trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if self.rebalance_freq == "monthly":
            monthly_dates = trading_dates.to_series().resample("M").last()
            return monthly_dates.index
        elif self.rebalance_freq == "quarterly":
            quarterly_dates = trading_dates.to_series().resample("Q").last()
            return quarterly_dates.index
        elif self.rebalance_freq == "weekly":
            weekly_dates = trading_dates.to_series().resample("W").last()
            return weekly_dates.index
        else:
            return trading_dates

    def _calculate_performance_metrics(
        self,
        dates: List[datetime],
        portfolio_values: List[float],
        returns: List[float],
    ) -> Dict:
        if len(portfolio_values) == 0:
            return {}

        portfolio_series = pd.Series(portfolio_values, index=dates)
        returns_series = pd.Series(returns, index=dates)

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        n_years = (dates[-1] - dates[0]).days / 365.25 if len(dates) > 1 else 1
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        annualized_volatility = returns_series.std() * np.sqrt(252)

        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        positive_returns = returns_series[returns_series > 0]
        negative_returns = returns_series[returns_series < 0]

        win_rate = len(positive_returns) / len(returns_series) if len(returns_series) > 0 else 0

        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "num_trades": len(returns_series),
        }

        return metrics

    def get_results(self) -> Dict:
        return self.backtest_results

    def save_results(self, filepath: str):
        if self.backtest_results is None:
            print("No results to save")
            return

        results_to_save = deepcopy(self.backtest_results)

        if "portfolio_values" in results_to_save:
            del results_to_save["portfolio_values"]
        if "weights_history" in results_to_save:
            del results_to_save["weights_history"]
        if "returns_history" in results_to_save:
            del results_to_save["returns_history"]

        pd.DataFrame([results_to_save]).to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
