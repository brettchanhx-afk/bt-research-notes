import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
from copy import deepcopy

warnings.filterwarnings("ignore")

from .csv_data_loader import CSVDataLoader
from .csv_factor_generator import CSVFactorGenerator
from .factor_exposure import FactorExposure
from .portfolio_optimizer import PortfolioOptimizer
from .macro_scoring import MacroScoring
from .config import (
    BACKTEST_CONFIG,
    PORTFOLIO_CONFIG,
    ASSETS_CONFIG,
)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class CSVBacktest:
    def __init__(
        self,
        start_date: str = "2013-01-01",
        end_date: str = "2024-12-31",
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

        self.csv_loader = CSVDataLoader()
        self.factor_generator = CSVFactorGenerator(self.csv_loader)
        self.portfolio_optimizer = PortfolioOptimizer()
        self.macro_scoring = MacroScoring()

        self.asset_returns = None
        self.factor_returns = None
        self.factor_exposure_matrix = None
        self.backtest_results = None

    def prepare_data(self):
        print("Loading asset returns from CSV data...")
        asset_prices = self.csv_loader.get_asset_prices()

        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)

        asset_prices = asset_prices[(asset_prices.index >= start_dt) & (asset_prices.index <= end_dt)]

        self.asset_returns = asset_prices.pct_change()
        self.asset_returns = self.asset_returns.replace([np.inf, -np.inf], np.nan)
        self.asset_returns = self.asset_returns.dropna(how='all')

        self.asset_returns = self.asset_returns.dropna(axis=1, how='all')

        print(f"Asset returns loaded: {len(self.asset_returns)} trading days")
        print(f"Assets: {list(self.asset_returns.columns)}")

        print("\nGenerating factor returns...")
        self.factor_returns = self.factor_generator.get_high_freq_factors()

        if len(self.factor_returns) > 0:
            self.factor_returns = self.factor_returns[(self.factor_returns.index >= start_dt) & (self.factor_returns.index <= end_dt)]
            self.factor_returns = self.factor_returns.replace([np.inf, -np.inf], np.nan)
            self.factor_returns = self.factor_returns.dropna(how='all', axis=1)

            print(f"Factor returns generated: {len(self.factor_returns)} days")
            print(f"Factors: {list(self.factor_returns.columns)}")

        aligned_dates = self.asset_returns.index.intersection(self.factor_returns.index) if len(self.factor_returns) > 0 else self.asset_returns.index

        if len(aligned_dates) > 0:
            self.asset_returns = self.asset_returns.loc[aligned_dates]
            self.factor_returns = self.factor_returns.loc[aligned_dates]

    def calculate_factor_exposure_single_period(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> np.ndarray:
        aligned_asset, aligned_factors = asset_returns.align(factor_returns, join="inner")

        valid_mask = ~(aligned_asset.isna() | aligned_factors.isna().any(axis=1))
        if valid_mask.sum() < 250:
            return np.full(factor_returns.shape[1], np.nan)

        clean_asset = aligned_asset[valid_mask].values
        clean_factors = aligned_factors[valid_mask].values

        scaler = StandardScaler()
        scaled_factors = scaler.fit_transform(clean_factors)

        model = LinearRegression()
        model.fit(scaled_factors, clean_asset)

        return model.coef_

    def calculate_factor_exposures(self, date: datetime) -> Optional[pd.DataFrame]:
        window_end = date
        window_start = date - pd.DateOffset(years=5)

        window_returns = self.asset_returns[
            (self.asset_returns.index >= window_start) & (self.asset_returns.index <= window_end)
        ]
        window_factors = self.factor_returns[
            (self.factor_returns.index >= window_start) & (self.factor_returns.index <= window_end)
        ]

        if len(window_returns) < 500 or len(window_factors) < 500:
            return None

        exposure_dict = {}
        for asset_col in window_returns.columns:
            asset_ret = window_returns[asset_col]
            exposure = self.calculate_factor_exposure_single_period(asset_ret, window_factors)

            if not np.any(np.isnan(exposure)):
                exposure_dict[asset_col] = exposure

        if exposure_dict:
            exposure_df = pd.DataFrame(
                exposure_dict,
                index=window_factors.columns
            ).T
            return exposure_df
        else:
            return None

    def run_backtest(
        self,
        use_macro_view: bool = False,
        macro_views: Optional[List[Dict[str, int]]] = None,
    ) -> Dict:
        print("\n" + "=" * 60)
        print("Starting Backtest...")
        print("=" * 60)

        self.prepare_data()

        if len(self.asset_returns) == 0:
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
            {asset: 1.0 / len(self.asset_returns.columns) for asset in self.asset_returns.columns}
        )

        print(f"\nRebalance frequency: {self.rebalance_freq}")
        print(f"Total rebalance periods: {len(rebalance_dates)}")

        for i, date in enumerate(rebalance_dates):
            if date not in self.asset_returns.index:
                date_idx = self.asset_returns.index[self.asset_returns.index >= date]
                if len(date_idx) == 0:
                    continue
                date = date_idx[0]

            exposure = self.calculate_factor_exposures(date)

            if exposure is not None and not exposure.empty:
                self.factor_exposure_matrix = exposure

            lookback_returns = self.asset_returns.loc[:date]
            if len(lookback_returns) < 60:
                continue

            target_weights = self.portfolio_optimizer.risk_parity_weights(lookback_returns)

            if use_macro_view and macro_views is not None and i < len(macro_views):
                factor_scores = macro_views[i]
                factor_volatility = self.macro_scoring.calculate_factor_volatility(
                    self.factor_returns.loc[:date]
                )

                if exposure is not None and not exposure.empty:
                    target_weights = self.macro_scoring.apply_macro_views_to_portfolio(
                        target_weights,
                        exposure,
                        factor_scores,
                        factor_volatility,
                    )

            new_weights = self.portfolio_optimizer.rebalance_portfolio(
                current_weights, target_weights
            )

            if date in self.asset_returns.index:
                date_returns = self.asset_returns.loc[date]
                aligned_weights = new_weights.reindex(date_returns.index).fillna(0)
                portfolio_return = (date_returns * aligned_weights).sum()

                if portfolio_return != portfolio_return or np.isinf(portfolio_return):
                    portfolio_return = 0

                turnover = (new_weights - current_weights).abs().sum()
                transaction_cost = turnover * self.commission_rate

                net_return = portfolio_return - transaction_cost
                current_value = current_value * (1 + net_return)

                portfolio_values.append(current_value)
                weights_history.append(new_weights)
                returns_history.append(net_return)
                dates_history.append(date)

                current_weights = new_weights

            if (i + 1) % 12 == 0 or i == 0:
                print(f"Progress: {i + 1}/{len(rebalance_dates)} | Value: {current_value:,.2f}")

        results = self._calculate_performance_metrics(
            dates_history, portfolio_values, returns_history
        )

        results["portfolio_values"] = pd.Series(portfolio_values, index=dates_history)
        results["weights_history"] = pd.DataFrame(weights_history, index=dates_history)
        results["returns_history"] = pd.Series(returns_history, index=dates_history)

        self.backtest_results = results

        print(f"\nBacktest completed!")
        print(f"Final portfolio value: {current_value:,.2f}")
        print(f"Total return: {results.get('total_return', 0) * 100:.2f}%")

        return results

    def run_benchmark_backtest(self) -> pd.Series:
        print("\nRunning benchmark backtest...")

        if self.asset_returns is None:
            self.prepare_data()

        trading_dates = self.asset_returns.index
        rebalance_dates = self._get_rebalance_dates(trading_dates)

        benchmark_values = []
        dates_history = []

        current_value = self.initial_capital
        equal_weights = pd.Series(
            {asset: 1.0 / len(self.asset_returns.columns) for asset in self.asset_returns.columns}
        )

        for date in rebalance_dates:
            if date not in self.asset_returns.index:
                date_idx = self.asset_returns.index[self.asset_returns.index >= date]
                if len(date_idx) == 0:
                    continue
                date = date_idx[0]

            if date in self.asset_returns.index:
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

        annualized_volatility = returns_series.std() * np.sqrt(12)

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


if __name__ == "__main__":
    backtest = CSVBacktest(
        start_date="2013-01-01",
        end_date="2024-12-31",
        initial_capital=100000000.0,
    )

    results = backtest.run_backtest(use_macro_view=False)
