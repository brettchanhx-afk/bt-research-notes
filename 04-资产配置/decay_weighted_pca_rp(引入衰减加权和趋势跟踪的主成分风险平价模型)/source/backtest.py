"""
Backtesting engine module for portfolio strategies.
Provides comprehensive backtesting functionality with performance analytics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List, Callable, Union
from pathlib import Path

from .config import OUTPUT_DIR, RISK_FREE_RATE
from .utils import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_calmar_ratio
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for portfolio strategies.
    Simulates portfolio trading and calculates performance metrics.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_name: str = "Strategy",
        initial_capital: float = 1000000.0,
        rebalance_freq: str = 'M',
        transaction_cost: float = 0.001,
        risk_free_rate: float = RISK_FREE_RATE
    ):
        """
        Initialize BacktestEngine.

        Args:
            returns: DataFrame of asset returns
            prices: DataFrame of asset prices
            strategy_name: Name of the strategy
            initial_capital: Initial capital
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')
            transaction_cost: Transaction cost as fraction
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.returns = returns
        self.prices = prices
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

        self.asset_names = returns.columns.tolist()
        self.dates = returns.index

        self.weights_history = None
        self.portfolio_returns = None
        self.cumulative_returns = None
        self.equity_curve = None

        self.metrics = None

    def set_weights_generator(
        self,
        weights_function: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
    ):
        """
        Set the function that generates portfolio weights.

        Args:
            weights_function: Function that takes (returns, prices) and returns weights
        """
        self.weights_generator = weights_function

    def generate_rebalance_dates(self) -> List[datetime]:
        """
        Generate rebalancing dates based on frequency.

        Returns:
            List of rebalancing dates
        """
        if self.rebalance_freq == 'D':
            return list(self.dates)
        elif self.rebalance_freq == 'W':
            return self.dates.to_list()[::5]
        elif self.rebalance_freq == 'M':
            monthly = self.dates.to_period('M').unique().to_timestamp()
            return monthly.tolist()
        elif self.rebalance_freq == 'Q':
            quarterly = self.dates.to_period('Q').unique().to_timestamp()
            return quarterly.tolist()
        else:
            return [self.dates[0]]

    def run_backtest(self, lookback_period: int = 60) -> Dict:
        """
        Run the backtest.

        Args:
            lookback_period: Minimum lookback period for model estimation

        Returns:
            Dict with backtest results
        """
        logger.info(f"Starting backtest for {self.strategy_name}")

        rebalance_dates = self.generate_rebalance_dates()

        portfolio_values = [self.initial_capital]
        portfolio_returns_list = []
        weights_history = []

        prev_weights = None

        for i, date in enumerate(self.dates):
            if date < rebalance_dates[0]:
                continue

            if date in rebalance_dates or i == 0:
                returns_window = self.returns.iloc[max(0, i-lookback_period):i]
                prices_window = self.prices.iloc[max(0, i-lookback_period):i]

                if len(returns_window) < lookback_period // 2:
                    weights = prev_weights if prev_weights is not None else np.ones(self.returns.shape[1]) / self.returns.shape[1]
                else:
                    try:
                        weights = self.weights_generator(returns_window, prices_window)
                    except Exception as e:
                        logger.warning(f"Weight generation failed on {date}: {str(e)}")
                        weights = prev_weights if prev_weights is not None else np.ones(self.returns.shape[1]) / self.returns.shape[1]

                weights = np.nan_to_num(weights, nan=1.0/len(weights))
                weights = np.clip(weights, 0, 1)
                weights = weights / np.sum(weights)

                if prev_weights is not None and self.transaction_cost > 0:
                    turnover = np.sum(np.abs(weights - prev_weights))
                    transaction_costs = turnover * self.transaction_cost
                else:
                    transaction_costs = 0

                prev_weights = weights
            else:
                weights = prev_weights if prev_weights is not None else np.ones(self.returns.shape[1]) / self.returns.shape[1]

            weights_history.append(weights)

            daily_return = np.dot(weights, self.returns.iloc[i].values)

            if i > 0 and date in rebalance_dates:
                daily_return -= transaction_costs

            portfolio_returns_list.append(daily_return)

            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)

        self.weights_history = pd.DataFrame(
            weights_history,
            index=self.dates[:len(weights_history)],
            columns=self.asset_names
        )

        self.portfolio_returns = pd.Series(
            portfolio_returns_list,
            index=self.dates[:len(portfolio_returns_list)]
        )

        self.cumulative_returns = (1 + self.portfolio_returns).cumprod() - 1

        self.equity_curve = pd.Series(
            portfolio_values[1:],
            index=self.dates[:len(portfolio_values)-1]
        )

        self.metrics = self.calculate_metrics()

        logger.info(f"Backtest completed. Total return: {self.metrics['total_return']:.2%}")

        return self.metrics

    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dict with performance metrics
        """
        if self.portfolio_returns is None:
            raise ValueError("Run backtest first")

        total_return = self.cumulative_returns.iloc[-1] if len(self.cumulative_returns) > 0 else 0

        annualized_return = (1 + total_return) ** (252 / len(self.portfolio_returns)) - 1 if len(self.portfolio_returns) > 0 else 0

        annualized_volatility = self.portfolio_returns.std() * np.sqrt(252) if len(self.portfolio_returns) > 0 else 0

        sharpe = calculate_sharpe_ratio(
            self.portfolio_returns,
            self.risk_free_rate,
            252
        ) if len(self.portfolio_returns) > 0 else 0

        max_dd = calculate_max_drawdown(self.equity_curve / self.equity_curve.iloc[0]) if len(self.equity_curve) > 0 else 0

        calmar = calculate_calmar_ratio(annualized_return, max_dd) if max_dd != 0 else 0

        win_rate = (self.portfolio_returns > 0).sum() / len(self.portfolio_returns) if len(self.portfolio_returns) > 0 else 0

        avg_win = self.portfolio_returns[self.portfolio_returns > 0].mean() if (self.portfolio_returns > 0).any() else 0
        avg_loss = self.portfolio_returns[self.portfolio_returns < 0].mean() if (self.portfolio_returns < 0).any() else 0

        return {
            'strategy_name': self.strategy_name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'num_trades': len(self.portfolio_returns),
            'final_value': self.equity_curve.iloc[-1] if len(self.equity_curve) > 0 else self.initial_capital
        }

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve."""
        return self.equity_curve

    def get_weights_history(self) -> pd.DataFrame:
        """Get weight history."""
        return self.weights_history

    def get_portfolio_returns(self) -> pd.Series:
        """Get portfolio returns series."""
        return self.portfolio_returns

    def save_results(self, output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
        """
        Save backtest results to files.

        Args:
            output_dir: Output directory

        Returns:
            Dict mapping result types to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"{self.strategy_name}_{timestamp}"

        results = {}

        metrics_df = pd.DataFrame([self.metrics])
        metrics_path = output_dir / f"{prefix}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        results['metrics'] = metrics_path

        if self.weights_history is not None:
            weights_path = output_dir / f"{prefix}_weights.csv"
            self.weights_history.to_csv(weights_path)
            results['weights'] = weights_path

        returns_path = output_dir / f"{prefix}_returns.csv"
        returns_df = pd.DataFrame({
            'date': self.portfolio_returns.index,
            'return': self.portfolio_returns.values,
            'cumulative_return': self.cumulative_returns.values,
            'equity_curve': self.equity_curve.values
        })
        returns_df.to_csv(returns_path, index=False)
        results['returns'] = returns_path

        logger.info(f"Results saved to {output_dir}")

        return results


class MultiStrategyBacktest:
    """
    Backtest multiple strategies for comparison.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        strategy_names: List[str],
        initial_capital: float = 1000000.0,
        rebalance_freq: str = 'M',
        transaction_cost: float = 0.001
    ):
        """
        Initialize MultiStrategyBacktest.

        Args:
            returns: DataFrame of asset returns
            prices: DataFrame of asset prices
            strategy_names: List of strategy names
            initial_capital: Initial capital
            rebalance_freq: Rebalancing frequency
            transaction_cost: Transaction cost
        """
        self.returns = returns
        self.prices = prices
        self.strategy_names = strategy_names
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

        self.backtests = {}
        self.comparison_df = None

    def add_strategy(
        self,
        name: str,
        weights_function: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
    ):
        """
        Add a strategy to the backtest.

        Args:
            name: Strategy name
            weights_function: Function to generate weights
        """
        engine = BacktestEngine(
            self.returns,
            self.prices,
            strategy_name=name,
            initial_capital=self.initial_capital,
            rebalance_freq=self.rebalance_freq,
            transaction_cost=self.transaction_cost
        )
        engine.set_weights_generator(weights_function)
        self.backtests[name] = engine

    def run_all(self) -> pd.DataFrame:
        """
        Run backtests for all strategies.

        Returns:
            DataFrame comparing all strategy metrics
        """
        all_metrics = []

        for name, engine in self.backtests.items():
            logger.info(f"Running backtest for {name}")
            engine.run_backtest()
            metrics = engine.metrics
            all_metrics.append(metrics)

        self.comparison_df = pd.DataFrame(all_metrics)
        self.comparison_df.set_index('strategy_name', inplace=True)

        return self.comparison_df

    def get_comparison(self) -> pd.DataFrame:
        """Get strategy comparison DataFrame."""
        return self.comparison_df

    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> str:
        """
        Get best performing strategy by metric.

        Args:
            metric: Metric to use for ranking

        Returns:
            Name of best strategy
        """
        if self.comparison_df is None:
            self.run_all()

        return self.comparison_df[metric].idxmax()


def plot_backtest_results(
    backtest: BacktestEngine,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot backtest results.

    Args:
        backtest: BacktestEngine instance
        save_path: Path to save plot
        show: Whether to display plot
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        equity = backtest.get_equity_curve()
        equity.plot(ax=axes[0], title=f"{backtest.strategy_name} - Equity Curve")
        axes[0].set_ylabel('Portfolio Value')

        backtest.get_portfolio_returns().hist(
            ax=axes[1],
            bins=50,
            title='Returns Distribution'
        )
        axes[1].set_xlabel('Daily Return')
        axes[0].set_ylabel('Frequency')

        weights = backtest.get_weights_history()
        if weights is not None:
            weights.plot.area(
                ax=axes[2],
                stacked=True,
                title='Portfolio Weights Over Time'
            )
            axes[2].set_ylabel('Weight')
            axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2017-01-01', periods=200, freq='D')
    assets = ['Asset_A', 'Asset_B', 'Asset_C']

    returns_data = np.random.randn(200, 3) * 0.02
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)

    prices_data = 100 + np.cumsum(returns_data, axis=0)
    prices = pd.DataFrame(prices_data, index=dates, columns=assets)

    def equal_weight_generator(returns_df, prices_df):
        return np.ones(returns_df.shape[1]) / returns_df.shape[1]

    engine = BacktestEngine(
        returns,
        prices,
        strategy_name="Equal Weight",
        initial_capital=1000000,
        rebalance_freq='M'
    )

    engine.set_weights_generator(equal_weight_generator)
    metrics = engine.run_backtest()

    print("\nBacktest Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")