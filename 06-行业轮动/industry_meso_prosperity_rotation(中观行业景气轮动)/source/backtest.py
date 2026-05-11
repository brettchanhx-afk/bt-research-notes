import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float


class Backtester:
    def __init__(
        self,
        initial_capital: float = 1000000,
        transaction_cost: float = 0.0,
        risk_free_rate: float = 0.03
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.portfolio_value = initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []

    def reset(self):
        self.portfolio_value = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []

    def rebalance(
        self,
        date: pd.Timestamp,
        selected_industries: List[str],
        weights: Optional[List[float]] = None,
        current_prices: Optional[Dict[str, float]] = None
    ):
        if not selected_industries:
            return

        if weights is None:
            weights = [1.0 / len(selected_industries)] * len(selected_industries)

        old_positions = self.positions.copy()
        self.positions = {
            industry: weight for industry, weight in zip(selected_industries, weights)
        }

        turnover = 0
        for industry in set(old_positions.keys()) | set(self.positions.keys()):
            old_weight = old_positions.get(industry, 0)
            new_weight = self.positions.get(industry, 0)
            turnover += abs(new_weight - old_weight)

        if self.transaction_cost > 0 and turnover > 0:
            cost = self.portfolio_value * turnover * self.transaction_cost
            self.portfolio_value -= cost

        self.trade_history.append({
            'date': date,
            'action': 'rebalance',
            'industries': selected_industries,
            'weights': weights,
            'turnover': turnover,
            'portfolio_value': self.portfolio_value
        })

    def update_portfolio(
        self,
        date: pd.Timestamp,
        returns: Dict[str, float]
    ):
        if not self.positions:
            return

        portfolio_return = 0
        for industry, weight in self.positions.items():
            if industry in returns:
                portfolio_return += returns[industry] * weight

        self.portfolio_value *= (1 + portfolio_return)

        self.portfolio_history.append({
            'date': date,
            'portfolio_return': portfolio_return,
            'portfolio_value': self.portfolio_value
        })

    def get_portfolio_value_series(self) -> pd.Series:
        if not self.portfolio_history:
            return pd.Series()

        df = pd.DataFrame(self.portfolio_history)
        return df.set_index('date')['portfolio_value']

    def get_returns_series(self) -> pd.Series:
        if not self.portfolio_history:
            return pd.Series()

        df = pd.DataFrame(self.portfolio_history)
        return df.set_index('date')['portfolio_return']


class PerformanceAnalyzer:
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        returns: pd.Series,
        periods_per_year: int = 12
    ) -> PerformanceMetrics:
        if len(returns) < 2:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0)

        annual_return = (1 + returns.mean()) ** periods_per_year - 1

        annual_volatility = returns.std() * np.sqrt(periods_per_year)

        risk_free_monthly = self.risk_free_rate / periods_per_year
        excess_returns = returns - risk_free_monthly
        sharpe_ratio = (
            excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
            if returns.std() > 0 else 0
        )

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        calmar_ratio = (
            annual_return / abs(max_drawdown)
            if max_drawdown != 0 else 0
        )

        win_rate = (returns > 0).sum() / len(returns)

        return PerformanceMetrics(
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate
        )

    def calculate_cumulative_returns(
        self,
        returns: pd.Series
    ) -> pd.Series:
        return (1 + returns).cumprod()

    def calculate_drawdown(
        self,
        cumulative_returns: pd.Series
    ) -> pd.Series:
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown

    def calculate_excess_returns(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> pd.Series:
        return portfolio_returns - benchmark_returns

    def calculate_tracking_error(
        self,
        excess_returns: pd.Series
    ) -> float:
        return excess_returns.std() * np.sqrt(12)

    def calculate_information_ratio(
        self,
        excess_returns: pd.Series
    ) -> float:
        tracking_error = self.calculate_tracking_error(excess_returns)
        if tracking_error == 0:
            return 0
        return excess_returns.mean() * 12 / tracking_error

    def get_summary_stats(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        metrics = self.calculate_metrics(returns)

        summary = {
            '年化收益': f"{metrics.annual_return:.2%}",
            '年化波动': f"{metrics.annual_volatility:.2%}",
            '夏普比率': f"{metrics.sharpe_ratio:.2f}",
            '最大回撤': f"{metrics.max_drawdown:.2%}",
            '卡玛比率': f"{metrics.calmar_ratio:.2f}",
            '胜率': f"{metrics.win_rate:.2%}"
        }

        if benchmark_returns is not None:
            excess = self.calculate_excess_returns(returns, benchmark_returns)
            summary['超额年化收益'] = f"{(excess.mean() * 12):.2%}"
            summary['信息比率'] = f"{self.calculate_information_ratio(excess):.2f}"

        return summary

    def plot_performance(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            cumulative = self.calculate_cumulative_returns(returns)
            axes[0].plot(cumulative.index, cumulative.values, label='Portfolio')
            if benchmark_returns is not None:
                bench_cumulative = self.calculate_cumulative_returns(benchmark_returns)
                axes[0].plot(bench_cumulative.index, bench_cumulative.values, label='Benchmark')
            axes[0].set_title('Cumulative Returns')
            axes[0].legend()
            axes[0].grid(True)

            drawdown = self.calculate_drawdown(cumulative)
            axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3)
            axes[1].set_title('Drawdown')
            axes[1].grid(True)

            rolling_sharpe = returns.rolling(12).apply(
                lambda x: (x.mean() - self.risk_free_rate/12) / x.std() * np.sqrt(12)
                if x.std() > 0 else 0
            )
            axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, label='Rolling Sharpe (12M)')
            axes[2].axhline(y=0, color='r', linestyle='--')
            axes[2].set_title('Rolling Sharpe Ratio')
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)

            return fig

        except ImportError:
            print("Matplotlib not available, skipping plot")
            return None


class IndustryBacktester:
    def __init__(
        self,
        backtester: Backtester,
        analyzer: PerformanceAnalyzer
    ):
        self.backtester = backtester
        self.analyzer = analyzer

    def run_backtest(
        self,
        industry_returns: Dict[str, pd.Series],
        selected_industries_history: List[Tuple[pd.Timestamp, List[str]]],
        weights_history: Optional[List[List[float]]] = None,
        rebalance_freq: str = 'M'
    ) -> Tuple[pd.Series, pd.DataFrame]:
        all_returns = pd.concat(industry_returns.values(), axis=1)
        all_returns = all_returns.sort_index()

        if weights_history is None:
            weights_history = [None] * len(selected_industries_history)

        for i, (date, selected) in enumerate(selected_industries_history):
            if i < len(selected_industries_history) - 1:
                next_date = selected_industries_history[i + 1][0]
            else:
                next_date = all_returns.index[-1]

            mask = (all_returns.index > date) & (all_returns.index <= next_date)

            weights = weights_history[i] if weights_history[i] else [1/len(selected)] * len(selected)

            for idx in all_returns[mask].index:
                daily_returns = all_returns.loc[idx]
                self.backtester.update_portfolio(idx, daily_returns.to_dict())

        self.backtester.rebalance(
            selected_industries_history[0][0],
            selected_industries_history[0][1],
            weights_history[0] if weights_history else None
        )

        for i in range(1, len(selected_industries_history)):
            date, selected = selected_industries_history[i]
            weights = weights_history[i] if weights_history[i] else [1/len(selected)] * len(selected)
            self.backtester.rebalance(date, selected, weights)

        returns = self.backtester.get_returns_series()
        portfolio_values = self.backtester.get_portfolio_value_series()

        return returns, portfolio_values

    def compare_with_benchmark(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> pd.DataFrame:
        portfolio_summary = self.analyzer.get_summary_stats(portfolio_returns, benchmark_returns)
        benchmark_summary = self.analyzer.get_summary_stats(benchmark_returns)

        comparison = pd.DataFrame({
            'Portfolio': portfolio_summary,
            'Benchmark': benchmark_summary
        })

        return comparison


def calculate_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    monthly = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    return monthly


def calculate_annual_returns(returns: pd.Series) -> pd.Series:
    annual = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    return annual


def generate_performance_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_name: str = "Strategy"
) -> Dict:
    analyzer = PerformanceAnalyzer()

    portfolio_metrics = analyzer.calculate_metrics(portfolio_returns)
    benchmark_metrics = analyzer.calculate_metrics(benchmark_returns)

    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(12)
    information_ratio = (
        excess_returns.mean() * 12 / tracking_error
        if tracking_error > 0 else 0
    )

    annual_return_diff = portfolio_metrics.annual_return - benchmark_metrics.annual_return

    win_rate = (portfolio_returns > benchmark_returns).sum() / len(portfolio_returns)

    report = {
        'Strategy': strategy_name,
        'Annual Return': f"{portfolio_metrics.annual_return:.2%}",
        'Annual Volatility': f"{portfolio_metrics.annual_volatility:.2%}",
        'Sharpe Ratio': f"{portfolio_metrics.sharpe_ratio:.2f}",
        'Max Drawdown': f"{portfolio_metrics.max_drawdown:.2%}",
        'Calmar Ratio': f"{portfolio_metrics.calmar_ratio:.2f}",
        'Win Rate': f"{portfolio_metrics.win_rate:.2%}",
        'Excess Annual Return': f"{annual_return_diff:.2%}",
        'Tracking Error': f"{tracking_error:.2%}",
        'Information Ratio': f"{information_ratio:.2f}",
        'Benchmark Annual Return': f"{benchmark_metrics.annual_return:.2%}",
        'Benchmark Sharpe': f"{benchmark_metrics.sharpe_ratio:.2f}",
    }

    return report


def print_performance_report(report: Dict):
    print("\n" + "=" * 60)
    print("Performance Report")
    print("=" * 60)
    for key, value in report.items():
        print(f"{key:25s}: {value}")
    print("=" * 60)


def main():
    print("Testing Backtest Module...")

    dates = pd.date_range('2019-01-01', '2022-06-30', freq='M')
    n = len(dates)

    np.random.seed(42)
    industry_returns = {
        f'industry_{i}': pd.Series(
            np.random.randn(n) * 0.05 + 0.01,
            index=dates
        )
        for i in range(5)
    }

    all_returns = pd.DataFrame(industry_returns)
    equal_weight = all_returns.mean(axis=1)

    benchmark = pd.Series(
        np.random.randn(n) * 0.03 + 0.005,
        index=dates,
        name='benchmark'
    )

    backtester = Backtester(initial_capital=1000000, transaction_cost=0.001)
    analyzer = PerformanceAnalyzer()

    metrics = analyzer.calculate_metrics(equal_weight)
    print(f"\nEqual Weight Portfolio:")
    print(f"  Annual Return: {metrics.annual_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")

    report = generate_performance_report(equal_weight, benchmark, "Equal Weight")
    print_performance_report(report)


if __name__ == "__main__":
    main()
