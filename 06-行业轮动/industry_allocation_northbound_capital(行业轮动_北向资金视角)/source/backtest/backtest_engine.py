"""
回测引擎模块
"""

from typing import Optional, Dict, List, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import json


class BacktestEngine:
    """
    回测引擎

    用于执行策略回测，计算业绩指标，生成回测报告
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 52,
    ):
        """
        初始化回测引擎

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            risk_free_rate: 无风险利率
            periods_per_year: 年化周期数
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        self.returns = None
        self.positions = None
        self.trades = None
        self.benchmark_returns = None

        self.portfolio_value = None
        self.cumulative_return = None
        self.metrics = None

    def set_returns(
        self,
        returns: pd.Series,
        positions: Optional[pd.DataFrame] = None,
    ) -> "BacktestEngine":
        """
        设置收益数据

        Args:
            returns: 策略收益序列
            positions: 持仓数据

        Returns:
            self
        """
        self.returns = returns.dropna()
        self.positions = positions
        return self

    def set_benchmark(
        self,
        benchmark_returns: pd.Series,
    ) -> "BacktestEngine":
        """
        设置基准收益

        Args:
            benchmark_returns: 基准收益序列

        Returns:
            self
        """
        self.benchmark_returns = benchmark_returns.reindex(self.returns.index).fillna(0)
        return self

    def run(self) -> "BacktestEngine":
        """
        执行回测

        Returns:
            self
        """
        if self.returns is None:
            raise ValueError("Returns not set. Please call set_returns first.")

        self.portfolio_value = self.initial_capital * (1 + self.returns).cumprod()
        self.cumulative_return = self.portfolio_value / self.initial_capital - 1

        self.metrics = self._calculate_metrics()

        return self

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        计算业绩指标

        Returns:
            业绩指标字典
        """
        if self.returns is None or len(self.returns) == 0:
            return {}

        total_return = self.cumulative_return.iloc[-1] if len(self.cumulative_return) > 0 else 0

        ann_return = (1 + total_return) ** (
            self.periods_per_year / len(self.returns)
        ) - 1 if len(self.returns) > 0 else 0

        ann_vol = self.returns.std() * np.sqrt(self.periods_per_year)

        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0

        wealth_index = self.cumulative_return + 1
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()

        drawdown_duration = self._calculate_drawdown_duration()
        max_drawdown_duration = drawdown_duration.max() if len(drawdown_duration) > 0 else 0

        win_rate = (self.returns > 0).sum() / len(self.returns)

        profit_loss_ratio = self._calculate_profit_loss_ratio()

        excess_returns = None
        excess_metrics = None
        if self.benchmark_returns is not None:
            aligned_benchmark = self.benchmark_returns.reindex(self.returns.index).fillna(0)
            excess_returns = self.returns - aligned_benchmark
            aligned_benchmark_cum = (1 + aligned_benchmark).cumprod() - 1
            strategy_cum = self.cumulative_return

            tracking_error = excess_returns.std() * np.sqrt(self.periods_per_year)
            info_ratio = (
                excess_returns.mean() * self.periods_per_year / tracking_error
                if tracking_error > 0
                else 0
            )

            excess_metrics = {
                "tracking_error": tracking_error,
                "information_ratio": info_ratio,
            }

        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "calmar_ratio": calmar,
        }

        if excess_metrics:
            metrics.update(excess_metrics)

        return metrics

    def _calculate_drawdown_duration(self) -> pd.Series:
        """计算回撤持续期"""
        if self.cumulative_return is None:
            return pd.Series()

        wealth_index = self.cumulative_return + 1
        previous_peaks = wealth_index.cummax()
        is_drawdown = wealth_index < previous_peaks

        drawdown_groups = (~is_drawdown).cumsum()
        drawdown_duration = drawdown_groups.map(drawdown_groups.value_counts())

        return drawdown_duration

    def _calculate_profit_loss_ratio(self) -> float:
        """计算盈亏比"""
        if self.returns is None or len(self.returns) == 0:
            return 0

        profits = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]

        avg_profit = profits.mean() if len(profits) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        return avg_profit / avg_loss if avg_loss > 0 else 0

    def get_results(self) -> Dict:
        """
        获取回测结果

        Returns:
            回测结果字典
        """
        return {
            "returns": self.returns,
            "positions": self.positions,
            "portfolio_value": self.portfolio_value,
            "cumulative_return": self.cumulative_return,
            "metrics": self.metrics,
        }

    def save_results(self, output_path: str) -> None:
        """
        保存回测结果

        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()

        if results["returns"] is not None:
            results["returns"].to_csv(output_path / "returns.csv")

        if results["cumulative_return"] is not None:
            results["cumulative_return"].to_csv(output_path / "cumulative_return.csv")

        if results["positions"] is not None:
            results["positions"].to_csv(output_path / "positions.csv")

        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results["metrics"], f, indent=2, ensure_ascii=False)

        print(f"回测结果已保存至: {output_path}")


class MultiStrategyBacktest:
    """
    多策略回测比较器
    """

    def __init__(self):
        self.strategies = {}

    def add_strategy(
        self,
        name: str,
        returns: pd.Series,
        positions: Optional[pd.DataFrame] = None,
    ) -> "MultiStrategyBacktest":
        """
        添加策略

        Args:
            name: 策略名称
            returns: 收益序列
            positions: 持仓数据

        Returns:
            self
        """
        engine = BacktestEngine(start_date="", end_date="")
        engine.set_returns(returns, positions)
        engine.run()

        self.strategies[name] = {
            "returns": returns,
            "positions": positions,
            "engine": engine,
            "metrics": engine.metrics,
            "cumulative_return": engine.cumulative_return,
        }

        return self

    def compare_strategies(self) -> pd.DataFrame:
        """
        比较各策略业绩

        Returns:
            策略比较表
        """
        comparison_data = []

        for name, data in self.strategies.items():
            metrics = data["metrics"]
            row = {"strategy": name}
            row.update(metrics)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index("strategy")

        return comparison_df

    def get_best_strategy(self, metric: str = "sharpe_ratio") -> str:
        """
        获取最佳策略

        Args:
            metric: 评估指标

        Returns:
            最佳策略名称
        """
        best_name = None
        best_value = -np.inf

        for name, data in self.strategies.items():
            value = data["metrics"].get(metric, -np.inf)
            if value > best_value:
                best_value = value
                best_name = name

        return best_name


def run_backtest(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    initial_capital: float = 1000000,
    output_path: Optional[str] = None,
) -> Dict:
    """
    便捷函数：运行回测

    Args:
        strategy_returns: 策略收益
        benchmark_returns: 基准收益
        initial_capital: 初始资金
        output_path: 输出路径

    Returns:
        回测结果
    """
    engine = BacktestEngine(
        start_date=strategy_returns.index[0] if len(strategy_returns) > 0 else "",
        end_date=strategy_returns.index[-1] if len(strategy_returns) > 0 else "",
        initial_capital=initial_capital,
    )

    engine.set_returns(strategy_returns)

    if benchmark_returns is not None:
        engine.set_benchmark(benchmark_returns)

    engine.run()

    if output_path:
        engine.save_results(output_path)

    return engine.get_results()
