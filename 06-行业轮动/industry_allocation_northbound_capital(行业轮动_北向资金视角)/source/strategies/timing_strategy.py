"""
择时策略模块

基于北向资金情绪指数的择时策略
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np


class SentimentTimingStrategy:
    """
    情绪择时策略

    基于北向资金情绪指数，当情绪指数高于阈值时持有，低于阈值时空仓
    """

    def __init__(self, long_threshold: float = 0.5, short_threshold: float = -0.5):
        """
        初始化择时策略

        Args:
            long_threshold: 做多阈值
            short_threshold: 做空阈值
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.positions = None
        self.returns = None

    def generate_signals(
        self,
        sentiment_index: pd.DataFrame,
        index_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            sentiment_index: 情绪指数
            index_returns: 指数收益

        Returns:
            信号数据
        """
        df = sentiment_index.copy()
        df = df.set_index("trade_date")
        df["index_return"] = index_returns

        df["signal"] = 0
        df.loc[df["sentiment_index_normalized"] > self.long_threshold, "signal"] = 1
        df.loc[df["sentiment_index_normalized"] < self.short_threshold, "signal"] = -1

        df["position"] = df["signal"].shift(1).fillna(0)

        df["strategy_return"] = df["position"] * df["index_return"]

        self.positions = df["position"]
        self.returns = df["strategy_return"]

        return df[["signal", "position", "strategy_return", "sentiment_index_normalized"]]

    def calculate_cumulative_return(self) -> pd.Series:
        """计算累计收益"""
        if self.returns is None:
            return pd.Series()
        return (1 + self.returns).cumprod() - 1

    def get_performance_metrics(
        self,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 52,
    ) -> Dict[str, float]:
        """
        计算业绩指标

        Args:
            risk_free_rate: 无风险利率
            periods_per_year: 年化周期数

        Returns:
            业绩指标字典
        """
        if self.returns is None or len(self.returns) == 0:
            return {}

        cumulative = self.calculate_cumulative_return()
        total_return = cumulative.iloc[-1] if len(cumulative) > 0 else 0

        ann_return = (1 + total_return) ** (periods_per_year / len(self.returns)) - 1
        ann_vol = self.returns.std() * np.sqrt(periods_per_year)

        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        wealth_index = 1 + cumulative
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()

        win_rate = (self.returns > 0).sum() / len(self.returns)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }


class MultiThresholdStrategy:
    """
    多阈值择时策略

    根据不同的情绪指数阈值构建多个策略
    """

    def __init__(self):
        self.strategies = {}

    def run_multiple_thresholds(
        self,
        sentiment_index: pd.DataFrame,
        index_returns: pd.Series,
        thresholds: List[float],
    ) -> Dict[str, Dict]:
        """
        运行多阈值策略

        Args:
            sentiment_index: 情绪指数
            index_returns: 指数收益
            thresholds: 阈值列表

        Returns:
            各阈值的策略结果
        """
        results = {}

        for threshold in thresholds:
            strategy = SentimentTimingStrategy(
                long_threshold=threshold,
                short_threshold=-threshold,
            )
            strategy.generate_signals(sentiment_index, index_returns)
            results[f"threshold_{threshold}"] = {
                "strategy": strategy,
                "signals": strategy.positions,
                "returns": strategy.returns,
                "metrics": strategy.get_performance_metrics(),
            }

        self.strategies = results
        return results

    def get_best_strategy(self, metric: str = "sharpe_ratio") -> str:
        """
        获取最佳策略

        Args:
            metric: 评估指标

        Returns:
            最佳策略名称
        """
        if not self.strategies:
            return None

        best_name = None
        best_value = -np.inf

        for name, data in self.strategies.items():
            value = data["metrics"].get(metric, -np.inf)
            if value > best_value:
                best_value = value
                best_name = name

        return best_name


def run_timing_strategy(
    sentiment_index: pd.DataFrame,
    benchmark_returns: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, any]:
    """
    运行择时策略

    Args:
        sentiment_index: 情绪指数
        benchmark_returns: 基准收益
        threshold: 阈值

    Returns:
        策略结果
    """
    strategy = SentimentTimingStrategy(
        long_threshold=threshold,
        short_threshold=-threshold,
    )

    signals = strategy.generate_signals(sentiment_index, benchmark_returns)
    metrics = strategy.get_performance_metrics()

    return {
        "signals": signals,
        "cumulative_return": strategy.calculate_cumulative_return(),
        "metrics": metrics,
        "strategy": strategy,
    }
