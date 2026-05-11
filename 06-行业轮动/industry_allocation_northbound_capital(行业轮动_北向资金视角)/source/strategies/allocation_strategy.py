"""
行业配置策略模块

基于北向资金因子的行业配置策略
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from ..utils.data_utils import rank_factor


class IndustryAllocationStrategy:
    """
    行业配置策略

    基于北向资金因子进行行业配置，选择因子值最高的行业
    """

    def __init__(self, n_industries: int = 3, frequency: str = "weekly"):
        """
        初始化行业配置策略

        Args:
            n_industries: 配置行业数量
            frequency: 调仓频率
        """
        self.n_industries = n_industries
        self.frequency = frequency
        self.positions = None
        self.returns = None

    def generate_signals(
        self,
        factor_data: pd.DataFrame,
        industry_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            factor_data: 因子数据
            industry_returns: 行业收益

        Returns:
            信号数据
        """
        df = factor_data.copy()
        df = df.sort_values(["trade_date", "factor"], ascending=[True, False])

        dates = df["trade_date"].unique()
        signals_list = []

        for date in dates:
            date_factor = df[df["trade_date"] == date]
            top_industries = date_factor.head(self.n_industries)["industry_code"].tolist()

            for ind in date_factor["industry_code"].unique():
                signals_list.append(
                    {
                        "trade_date": date,
                        "industry_code": ind,
                        "signal": 1 if ind in top_industries else 0,
                    }
                )

        signals_df = pd.DataFrame(signals_list)
        signals_df = signals_df.merge(
            industry_returns, on=["trade_date", "industry_code"], how="left"
        )
        signals_df["position"] = signals_df.groupby("trade_date")["signal"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )

        signals_df["strategy_return"] = signals_df["position"].shift(1) * signals_df["return"]

        self.positions = signals_df.pivot(
            index="trade_date", columns="industry_code", values="position"
        )
        self.returns = signals_df.groupby("trade_date")["strategy_return"].sum()

        return signals_df

    def calculate_portfolio_return(self) -> pd.Series:
        """计算组合收益"""
        if self.returns is None:
            return pd.Series()
        return (1 + self.returns).cumprod() - 1

    def get_performance_metrics(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 52,
    ) -> Dict[str, float]:
        """
        计算业绩指标

        Args:
            benchmark_returns: 基准收益
            risk_free_rate: 无风险利率
            periods_per_year: 年化周期数

        Returns:
            业绩指标
        """
        if self.returns is None or len(self.returns) == 0:
            return {}

        cumulative = self.calculate_portfolio_return()
        total_return = cumulative.iloc[-1] if len(cumulative) > 0 else 0

        ann_return = (1 + total_return) ** (periods_per_year / len(self.returns)) - 1
        ann_vol = self.returns.std() * np.sqrt(periods_per_year)

        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        wealth_index = 1 + cumulative
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()

        excess_return = None
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(self.returns.index).fillna(0)
            excess_return = (self.returns - aligned_benchmark).mean() * periods_per_year

        win_rate = (self.returns > 0).sum() / len(self.returns)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "excess_return": excess_return,
            "win_rate": win_rate,
        }


class CompositeFactorStrategy:
    """
    复合因子策略

    将多个因子复合后进行行业配置
    """

    def __init__(
        self,
        n_industries: int = 3,
        frequency: str = "weekly",
    ):
        """
        初始化复合因子策略

        Args:
            n_industries: 配置行业数量
            frequency: 调仓频率
        """
        self.n_industries = n_industries
        self.frequency = frequency

    def composite_factors(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        method: str = "rank_avg",
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        复合因子

        Args:
            factors_dict: 因子字典
            method: 复合方法 (rank_avg, weighted_avg)
            weights: 因子权重

        Returns:
            复合因子
        """
        merged = None

        for name, df in factors_dict.items():
            df = df.copy()
            df["factor_rank"] = df.groupby("trade_date")["factor"].rank(pct=True)

            if merged is None:
                merged = df[["trade_date", "industry_code", "factor_rank"]].copy()
                merged = merged.rename(columns={"factor_rank": name})
            else:
                temp = df[["trade_date", "industry_code", "factor_rank"]].copy()
                temp = temp.rename(columns={"factor_rank": name})
                merged = merged.merge(temp, on=["trade_date", "industry_code"], how="outer")

        if merged is None:
            return pd.DataFrame()

        factor_cols = list(factors_dict.keys())

        if weights is None:
            weights = {col: 1.0 / len(factor_cols) for col in factor_cols}

        if method == "rank_avg":
            merged["composite_factor"] = merged[factor_cols].mean(axis=1)
        elif method == "weighted_avg":
            merged["composite_factor"] = sum(
                merged[col] * weights[col] for col in factor_cols
            )

        return merged[["trade_date", "industry_code", "composite_factor"]]

    def run_strategy(
        self,
        composite_factor: pd.DataFrame,
        industry_returns: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        运行策略

        Args:
            composite_factor: 复合因子
            industry_returns: 行业收益

        Returns:
            (信号数据, 业绩指标)
        """
        strategy = IndustryAllocationStrategy(
            n_industries=self.n_industries,
            frequency=self.frequency,
        )

        signals = strategy.generate_signals(composite_factor, industry_returns)
        metrics = strategy.get_performance_metrics()

        return signals, metrics


class LayerBacktestStrategy:
    """
    分层回测策略

    按因子值大小将行业分为5层，回测各层表现
    """

    def __init__(self, n_layers: int = 5):
        """
        初始化分层回测策略

        Args:
            n_layers: 分层数量
        """
        self.n_layers = n_layers
        self.layer_returns = None
        self.layer_metrics = None

    def run_layer_backtest(
        self,
        factor_data: pd.DataFrame,
        industry_returns: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        运行分层回测

        Args:
            factor_data: 因子数据
            industry_returns: 行业收益

        Returns:
            各层回测结果
        """
        df = factor_data.copy()
        df = df.merge(
            industry_returns, on=["trade_date", "industry_code"], how="left"
        )

        df["layer"] = df.groupby("trade_date")["factor"].transform(
            lambda x: pd.qcut(x, q=self.n_layers, labels=False, duplicates="drop") + 1
        )

        layer_results = {}
        for layer in range(1, self.n_layers + 1):
            layer_df = df[df["layer"] == layer].copy()
            layer_df["position"] = 1 / len(layer_df)

            layer_df["strategy_return"] = layer_df["position"].shift(1) * layer_df["return"]
            layer_returns = layer_df.groupby("trade_date")["strategy_return"].sum()

            layer_cum_return = (1 + layer_returns).cumprod() - 1
            layer_results[f"layer_{layer}"] = {
                "returns": layer_returns,
                "cumulative_return": layer_cum_return,
                "metrics": self._calculate_layer_metrics(layer_returns),
            }

        df["long_short_return"] = (
            df[df["layer"] == 1].groupby("trade_date")["return"].mean()
            - df[df["layer"] == self.n_layers].groupby("trade_date")["return"].mean()
        )

        self.layer_returns = layer_returns
        self.layer_metrics = layer_results

        return layer_results

    def _calculate_layer_metrics(
        self,
        returns: pd.Series,
        periods_per_year: int = 52,
    ) -> Dict[str, float]:
        """计算分层业绩指标"""
        if len(returns) == 0:
            return {}

        cumulative = (1 + returns).cumprod() - 1
        total_return = cumulative.iloc[-1] if len(cumulative) > 0 else 0
        ann_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(periods_per_year)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
        }


def run_allocation_strategy(
    factor_data: pd.DataFrame,
    industry_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_industries: int = 3,
) -> Dict:
    """
    运行行业配置策略

    Args:
        factor_data: 因子数据
        industry_returns: 行业收益
        benchmark_returns: 基准收益
        n_industries: 配置行业数量

    Returns:
        策略结果
    """
    strategy = IndustryAllocationStrategy(n_industries=n_industries)
    signals = strategy.generate_signals(factor_data, industry_returns)
    metrics = strategy.get_performance_metrics(benchmark_returns)

    return {
        "signals": signals,
        "cumulative_return": strategy.calculate_portfolio_return(),
        "metrics": metrics,
        "positions": strategy.positions,
        "returns": strategy.returns,
    }
