"""
回测和评价模块
实现基于景气度指数的行业择时回测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

from .utils import calculate_direction_accuracy


@dataclass
class BacktestResult:
    """
    回测结果类
    """
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    excess_returns: pd.Series
    cumulative_returns: pd.Series


class IndustryTimingBacktest:
    """
    基于行业景气度的单行业择时回测

    基于研报中的描述：
    - 假设当季末能够知道该季度ROE_TTM变化方向，即"上帝视角"
    - 分别使用景气度指数趋势和"上帝视角"对行业超额指数进行择时
    """

    def __init__(self, initial_capital: float = 1000000.0,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.0001):
        """
        初始化回测引擎

        Parameters:
        -----------
        initial_capital : float
            初始资金
        commission_rate : float
            佣金费率
        slippage : float
            滑点
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

        self.position: int = 0
        self.cash: float = initial_capital
        self.total_capital: float = initial_capital

        self.trade_history: List[Dict] = []
        self.capital_history: List[Dict] = []

    def reset(self):
        """重置回测状态"""
        self.position = 0
        self.cash = self.initial_capital
        self.total_capital = self.initial_capital
        self.trade_history = []
        self.capital_history = []

    def execute_trade(self, date: pd.Timestamp, signal: int,
                       price: float, n_shares: int = 100):
        """
        执行交易

        Parameters:
        -----------
        date : pd.Timestamp
            交易日期
        signal : int
            交易信号 (1买入, -1卖出, 0持有)
        price : float
            价格
        n_shares : int
            交易股数（默认100的倍数）
        """
        target_position = signal * n_shares

        if target_position == self.position:
            return

        if target_position > self.position:
            shares_to_buy = target_position - self.position
            cost = shares_to_buy * price * (1 + self.commission_rate + self.slippage)

            if cost <= self.cash:
                self.cash -= cost
                self.position = target_position

                self.trade_history.append({
                    'date': date,
                    'action': 'buy',
                    'shares': shares_to_buy,
                    'price': price,
                    'cost': cost
                })

        elif target_position < self.position:
            shares_to_sell = self.position - target_position
            proceeds = shares_to_sell * price * (1 - self.commission_rate - self.slippage)

            self.cash += proceeds
            self.position = target_position

            self.trade_history.append({
                'date': date,
                'action': 'sell',
                'shares': shares_to_sell,
                'price': price,
                'proceeds': proceeds
            })

        self.total_capital = self.cash + self.position * price

        self.capital_history.append({
            'date': date,
            'cash': self.cash,
            'position_value': self.position * price,
            'total_capital': self.total_capital,
            'position': self.position
        })

    def run_backtest(self, sentiment_index: pd.Series,
                      price_series: pd.Series,
                      benchmark_returns: Optional[pd.Series] = None,
                      sentiment_threshold: float = 0.0) -> BacktestResult:
        """
        运行回测

        Parameters:
        -----------
        sentiment_index : pd.Series
            景气度指数
        price_series : pd.Series
            价格序列
        benchmark_returns : pd.Series, optional
            基准收益序列
        sentiment_threshold : float
            景气度阈值

        Returns:
        --------
        BacktestResult
            回测结果
        """
        self.reset()

        common_idx = sentiment_index.index.intersection(price_series.index)
        if len(common_idx) < 5:
            raise ValueError("有效数据点不足")

        sentiment_aligned = sentiment_index.loc[common_idx]
        price_aligned = price_series.loc[common_idx]

        sentiment_diff = sentiment_aligned.diff()
        sentiment_direction = np.sign(sentiment_diff)

        returns = price_aligned.pct_change().fillna(0)

        signals = (sentiment_direction > sentiment_threshold).astype(int)
        signals[sentiment_direction < -sentiment_threshold] = -1

        strategy_returns = []
        benchmark_aligned = benchmark_returns.loc[common_idx] if benchmark_returns is not None else returns

        for i in range(1, len(common_idx)):
            date = common_idx[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            ret = returns.iloc[i] if i < len(returns) else 0

            if signal > 0:
                strategy_ret = ret
            elif signal < 0:
                strategy_ret = -ret
            else:
                strategy_ret = 0

            strategy_returns.append(strategy_ret)

            self.execute_trade(date, signal, price_aligned.iloc[i])

        strategy_returns_series = pd.Series(
            strategy_returns,
            index=common_idx[1:]
        )

        cumulative_returns = (1 + strategy_returns_series).cumprod()
        cumulative_returns = cumulative_returns.fillna(1.0)

        total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

        n_periods = len(cumulative_returns)
        annual_return = (1 + total_return) ** (12 / n_periods) - 1 if n_periods > 0 else 0

        excess_returns = strategy_returns_series - benchmark_aligned.iloc[1:] if benchmark_returns is not None else strategy_returns_series

        sharpe_ratio = self._calculate_sharpe_ratio(excess_returns)

        max_drawdown = self._calculate_max_drawdown(cumulative_returns)

        win_rate = (strategy_returns_series > 0).sum() / len(strategy_returns_series) if len(strategy_returns_series) > 0 else 0

        n_trades = len([t for t in self.trade_history if t['action'] in ['buy', 'sell']])

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            n_trades=n_trades,
            excess_returns=excess_returns,
            cumulative_returns=cumulative_returns
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series,
                                  risk_free_rate: float = 0.03) -> float:
        """
        计算夏普比率

        Parameters:
        -----------
        returns : pd.Series
            收益序列
        risk_free_rate : float
            无风险利率（年化）

        Returns:
        --------
        float
            夏普比率
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 12

        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()

        if std_excess < 1e-10:
            return 0.0

        sharpe = mean_excess / std_excess * np.sqrt(12)

        return sharpe

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        计算最大回撤

        Parameters:
        -----------
        cumulative_returns : pd.Series
            累积收益序列

        Returns:
        --------
        float
            最大回撤
        """
        if len(cumulative_returns) < 2:
            return 0.0

        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak

        max_dd = drawdown.min()

        return max_dd


class GodViewBacktest:
    """
    "上帝视角"回测 - 假设完美预知未来

    用于对比分析
    """

    def __init__(self, initial_capital: float = 1000000.0):
        """
        初始化上帝视角回测

        Parameters:
        -----------
        initial_capital : float
            初始资金
        """
        self.initial_capital = initial_capital

    def run_backtest(self, true_direction: pd.Series,
                      price_series: pd.Series) -> Dict:
        """
        运行上帝视角回测

        Parameters:
        -----------
        true_direction : pd.Series
            真实方向（1上升，-1下降）
        price_series : pd.Series
            价格序列

        Returns:
        --------
        Dict
            回测结果
        """
        common_idx = true_direction.index.intersection(price_series.index)

        if len(common_idx) < 5:
            return {}

        direction_aligned = true_direction.loc[common_idx]
        price_aligned = price_series.loc[common_idx]

        returns = price_aligned.pct_change().fillna(0)

        strategy_returns = []
        for i in range(1, len(common_idx)):
            signal = direction_aligned.iloc[i]
            ret = returns.iloc[i]

            if signal > 0:
                strategy_ret = ret
            elif signal < 0:
                strategy_ret = -ret
            else:
                strategy_ret = 0

            strategy_returns.append(strategy_ret)

        strategy_returns_series = pd.Series(
            strategy_returns,
            index=common_idx[1:]
        )

        cumulative_returns = (1 + strategy_returns_series).cumprod()
        cumulative_returns = cumulative_returns.fillna(1.0)

        total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

        return {
            'total_return': total_return,
            'cumulative_returns': cumulative_returns,
            'n_periods': len(strategy_returns)
        }


class TimingComparison:
    """
    择时效果对比分析
    """

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def add_result(self, name: str, result: BacktestResult):
        """
        添加回测结果

        Parameters:
        -----------
        name : str
            结果名称
        result : BacktestResult
            回测结果
        """
        self.results[name] = {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'n_trades': result.n_trades
        }

    def compare(self) -> pd.DataFrame:
        """
        对比所有结果

        Returns:
        --------
        pd.DataFrame
            对比表格
        """
        comparison_data = []

        for name, result in self.results.items():
            comparison_data.append({
                'strategy': name,
                **result
            })

        return pd.DataFrame(comparison_data).set_index('strategy')

    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> str:
        """
        获取最佳策略

        Parameters:
        -----------
        metric : str
            评估指标

        Returns:
        --------
        str
            最佳策略名称
        """
        if not self.results:
            return ""

        comparison = self.compare()

        if metric not in comparison.columns:
            return ""

        return comparison[metric].idxmax()
