"""
回测引擎模块
实现趋势追踪策略的回测功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

@dataclass
class BacktestResult:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    monthly_win_rate: float
    equity_curve: pd.Series
    monthly_returns: pd.Series
    yearly_returns: pd.Series

class BacktestEngine:
    def __init__(self, rebalance_freq: str = 'monthly',
                 target_volatility: Optional[float] = None,
                 commission_rate: float = 0.001):
        self.rebalance_freq = rebalance_freq
        self.target_volatility = target_volatility
        self.commission_rate = commission_rate

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        return returns

    def generate_signals(self, indicator_values: pd.Series) -> pd.Series:
        signals = pd.Series(index=indicator_values.index, dtype=int)
        signals[indicator_values > 0] = 1
        signals[indicator_values <= 0] = 0
        return signals

    def resample_signals(self, signals: pd.Series, freq: str = 'monthly') -> pd.Series:
        if freq == 'monthly':
            resampled = signals.resample('MS').last()
        elif freq == 'weekly':
            resampled = signals.resample('WE').last()
        else:
            resampled = signals
        return resampled

    def time_series_momentum_backtest(self, prices: pd.DataFrame,
                                       signals: pd.DataFrame,
                                       initial_capital: float = 1000000) -> BacktestResult:
        """
        时序动量策略回测
        配置所有发出买入信号的资产
        """
        n_days = len(prices)
        n_assets = len(prices.columns)

        equity = pd.Series(index=prices.index, dtype=float)
        equity.iloc[0] = initial_capital

        position = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for t in range(1, n_days):
            current_prices = prices.iloc[t]
            prev_prices = prices.iloc[t - 1]

            if t > 1:
                prev_equity = equity.iloc[t - 1]
                current_signals = signals.iloc[t - 1]

                selected_assets = current_signals[current_signals == 1].index

                if len(selected_assets) > 0:
                    weight = 1.0 / len(selected_assets)
                    position.loc[prices.index[t], selected_assets] = weight

                    asset_returns = (current_prices[selected_assets] -
                                   prev_prices[selected_assets]) / prev_prices[selected_assets]

                    strategy_return = (asset_returns * weight).sum()

                    if self.commission_rate > 0:
                        turnover = weight * len(selected_assets)
                        strategy_return -= self.commission_rate * turnover

                    equity.iloc[t] = equity.iloc[t - 1] * (1 + strategy_return)
                else:
                    equity.iloc[t] = equity.iloc[t - 1]
            else:
                equity.iloc[t] = initial_capital

        result = self._calculate_metrics(equity, prices)
        return result

    def cross_section_momentum_backtest(self, prices: pd.DataFrame,
                                        signals: pd.DataFrame,
                                        n_select: int = 5,
                                        initial_capital: float = 1000000) -> BacktestResult:
        """
        截面动量策略回测
        选择动量排名前n的资产进行配置
        """
        n_days = len(prices)
        n_assets = len(prices.columns)

        equity = pd.Series(index=prices.index, dtype=float)
        equity.iloc[0] = initial_capital

        position = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for t in range(1, n_days):
            current_prices = prices.iloc[t]
            prev_prices = prices.iloc[t - 1]

            if t > 1:
                current_signals = signals.iloc[t - 1]

                selected_assets = current_signals[current_signals == 1]
                selected_assets = selected_assets.sort_values(ascending=False).head(n_select).index

                if len(selected_assets) > 0:
                    weight = 1.0 / len(selected_assets)
                    position.loc[prices.index[t], selected_assets] = weight

                    asset_returns = (current_prices[selected_assets] -
                                   prev_prices[selected_assets]) / prev_prices[selected_assets]

                    strategy_return = (asset_returns * weight).sum()

                    if self.commission_rate > 0:
                        turnover = weight * len(selected_assets)
                        strategy_return -= self.commission_rate * turnover

                    equity.iloc[t] = equity.iloc[t - 1] * (1 + strategy_return)
                else:
                    equity.iloc[t] = equity.iloc[t - 1]
            else:
                equity.iloc[t] = initial_capital

        result = self._calculate_metrics(equity, prices)
        return result

    def risk_parity_allocation(self, returns_df: pd.DataFrame,
                               lookback: int = 40) -> pd.DataFrame:
        """
        风险平价权重分配
        """
        cov_matrix = returns_df.iloc[-lookback:].cov()
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        risk_weights = inv_vol / inv_vol.sum()

        asset_classes = self._get_asset_classes(len(risk_weights))
        class_weights = self._allocate_to_classes(asset_classes, risk_weights)

        return class_weights

    def _get_asset_classes(self, n_assets: int) -> Dict[str, List[int]]:
        """
        假设资产按顺序分配到大类: 股票、债券、商品
        """
        n_stock = max(1, n_assets // 3)
        n_bond = max(1, n_assets // 3)
        n_commodity = n_assets - n_stock - n_bond

        return {
            'stock': list(range(n_stock)),
            'bond': list(range(n_stock, n_stock + n_bond)),
            'commodity': list(range(n_stock + n_bond, n_assets))
        }

    def _allocate_to_classes(self, asset_classes: Dict[str, List[int]],
                            risk_weights: np.ndarray) -> np.ndarray:
        """
        大类资产内部等分风险
        """
        final_weights = np.zeros_like(risk_weights)
        n_classes = len(asset_classes)
        equal_class_weight = 1.0 / n_classes

        for class_name, indices in asset_classes.items():
            if len(indices) > 0:
                class_weight = equal_class_weight / len(indices)
                for idx in indices:
                    final_weights[idx] = class_weight

        return final_weights

    def _calculate_metrics(self, equity: pd.Series,
                           prices: pd.DataFrame) -> BacktestResult:
        """
        计算回测指标
        """
        returns = equity.pct_change().dropna()

        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()

        monthly_returns = equity.resample('MS').last().pct_change().dropna()
        monthly_win_rate = (monthly_returns > 0).sum() / len(monthly_returns)

        yearly_returns = equity.resample('YS').last().pct_change().dropna()

        return BacktestResult(
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            monthly_win_rate=monthly_win_rate,
            equity_curve=equity,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns
        )

    def run_backtest(self, prices: pd.DataFrame,
                     strategy_type: str = 'time_series',
                     n_select: Optional[int] = None) -> BacktestResult:
        """
        运行回测
        """
        if strategy_type == 'time_series':
            signals = (prices.pct_change() > 0).astype(int)
            return self.time_series_momentum_backtest(prices, signals)
        elif strategy_type == 'cross_section':
            if n_select is None:
                n_select = len(prices.columns) // 2
            signals = prices.pct_change()
            return self.cross_section_momentum_backtest(prices, signals, n_select)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

class StrategyEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_indicator(self, prices: pd.DataFrame,
                           indicator_values: pd.Series,
                           strategy_type: str = 'time_series',
                           n_select: Optional[int] = None) -> Dict:
        """
        评估单个指标表现
        """
        engine = BacktestEngine(rebalance_freq='monthly')

        signals = engine.generate_signals(indicator_values)
        signals_aligned = signals.reindex(prices.index, method='ffill')

        if strategy_type == 'time_series':
            result = engine.time_series_momentum_backtest(prices, signals_aligned.to_frame())
        else:
            if n_select is None:
                n_select = len(prices.columns) // 2
            result = engine.cross_section_momentum_backtest(prices, signals_aligned.to_frame(), n_select)

        return {
            'annual_return': result.annual_return,
            'annual_volatility': result.annual_volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'monthly_win_rate': result.monthly_win_rate
        }

    def evaluate_multiple_indicators(self, prices: pd.DataFrame,
                                    signals_df: pd.DataFrame,
                                    strategy_type: str = 'time_series',
                                    n_select: Optional[int] = None) -> pd.DataFrame:
        """
        批量评估指标
        """
        results = []
        for col in signals_df.columns:
            try:
                metrics = self.evaluate_indicator(prices, signals_df[col], strategy_type, n_select)
                metrics['indicator'] = col
                results.append(metrics)
            except Exception as e:
                print(f"评估 {col} 时出错: {e}")

        return pd.DataFrame(results).set_index('indicator')

if __name__ == "__main__":
    print("测试回测引擎...")

    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'asset1': 100 + np.random.randn(100).cumsum(),
        'asset2': 100 + np.random.randn(100).cumsum(),
        'asset3': 100 + np.random.randn(100).cumsum()
    }, index=dates)

    prices = prices.clip(lower=1)

    engine = BacktestEngine()
    result = engine.run_backtest(prices, strategy_type='time_series')

    print(f"年化收益率: {result.annual_return:.2%}")
    print(f"年化波动率: {result.annual_volatility:.2%}")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2%}")

    print("回测引擎测试完成!")