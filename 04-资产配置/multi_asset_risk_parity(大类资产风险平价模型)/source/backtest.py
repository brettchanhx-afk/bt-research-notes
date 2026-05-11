"""
回测引擎模块

提供统一的回测框架，支持多种策略的回测：
- 风险平价策略
- 波动率倒数策略
- 固定资产比例策略
- 等权重策略
- 风险预算策略
- 因子风险平价策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

try:
    from risk_parity import (
        risk_parity_portfolio,
        equal_weight_portfolio,
        fixed_ratio_portfolio,
        calculate_covariance_matrix,
        solve_risk_parity_weights,
        calculate_volatility_inverse_weights
    )
except ImportError:
    from .risk_parity import (
        risk_parity_portfolio,
        equal_weight_portfolio,
        fixed_ratio_portfolio,
        calculate_covariance_matrix,
        solve_risk_parity_weights,
        calculate_volatility_inverse_weights
    )

try:
    from risk_budget import (
        sharpe_squared_risk_budget_portfolio,
        leveraged_risk_parity_portfolio,
        custom_risk_budget_portfolio
    )
except ImportError:
    from .risk_budget import (
        sharpe_squared_risk_budget_portfolio,
        leveraged_risk_parity_portfolio,
        custom_risk_budget_portfolio
    )

try:
    from factor_risk_parity import (
        principal_component_risk_parity_portfolio
    )
except ImportError:
    from .factor_risk_parity import (
        principal_component_risk_parity_portfolio
    )


@dataclass
class BacktestConfig:
    """回测配置参数"""
    start_date: str = '2009-01-01'
    end_date: str = '2023-04-30'
    lookback_period: int = 126
    rebalance_freq: str = 'M'
    cov_method: str = 'sample'
    risk_free_rate: float = 0.03
    weight_bounds: Tuple[float, float] = (0.0, 1.0)


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    weights: pd.DataFrame
    portfolio_returns: pd.Series
    config: BacktestConfig
    leverage: Optional[pd.Series] = None


class BacktestEngine:
    """
    回测引擎类

    支持多种策略的回测，并提供统一的接口
    """

    def __init__(self, prices: pd.DataFrame, config: Optional[BacktestConfig] = None):
        """
        初始化回测引擎

        Parameters:
        -----------
        prices : pd.DataFrame
            价格数据
        config : BacktestConfig, optional
            回测配置
        """
        self.prices = prices
        self.config = config if config is not None else BacktestConfig()

        if not isinstance(prices.index, pd.DatetimeIndex):
            self.prices.index = pd.to_datetime(self.prices.index)

        self.returns = self.prices.pct_change().dropna()

        self.results: Dict[str, BacktestResult] = {}

    def run_risk_parity_strategy(self,
                                  name: str = 'RiskParity',
                                  **kwargs) -> BacktestResult:
        """
        运行风险平价策略回测

        Parameters:
        -----------
        name : str
            策略名称
        **kwargs : dict
            传递给风险平价函数的额外参数

        Returns:
        --------
        BacktestResult
            回测结果
        """
        config = BacktestConfig(
            lookback_period=kwargs.get('lookback_period', self.config.lookback_period),
            rebalance_freq=kwargs.get('rebalance_freq', self.config.rebalance_freq),
            cov_method=kwargs.get('cov_method', self.config.cov_method),
            weight_bounds=kwargs.get('weight_bounds', self.config.weight_bounds)
        )

        weights, portfolio_returns = risk_parity_portfolio(
            self.prices,
            lookback_period=config.lookback_period,
            rebalance_freq=config.rebalance_freq,
            cov_method=config.cov_method,
            weight_bounds=config.weight_bounds
        )

        result = BacktestResult(
            strategy_name=name,
            weights=weights,
            portfolio_returns=portfolio_returns,
            config=config
        )

        self.results[name] = result
        return result

    def run_volatility_inverse_strategy(self,
                                        name: str = 'VolInv') -> BacktestResult:
        """
        运行波动率倒数策略回测

        Parameters:
        -----------
        name : str
            策略名称

        Returns:
        --------
        BacktestResult
            回测结果
        """
        returns = self.returns.copy()
        lookback = self.config.lookback_period

        if self.config.rebalance_freq == 'M':
            rebalance_dates = returns.resample('M').last().index
        elif self.config.rebalance_freq == 'W':
            rebalance_dates = returns.resample('W').last().index
        else:
            rebalance_dates = returns.index

        weights_df = pd.DataFrame(index=rebalance_dates, columns=self.prices.columns)
        portfolio_returns = pd.Series(index=returns.index[lookback:])

        valid_returns = returns[lookback:]

        for i, date in enumerate(rebalance_dates):
            if date < valid_returns.index[0]:
                continue

            hist_returns = valid_returns[valid_returns.index < date]
            if len(hist_returns) < lookback // 2:
                continue

            hist_returns_subset = hist_returns.tail(lookback)

            cov_matrix = calculate_covariance_matrix(
                hist_returns_subset,
                method=self.config.cov_method,
                lookback_period=lookback
            )

            volatilities = np.sqrt(np.diag(cov_matrix))
            weights = calculate_volatility_inverse_weights(volatilities)

            weights_df.loc[date] = weights

            if i + 1 < len(rebalance_dates):
                next_date = rebalance_dates[i + 1]
            else:
                next_date = valid_returns.index[-1]

            period_returns = valid_returns.loc[date:next_date]
            if len(period_returns) > 0:
                portfolio_returns.loc[period_returns.index] = period_returns.values @ weights

        weights_df = weights_df.dropna()
        portfolio_returns = portfolio_returns.dropna()

        result = BacktestResult(
            strategy_name=name,
            weights=weights_df,
            portfolio_returns=portfolio_returns,
            config=self.config
        )

        self.results[name] = result
        return result

    def run_fixed_ratio_strategy(self,
                                  name: str = 'FixedRatio',
                                  stock_bond_ratio: Tuple[int, int, int] = (1, 8, 1)) -> BacktestResult:
        """
        运行固定资产比例策略回测

        Parameters:
        -----------
        name : str
            策略名称
        stock_bond_ratio : tuple
            股债商比例

        Returns:
        --------
        BacktestResult
            回测结果
        """
        weights, portfolio_returns = fixed_ratio_portfolio(
            self.prices,
            stock_bond_ratio=stock_bond_ratio,
            rebalance_freq=self.config.rebalance_freq
        )

        result = BacktestResult(
            strategy_name=name,
            weights=weights,
            portfolio_returns=portfolio_returns,
            config=self.config
        )

        self.results[name] = result
        return result

    def run_equal_weight_strategy(self,
                                    name: str = 'EqualWeight') -> BacktestResult:
        """
        运行等权重策略回测

        Parameters:
        -----------
        name : str
            策略名称

        Returns:
        --------
        BacktestResult
            回测结果
        """
        weights, portfolio_returns = equal_weight_portfolio(
            self.prices,
            rebalance_freq=self.config.rebalance_freq
        )

        result = BacktestResult(
            strategy_name=name,
            weights=weights,
            portfolio_returns=portfolio_returns,
            config=self.config
        )

        self.results[name] = result
        return result

    def run_sharpe_budget_strategy(self,
                                    name: str = 'SharpeBudget',
                                    **kwargs) -> BacktestResult:
        """
        运行基于夏普率平方的风险预算策略回测

        Parameters:
        -----------
        name : str
            策略名称
        **kwargs : dict
            传递给夏普预算函数的额外参数

        Returns:
        --------
        BacktestResult
            回测结果
        """
        config = BacktestConfig(
            lookback_period=kwargs.get('lookback_period', self.config.lookback_period),
            rebalance_freq=kwargs.get('rebalance_freq', self.config.rebalance_freq),
            cov_method=kwargs.get('cov_method', self.config.cov_method),
            risk_free_rate=kwargs.get('risk_free_rate', self.config.risk_free_rate),
            weight_bounds=kwargs.get('weight_bounds', self.config.weight_bounds)
        )

        weights, portfolio_returns = sharpe_squared_risk_budget_portfolio(
            self.prices,
            lookback_period=config.lookback_period,
            rebalance_freq=config.rebalance_freq,
            risk_free_rate=config.risk_free_rate,
            cov_method=config.cov_method,
            weight_bounds=config.weight_bounds
        )

        result = BacktestResult(
            strategy_name=name,
            weights=weights,
            portfolio_returns=portfolio_returns,
            config=config
        )

        self.results[name] = result
        return result

    def run_leveraged_risk_parity_strategy(self,
                                            name: str = 'LeveragedRP',
                                            target_volatility: float = 0.03,
                                            **kwargs) -> BacktestResult:
        """
        运行加杠杆的风险平价策略回测

        Parameters:
        -----------
        name : str
            策略名称
        target_volatility : float
            目标波动率
        **kwargs : dict
            传递给杠杆风险平价函数的额外参数

        Returns:
        --------
        BacktestResult
            回测结果
        """
        config = BacktestConfig(
            lookback_period=kwargs.get('lookback_period', self.config.lookback_period),
            rebalance_freq=kwargs.get('rebalance_freq', self.config.rebalance_freq),
            cov_method=kwargs.get('cov_method', self.config.cov_method),
            weight_bounds=kwargs.get('weight_bounds', self.config.weight_bounds)
        )

        weights, portfolio_returns, leverage = leveraged_risk_parity_portfolio(
            self.prices,
            lookback_period=config.lookback_period,
            rebalance_freq=config.rebalance_freq,
            target_volatility=target_volatility,
            **kwargs
        )

        result = BacktestResult(
            strategy_name=name,
            weights=weights,
            portfolio_returns=portfolio_returns,
            config=config,
            leverage=leverage
        )

        self.results[name] = result
        return result

    def run_factor_risk_parity_strategy(self,
                                         name: str = 'FactorRP',
                                         n_factors: int = None,
                                         **kwargs) -> BacktestResult:
        """
        运行因子风险平价策略回测

        Parameters:
        -----------
        name : str
            策略名称
        n_factors : int, optional
            因子数量
        **kwargs : dict
            传递给因子风险平价函数的额外参数

        Returns:
        --------
        BacktestResult
            回测结果
        """
        config = BacktestConfig(
            lookback_period=kwargs.get('lookback_period', self.config.lookback_period),
            rebalance_freq=kwargs.get('rebalance_freq', self.config.rebalance_freq),
            cov_method=kwargs.get('cov_method', self.config.cov_method),
            weight_bounds=kwargs.get('weight_bounds', self.config.weight_bounds)
        )

        weights, portfolio_returns = principal_component_risk_parity_portfolio(
            self.prices,
            lookback_period=config.lookback_period,
            n_factors=n_factors,
            rebalance_freq=config.rebalance_freq,
            cov_method=config.cov_method,
            weight_bounds=config.weight_bounds
        )

        result = BacktestResult(
            strategy_name=name,
            weights=weights,
            portfolio_returns=portfolio_returns,
            config=config
        )

        self.results[name] = result
        return result

    def run_all_strategies(self) -> Dict[str, BacktestResult]:
        """
        运行所有标准策略回测

        Returns:
        --------
        dict
            所有策略的回测结果
        """
        print("开始运行所有策略回测...")

        print("\n1. 风险平价策略...")
        self.run_risk_parity_strategy()

        print("2. 波动率倒数策略...")
        self.run_volatility_inverse_strategy()

        print("3. 固定资产比例策略 (1:8:1)...")
        self.run_fixed_ratio_strategy()

        print("4. 等权重策略...")
        self.run_equal_weight_strategy()

        print("5. 夏普预算策略...")
        self.run_sharpe_budget_strategy()

        print("6. 加杠杆风险平价策略...")
        self.run_leveraged_risk_parity_strategy(target_volatility=0.03)

        print("7. 因子风险平价策略...")
        self.run_factor_risk_parity_strategy()

        print("\n所有策略回测完成!")
        return self.results

    def get_result(self, name: str) -> Optional[BacktestResult]:
        """
        获取指定策略的回测结果

        Parameters:
        -----------
        name : str
            策略名称

        Returns:
        --------
        BacktestResult or None
        """
        return self.results.get(name)

    def get_all_results(self) -> Dict[str, BacktestResult]:
        """
        获取所有策略的回测结果

        Returns:
        --------
        dict
        """
        return self.results


def calculate_portfolio_nav(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    计算组合净值序列

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    initial_value : float
        初始净值

    Returns:
    --------
    pd.Series
        净值序列
    """
    nav = (1 + returns).cumprod() * initial_value
    return nav


def run_comparative_backtest(prices: pd.DataFrame,
                              config: Optional[BacktestConfig] = None) -> Dict[str, BacktestResult]:
    """
    运行比较回测

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    config : BacktestConfig, optional
        回测配置

    Returns:
    --------
    dict
        所有策略的回测结果
    """
    engine = BacktestEngine(prices, config)
    results = engine.run_all_strategies()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("回测引擎模块测试")
    print("=" * 60)

    np.random.seed(42)
    n_assets = 6
    n_periods = 500

    dates = pd.date_range(start='2008-01-01', periods=n_periods, freq='B')
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(n_periods, n_assets) * 0.01, axis=0),
        index=dates,
        columns=['CSI300', 'SPX', 'HSI', 'CBCE', 'NHCI', 'GC']
    )

    print(f"价格数据形状: {prices.shape}")
    print(f"时间范围: {prices.index[0]} 至 {prices.index[-1]}")

    config = BacktestConfig(
        start_date='2008-01-01',
        end_date='2023-04-30',
        lookback_period=126,
        rebalance_freq='M'
    )

    engine = BacktestEngine(prices, config)

    print("\n运行风险平价策略...")
    rp_result = engine.run_risk_parity_strategy()
    print(f"  策略名称: {rp_result.strategy_name}")
    print(f"  权重形状: {rp_result.weights.shape}")
    print(f"  收益率序列长度: {len(rp_result.portfolio_returns)}")

    print("\n运行波动率倒数策略...")
    vi_result = engine.run_volatility_inverse_strategy()
    print(f"  策略名称: {vi_result.strategy_name}")

    print("\n运行等权重策略...")
    ew_result = engine.run_equal_weight_strategy()
    print(f"  策略名称: {ew_result.strategy_name}")

    print("\n回测引擎测试完成!")