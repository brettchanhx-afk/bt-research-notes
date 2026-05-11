"""
风险预算模型模块

实现基于风险预算的大类资产配置策略，包括：
- 通用风险预算优化
- 基于夏普率平方的风险预算策略
- 加杠杆的风险平价模型
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

try:
    from risk_parity import (
        calculate_covariance_matrix,
        calculate_portfolio_volatility,
        calculate_marginal_risk_contribution,
        calculate_total_risk_contribution,
        calculate_volatility_inverse_weights,
        solve_risk_parity_weights
    )
except ImportError:
    from .risk_parity import (
        calculate_covariance_matrix,
        calculate_portfolio_volatility,
        calculate_marginal_risk_contribution,
        calculate_total_risk_contribution,
        calculate_volatility_inverse_weights,
        solve_risk_parity_weights
    )


def risk_budget_objective(weights: np.ndarray, cov_matrix: np.ndarray,
                           target_budgets: np.ndarray) -> float:
    """
    风险预算优化目标函数

    最小化各资产风险贡献与目标风险预算之间的平方误差

    Parameters:
    -----------
    weights : np.ndarray
        资产权重
    cov_matrix : np.ndarray
        协方差矩阵
    target_budgets : np.ndarray
        目标风险预算比例

    Returns:
    --------
    float
        目标函数值
    """
    n_assets = len(weights)
    if np.abs(weights.sum() - 1.0) > 1e-6:
        weights = weights / weights.sum()

    portfolio_volatility = calculate_portfolio_volatility(cov_matrix, weights)
    if portfolio_volatility == 0:
        return 1e10

    total_risk_contrib = calculate_total_risk_contribution(cov_matrix, weights)
    risk_contrib_proportions = total_risk_contrib / portfolio_volatility

    error = np.sum((risk_contrib_proportions - target_budgets) ** 2)
    return error


def solve_risk_budget_weights(cov_matrix: np.ndarray,
                              target_budgets: np.ndarray,
                              weight_bounds: Tuple[float, float] = (0.0, 1.0),
                              max_iter: int = 1000) -> np.ndarray:
    """
    求解风险预算权重

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵
    target_budgets : np.ndarray
        目标风险预算比例，必须加总为1
    weight_bounds : tuple
        权重上下限
    max_iter : int
        最大迭代次数

    Returns:
    --------
    np.ndarray
        最优资产权重
    """
    n_assets = cov_matrix.shape[0]

    if len(target_budgets) != n_assets:
        raise ValueError(f"目标预算长度({len(target_budgets)})必须等于资产数量({n_assets})")

    if np.abs(target_budgets.sum() - 1.0) > 1e-6:
        target_budgets = target_budgets / target_budgets.sum()

    initial_weights = np.ones(n_assets) / n_assets

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]

    bounds = [(weight_bounds[0], weight_bounds[1]) for _ in range(n_assets)]

    result = minimize(
        risk_budget_objective,
        initial_weights,
        args=(cov_matrix, target_budgets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )

    if not result.success:
        print(f"风险预算优化警告: {result.message}")

    optimal_weights = result.x
    optimal_weights = np.clip(optimal_weights, weight_bounds[0], weight_bounds[1])
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights


def solve_two_asset_risk_budget(cov_matrix: np.ndarray,
                                 risk_budgets: np.ndarray) -> np.ndarray:
    """
    两资产风险预算的解析解

    对于两资产情况，可以给出解析解：
    w_1* = [(1-b_1)*σ_2² + ρ*σ_1*σ_2*b_1] / [σ_1² + σ_2² - 2*ρ*σ_1*σ_2]

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵（2x2）
    risk_budgets : np.ndarray
        风险预算 [b_1, b_2]，b_1 + b_2 = 1

    Returns:
    --------
    np.ndarray
        最优权重 [w_1, w_2]
    """
    if cov_matrix.shape != (2, 2):
        raise ValueError("两资产解析解仅适用于2x2协方差矩阵")

    sigma1 = np.sqrt(cov_matrix[0, 0])
    sigma2 = np.sqrt(cov_matrix[1, 1])
    rho = cov_matrix[0, 1] / (sigma1 * sigma2) if sigma1 * sigma2 > 0 else 0

    b1 = risk_budgets[0]

    if abs(1 - rho**2) < 1e-10:
        w1 = b1
    else:
        numerator = (1 - b1) * sigma2**2 + rho * sigma1 * sigma2 * b1
        denominator = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
        w1 = numerator / denominator

    w1 = np.clip(w1, 0, 1)
    w2 = 1 - w1

    return np.array([w1, w2])


def calculate_sharpe_ratio_weights(returns: pd.DataFrame, risk_free_rate: float = 0.03) -> np.ndarray:
    """
    计算基于历史夏普比率的权重

    根据各资产的历史夏普比率计算风险预算权重

    Parameters:
    -----------
    returns : pd.DataFrame
        收益率数据
    risk_free_rate : float
        年化无风险利率

    Returns:
    --------
    np.ndarray
        基于夏普比率平方的风险预算权重
    """
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)

    sharpe_ratios = (annual_returns - risk_free_rate) / annual_vol

    sharpe_ratios = np.maximum(sharpe_ratios, 0.0)

    sharpe_squared = sharpe_ratios ** 2
    budgets = sharpe_squared / sharpe_squared.sum()

    return np.array(budgets)


def sharpe_squared_risk_budget_portfolio(prices: pd.DataFrame,
                                          lookback_period: int = 252,
                                          rebalance_freq: str = 'M',
                                          risk_free_rate: float = 0.03,
                                          cov_method: str = 'sample',
                                          weight_bounds: Tuple[float, float] = (0.0, 1.0)) -> Tuple[pd.DataFrame, pd.Series]:
    """
    基于夏普率平方的风险预算策略

    各资产的风险预算与其夏普比率的平方成正比

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    lookback_period : int
        计算回望期
    rebalance_freq : str
        调仓频率
    risk_free_rate : float
        年化无风险利率
    cov_method : str
        协方差矩阵计算方法
    weight_bounds : tuple
        权重上下限

    Returns:
    --------
    tuple
        (权重数据框, 组合收益率序列)
    """
    returns = prices.pct_change().dropna()

    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').apply(lambda x: x.index[-1])
    elif rebalance_freq == 'W':
        rebalance_dates = returns.resample('W').apply(lambda x: x.index[-1])
    else:
        rebalance_dates = returns.index

    weights_df = pd.DataFrame(index=rebalance_dates, columns=prices.columns)
    portfolio_returns = pd.Series(index=returns.index[lookback_period:])

    valid_returns = returns[lookback_period:]

    for i, date in enumerate(rebalance_dates):
        if date < valid_returns.index[0]:
            continue

        hist_returns = valid_returns[valid_returns.index < date]
        if len(hist_returns) < lookback_period // 2:
            continue

        hist_returns_subset = hist_returns.tail(lookback_period)

        cov_matrix = calculate_covariance_matrix(hist_returns_subset, method=cov_method,
                                                  lookback_period=lookback_period)

        sharpe_budgets = calculate_sharpe_ratio_weights(hist_returns_subset, risk_free_rate)

        weights = solve_risk_budget_weights(cov_matrix, sharpe_budgets, weight_bounds=weight_bounds)

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

    return weights_df, portfolio_returns


def leveraged_risk_parity_portfolio(prices: pd.DataFrame,
                                     lookback_period: int = 126,
                                     rebalance_freq: str = 'M',
                                     target_volatility: float = 0.03,
                                     leverage_cost: float = 0.03,
                                     min_leverage: float = 0.8,
                                     max_leverage: float = 1.4,
                                     cov_method: str = 'sample',
                                     weight_bounds: Tuple[float, float] = (0.0, 1.0)) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    加杠杆的风险平价模型

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    lookback_period : int
        计算回望期
    rebalance_freq : str
        调仓频率
    target_volatility : float
        目标波动率
    leverage_cost : float
        杠杆资金成本（年化）
    min_leverage : float
        最小杠杆率
    max_leverage : float
        最大杠杆率
    cov_method : str
        协方差矩阵计算方法
    weight_bounds : tuple
        权重上下限

    Returns:
    --------
    tuple
        (权重数据框, 组合收益率序列, 杠杆率序列)
    """
    returns = prices.pct_change().dropna()

    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').apply(lambda x: x.index[-1])
    elif rebalance_freq == 'W':
        rebalance_dates = returns.resample('W').apply(lambda x: x.index[-1])
    else:
        rebalance_dates = returns.index

    weights_df = pd.DataFrame(index=rebalance_dates, columns=prices.columns)
    portfolio_returns = pd.Series(index=returns.index[lookback_period:])
    leverage_series = pd.Series(index=rebalance_dates)

    valid_returns = returns[lookback_period:]

    for i, date in enumerate(rebalance_dates):
        if date < valid_returns.index[0]:
            continue

        hist_returns = valid_returns[valid_returns.index < date]
        if len(hist_returns) < lookback_period // 2:
            continue

        hist_returns_subset = hist_returns.tail(lookback_period)

        cov_matrix = calculate_covariance_matrix(hist_returns_subset, method=cov_method,
                                                  lookback_period=lookback_period)

        full_weights = solve_risk_parity_weights(cov_matrix, weight_bounds=weight_bounds)

        full_portfolio_vol = calculate_portfolio_volatility(cov_matrix, full_weights)

        if full_portfolio_vol > 0:
            base_leverage = target_volatility / full_portfolio_vol
        else:
            base_leverage = 1.0

        base_leverage = np.clip(base_leverage, min_leverage, max_leverage)

        actual_weights = full_weights * base_leverage

        leverage_cost_factor = 1 - (base_leverage - 1) * leverage_cost * (1/252)

        weights_df.loc[date] = actual_weights
        leverage_series.loc[date] = base_leverage

        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
        else:
            next_date = valid_returns.index[-1]

        period_returns = valid_returns.loc[date:next_date]
        if len(period_returns) > 0:
            raw_returns = period_returns.values @ actual_weights
            adjusted_returns = raw_returns * leverage_cost_factor
            portfolio_returns.loc[period_returns.index] = adjusted_returns

    weights_df = weights_df.dropna()
    portfolio_returns = portfolio_returns.dropna()
    leverage_series = leverage_series.dropna()

    return weights_df, portfolio_returns, leverage_series


def custom_risk_budget_portfolio(prices: pd.DataFrame,
                                  budget_func,
                                  lookback_period: int = 126,
                                  rebalance_freq: str = 'M',
                                  cov_method: str = 'sample',
                                  weight_bounds: Tuple[float, float] = (0.0, 1.0),
                                  **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """
    自定义风险预算策略的通用接口

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    budget_func : callable
        计算风险预算的函数，输入收益率数据，输出预算数组
    lookback_period : int
        计算回望期
    rebalance_freq : str
        调仓频率
    cov_method : str
        协方差矩阵计算方法
    weight_bounds : tuple
        权重上下限
    **kwargs : dict
        传递给budget_func的额外参数

    Returns:
    --------
    tuple
        (权重数据框, 组合收益率序列)
    """
    returns = prices.pct_change().dropna()

    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').apply(lambda x: x.index[-1])
    elif rebalance_freq == 'W':
        rebalance_dates = returns.resample('W').apply(lambda x: x.index[-1])
    else:
        rebalance_dates = returns.index

    weights_df = pd.DataFrame(index=rebalance_dates, columns=prices.columns)
    portfolio_returns = pd.Series(index=returns.index[lookback_period:])

    valid_returns = returns[lookback_period:]

    for i, date in enumerate(rebalance_dates):
        if date < valid_returns.index[0]:
            continue

        hist_returns = valid_returns[valid_returns.index < date]
        if len(hist_returns) < lookback_period // 2:
            continue

        hist_returns_subset = hist_returns.tail(lookback_period)

        cov_matrix = calculate_covariance_matrix(hist_returns_subset, method=cov_method,
                                                  lookback_period=lookback_period)

        target_budgets = budget_func(hist_returns_subset, **kwargs)

        if np.abs(target_budgets.sum() - 1.0) > 1e-6:
            target_budgets = target_budgets / target_budgets.sum()

        weights = solve_risk_budget_weights(cov_matrix, target_budgets, weight_bounds=weight_bounds)

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

    return weights_df, portfolio_returns


if __name__ == "__main__":
    print("=" * 60)
    print("风险预算模型模块测试")
    print("=" * 60)

    np.random.seed(42)
    n_assets = 3
    n_periods = 252

    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.01,
        columns=['Asset1', 'Asset2', 'Asset3']
    )

    cov_matrix = calculate_covariance_matrix(returns)
    print(f"\n协方差矩阵:\n{cov_matrix}")

    target_budgets = np.array([0.4, 0.35, 0.25])
    print(f"\n目标风险预算: {target_budgets}")

    rb_weights = solve_risk_budget_weights(cov_matrix, target_budgets)
    print(f"风险预算权重: {rb_weights}")

    from risk_parity import risk_contribution_decomposition
    rb_vol = calculate_portfolio_volatility(cov_matrix, rb_weights)
    rb_trc = calculate_total_risk_contribution(cov_matrix, rb_weights)
    print(f"风险预算组合波动率: {rb_vol:.4f}")
    print(f"各资产风险贡献比例: {rb_trc / rb_vol if rb_vol > 0 else 'N/A'}")

    print("\n夏普平方风险预算测试...")
    sharpe_budgets = calculate_sharpe_ratio_weights(returns)
    print(f"基于夏普比率平方的预算: {sharpe_budgets}")

    print("\n两资产解析解测试...")
    cov_2x2 = cov_matrix[:2, :2]
    budgets_2 = np.array([0.6, 0.4])
    w_2asset = solve_two_asset_risk_budget(cov_2x2, budgets_2)
    print(f"两资产风险预算权重: {w_2asset}")