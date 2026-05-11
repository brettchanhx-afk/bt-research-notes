"""
风险平价模型核心模块

实现基于风险平价的大类资产配置策略，包括：
- 协方差矩阵计算
- 风险贡献计算（边际风险贡献、总风险贡献）
- 风险平价优化求解
- 波动率倒数加权
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


def calculate_covariance_matrix(returns: pd.DataFrame, method: str = 'sample',
                                 lookback_period: int = 126) -> np.ndarray:
    """
    计算协方差矩阵

    Parameters:
    -----------
    returns : pd.DataFrame
        收益率数据
    method : str
        计算方法，'sample'（样本协方差）、'ewma'（指数加权移动平均）
    lookback_period : int
        回望期，用于EWMA方法

    Returns:
    --------
    np.ndarray
        协方差矩阵
    """
    if method == 'sample':
        cov_matrix = returns.cov().values
    elif method == 'ewma':
        decay_factor = 0.94
        ewm_cov = returns.ewm(halflife=lookback_period).cov()
        cov_matrix = ewm_cov.iloc[-len(returns.columns):].values
        if cov_matrix.shape[0] != len(returns.columns):
            cov_matrix = returns.cov().values
    else:
        cov_matrix = returns.cov().values

    return cov_matrix


def calculate_volatility_inverse_weights(volatilities: np.ndarray) -> np.ndarray:
    """
    计算波动率倒数权重（不考虑相关性）

    Parameters:
    -----------
    volatilities : np.ndarray
        各资产波动率

    Returns:
    --------
    np.ndarray
        权重数组
    """
    inv_vol = 1.0 / volatilities
    weights = inv_vol / inv_vol.sum()
    return weights


def calculate_marginal_risk_contribution(cov_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    计算边际风险贡献 (Marginal Risk Contribution, MRC)

    MRC_i = (Σ * w)_i / σ_p

    其中 σ_p = sqrt(w' * Σ * w)

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵
    weights : np.ndarray
        资产权重

    Returns:
    --------
    np.ndarray
        各资产的边际风险贡献
    """
    portfolio_volatility = calculate_portfolio_volatility(cov_matrix, weights)
    if portfolio_volatility == 0:
        return np.zeros_like(weights)

    marginal_risk = (cov_matrix @ weights) / portfolio_volatility
    return marginal_risk


def calculate_total_risk_contribution(cov_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    计算总风险贡献 (Total Risk Contribution, TRC)

    TRC_i = w_i * MRC_i

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵
    weights : np.ndarray
        资产权重

    Returns:
    --------
    np.ndarray
        各资产的总风险贡献
    """
    marginal_risk = calculate_marginal_risk_contribution(cov_matrix, weights)
    total_risk_contribution = weights * marginal_risk
    return total_risk_contribution


def calculate_portfolio_volatility(cov_matrix: np.ndarray, weights: np.ndarray) -> float:
    """
    计算投资组合波动率

    σ_p = sqrt(w' * Σ * w)

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵
    weights : np.ndarray
        资产权重

    Returns:
    --------
    float
        组合波动率
    """
    variance = weights @ cov_matrix @ weights
    if variance < 0:
        return 0.0
    return np.sqrt(variance)


def risk_contribution_objective(weights: np.ndarray, cov_matrix: np.ndarray,
                                target_contributions: np.ndarray) -> float:
    """
    风险平价优化目标函数

    最小化各资产风险贡献与目标风险贡献之间的平方误差

    Parameters:
    -----------
    weights : np.ndarray
        资产权重
    cov_matrix : np.ndarray
        协方差矩阵
    target_contributions : np.ndarray
        目标风险贡献比例

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

    error = np.sum((risk_contrib_proportions - target_contributions) ** 2)
    return error


def solve_risk_parity_weights(cov_matrix: np.ndarray,
                              target_risk_contributions: Optional[np.ndarray] = None,
                              weight_bounds: Tuple[float, float] = (0.0, 1.0),
                              max_iter: int = 1000) -> np.ndarray:
    """
    求解风险平价权重

    使用凸优化方法求解风险平价条件下的最优权重

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵
    target_risk_contributions : np.ndarray, optional
        目标风险贡献，默认等权（1/n）
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

    if target_risk_contributions is None:
        target_risk_contributions = np.ones(n_assets) / n_assets

    initial_weights = np.ones(n_assets) / n_assets

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]

    bounds = [(weight_bounds[0], weight_bounds[1]) for _ in range(n_assets)]

    result = minimize(
        risk_contribution_objective,
        initial_weights,
        args=(cov_matrix, target_risk_contributions),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )

    if not result.success:
        print(f"优化警告: {result.message}")

    optimal_weights = result.x
    optimal_weights = np.clip(optimal_weights, weight_bounds[0], weight_bounds[1])
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights


def solve_risk_parity_weights_iterative(cov_matrix: np.ndarray,
                                        max_iter: int = 100,
                                        tol: float = 1e-8) -> np.ndarray:
    """
    使用迭代方法求解风险平价权重（当相关系数为0时）

    对于两资产情况：w_i = 1/σ_i / sum(1/σ_j)

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵
    max_iter : int
        最大迭代次数
    tol : float
        收敛容忍度

    Returns:
    --------
    np.ndarray
        最优资产权重
    """
    n_assets = cov_matrix.shape[0]
    volatilities = np.sqrt(np.diag(cov_matrix))

    weights = np.ones(n_assets) / n_assets

    for iteration in range(max_iter):
        portfolio_volatility = calculate_portfolio_volatility(cov_matrix, weights)
        if portfolio_volatility == 0:
            break

        marginal_risk = calculate_marginal_risk_contribution(cov_matrix, weights)
        risk_contributions = weights * marginal_risk
        risk_contrib_proportions = risk_contributions / portfolio_volatility

        new_weights = np.ones(n_assets) / (volatilities * np.sqrt(n_assets))
        inv_vol_sum = np.sum(1.0 / (volatilities * np.sqrt(n_assets)))
        new_weights = (1.0 / (volatilities * np.sqrt(n_assets))) / inv_vol_sum

        if np.max(np.abs(new_weights - weights)) < tol:
            weights = new_weights
            break

        weights = new_weights

    return weights


def risk_parity_portfolio(prices: pd.DataFrame,
                          lookback_period: int = 126,
                          rebalance_freq: str = 'M',
                          cov_method: str = 'sample',
                          weight_bounds: Tuple[float, float] = (0.0, 1.0)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建风险平价组合

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    lookback_period : int
        计算协方差矩阵的回望期
    rebalance_freq : str
        调仓频率，'M'月度、'W'周度、'D'日度
    cov_method : str
        协方差矩阵计算方法
    weight_bounds : tuple
        权重上下限

    Returns:
    --------
    tuple
        (权重数据框, 组合收益率数据框)
    """
    returns = prices.pct_change().dropna()

    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').last().index
    elif rebalance_freq == 'W':
        rebalance_dates = returns.resample('W').last().index
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

        weights = solve_risk_parity_weights(cov_matrix, weight_bounds=weight_bounds)

        weights_df.loc[date] = weights

        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1]
        period_returns = valid_returns.loc[date:next_date]

        if len(period_returns) > 0:
            portfolio_returns.loc[period_returns.index] = (
                period_returns.values @ weights
            )

    weights_df = weights_df.dropna()
    portfolio_returns = portfolio_returns.dropna()

    return weights_df, portfolio_returns


def equal_weight_portfolio(prices: pd.DataFrame, rebalance_freq: str = 'M') -> Tuple[pd.DataFrame, pd.Series]:
    """
    构建等权重组合

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    rebalance_freq : str
        调仓频率

    Returns:
    --------
    tuple
        (权重数据框, 组合收益率序列)
    """
    returns = prices.pct_change().dropna()

    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').last().index
    elif rebalance_freq == 'W':
        rebalance_dates = returns.resample('W').last().index
    else:
        rebalance_dates = returns.index

    n_assets = prices.shape[1]
    equal_weight = np.ones(n_assets) / n_assets

    weights_df = pd.DataFrame(index=rebalance_dates, columns=prices.columns, data=equal_weight)

    valid_returns = returns[returns.index >= rebalance_dates[0]]

    portfolio_returns = pd.Series(index=valid_returns.index)
    for i, date in enumerate(rebalance_dates):
        if i + 1 < len(rebalance_dates):
            period_end = rebalance_dates[i + 1]
        else:
            period_end = valid_returns.index[-1]

        period_returns = valid_returns.loc[date:period_end]
        if len(period_returns) > 0:
            portfolio_returns.loc[period_returns.index] = period_returns.values @ equal_weight

    portfolio_returns = portfolio_returns.dropna()

    return weights_df, portfolio_returns


def fixed_ratio_portfolio(prices: pd.DataFrame,
                          stock_bond_ratio: Tuple[int, int, int] = (1, 8, 1),
                          rebalance_freq: str = 'M') -> Tuple[pd.DataFrame, pd.Series]:
    """
    固定资产比例组合

    股债商比例为 股票:债券:商品 = 1:8:1
    每类资产内部等权分配

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据，列顺序需为 [CSI300, SPX, HSI, CBCE, NHCI, GC]
    stock_bond_ratio : tuple
        股票、债券、商品的大类资产比例
    rebalance_freq : str
        调仓频率

    Returns:
    --------
    tuple
        (权重数据框, 组合收益率序列)
    """
    returns = prices.pct_change().dropna()

    if rebalance_freq == 'M':
        rebalance_dates = returns.resample('M').last().index
    elif rebalance_freq == 'W':
        rebalance_dates = returns.resample('W').last().index
    else:
        rebalance_dates = returns.index

    asset_classes = {
        'stock': ['CSI300', 'SPX', 'HSI'],
        'bond': ['CBCE'],
        'commodity': ['NHCI', 'GC']
    }

    total_ratio = sum(stock_bond_ratio)
    stock_weight = stock_bond_ratio[0] / total_ratio
    bond_weight = stock_bond_ratio[1] / total_ratio
    commodity_weight = stock_bond_ratio[2] / total_ratio

    class_weights = {
        'stock': stock_weight / len(asset_classes['stock']),
        'bond': bond_weight / len(asset_classes['bond']),
        'commodity': commodity_weight / len(asset_classes['commodity'])
    }

    final_weights = np.zeros(prices.shape[1])
    for i, col in enumerate(prices.columns):
        if col in asset_classes['stock']:
            final_weights[i] = class_weights['stock']
        elif col in asset_classes['bond']:
            final_weights[i] = class_weights['bond']
        elif col in asset_classes['commodity']:
            final_weights[i] = class_weights['commodity']

    weights_df = pd.DataFrame(index=rebalance_dates, columns=prices.columns, data=final_weights)

    valid_returns = returns[returns.index >= rebalance_dates[0]]

    portfolio_returns = pd.Series(index=valid_returns.index)
    for i, date in enumerate(rebalance_dates):
        if i + 1 < len(rebalance_dates):
            period_end = rebalance_dates[i + 1]
        else:
            period_end = valid_returns.index[-1]

        period_returns = valid_returns.loc[date:period_end]
        if len(period_returns) > 0:
            portfolio_returns.loc[period_returns.index] = period_returns.values @ final_weights

    portfolio_returns = portfolio_returns.dropna()

    return weights_df, portfolio_returns


def risk_contribution_decomposition(cov_matrix: np.ndarray, weights: np.ndarray) -> Dict:
    """
    风险贡献分解

    Parameters:
    -----------
    cov_matrix : np.ndarray
        协方差矩阵
    weights : np.ndarray
        资产权重

    Returns:
    --------
    dict
        包含各种风险分解结果
    """
    portfolio_volatility = calculate_portfolio_volatility(cov_matrix, weights)
    marginal_risk = calculate_marginal_risk_contribution(cov_matrix, weights)
    total_risk_contrib = calculate_total_risk_contribution(cov_matrix, weights)
    risk_contrib_proportions = total_risk_contrib / portfolio_volatility if portfolio_volatility > 0 else np.zeros_like(total_risk_contrib)

    return {
        'portfolio_volatility': portfolio_volatility,
        'marginal_risk_contribution': marginal_risk,
        'total_risk_contribution': total_risk_contrib,
        'risk_contribution_proportions': risk_contrib_proportions
    }


if __name__ == "__main__":
    print("=" * 60)
    print("风险平价模型核心模块测试")
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

    weights = np.array([0.5, 0.2, 0.3])
    print(f"\n初始权重: {weights}")

    vol = calculate_portfolio_volatility(cov_matrix, weights)
    print(f"组合波动率: {vol:.4f}")

    risk_decomp = risk_contribution_decomposition(cov_matrix, weights)
    print(f"\n风险贡献分解:")
    print(f"  边际风险贡献: {risk_decomp['marginal_risk_contribution']}")
    print(f"  总风险贡献: {risk_decomp['total_risk_contribution']}")
    print(f"  风险贡献比例: {risk_decomp['risk_contribution_proportions']}")

    print("\n风险平价优化...")
    rp_weights = solve_risk_parity_weights(cov_matrix)
    print(f"风险平价权重: {rp_weights}")

    rp_vol = calculate_portfolio_volatility(cov_matrix, rp_weights)
    rp_trc = calculate_total_risk_contribution(cov_matrix, rp_weights)
    print(f"风险平价组合波动率: {rp_vol:.4f}")
    print(f"风险平价各资产风险贡献: {rp_trc}")
    print(f"风险平价各资产风险贡献比例: {rp_trc / rp_vol if rp_vol > 0 else 'N/A'}")