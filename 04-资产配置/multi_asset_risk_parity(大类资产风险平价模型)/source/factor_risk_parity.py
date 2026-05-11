"""
因子风险平价模型模块

实现基于主成分的因子风险平价策略：
- 使用主成分分析（PCA）提取风险因子
- 在因子层面实现风险平价
- 资产权重通过因子暴露和因子风险预算推导
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


def extract_risk_factors(returns: pd.DataFrame, n_factors: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用PCA提取风险因子

    Parameters:
    -----------
    returns : pd.DataFrame
        资产收益率数据
    n_factors : int, optional
        因子数量，默认为资产数量

    Returns:
    --------
    tuple
        (因子收益率矩阵, 因子载荷矩阵)
    """
    if n_factors is None:
        n_factors = returns.shape[1]

    returns_centered = returns - returns.mean()

    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns_centered)

    factor_loadings = pca.components_.T

    explained_variance_ratio = pca.explained_variance_ratio_

    return factor_returns, factor_loadings, explained_variance_ratio


def calculate_factor_covariance(factor_returns: np.ndarray) -> np.ndarray:
    """
    计算因子收益率的协方差矩阵

    Parameters:
    -----------
    factor_returns : np.ndarray
        因子收益率

    Returns:
    --------
    np.ndarray
        因子协方差矩阵
    """
    return np.cov(factor_returns.T)


def calculate_portfolio_factor_exposure(weights: np.ndarray, factor_loadings: np.ndarray) -> np.ndarray:
    """
    计算投资组合的因子暴露

    Parameters:
    -----------
    weights : np.ndarray
        资产权重
    factor_loadings : np.ndarray
        因子载荷矩阵（资产 x 因子）

    Returns:
    --------
    np.ndarray
        因子暴露向量
    """
    factor_exposure = factor_loadings.T @ weights
    return factor_exposure


def calculate_portfolio_total_volatility(weights: np.ndarray,
                                         factor_loadings: np.ndarray,
                                         factor_cov: np.ndarray,
                                         idiosyncratic_vol: np.ndarray) -> float:
    """
    计算投资组合的总波动率

    σ²_p = b' * Σ_f * b + σ²_ε

    其中：
    - b: 因子暴露
    - Σ_f: 因子协方差矩阵
    - σ²_ε: 特异风险（残差波动率）

    Parameters:
    -----------
    weights : np.ndarray
        资产权重
    factor_loadings : np.ndarray
        因子载荷矩阵
    factor_cov : np.ndarray
        因子协方差矩阵
    idiosyncratic_vol : np.ndarray
        特异风险向量

    Returns:
    --------
    float
        组合波动率
    """
    factor_exposure = calculate_portfolio_factor_exposure(weights, factor_loadings)

    systematic_var = factor_exposure @ factor_cov @ factor_exposure

    idiosyncratic_var = weights @ (idiosyncratic_vol ** 2)

    total_var = systematic_var + idiosyncratic_var

    if total_var < 0:
        return 0.0

    return np.sqrt(total_var)


def calculate_factor_risk_contribution(weights: np.ndarray,
                                        factor_loadings: np.ndarray,
                                        factor_cov: np.ndarray,
                                        idiosyncratic_vol: np.ndarray) -> np.ndarray:
    """
    计算各因子对组合风险的贡献

    Parameters:
    -----------
    weights : np.ndarray
        资产权重
    factor_loadings : np.ndarray
        因子载荷矩阵
    factor_cov : np.ndarray
        因子协方差矩阵
    idiosyncratic_vol : np.ndarray
        特异风险向量

    Returns:
    --------
    np.ndarray
        各因子风险贡献
    """
    portfolio_vol = calculate_portfolio_total_volatility(
        weights, factor_loadings, factor_cov, idiosyncratic_vol
    )

    if portfolio_vol == 0:
        return np.zeros(factor_loadings.shape[1])

    factor_exposure = calculate_portfolio_factor_exposure(weights, factor_loadings)

    factor_marginal_risk = factor_cov @ factor_exposure

    factor_risk_contrib = factor_marginal_risk * factor_exposure

    idiosyncratic_risk = weights * idiosyncratic_vol**2

    total_factor_risk = np.sum(factor_risk_contrib) + np.sum(idiosyncratic_risk)

    if total_factor_risk > 0:
        factor_risk_contrib = factor_risk_contrib * (portfolio_vol**2) / total_factor_risk

    return factor_risk_contrib


def factor_risk_parity_objective(weights: np.ndarray,
                                  factor_loadings: np.ndarray,
                                  factor_cov: np.ndarray,
                                  idiosyncratic_vol: np.ndarray,
                                  target_factor_budgets: np.ndarray,
                                  n_factors: int) -> float:
    """
    因子风险平价优化目标函数

    Parameters:
    -----------
    weights : np.ndarray
        资产权重
    factor_loadings : np.ndarray
        因子载荷矩阵
    factor_cov : np.ndarray
        因子协方差矩阵
    idiosyncratic_vol : np.ndarray
        特异风险向量
    target_factor_budgets : np.ndarray
        目标因子风险预算
    n_factors : int
        因子数量

    Returns:
    --------
    float
        目标函数值
    """
    if np.abs(weights.sum() - 1.0) > 1e-6:
        weights = weights / weights.sum()

    factor_risk_contrib = calculate_factor_risk_contribution(
        weights, factor_loadings, factor_cov, idiosyncratic_vol
    )

    factor_risk_proportions = np.zeros(n_factors)
    total_factor_risk = np.sum(factor_risk_contrib)

    if total_factor_risk > 0:
        factor_risk_proportions = factor_risk_contrib / total_factor_risk

    residual_risk = weights @ (idiosyncratic_vol ** 2)
    total_portfolio_var = np.sum(factor_risk_contrib) + residual_risk

    if total_portfolio_var > 0:
        residual_proportion = residual_risk / total_portfolio_var
    else:
        residual_proportion = 0

    all_budgets = np.concatenate([target_factor_budgets, [residual_proportion]])

    error = np.sum((factor_risk_proportions - target_factor_budgets) ** 2)
    error += (residual_proportion - all_budgets[-1]) ** 2

    return error


def solve_factor_risk_parity_weights(factor_loadings: np.ndarray,
                                      factor_cov: np.ndarray,
                                      idiosyncratic_vol: np.ndarray,
                                      target_factor_budgets: Optional[np.ndarray] = None,
                                      weight_bounds: Tuple[float, float] = (0.0, 1.0),
                                      max_iter: int = 1000) -> np.ndarray:
    """
    求解因子风险平价权重

    Parameters:
    -----------
    factor_loadings : np.ndarray
        因子载荷矩阵（n_assets x n_factors）
    factor_cov : np.ndarray
        因子协方差矩阵
    idiosyncratic_vol : np.ndarray
        特异风险向量
    target_factor_budgets : np.ndarray, optional
        目标因子风险预算，默认等权
    weight_bounds : tuple
        权重上下限
    max_iter : int
        最大迭代次数

    Returns:
    --------
    np.ndarray
        最优资产权重
    """
    n_assets = factor_loadings.shape[0]
    n_factors = factor_loadings.shape[1]

    if target_factor_budgets is None:
        target_factor_budgets = np.ones(n_factors) / n_factors

    if len(target_factor_budgets) != n_factors:
        raise ValueError(f"目标预算长度({len(target_factor_budgets)})必须等于因子数量({n_factors})")

    target_factor_budgets = target_factor_budgets / target_factor_budgets.sum()

    initial_weights = np.ones(n_assets) / n_assets

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]

    bounds = [(weight_bounds[0], weight_bounds[1]) for _ in range(n_assets)]

    result = minimize(
        factor_risk_parity_objective,
        initial_weights,
        args=(factor_loadings, factor_cov, idiosyncratic_vol,
              target_factor_budgets, n_factors),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )

    if not result.success:
        print(f"因子风险平价优化警告: {result.message}")

    optimal_weights = result.x
    optimal_weights = np.clip(optimal_weights, weight_bounds[0], weight_bounds[1])
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights


def principal_component_risk_parity_portfolio(
        prices: pd.DataFrame,
        lookback_period: int = 126,
        n_factors: int = None,
        rebalance_freq: str = 'M',
        cov_method: str = 'sample',
        weight_bounds: Tuple[float, float] = (0.0, 1.0)) -> Tuple[pd.DataFrame, pd.Series]:
    """
    基于主成分的因子风险平价策略

    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    lookback_period : int
        计算回望期
    n_factors : int, optional
        因子数量，默认为资产数量
    rebalance_freq : str
        调仓频率
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

    if n_factors is None:
        n_factors = prices.shape[1]

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

        factor_returns, factor_loadings, explained_variance = extract_risk_factors(
            hist_returns_subset, n_factors=n_factors
        )

        factor_cov = calculate_factor_covariance(factor_returns)

        asset_cov = hist_returns_subset.cov().values
        predicted_cov = factor_loadings @ factor_cov @ factor_loadings.T
        residual_cov = asset_cov - predicted_cov
        np.fill_diagonal(residual_cov, np.maximum(np.diag(residual_cov), 0))
        idiosyncratic_vol = np.sqrt(np.abs(np.diag(residual_cov)))

        idiosyncratic_vol = np.maximum(idiosyncratic_vol, 1e-6)

        target_budgets = np.ones(n_factors) / n_factors

        weights = solve_factor_risk_parity_weights(
            factor_loadings,
            factor_cov,
            idiosyncratic_vol,
            target_budgets=target_budgets,
            weight_bounds=weight_bounds
        )

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


def factor_analysis_report(returns: pd.DataFrame, weights: np.ndarray,
                           n_factors: int = None) -> Dict:
    """
    生成因子分析报告

    Parameters:
    -----------
    returns : pd.DataFrame
        收益率数据
    weights : np.ndarray
        资产权重
    n_factors : int, optional
        因子数量

    Returns:
    --------
    dict
        包含因子分析结果的字典
    """
    if n_factors is None:
        n_factors = returns.shape[1]

    factor_returns, factor_loadings, explained_variance = extract_risk_factors(returns, n_factors)
    factor_cov = calculate_factor_covariance(factor_returns)

    asset_cov = returns.cov().values
    predicted_cov = factor_loadings @ factor_cov @ factor_loadings.T
    residual_cov = asset_cov - predicted_cov
    np.fill_diagonal(residual_cov, np.maximum(np.diag(residual_cov), 0))
    idiosyncratic_vol = np.sqrt(np.abs(np.diag(residual_cov)))

    portfolio_factor_exposure = calculate_portfolio_factor_exposure(weights, factor_loadings)
    portfolio_vol = calculate_portfolio_total_volatility(weights, factor_loadings, factor_cov, idiosyncratic_vol)
    factor_risk_contrib = calculate_factor_risk_contribution(weights, factor_loadings, factor_cov, idiosyncratic_vol)

    systematic_risk = portfolio_factor_exposure @ factor_cov @ portfolio_factor_exposure
    idiosyncratic_risk = weights @ (idiosyncratic_vol ** 2)
    total_risk = systematic_risk + idiosyncratic_risk

    return {
        'factor_loadings': factor_loadings,
        'factor_covariance': factor_cov,
        'factor_returns': factor_returns,
        'explained_variance_ratio': explained_variance,
        'portfolio_factor_exposure': portfolio_factor_exposure,
        'portfolio_volatility': portfolio_vol,
        'factor_risk_contribution': factor_risk_contrib,
        'systematic_risk': systematic_risk,
        'idiosyncratic_risk': idiosyncratic_risk,
        'total_risk': total_risk
    }


if __name__ == "__main__":
    print("=" * 60)
    print("因子风险平价模型模块测试")
    print("=" * 60)

    np.random.seed(42)
    n_assets = 6
    n_periods = 500

    prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(n_periods, n_assets) * 0.01, axis=0),
        columns=['CSI300', 'SPX', 'HSI', 'CBCE', 'NHCI', 'GC']
    )

    returns = prices.pct_change().dropna()

    print(f"\n资产数量: {n_assets}")
    print(f"收益率数据形状: {returns.shape}")

    factor_returns, factor_loadings, explained_var = extract_risk_factors(returns, n_factors=3)
    print(f"\n因子载荷矩阵形状: {factor_loadings.shape}")
    print(f"各因子解释方差比例: {explained_var}")

    print("\n因子协方差矩阵:")
    factor_cov = calculate_factor_covariance(factor_returns)
    print(factor_cov)

    asset_cov = returns.cov().values
    predicted_cov = factor_loadings @ factor_cov @ factor_loadings.T
    residual_cov = asset_cov - predicted_cov
    np.fill_diagonal(residual_cov, np.maximum(np.diag(residual_cov), 0))
    idiosyncratic_vol = np.sqrt(np.abs(np.diag(residual_cov)))

    equal_weights = np.ones(n_assets) / n_assets
    print(f"\n等权重的因子暴露: {calculate_portfolio_factor_exposure(equal_weights, factor_loadings)}")

    print("\n求解因子风险平价权重...")
    rp_weights = solve_factor_risk_parity_weights(
        factor_loadings, factor_cov, idiosyncratic_vol,
        target_factor_budgets=np.ones(3) / 3
    )
    print(f"因子风险平价权重: {rp_weights}")

    analysis = factor_analysis_report(returns, rp_weights, n_factors=3)
    print(f"\n组合波动率: {analysis['portfolio_volatility']:.4f}")
    print(f"因子风险贡献: {analysis['factor_risk_contribution']}")
    print(f"系统性风险: {analysis['systematic_risk']:.6f}")
    print(f"特异风险: {analysis['idiosyncratic_risk']:.6f}")