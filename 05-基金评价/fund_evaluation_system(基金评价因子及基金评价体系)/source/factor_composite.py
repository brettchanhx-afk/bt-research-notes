# -*- coding: utf-8 -*-
"""
因子复合模块

功能：
- 等权合成
- 最大ICIR合成
- 动态权重调整
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 等权合成
# ============================================================
def equal_weight_composite(
    factor_df: pd.DataFrame,
    factor_names: List[str] = None
) -> pd.Series:
    """
    等权合成因子
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子数据，columns为因子名称
    factor_names : List[str]
        参与合成的因子名称
        
    Returns
    -------
    pd.Series
        复合因子值
    """
    if factor_names is None:
        factor_names = factor_df.columns.tolist()
    
    # 选取因子
    df = factor_df[factor_names].copy()
    
    # 标准化（排名归一化）
    df_normalized = df.rank(pct=True)
    
    # 等权平均
    composite = df_normalized.mean(axis=1)
    
    return composite


# ============================================================
# 2. 最大ICIR合成
# ============================================================
def max_icir_composite(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    factor_names: List[str] = None,
    weight_min: float = 0.10,
    weight_max: float = 0.30,
    window: int = 6
) -> Tuple[pd.Series, np.ndarray]:
    """
    最大ICIR合成因子
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子数据
    return_df : pd.DataFrame
        下期收益率
    factor_names : List[str]
        参与合成的因子
    weight_min, weight_max : float
        权重约束
    window : int
        滚动窗口（月）
        
    Returns
    -------
    Tuple[pd.Series, np.ndarray]
        (复合因子, 权重)
    """
    if factor_names is None:
        factor_names = factor_df.columns.tolist()
    
    n_factors = len(factor_names)
    
    # 计算IC矩阵
    from .backtest_engine import calc_rank_ic
    
    ic_matrix = []
    
    for date in factor_df.index:
        if date not in return_df.index:
            continue
        
        factor_values = factor_df.loc[date, factor_names]
        fwd_ret = return_df.loc[date]
        
        row = []
        for factor in factor_names:
            ic = calc_rank_ic(factor_values[factor], fwd_ret)
            row.append(ic)
        
        ic_matrix.append(row)
    
    ic_df = pd.DataFrame(ic_matrix, index=factor_df.index, columns=factor_names)
    
    # 滚动计算权重
    weights_series = []
    composite_series = []
    
    for i in range(window, len(ic_df)):
        # 滚动窗口IC
        ic_window = ic_df.iloc[i - window:i].dropna()
        
        if len(ic_window) < window // 2:
            # 默认等权
            weights = np.ones(n_factors) / n_factors
        else:
            # 优化权重
            weights = _optimize_icir_weights(
                ic_window.values,
                weight_min,
                weight_max
            )
        
        weights_series.append(weights)
        
        # 计算复合因子
        factor_values = factor_df.iloc[i][factor_names]
        factor_normalized = factor_values.rank(pct=True)
        composite = (factor_normalized * weights).sum()
        composite_series.append(composite)
    
    weights_array = np.array(weights_series)
    composite = pd.Series(composite_series, index=factor_df.index[window:])
    
    return composite, weights_array


def _optimize_icir_weights(
    ic_matrix: np.ndarray,
    weight_min: float,
    weight_max: float
) -> np.ndarray:
    """
    优化ICIR权重
    
    Parameters
    ----------
    ic_matrix : np.ndarray
        IC矩阵 (T x N)
    weight_min, weight_max : float
        权重约束
        
    Returns
    -------
    np.ndarray
        最优权重
    """
    n_factors = ic_matrix.shape[1]
    
    # IC均值和协方差
    ic_mean = np.nanmean(ic_matrix, axis=0)
    ic_cov = np.cov(ic_matrix.T)
    
    # 目标函数：最小化 -ICIR
    def neg_icir(weights):
        portfolio_ic = ic_mean @ weights
        portfolio_var = weights @ ic_cov @ weights
        portfolio_std = np.sqrt(portfolio_var) if portfolio_var > 0 else 1e-10
        return -portfolio_ic / portfolio_std
    
    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 权重和为1
    ]
    
    # 边界条件
    bounds = [(weight_min, weight_max) for _ in range(n_factors)]
    
    # 初始值
    w0 = np.ones(n_factors) / n_factors
    
    # 优化
    try:
        result = minimize(
            neg_icir,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return result.x
        else:
            return w0
    except Exception:
        return w0


# ============================================================
# 3. 动态权重
# ============================================================
def calculate_dynamic_weights(
    ic_history: pd.DataFrame,
    factor_names: List[str],
    window: int = 6,
    weight_min: float = 0.10,
    weight_max: float = 0.30
) -> pd.DataFrame:
    """
    计算动态权重
    
    Parameters
    ----------
    ic_history : pd.DataFrame
        IC历史数据
    factor_names : List[str]
        因子名称
    window : int
        滚动窗口
    weight_min, weight_max : float
        权重约束
        
    Returns
    -------
    pd.DataFrame
        动态权重
    """
    weights_list = []
    dates = []
    
    for i in range(window, len(ic_history)):
        ic_window = ic_history.iloc[i - window:i][factor_names]
        
        weights = _optimize_icir_weights(
            ic_window.values,
            weight_min,
            weight_max
        )
        
        weights_list.append(weights)
        dates.append(ic_history.index[i])
    
    weights_df = pd.DataFrame(weights_list, index=dates, columns=factor_names)
    
    return weights_df
