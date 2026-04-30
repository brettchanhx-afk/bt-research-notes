# -*- coding: utf-8 -*-
"""
共线性诊断与处理模块

功能：
  - VIF检验（方差膨胀因子）
  - 相关系数矩阵分析
  - 主成分分析（PCA）降维
  - 因子正交化
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats


# ============================================================
# 1. VIF检验
# ============================================================
def calculate_vif(
    X: pd.DataFrame,
    factor_names: List[str] = None
) -> pd.DataFrame:
    """计算方差膨胀因子（VIF）。
    
    VIF = 1 / (1 - R²)
    
    VIF > 10 表示存在严重共线性
    VIF > 5 表示存在中度共线性
    
    Parameters
    ----------
    X : pd.DataFrame
        因子数据
    factor_names : List[str]
        因子名称列表
    
    Returns
    -------
    pd.DataFrame
        VIF结果
    """
    if factor_names is None:
        factor_names = X.columns.tolist()
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # 准备数据
    X_matrix = X[factor_names].dropna().values
    
    if len(X_matrix) == 0:
        return pd.DataFrame()
    
    vif_results = []
    
    for i, name in enumerate(factor_names):
        try:
            vif = variance_inflation_factor(X_matrix, i)
            vif_results.append({
                'factor': name,
                'VIF': vif,
                'collinearity': '严重' if vif > 10 else ('中度' if vif > 5 else '无')
            })
        except Exception:
            vif_results.append({
                'factor': name,
                'VIF': np.inf,
                'collinearity': '无法计算'
            })
    
    return pd.DataFrame(vif_results)


def check_collinearity(
    X: pd.DataFrame,
    threshold: float = 10.0
) -> Tuple[pd.DataFrame, bool]:
    """检查共线性问题。
    
    Parameters
    ----------
    X : pd.DataFrame
        因子数据
    threshold : float
        VIF阈值
    
    Returns
    -------
    Tuple[pd.DataFrame, bool]
        (VIF结果, 是否存在共线性)
    """
    vif_df = calculate_vif(X)
    
    if len(vif_df) == 0:
        return vif_df, False
    
    has_collinearity = (vif_df['VIF'] > threshold).any()
    
    return vif_df, has_collinearity


# ============================================================
# 2. 相关系数矩阵
# ============================================================
def calculate_correlation_matrix(
    X: pd.DataFrame,
    factor_names: List[str] = None
) -> pd.DataFrame:
    """计算因子相关系数矩阵。
    
    Parameters
    ----------
    X : pd.DataFrame
        因子数据
    factor_names : List[str]
        因子名称列表
    
    Returns
    -------
    pd.DataFrame
        相关系数矩阵
    """
    if factor_names is None:
        factor_names = X.columns.tolist()
    
    return X[factor_names].corr()


def find_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7
) -> pd.DataFrame:
    """找出高相关性的因子对。
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        相关系数矩阵
    threshold : float
        相关性阈值
    
    Returns
    -------
    pd.DataFrame
        高相关因子对
    """
    pairs = []
    n = len(corr_matrix)
    
    for i in range(n):
        for j in range(i+1, n):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                pairs.append({
                    'factor_1': corr_matrix.index[i],
                    'factor_2': corr_matrix.columns[j],
                    'correlation': corr
                })
    
    if pairs:
        return pd.DataFrame(pairs).sort_values('correlation', key=abs, ascending=False)
    else:
        return pd.DataFrame()


# ============================================================
# 3. 主成分分析（PCA）
# ============================================================
def apply_pca(
    X: pd.DataFrame,
    n_components: int = None,
    variance_threshold: float = 0.95
) -> Tuple[pd.DataFrame, dict]:
    """应用PCA降维处理共线性。
    
    Parameters
    ----------
    X : pd.DataFrame
        因子数据
    n_components : int
        主成分数量（可选）
    variance_threshold : float
        累计方差解释阈值
    
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (主成分数据, PCA信息)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.dropna())
    
    # PCA
    if n_components is None:
        n_components = min(X.shape[1], X.shape[0])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 确定保留的主成分数量
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = np.argmax(cumvar >= variance_threshold) + 1
    
    # 主成分数据
    pc_df = pd.DataFrame(
        X_pca[:, :n_keep],
        index=X.dropna().index,
        columns=[f'PC{i+1}' for i in range(n_keep)]
    )
    
    # PCA信息
    pca_info = {
        'n_components': n_keep,
        'explained_variance_ratio': pca.explained_variance_ratio_[:n_keep],
        'cumulative_variance': cumvar[n_keep-1],
        'loadings': pca.components_[:n_keep].T,
        'original_features': X.columns.tolist(),
    }
    
    return pc_df, pca_info


# ============================================================
# 4. 因子正交化
# ============================================================
def orthogonalize_factors(
    X: pd.DataFrame,
    factor_names: List[str] = None,
    method: str = 'gram_schmidt'
) -> pd.DataFrame:
    """因子正交化处理。
    
    Parameters
    ----------
    X : pd.DataFrame
        因子数据
    factor_names : List[str]
        因子名称列表
    method : str
        正交化方法：'gram_schmidt', 'cholesky'
    
    Returns
    -------
    pd.DataFrame
        正交化后的因子
    """
    if factor_names is None:
        factor_names = X.columns.tolist()
    
    X_matrix = X[factor_names].dropna().values
    
    if method == 'gram_schmidt':
        # Gram-Schmidt正交化
        Q, R = np.linalg.qr(X_matrix)
        X_ortho = Q
    elif method == 'cholesky':
        # Cholesky分解
        cov = np.cov(X_matrix.T)
        L = np.linalg.cholesky(cov)
        X_ortho = X_matrix @ np.linalg.inv(L.T)
    else:
        X_ortho = X_matrix
    
    result = pd.DataFrame(
        X_ortho,
        index=X.dropna().index,
        columns=[f'{name}_ortho' for name in factor_names]
    )
    
    return result


# ============================================================
# 5. 综合诊断报告
# ============================================================
def diagnose_collinearity(
    X: pd.DataFrame,
    factor_names: List[str] = None,
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.7
) -> dict:
    """综合共线性诊断。
    
    Parameters
    ----------
    X : pd.DataFrame
        因子数据
    factor_names : List[str]
        因子名称列表
    vif_threshold : float
        VIF阈值
    corr_threshold : float
        相关性阈值
    
    Returns
    -------
    dict
        诊断报告
    """
    if factor_names is None:
        factor_names = X.columns.tolist()
    
    # VIF检验
    vif_df, has_vif_issue = check_collinearity(X[factor_names], vif_threshold)
    
    # 相关系数矩阵
    corr_matrix = calculate_correlation_matrix(X, factor_names)
    
    # 高相关因子对
    high_corr_pairs = find_high_correlation_pairs(corr_matrix, corr_threshold)
    
    # 条件数
    try:
        cond_number = np.linalg.cond(X[factor_names].dropna().values)
    except Exception:
        cond_number = np.inf
    
    report = {
        'vif': vif_df,
        'has_vif_issue': has_vif_issue,
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'condition_number': cond_number,
        'has_collinearity': has_vif_issue or len(high_corr_pairs) > 0 or cond_number > 30,
        'recommendation': [],
    }
    
    # 建议
    if has_vif_issue:
        report['recommendation'].append('VIF检验发现问题，建议移除高VIF因子或使用PCA降维')
    
    if len(high_corr_pairs) > 0:
        report['recommendation'].append(f'发现{len(high_corr_pairs)}对高相关因子，建议移除其中一个或进行正交化')
    
    if cond_number > 30:
        report['recommendation'].append(f'条件数={cond_number:.1f}过大，矩阵接近奇异，建议正则化处理')
    
    if not report['has_collinearity']:
        report['recommendation'].append('未发现明显共线性问题，可直接进行回归')
    
    return report
