"""
工具函数模块

提供数据清洗、格式转换、统计检验等通用功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    计算基金业绩指标
    
    Parameters
    ----------
    returns : pd.Series
        基金收益率序列（日度或月度）
        
    Returns
    -------
    Dict[str, float]
        包含各项业绩指标的字典
    """
    if len(returns) == 0 or returns.isna().all():
        return {}
    
    # 年化收益率
    annual_return = (1 + returns.mean()) ** 12 - 1 if returns.mean() < 1 else returns.mean() * 12
    
    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(12)
    
    # 夏普比率（假设无风险利率为2%）
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
    }


def format_results(regression_results: Dict, factor_names: List[str]) -> pd.DataFrame:
    """
    格式化回归结果
    
    Parameters
    ----------
    regression_results : Dict
        回归结果字典
    factor_names : List[str]
        因子名称列表
        
    Returns
    -------
    pd.DataFrame
        格式化的结果表格
    """
    results = []
    
    for factor in factor_names:
        if factor in regression_results.get('params', {}):
            coef = regression_results['params'][factor]
            t_stat = regression_results.get('t_values', {}).get(factor, np.nan)
            p_value = regression_results.get('p_values', {}).get(factor, np.nan)
            
            # 显著性标记
            significance = ''
            if p_value < 0.01:
                significance = '***'
            elif p_value < 0.05:
                significance = '**'
            elif p_value < 0.1:
                significance = '*'
            
            results.append({
                'factor': factor,
                'coefficient': coef,
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': significance,
            })
    
    # 添加Alpha
    if 'alpha' in regression_results:
        results.append({
            'factor': 'Alpha',
            'coefficient': regression_results['alpha'],
            't_statistic': regression_results.get('alpha_t', np.nan),
            'p_value': regression_results.get('alpha_p', np.nan),
            'significance': '***' if regression_results.get('alpha_p', 1) < 0.01 else 
                           ('**' if regression_results.get('alpha_p', 1) < 0.05 else
                            ('*' if regression_results.get('alpha_p', 1) < 0.1 else '')),
        })
    
    df = pd.DataFrame(results)
    return df


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    对序列进行缩尾处理
    
    Parameters
    ----------
    series : pd.Series
        输入序列
    lower : float
        下分位数
    upper : float
        上分位数
        
    Returns
    -------
    pd.Series
        缩尾处理后的序列
    """
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)
    return series.clip(lower=lower_val, upper=upper_val)


def standardize_series(series: pd.Series) -> pd.Series:
    """
    对序列进行标准化（z-score）
    
    Parameters
    ----------
    series : pd.Series
        输入序列
        
    Returns
    -------
    pd.Series
        标准化后的序列
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def calculate_ic_series(factor_values: pd.Series, forward_returns: pd.Series) -> pd.Series:
    """
    计算因子IC（信息系数）时间序列
    
    Parameters
    ----------
    factor_values : pd.Series
        因子值
    forward_returns : pd.Series
        未来收益率
        
    Returns
    -------
    pd.Series
        IC时间序列
    """
    return factor_values.corr(forward_returns, method='spearman')


def newey_west_t_stat(residuals: np.ndarray, X: np.ndarray, lags: int = 3) -> np.ndarray:
    """
    计算Newey-West调整后的t统计量
    
    用于处理时间序列回归中的异方差和自相关问题
    
    Parameters
    ----------
    residuals : np.ndarray
        残差
    X : np.ndarray
        自变量矩阵
    lags : int
        滞后阶数
        
    Returns
    -------
    np.ndarray
        调整后的t统计量
    """
    n = len(residuals)
    k = X.shape[1] if len(X.shape) > 1 else 1
    
    # 计算标准OLS的协方差矩阵
    XTX_inv = np.linalg.inv(X.T @ X)
    
    # 计算Newey-West调整后的协方差矩阵
    S0 = (residuals ** 2).mean() * (X.T @ X) / n
    
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        for t in range(lag, n):
            S0 += weight * residuals[t] * residuals[t-lag] * (X[t:t+1].T @ X[t-lag:t-lag+1]) / n
    
    cov_matrix = XTX_inv @ S0 @ XTX_inv
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # 计算t统计量
    beta = XTX_inv @ X.T @ residuals
    t_stats = beta / std_errors
    
    return t_stats


def rolling_regression(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = 12,
    min_periods: int = 6
) -> pd.DataFrame:
    """
    滚动窗口回归
    
    Parameters
    ----------
    y : pd.Series
        因变量
    X : pd.DataFrame
        自变量
    window : int
        滚动窗口大小
    min_periods : int
        最小观测数
        
    Returns
    -------
    pd.DataFrame
        滚动回归系数
    """
    from sklearn.linear_model import LinearRegression
    
    results = []
    index = []
    
    for i in range(window, len(y) + 1):
        y_window = y.iloc[i-window:i]
        X_window = X.iloc[i-window:i]
        
        if len(y_window.dropna()) < min_periods:
            continue
            
        # 删除缺失值
        mask = ~(y_window.isna() | X_window.isna().any(axis=1))
        y_clean = y_window[mask]
        X_clean = X_window[mask]
        
        if len(y_clean) < min_periods:
            continue
        
        # 回归
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        results.append(model.coef_)
        index.append(y.index[i-1])
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results, columns=X.columns, index=index)
    return result_df


def calculate_factor_contribution(
    factor_returns: pd.DataFrame,
    factor_exposures: pd.Series
) -> pd.DataFrame:
    """
    计算各因子对收益的贡献
    
    Parameters
    ----------
    factor_returns : pd.DataFrame
        因子收益率
    factor_exposures : pd.Series
        因子暴露
        
    Returns
    -------
    pd.DataFrame
        各因子贡献
    """
    contributions = {}
    
    for factor in factor_exposures.index:
        if factor in factor_returns.columns:
            # 因子贡献 = 因子暴露 × 因子收益率均值
            avg_return = factor_returns[factor].mean()
            contribution = factor_exposures[factor] * avg_return
            contributions[factor] = {
                'exposure': factor_exposures[factor],
                'avg_factor_return': avg_return,
                'contribution': contribution,
            }
    
    return pd.DataFrame(contributions).T


def print_regression_summary(results: Dict):
    """
    打印回归结果摘要
    
    Parameters
    ----------
    results : Dict
        回归结果字典
    """
    print("\n" + "="*60)
    print("Fama-French五因子回归结果")
    print("="*60)
    
    # 因子暴露
    print("\n因子暴露系数:")
    print("-" * 50)
    print(f"{'因子':<15} {'系数':>12} {'t统计量':>12} {'显著性':>8}")
    print("-" * 50)
    
    factor_names = ['R_M', 'SMB', 'HML', 'RMW', 'CMA']
    factor_labels = {
        'R_M': '市场因子',
        'SMB': '市值因子(SMB)',
        'HML': '价值因子(HML)',
        'RMW': '盈利因子(RMW)',
        'CMA': '投资因子(CMA)',
    }
    
    for factor in factor_names:
        if factor in results.get('params', {}):
            coef = results['params'][factor]
            t_stat = results.get('t_values', {}).get(factor, np.nan)
            p_val = results.get('p_values', {}).get(factor, np.nan)
            
            sig = ''
            if p_val < 0.01:
                sig = '***'
            elif p_val < 0.05:
                sig = '**'
            elif p_val < 0.1:
                sig = '*'
            
            print(f"{factor_labels.get(factor, factor):<15} {coef:>12.4f} {t_stat:>12.2f} {sig:>8}")
    
    # Alpha
    if 'alpha' in results:
        alpha = results['alpha']
        alpha_t = results.get('alpha_t', np.nan)
        alpha_p = results.get('alpha_p', np.nan)
        
        sig = ''
        if alpha_p < 0.01:
            sig = '***'
        elif alpha_p < 0.05:
            sig = '**'
        elif alpha_p < 0.1:
            sig = '*'
        
        print("-" * 50)
        print(f"{'Alpha (年化)':<15} {alpha*12:>12.4f} {alpha_t:>12.2f} {sig:>8}")
    
    # 模型拟合度
    print("\n" + "-" * 50)
    print(f"R-squared:       {results.get('r_squared', 0):.4f}")
    print(f"Adj R-squared:   {results.get('adj_r_squared', 0):.4f}")
    print(f"观测数:          {results.get('n_obs', 0)}")
    print("="*60)


if __name__ == '__main__':
    # 测试
    print("Utils module loaded successfully!")
