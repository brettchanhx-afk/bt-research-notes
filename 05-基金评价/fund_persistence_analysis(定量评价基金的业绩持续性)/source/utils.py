# -*- coding: utf-8 -*-
"""
工具函数模块
===================================
包含：数据清洗、格式转换、日志等工具函数
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import json

# ============================================================
# 数据清洗函数
# ============================================================

def clean_fund_data(df, date_col='date', return_col='return'):
    """
    清洗基金净值/收益数据
    
    Parameters:
    -----------
    df : DataFrame
        原始数据框
    date_col : str
        日期列名
    return_col : str
        收益列名
        
    Returns:
    --------
    DataFrame : 清洗后的数据
    """
    df = df.copy()
    
    # 确保日期列是datetime类型
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
    
    # 删除缺失值
    if return_col in df.columns:
        df = df.dropna(subset=[return_col])
    
    # 删除异常值（超过10倍标准差）
    if return_col in df.columns:
        mean_val = df[return_col].mean()
        std_val = df[return_col].std()
        df = df[(df[return_col] > mean_val - 10*std_val) & 
                 (df[return_col] < mean_val + 10*std_val)]
    
    return df


def calculate_excess_return(fund_returns, benchmark_returns, risk_free_rate=0.03):
    """
    计算超额收益
    
    公式：超额收益 = 基金收益 - 无风险收益（或基准收益）
    
    Parameters:
    -----------
    fund_returns : Series
        基金收益率序列
    benchmark_returns : Series, optional
        基准收益率序列（若提供则为基金-基准）
    risk_free_rate : float
        年化无风险利率（若不提供基准则用此计算）
        
    Returns:
    --------
    Series : 超额收益率序列
    """
    if benchmark_returns is not None:
        # 基金超额收益 = 基金收益 - 基准收益
        return fund_returns - benchmark_returns
    else:
        # 假设无风险收益按日计算
        daily_rf = risk_free_rate / 252
        return fund_returns - daily_rf


def split_periods(returns, n_periods=2):
    """
    将收益序列划分为多个等长期间
    
    Parameters:
    -----------
    returns : Series
        收益率序列
    n_periods : int
        划分期数
        
    Returns:
    --------
    list of Series : 各期间收益率列表
    """
    n = len(returns)
    period_len = n // n_periods
    
    periods = []
    for i in range(n_periods):
        start_idx = i * period_len
        end_idx = (i + 1) * period_len if i < n_periods - 1 else n
        period_returns = returns.iloc[start_idx:end_idx]
        periods.append(period_returns)
    
    return periods


def calculate_period_return(returns):
    """
    计算期间累计收益
    
    Parameters:
    -----------
    returns : Series
        日收益率序列
        
    Returns:
    --------
    float : 期间累计收益
    """
    # 累计收益 = (1+r1)*(1+r2)*...*(1+rn) - 1
    cumulative = (1 + returns).prod() - 1
    return cumulative


def classify_winners_losers(fund_returns, median_return):
    """
    分类赢家和输家
    
    收益率高于中位数 = 赢家(W)
    收益率低于或等于中位数 = 输家(L)
    
    Parameters:
    -----------
    fund_returns : float
        基金期间收益率
    median_return : float
        同类基金中位数收益率
        
    Returns:
    --------
    str : 'W' 或 'L'
    """
    return 'W' if fund_returns > median_return else 'L'


# ============================================================
# 统计分析函数
# ============================================================

def regression_with_stats(y, X):
    """
    简单线性回归（带统计量）
    
    Parameters:
    -----------
    y : array-like
        因变量
    X : array-like
        自变量（会添加常数项）
        
    Returns:
    --------
    dict : 包含系数、截距、R方、p值等
    """
    from scipy import stats
    
    # 添加常数项
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # 回归
    try:
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        residuals = y - X_with_const @ coeffs
        n = len(y)
        k = X_with_const.shape[1]
        
        # 计算统计量
        se = np.sqrt(np.sum(residuals**2) / (n - k))  # 标准误
        
        # 系数矩阵的协方差矩阵
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        se_coef = se * np.sqrt(np.diag(XtX_inv))
        
        # t统计量和p值
        t_stats = coeffs / se_coef
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
        
        # R方
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        
        # 调整R方
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        return {
            'intercept': coeffs[0],
            'slope': coeffs[1],
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'se': se,
            't_intercept': t_stats[0],
            't_slope': t_stats[1],
            'p_intercept': p_values[0],
            'p_slope': p_values[1],
            'n_obs': n,
            'residuals': residuals
        }
    except Exception as e:
        return {
            'error': str(e),
            'intercept': np.nan,
            'slope': np.nan,
            'r_squared': np.nan
        }


# ============================================================
# Hurst指数计算
# ============================================================

def calculate_r_s(series, n):
    """
    计算长度为n的子序列的R/S值
    
    Parameters:
    -----------
    series : array-like
        时间序列
    n : int
        子序列长度
        
    Returns:
    --------
    float : R/S值
    """
    series = np.array(series)
    a = len(series) // n  # 子集个数
    
    if a == 0:
        return np.nan
    
    rs_values = []
    
    for i in range(a):
        subseries = series[i*n:(i+1)*n]
        mean_val = np.mean(subseries)
        
        # 累计离差
        cumdev = np.cumsum(subseries - mean_val)
        
        # 极差
        r = np.max(cumdev) - np.min(cumdev)
        
        # 标准差
        s = np.std(subseries, ddof=1)
        
        if s > 0:
            rs_values.append(r / s)
    
    if len(rs_values) == 0:
        return np.nan
    
    return np.mean(rs_values)


def estimate_hurst_exponent(returns, n_values=[4, 8, 16, 32], n_estimators=8):
    """
    估计Hurst指数
    
    公式: log((R/S)_n) = log(c) + H * log(n)
    
    Parameters:
    -----------
    returns : array-like
        对数收益率序列
    n_values : list
        子集长度候选值
    n_estimators : int
        每个n值的估计次数（取平均）
        
    Returns:
    --------
    dict : 包含Hurst指数H、c值、R方等
    """
    returns = np.array(returns)
    n = len(returns)
    
    # 过滤有效的n值
    valid_n = [k for k in n_values if k <= n // 2]
    
    if len(valid_n) < 2:
        return {'H': np.nan, 'c': np.nan, 'r_squared': np.nan}
    
    # 计算各n值对应的(R/S)_n
    log_n_list = []
    log_rs_list = []
    
    for n_val in valid_n:
        rs_estimates = []
        for _ in range(n_estimators):
            # 随机起始位置取平均
            start = np.random.randint(0, n - n_val)
            rs = calculate_r_s(returns[start:start+n_val], n_val)
            if not np.isnan(rs) and rs > 0:
                rs_estimates.append(rs)
        
        if len(rs_estimates) > 0:
            avg_rs = np.mean(rs_estimates)
            log_n_list.append(np.log(n_val))
            log_rs_list.append(np.log(avg_rs))
    
    if len(log_n_list) < 2:
        return {'H': np.nan, 'c': np.nan, 'r_squared': np.nan}
    
    # 回归: log((R/S)_n) = log(c) + H * log(n)
    result = regression_with_stats(np.array(log_rs_list), np.array(log_n_list))
    
    return {
        'H': result['slope'],
        'c': np.exp(result['intercept']),
        'r_squared': result['r_squared'],
        'p_value': result.get('p_slope', np.nan),
        'n_valid': len(log_n_list)
    }


# ============================================================
# 结果输出函数
# ============================================================

def save_results(results, fund_code, method, output_dir):
    """
    保存分析结果
    
    Parameters:
    -----------
    results : dict
        分析结果字典
    fund_code : str
        基金代码
    method : str
        方法名称
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON格式
    json_path = os.path.join(output_dir, f"{fund_code}_{method}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        results_clean = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                results_clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                results_clean[k] = v.tolist()
            else:
                results_clean[k] = v
        json.dump(results_clean, f, ensure_ascii=False, indent=2)
    
    # CSV格式（如果结果包含DataFrame）
    if 'details' in results and isinstance(results['details'], pd.DataFrame):
        csv_path = os.path.join(output_dir, f"{fund_code}_{method}_details.csv")
        results['details'].to_csv(csv_path, encoding='utf-8-sig')
    
    return json_path, csv_path if 'details' in results else None


def print_persistence_summary(fund_code, results):
    """
    打印业绩持续性分析摘要
    """
    print(f"\n{'='*60}")
    print(f"基金 {fund_code} 业绩持续性分析结果")
    print(f"{'='*60}")
    
    for method, result in results.items():
        print(f"\n【{method}】")
        if isinstance(result, dict):
            for k, v in result.items():
                if k not in ['residuals', 'details']:
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")


# ============================================================
# 滚动窗口分析
# ============================================================

def rolling_persistence_analysis(fund_returns, method='hurst', window=60, step=20):
    """
    滚动窗口业绩持续性分析
    
    Parameters:
    -----------
    fund_returns : Series
        基金收益率序列
    method : str
        分析方法 ('cross_section', 'cpr', 'hurst')
    window : int
        滚动窗口大小
    step : int
        滚动步长
        
    Returns:
    --------
    DataFrame : 各窗口的分析结果
    """
    results_list = []
    
    for i in range(0, len(fund_returns) - window, step):
        window_returns = fund_returns.iloc[i:i+window]
        start_date = fund_returns.index[i]
        end_date = fund_returns.index[i+window]
        
        result = {'start_date': start_date, 'end_date': end_date}
        
        if method == 'hurst':
            hurst_result = estimate_hurst_exponent(window_returns)
            result.update({
                'H': hurst_result.get('H'),
                'c': hurst_result.get('c'),
                'persistence': '持续' if hurst_result.get('H', 0.5) > 0.5 else '反转'
            })
        
        results_list.append(result)
    
    return pd.DataFrame(results_list)
