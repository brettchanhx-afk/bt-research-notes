# -*- coding: utf-8 -*-
"""
因子计算模块
===================================
实现三种业绩持续性评价方法：
1. 横截面分析法 (Cross-Section Analysis)
2. 交叉积比率法 (Cross-Product Ratio, CPR)
3. Hurst指数法 (Hurst Exponent)
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.utils import (
    calculate_excess_return,
    split_periods,
    calculate_period_return,
    classify_winners_losers,
    regression_with_stats,
    estimate_hurst_exponent,
    calculate_r_s
)
from config import RISK_FREE_RATE, CROSS_SECTION_CONFIG, CPR_CONFIG, HURST_CONFIG


# ============================================================
# 横截面分析法
# ============================================================

def cross_section_analysis(fund_returns, benchmark_returns=None, 
                          risk_free_rate=RISK_FREE_RATE,
                          min_periods=CROSS_SECTION_CONFIG['min_periods']):
    """
    横截面分析法
    
    原理：将样本期划分为两个等长子期间，
    检验评价期超额收益与持有期超额收益的正相关性
    
    公式: α₂ᵢ = α + β × α₁ᵢ
    
    其中:
    - α₁ᵢ: 基金i在评价期的超额收益
    - α₂ᵢ: 基金i在持有期的超额收益
    - α: 截距
    - β: 持续性系数（若显著为正，说明业绩有持续性）
    
    Parameters:
    -----------
    fund_returns : DataFrame
        基金收益率数据，列为基金代码，行为日期
    benchmark_returns : Series, optional
        基准收益率
    risk_free_rate : float
        年化无风险利率
    min_periods : int
        最少需要的期数
        
    Returns:
    --------
    dict : 包含:
        - alpha (截距)
        - beta (持续性系数)
        - p_value (p值)
        - r_squared (R方)
        - persistence_verdict (持续性判断: '有持续性'/'无持续性'/'有反转倾向')
    """
    # 确保有足够的数据
    if len(fund_returns) < min_periods * 2:
        print(f"  [警告] 数据不足，最少需要 {min_periods * 2} 期，当前仅有 {len(fund_returns)} 期")
        return None
    
    # 划分为两个等长子期间
    n = len(fund_returns)
    mid_point = n // 2
    
    period1_returns = fund_returns.iloc[:mid_point]
    period2_returns = fund_returns.iloc[mid_point:]
    
    results = {}
    
    # 对每只基金计算超额收益
    for col in fund_returns.columns:
        fund_p1 = period1_returns[col].dropna()
        fund_p2 = period2_returns[col].dropna()
        
        if len(fund_p1) < min_periods or len(fund_p2) < min_periods:
            continue
        
        # 计算期间超额收益（使用累计收益）
        alpha1 = calculate_period_return(fund_p1)  # 评价期超额收益
        alpha2 = calculate_period_return(fund_p2)  # 持有期超额收益
        
        # 如果有基准，使用基金-基准的超额收益
        if benchmark_returns is not None:
            bm_p1 = benchmark_returns.iloc[:mid_point].reindex(fund_p1.index)
            bm_p2 = benchmark_returns.iloc[mid_point:].reindex(fund_p2.index)
            
            alpha1 = calculate_period_return(fund_p1) - calculate_period_return(bm_p1.dropna())
            alpha2 = calculate_period_return(fund_p2) - calculate_period_return(bm_p2.dropna())
        
        results[col] = {
            'alpha1': alpha1,
            'alpha2': alpha2
        }
    
    if len(results) < 3:  # 至少需要3只基金
        print(f"  [警告] 基金数量不足，至少需要3只，当前仅有 {len(results)} 只")
        return None
    
    # 构建回归数据
    df_results = pd.DataFrame(results).T
    X = df_results['alpha1'].values
    y = df_results['alpha2'].values
    
    # 回归分析
    reg_result = regression_with_stats(y, X)
    
    # 判断持续性
    beta = reg_result['slope']
    p_value = reg_result['p_slope']
    
    if p_value < 0.05 and beta > 0:
        verdict = "有持续性"
    elif p_value < 0.05 and beta < 0:
        verdict = "有反转倾向"
    else:
        verdict = "无显著持续性"
    
    return {
        'alpha': reg_result['intercept'],
        'beta': beta,
        't_stat': reg_result['t_slope'],
        'p_value': p_value,
        'r_squared': reg_result['r_squared'],
        'n_funds': len(results),
        'persistence_verdict': verdict,
        'details': df_results
    }


def cross_section_single_fund(fund_returns, benchmark_returns=None, 
                              risk_free_rate=RISK_FREE_RATE,
                              min_periods=CROSS_SECTION_CONFIG['min_periods']):
    """
    单只基金的横截面持续性分析（与传统方法的区别是不依赖同类基金池）
    
    Parameters:
    -----------
    fund_returns : Series
        基金收益率序列
    benchmark_returns : Series, optional
        基准收益率
    risk_free_rate : float
        年化无风险利率
    min_periods : int
        最少需要的期数
        
    Returns:
    --------
    dict : 分析结果
    """
    if len(fund_returns) < min_periods * 2:
        return None
    
    n = len(fund_returns)
    mid_point = n // 2
    
    # 计算两个期间的累计超额收益
    if benchmark_returns is not None:
        # 与基准对比的超额收益
        excess = fund_returns - benchmark_returns
    else:
        # 与无风险利率对比的超额收益
        daily_rf = risk_free_rate / 252
        excess = fund_returns - daily_rf
    
    period1 = excess.iloc[:mid_point]
    period2 = excess.iloc[mid_point:]
    
    alpha1 = calculate_period_return(period1)
    alpha2 = calculate_period_return(period2)
    
    # 判断逻辑：如果两个期间超额收益同号，说明有持续性
    same_sign = (alpha1 > 0 and alpha2 > 0) or (alpha1 < 0 and alpha2 < 0)
    
    return {
        'alpha1': alpha1,
        'alpha2': alpha2,
        'period1_return': calculate_period_return(period1 + (benchmark_returns.iloc[:mid_point] if benchmark_returns is not None else 0)),
        'period2_return': calculate_period_return(period2 + (benchmark_returns.iloc[mid_point:] if benchmark_returns is not None else 0)),
        'same_sign': same_sign,
        'persistence': '有持续性' if same_sign else '无持续性'
    }


# ============================================================
# 交叉积比率法 (CPR)
# ============================================================

def calculate_cpr(fund_returns_list, n_periods=4):
    """
    交叉积比率法 (Cross-Product Ratio)
    
    原理：将样本期分为多个等长期间，比较相邻期间的赢家/输家组合
    
    公式: CPR = (WW × LL) / (WL × LW)
    
    其中:
    - WW: 连续两个期间均为赢家
    - LL: 连续两个期间均为输家
    - WL: 第一期间赢，第二期间输
    - LW: 第一期间输，第二期间赢
    
    判断:
    - CPR ≈ 1: 业绩不具有持续性
    - CPR > 1: 业绩有持续性（CPR越大，持续性越强）
    - CPR < 1: 业绩有反转倾向
    
    Parameters:
    -----------
    fund_returns_list : list of (fund_code, returns_series) tuples
        基金代码和收益率序列的列表
    n_periods : int
        划分期数
        
    Returns:
    --------
    dict : 包含 CPR值 及 各组合数量
    """
    # 将数据分为n_periods个期间
    n_funds = len(fund_returns_list)
    
    # 计算每只基金在每个期间的收益率
    fund_period_returns = []
    
    for fund_code, returns in fund_returns_list:
        returns = returns.dropna()
        n = len(returns)
        period_len = n // n_periods
        
        period_rets = []
        for i in range(n_periods):
            start_idx = i * period_len
            end_idx = (i + 1) * period_len if i < n_periods - 1 else n
            period_ret = calculate_period_return(returns.iloc[start_idx:end_idx])
            period_rets.append(period_ret)
        
        fund_period_returns.append((fund_code, period_rets))
    
    # 计算所有基金在各期间的中位数
    n_periods_actual = len(fund_period_returns[0][1])
    
    # 分类赢家和输家
    period_winners_losers = []
    
    for period_idx in range(n_periods_actual):
        # 该期间所有基金的收益率
        period_returns = [fund[1][period_idx] for fund in fund_period_returns]
        median_return = np.median(period_returns)
        
        # 分类
        wl = []
        for fund_code, period_rets in fund_period_returns:
            classification = 'W' if period_rets[period_idx] > median_return else 'L'
            wl.append((fund_code, classification))
        
        period_winners_losers.append(wl)
    
    # 计算各组合数量
    WW = 0  # 连续赢
    LL = 0  # 连续输
    WL = 0  # 先赢后输
    LW = 0  # 先输后赢
    
    for fund_code, _ in fund_period_returns:
        for i in range(len(period_winners_losers) - 1):
            # 找到该基金在各期间的分类
            current_period = period_winners_losers[i]
            next_period = period_winners_losers[i + 1]
            
            current_class = None
            next_class = None
            
            for code, cls in current_period:
                if code == fund_code:
                    current_class = cls
                    break
            
            for code, cls in next_period:
                if code == fund_code:
                    next_class = cls
                    break
            
            if current_class is None or next_class is None:
                continue
            
            # 统计组合
            if current_class == 'W' and next_class == 'W':
                WW += 1
            elif current_class == 'L' and next_class == 'L':
                LL += 1
            elif current_class == 'W' and next_class == 'L':
                WL += 1
            elif current_class == 'L' and next_class == 'W':
                LW += 1
    
    # 计算CPR
    if WL == 0 or LW == 0:
        # 如果没有WL或LW，CPR无法计算
        if CPR_CONFIG['require_nonzero']:
            print(f"  [警告] WL或LW为0，CPR无法计算")
            cpr = np.nan
        else:
            # 使用拉普拉斯平滑
            cpr = (WW + 1) * (LL + 1) / ((WL + 1) * (LW + 1))
    else:
        cpr = (WW * LL) / (WL * LW)
    
    # 判断持续性
    if np.isnan(cpr):
        verdict = "无法判断"
    elif cpr > 1:
        verdict = "有持续性"
    elif cpr < 1:
        verdict = "有反转倾向"
    else:
        verdict = "无显著持续性"
    
    return {
        'CPR': cpr,
        'WW': WW,
        'LL': LL,
        'WL': WL,
        'LW': LW,
        'n_periods': n_periods_actual,
        'n_funds': n_funds,
        'persistence_verdict': verdict
    }


def calculate_cpr_single_fund(fund_returns, n_periods=4, market_median_returns=None):
    """
    单只基金的CPR分析（需要市场/同类基金收益率中位数作为基准）
    
    Parameters:
    -----------
    fund_returns : Series
        基金收益率序列
    n_periods : int
        划分期数
    market_median_returns : Series, optional
        市场/同类基金收益率中位数序列
        
    Returns:
    --------
    dict : 分析结果
    """
    returns = fund_returns.dropna()
    n = len(returns)
    
    if n < n_periods * 2:
        return None
    
    period_len = n // n_periods
    
    # 计算各期间收益率
    period_rets = []
    for i in range(n_periods):
        start_idx = i * period_len
        end_idx = (i + 1) * period_len if i < n_periods - 1 else n
        period_ret = calculate_period_return(returns.iloc[start_idx:end_idx])
        period_rets.append(period_ret)
    
    # 如果有市场中位数，使用它作为基准
    if market_median_returns is not None:
        market_period_rets = []
        for i in range(n_periods):
            start_idx = i * period_len
            end_idx = (i + 1) * period_len if i < n_periods - 1 else n
            mkt_rets = market_median_returns.reindex(returns.iloc[start_idx:end_idx].index).dropna()
            if len(mkt_rets) > 0:
                market_period_rets.append(calculate_period_return(mkt_rets))
            else:
                market_period_rets.append(np.nan)
        
        # 使用基金相对于市场的超额收益
        period_rets = [f - m for f, m in zip(period_rets, market_period_rets)]
    
    # 计算CPR所需的四格表
    # 相比简单的是否为正，使用中位数划分
    median_ret = np.median(period_rets)
    
    # 标记赢家和输家
    classifications = ['W' if r > median_ret else 'L' for r in period_rets]
    
    # 计算各组合数量
    WW = sum(1 for i in range(len(classifications)-1) 
             if classifications[i] == 'W' and classifications[i+1] == 'W')
    LL = sum(1 for i in range(len(classifications)-1) 
             if classifications[i] == 'L' and classifications[i+1] == 'L')
    WL = sum(1 for i in range(len(classifications)-1) 
             if classifications[i] == 'W' and classifications[i+1] == 'L')
    LW = sum(1 for i in range(len(classifications)-1) 
             if classifications[i] == 'L' and classifications[i+1] == 'W')
    
    if WL == 0 or LW == 0:
        cpr = np.nan
    else:
        cpr = (WW * LL) / (WL * LW)
    
    return {
        'CPR': cpr,
        'WW': WW,
        'LL': LL,
        'WL': WL,
        'LW': LW,
        'period_returns': period_rets,
        'classifications': classifications,
        'periods': n_periods
    }


# ============================================================
# Hurst指数法
# ============================================================

def hurst_analysis(fund_returns, log_returns=None,
                  n_values=HURST_CONFIG['n_values'],
                  n_estimators=HURST_CONFIG['n_estimators'],
                  min_periods=HURST_CONFIG['min_periods']):
    """
    Hurst指数法
    
    原理：研究时间序列历史取值对未来取值的影响力（长记忆性）
    
    公式: log((R/S)_n) = log(c) + H × log(n)
    
    判断:
    - 0.5 < H < 1: 业绩有正向持续性（越接近1，持续性越强）
    - H = 0.5: 收益随机波动，不具备持续性
    - 0 < H < 0.5: 业绩有反转倾向（越接近0，反转性越强）
    
    Parameters:
    -----------
    fund_returns : Series
        基金收益率序列（日收益率）
    log_returns : Series, optional
        对数收益率序列（如果未提供会自动计算）
    n_values : list
        子集长度候选值
    n_estimators : int
        每个n值的估计次数
    min_periods : int
        最少需要的期数
        
    Returns:
    --------
    dict : 包含:
        - H: Hurst指数
        - c: 常数项
        - r_squared: 回归R方
        - persistence_verdict: 持续性判断
    """
    # 使用对数收益率（更符合Hurst分析假设）
    if log_returns is not None:
        returns = log_returns.dropna()
    else:
        returns = np.log(1 + fund_returns).dropna()
    
    if len(returns) < min_periods:
        print(f"  [警告] 数据不足，最少需要 {min_periods} 期，当前仅有 {len(returns)} 期")
        return None
    
    # 估计Hurst指数
    hurst_result = estimate_hurst_exponent(
        returns.values, 
        n_values=n_values,
        n_estimators=n_estimators
    )
    
    H = hurst_result.get('H', np.nan)
    
    # 判断持续性
    if np.isnan(H):
        verdict = "无法判断"
    elif H > 0.5:
        if H > 0.7:
            verdict = "强持续性"
        else:
            verdict = "有持续性"
    elif H == 0.5:
        verdict = "无持续性(随机)"
    else:
        if H < 0.3:
            verdict = "强反转倾向"
        else:
            verdict = "有反转倾向"
    
    return {
        'H': H,
        'c': hurst_result.get('c', np.nan),
        'r_squared': hurst_result.get('r_squared', np.nan),
        'p_value': hurst_result.get('p_value', np.nan),
        'n_observations': len(returns),
        'persistence_verdict': verdict,
        'hurst_category': _categorize_hurst(H)
    }


def _categorize_hurst(H):
    """
    将Hurst指数分类
    
    Returns:
    --------
    str : 分类标签
    """
    if np.isnan(H):
        return "未知"
    elif H > 0.8:
        return "极强持续"
    elif H > 0.6:
        return "强持续"
    elif H > 0.5:
        return "中等持续"
    elif H == 0.5:
        return "随机"
    elif H > 0.4:
        return "弱反转"
    elif H > 0.2:
        return "中等反转"
    else:
        return "强反转"


# ============================================================
# 综合分析
# ============================================================

def comprehensive_persistence_analysis(fund_returns, benchmark_returns=None,
                                      fund_pool_returns=None):
    """
    综合使用三种方法分析基金业绩持续性
    
    Parameters:
    -----------
    fund_returns : Series
        基金收益率序列
    benchmark_returns : Series, optional
        基准收益率序列
    fund_pool_returns : DataFrame, optional
        基金池收益率（用于CPR分析）
        
    Returns:
    --------
    dict : 综合分析结果
    """
    results = {}
    
    # 1. 横截面分析法（单基金版本）
    cs_result = cross_section_single_fund(
        fund_returns, 
        benchmark_returns
    )
    if cs_result:
        results['横截面分析法'] = cs_result
    
    # 2. CPR法
    if fund_pool_returns is not None:
        # 使用基金池计算CPR
        fund_returns_list = [(col, fund_pool_returns[col]) for col in fund_pool_returns.columns]
        cpr_result = calculate_cpr(fund_returns_list, n_periods=CPR_CONFIG.get('min_periods', 4))
        results['交叉积比率法'] = cpr_result
    else:
        # 单基金CPR
        cpr_result = calculate_cpr_single_fund(fund_returns, n_periods=4)
        if cpr_result:
            results['交叉积比率法'] = cpr_result
    
    # 3. Hurst指数法
    hurst_result = hurst_analysis(fund_returns)
    if hurst_result:
        results['Hurst指数法'] = hurst_result
    
    # 综合判断
    verdicts = [r.get('persistence_verdict', '') for r in results.values() 
                if isinstance(r, dict) and 'persistence_verdict' in r]
    
    # 统计各判断的数量
    persistence_count = sum(1 for v in verdicts if '持续' in v and '无' not in v)
    reversal_count = sum(1 for v in verdicts if '反转' in v)
    
    if persistence_count > reversal_count and persistence_count >= 2:
        overall_verdict = "业绩整体有持续性"
    elif reversal_count > persistence_count and reversal_count >= 2:
        overall_verdict = "业绩整体有反转倾向"
    elif persistence_count == reversal_count and persistence_count > 0:
        overall_verdict = "业绩持续性不明确"
    else:
        overall_verdict = "无法判断"
    
    results['综合判断'] = overall_verdict
    
    return results


def _categorize_hurst(H):
    if np.isnan(H):
        return "未知"
    elif H > 0.8:
        return "极强持续"
    elif H > 0.6:
        return "强持续"
    elif H > 0.5:
        return "中等持续"
    elif H == 0.5:
        return "随机"
    elif H > 0.4:
        return "弱反转"
    elif H > 0.2:
        return "中等反转"
    else:
        return "强反转"
