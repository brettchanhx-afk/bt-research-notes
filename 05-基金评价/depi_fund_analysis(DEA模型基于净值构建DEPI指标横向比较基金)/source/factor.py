# -*- coding: utf-8 -*-
"""
因子计算模块
计算 DEA/DEPI 所需的全部投入指标：
  1. 区间超额收益 R_j（产出指标）
  2. volatility  年化波动率（投入指标）
  3. fee_rate    基金费率（投入指标）
  4. timing_alpha  C-L 模型 alpha（投入指标）
  5. timing_beta   C-L 模型 beta2-beta1（投入指标）
"""
import pandas as pd
import numpy as np
from scipy import stats


# ============================================================
# 收益率计算
# ============================================================
def calc_returns(nav_df: pd.DataFrame) -> pd.Series:
    """从净值序列计算日收益率序列。
    
    Parameters
    ----------
    nav_df : pd.DataFrame
        包含 '净值' 列的 DataFrame，索引为日期
    
    Returns
    -------
    pd.Series
        日收益率序列
    """
    if '净值' in nav_df.columns:
        nav = nav_df['净值'].dropna().sort_index()
    else:
        nav = nav_df.iloc[:, 0].dropna().sort_index()
    returns = nav.pct_change().dropna()
    returns.name = '日收益率'
    return returns


def calc_excess_return(fund_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       risk_free_rate: float = 0.03) -> float:
    """计算区间年化超额收益（产出指标 R_j）。
    
    Parameters
    ----------
    fund_returns, benchmark_returns : pd.Series
        日收益率序列
    risk_free_rate : float
        年化无风险利率
    
    Returns
    -------
    float
        区间超额收益（年化）
    """
    rf_daily = risk_free_rate / 252
    fund_excess = fund_returns - rf_daily
    bench_excess = benchmark_returns - rf_daily
    ann_fund = (1 + fund_returns.mean()) ** 252 - 1
    ann_bench = (1 + benchmark_returns.mean()) ** 252 - 1
    return ann_fund - ann_bench


# ============================================================
# 波动率（年化）
# ============================================================
def calc_volatility(fund_returns: pd.Series) -> float:
    """计算日收益率标准差（年化波动率）。
    
    Parameters
    ----------
    fund_returns : pd.Series
        日收益率序列
    
    Returns
    -------
    float
        年化波动率
    """
    return fund_returns.std() * np.sqrt(252)


# ============================================================
# C-L 择时能力模型（与研报一致）
# ============================================================
def calc_timing_ability(fund_returns: pd.Series,
                        benchmark_returns: pd.Series,
                        window: int = 60) -> dict:
    """计算 C-L 模型择时能力指标。
    
    公式：R_p - R_f = alpha + beta1*(R_m - R_f) + beta2*(R_m - R_f)^2 + epsilon
    
    其中：
      alpha   = 选股能力（独立于市场的超额收益）
      beta2   = 择时能力（市场上涨时放大暴露，下跌时减少暴露）
      beta2 > 0 表示具备正向择时能力
    
    Parameters
    ----------
    fund_returns, benchmark_returns : pd.Series
        日收益率序列
    window : int
        滚动窗口天数（用于滚动回归）
    
    Returns
    -------
    dict
        {'alpha': ..., 'beta2': ..., 'timing_beta': beta2}
        timing_beta = beta2（与研报一致）
    """
    # 对齐日期
    aligned = pd.DataFrame({
        'fund': fund_returns,
        'benchmark': benchmark_returns,
    }).dropna()
    
    if len(aligned) < window:
        return {'alpha': 0.0, 'beta2': 0.0, 'timing_beta': 0.0}

    rf_daily = 0.03 / 252
    excess_fund = aligned['fund'] - rf_daily
    excess_bench = aligned['benchmark'] - rf_daily

    # OLS 回归：R_fund = alpha + beta1*R_bench + beta2*R_bench^2
    X = np.column_stack([
        np.ones(len(excess_bench)),
        excess_bench.values,
        excess_bench.values ** 2,
    ])
    y = excess_fund.values

    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        alpha = beta[0]
        beta2 = beta[2]
        return {'alpha': float(alpha), 'beta2': float(beta2),
                'timing_beta': float(beta2)}
    except Exception:
        return {'alpha': 0.0, 'beta2': 0.0, 'timing_beta': 0.0}


# ============================================================
# 构建因子表格
# ============================================================
def build_factor_table(nav_dict: dict,
                       benchmark_returns: pd.Series,
                       fee_df: pd.DataFrame,
                       risk_free_rate: float = 0.03) -> pd.DataFrame:
    """构建 DEPI 分析所需的因子表格。
    
    包含列：基金代码, 超额收益R, volatility, fee_rate, timing_alpha, timing_beta
    每行是一只基金在观测期的截面数据。
    
    Parameters
    ----------
    nav_dict : dict
        {基金代码: nav_df} 净值历史
    benchmark_returns : pd.Series
        基准指数日收益率
    fee_df : pd.DataFrame
        费率数据
    risk_free_rate : float
        年化无风险利率
    
    Returns
    -------
    pd.DataFrame
        因子表格
    """
    rows = []
    for code, nav_df in nav_dict.items():
        fund_returns = calc_returns(nav_df)
        if len(fund_returns) < 30:
            continue
        
        # 对齐基准
        aligned_bench = benchmark_returns.reindex(fund_returns.index).dropna()
        aligned_fund = fund_returns.reindex(aligned_bench.index)
        if len(aligned_fund) < 30:
            continue

        excess_ret = calc_excess_return(aligned_fund, aligned_bench, risk_free_rate)
        volatility = calc_volatility(aligned_fund)
        timing = calc_timing_ability(aligned_fund, aligned_bench)

        fee_row = fee_df[fee_df['基金代码'] == code]
        fee_rate = float(fee_row['总费率'].values[0]) if len(fee_row) > 0 else 0.0175

        rows.append({
            '基金代码': code,
            '超额收益R': excess_ret,
            'volatility': volatility,
            'fee_rate': fee_rate,
            'timing_alpha': timing['alpha'],
            'timing_beta': timing['timing_beta'],
        })

    df = pd.DataFrame(rows)
    print(f'  [因子表] 有效基金 {len(df)} 只')
    return df.reset_index(drop=True)
