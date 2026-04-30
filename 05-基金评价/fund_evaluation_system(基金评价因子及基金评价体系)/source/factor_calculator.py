# -*- coding: utf-8 -*-
"""
基金评价因子计算模块

包含31个选基因子的计算逻辑：
- 收益获取能力（1个）
- 风险控制能力（6个）
- 风险调整收益（4个）
- 牛熊市表现（4个）
- 选股能力（3个）
- 择时能力（2个）
- 规模变化（4个）
- 投资者结构（4个）
- 交易能力（2个）
- 业绩持续性（1个）
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 收益获取能力因子
# ============================================================
def calc_annual_return(nav_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    年化收益率
    
    公式: R_annual = (1 + R_total)^(periods_per_year/n) - 1
    
    Parameters
    ----------
    nav_series : pd.Series
        基金净值序列
    periods_per_year : int
        年交易日数
        
    Returns
    -------
    float
        年化收益率
    """
    if len(nav_series) < 2:
        return np.nan
    
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n_periods = len(nav_series)
    
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    return annual_return


# ============================================================
# 2. 风险控制能力因子
# ============================================================
def calc_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    年化波动率
    
    公式: σ_annual = σ_daily * sqrt(periods_per_year)
    """
    if len(returns) < 2:
        return np.nan
    
    return returns.std() * np.sqrt(periods_per_year)


def calc_downside_risk(returns: pd.Series, mar: float = 0, periods_per_year: int = 252) -> float:
    """
    下行风险
    
    公式: downside_risk = sqrt(sum((min(r-mar, 0))^2) / n) * sqrt(periods_per_year)
    
    Parameters
    ----------
    returns : pd.Series
        日收益率序列
    mar : float
        最低可接受收益（Minimum Acceptable Return）
    """
    if len(returns) < 2:
        return np.nan
    
    downside = returns[returns < mar] - mar
    if len(downside) == 0:
        return 0.0
    
    downside_risk = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
    
    return downside_risk


def calc_max_drawdown(nav_series: pd.Series) -> float:
    """
    最大回撤
    
    公式: MaxDD = max((peak - trough) / peak)
    """
    if len(nav_series) < 2:
        return np.nan
    
    cummax = nav_series.cummax()
    drawdown = (cummax - nav_series) / cummax
    
    return drawdown.max()


def calc_max_recovery_days(nav_series: pd.Series) -> int:
    """
    回撤最大回补天数
    """
    if len(nav_series) < 2:
        return np.nan
    
    cummax = nav_series.cummax()
    drawdown = (cummax - nav_series) / cummax
    
    # 找到回撤期
    in_drawdown = drawdown > 0
    
    if not in_drawdown.any():
        return 0
    
    # 计算每个回撤期的持续天数
    recovery_days = []
    current_days = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_days += 1
        else:
            if current_days > 0:
                recovery_days.append(current_days)
            current_days = 0
    
    if current_days > 0:
        recovery_days.append(current_days)
    
    return max(recovery_days) if recovery_days else 0


def calc_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    VaR（在险价值）
    
    在置信水平1-alpha下的最大可能损失
    """
    if len(returns) < 10:
        return np.nan
    
    return -np.percentile(returns, alpha * 100)


def calc_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta系数
    
    公式: β = Cov(r_fund, r_bench) / Var(r_bench)
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return np.nan
    
    # 对齐
    common_idx = returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2:
        return np.nan
    
    r_fund = returns.loc[common_idx]
    r_bench = benchmark_returns.loc[common_idx]
    
    cov = np.cov(r_fund, r_bench)[0, 1]
    var = np.var(r_bench)
    
    if var == 0:
        return np.nan
    
    return cov / var


# ============================================================
# 3. 风险调整收益因子
# ============================================================
def calc_sharpe_ratio(returns: pd.Series, rf: float = 0, periods_per_year: int = 252) -> float:
    """
    夏普比率
    
    公式: Sharpe = (E(R) - Rf) / σ
    """
    if len(returns) < 2:
        return np.nan
    
    excess_return = returns.mean() * periods_per_year - rf
    volatility = calc_volatility(returns, periods_per_year)
    
    if volatility == 0:
        return np.nan
    
    return excess_return / volatility


def calc_sortino_ratio(returns: pd.Series, rf: float = 0, mar: float = 0) -> float:
    """
    索提诺比率
    
    公式: Sortino = (E(R) - Rf) / downside_risk
    """
    if len(returns) < 2:
        return np.nan
    
    excess_return = returns.mean() * 252 - rf
    downside = calc_downside_risk(returns, mar)
    
    if downside == 0:
        return np.nan
    
    return excess_return / downside


def calc_calmar_ratio(returns: pd.Series, nav_series: pd.Series) -> float:
    """
    卡玛比率
    
    公式: Calmar = annual_return / max_drawdown
    """
    if len(returns) < 2 or len(nav_series) < 2:
        return np.nan
    
    annual_ret = calc_annual_return(nav_series)
    max_dd = calc_max_drawdown(nav_series)
    
    if max_dd == 0:
        return np.nan
    
    return annual_ret / max_dd


def calc_treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series, rf: float = 0) -> float:
    """
    特雷诺比率
    
    公式: Treynor = (E(R) - Rf) / β
    """
    if len(returns) < 2:
        return np.nan
    
    excess_return = returns.mean() * 252 - rf
    beta = calc_beta(returns, benchmark_returns)
    
    if beta == 0 or np.isnan(beta):
        return np.nan
    
    return excess_return / beta


# ============================================================
# 4. 牛熊市表现因子
# ============================================================
def calc_bull_bear_returns(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252
) -> Tuple[float, float, float, float]:
    """
    计算牛熊市表现因子
    
    Parameters
    ----------
    returns : pd.Series
        基金收益率
    market_returns : pd.Series
        市场收益率（万得全A）
    window : int
        回看窗口（日）
        
    Returns
    -------
    Tuple[float, float, float, float]
        (顺境收益率, 逆境收益率, 顺境战胜市场胜率, 逆境战胜市场胜率)
    """
    if len(returns) < window or len(market_returns) < window:
        return np.nan, np.nan, np.nan, np.nan
    
    # 对齐
    common_idx = returns.index.intersection(market_returns.index)
    if len(common_idx) < window:
        return np.nan, np.nan, np.nan, np.nan
    
    r_fund = returns.loc[common_idx[-window:]]
    r_market = market_returns.loc[common_idx[-window:]]
    
    # 定义牛熊市（收益率高于中位数为牛市）
    median_ret = r_market.median()
    
    bull_days = r_market > median_ret
    bear_days = r_market <= median_ret
    
    # 顺境收益率（牛市）
    bull_returns = r_fund[bull_days]
    bull_return = (1 + bull_returns).prod() - 1 if len(bull_returns) > 0 else 0
    
    # 逆境收益率（熊市）
    bear_returns = r_fund[bear_days]
    bear_return = (1 + bear_returns).prod() - 1 if len(bear_returns) > 0 else 0
    
    # 顺境战胜市场胜率
    bull_win_rate = (bull_returns > 0).sum() / len(bull_returns) if len(bull_returns) > 0 else 0
    
    # 逆境战胜市场胜率
    bear_win_rate = (bear_returns > 0).sum() / len(bear_returns) if len(bear_returns) > 0 else 0
    
    return bull_return, bear_return, bull_win_rate, bear_win_rate


# ============================================================
# 5. 选股能力因子（Alpha）
# ============================================================
def calc_single_factor_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0
) -> float:
    """
    单因子模型Alpha
    
    公式: α = R_fund - β * (R_bench - Rf) - Rf
    """
    if len(returns) < 2:
        return np.nan
    
    beta = calc_beta(returns, benchmark_returns)
    
    if np.isnan(beta):
        return np.nan
    
    fund_return = returns.mean() * 252
    bench_return = benchmark_returns.mean() * 252
    
    alpha = fund_return - beta * (bench_return - rf) - rf
    
    return alpha


def calc_tm_model(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0
) -> Tuple[float, float]:
    """
    T-M模型（Treynor-Mazuy）
    
    公式: R_fund - Rf = α + β1*(R_bench - Rf) + β2*(R_bench - Rf)^2
    
    Returns
    -------
    Tuple[float, float]
        (alpha, timing) - 选股能力、择时能力
    """
    from scipy import stats as sp_stats
    
    if len(returns) < 10:
        return np.nan, np.nan
    
    # 对齐
    common_idx = returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 10:
        return np.nan, np.nan
    
    y = returns.loc[common_idx] - rf / 252
    x1 = benchmark_returns.loc[common_idx] - rf / 252
    x2 = x1 ** 2
    
    # 回归
    X = np.column_stack([np.ones(len(x1)), x1.values, x2.values])
    
    try:
        result = np.linalg.lstsq(X, y.values, rcond=None)
        alpha, beta1, beta2 = result[0]
        return alpha * 252, beta2  # 年化alpha, 择时系数
    except Exception:
        return np.nan, np.nan


def calc_hm_model(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0
) -> Tuple[float, float]:
    """
    H-M模型（Henriksson-Merton）
    
    公式: R_fund - Rf = α + β1*(R_bench - Rf) + β2*D*(R_bench - Rf)
    其中 D = 1 if R_bench > Rf, else 0
    
    Returns
    -------
    Tuple[float, float]
        (alpha, timing) - 选股能力、择时能力
    """
    if len(returns) < 10:
        return np.nan, np.nan
    
    # 对齐
    common_idx = returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 10:
        return np.nan, np.nan
    
    y = returns.loc[common_idx] - rf / 252
    x1 = benchmark_returns.loc[common_idx] - rf / 252
    
    # D变量
    D = (x1 > 0).astype(float)
    
    # 回归
    X = np.column_stack([np.ones(len(x1)), x1.values, (D * x1).values])
    
    try:
        result = np.linalg.lstsq(X, y.values, rcond=None)
        alpha, beta1, beta2 = result[0]
        return alpha * 252, beta2  # 年化alpha, 择时系数
    except Exception:
        return np.nan, np.nan


# ============================================================
# 6. 规模因子
# ============================================================
def calc_fund_scale(fund_info: pd.Series) -> float:
    """基金规模（亿元）"""
    return fund_info.get('size', np.nan)


def calc_fund_shares(fund_info: pd.Series) -> float:
    """基金份额（亿份）"""
    return fund_info.get('shares', np.nan)


def calc_scale_growth(fund_info_current: pd.Series, fund_info_prev: pd.Series) -> float:
    """基金规模增长率"""
    scale_current = fund_info_current.get('size', np.nan)
    scale_prev = fund_info_prev.get('size', np.nan)
    
    if np.isnan(scale_current) or np.isnan(scale_prev) or scale_prev == 0:
        return np.nan
    
    return (scale_current - scale_prev) / scale_prev


def calc_shares_growth(fund_info_current: pd.Series, fund_info_prev: pd.Series) -> float:
    """基金份额增长率"""
    shares_current = fund_info_current.get('shares', np.nan)
    shares_prev = fund_info_prev.get('shares', np.nan)
    
    if np.isnan(shares_current) or np.isnan(shares_prev) or shares_prev == 0:
        return np.nan
    
    return (shares_current - shares_prev) / shares_prev


# ============================================================
# 7. 投资者结构因子
# ============================================================
def calc_holder_structure(fund_info: pd.Series) -> Tuple[float, float, float, float]:
    """
    投资者结构因子
    
    Returns
    -------
    Tuple[float, float, float, float]
        (管理人员工持有占比, 机构投资者占比, 个人投资者占比, 户均持有份额)
    """
    manager_holding = fund_info.get('manager_holding_ratio', np.nan)
    inst_holding = fund_info.get('inst_holding_ratio', np.nan)
    personal_holding = fund_info.get('personal_holding_ratio', np.nan)
    avg_shares = fund_info.get('avg_shares_per_holder', np.nan)
    
    return manager_holding, inst_holding, personal_holding, avg_shares


# ============================================================
# 8. 交易能力因子
# ============================================================
def calc_invisible_trading_ability(
    fund_returns: pd.Series,
    simulated_returns: pd.Series
) -> float:
    """
    隐形交易能力
    
    公式: IR = E(R_fund - R_sim) / std(R_fund - R_sim)
    
    Parameters
    ----------
    fund_returns : pd.Series
        基金实际收益率
    simulated_returns : pd.Series
        模拟持仓收益率
    """
    if len(fund_returns) < 10 or len(simulated_returns) < 10:
        return np.nan
    
    # 对齐
    common_idx = fund_returns.index.intersection(simulated_returns.index)
    if len(common_idx) < 10:
        return np.nan
    
    excess = fund_returns.loc[common_idx] - simulated_returns.loc[common_idx]
    
    mean_excess = excess.mean() * 252
    std_excess = excess.std() * np.sqrt(252)
    
    if std_excess == 0:
        return np.nan
    
    return mean_excess / std_excess


def calc_turnover_rate(fund_info: pd.Series) -> float:
    """换手率"""
    return fund_info.get('turnover_rate', np.nan)


# ============================================================
# 9. 业绩持续性因子
# ============================================================
def calc_hurst_exponent(returns: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst指数
    
    用于衡量时间序列的长期记忆性
    H > 0.5: 持续性（趋势延续）
    H = 0.5: 随机游走
    H < 0.5: 均值回归
    
    Parameters
    ----------
    returns : pd.Series
        收益率序列
    max_lag : int
        最大滞后阶数
    """
    if len(returns) < max_lag * 2:
        return np.nan
    
    returns = returns.dropna().values
    
    # R/S分析
    lags = range(2, max_lag + 1)
    rs_values = []
    
    for lag in lags:
        # 分割为lag长度的子序列
        n_subsequences = len(returns) // lag
        if n_subsequences == 0:
            continue
        
        rs_list = []
        
        for i in range(n_subsequences):
            sub_seq = returns[i * lag:(i + 1) * lag]
            
            # 累积离差
            mean = np.mean(sub_seq)
            cum_dev = np.cumsum(sub_seq - mean)
            
            # R = max - min
            R = np.max(cum_dev) - np.min(cum_dev)
            
            # S = std
            S = np.std(sub_seq)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 2:
        return np.nan
    
    # 回归 log(R/S) = log(c) + H * log(n)
    log_rs = np.log(rs_values)
    log_n = np.log(list(range(2, len(rs_values) + 2)))
    
    # 线性回归
    try:
        slope, intercept = np.polyfit(log_n, log_rs, 1)
        return slope  # H指数
    except Exception:
        return np.nan


# ============================================================
# 10. 综合因子计算函数
# ============================================================
def calc_all_factors(
    nav_series: pd.Series,
    benchmark_returns: pd.Series,
    fund_info: Optional[pd.Series] = None,
    window: int = 252
) -> pd.Series:
    """
    计算所有因子
    
    Parameters
    ----------
    nav_series : pd.Series
        基金净值序列
    benchmark_returns : pd.Series
        基准收益率（同类基金中位数或市场指数）
    fund_info : pd.Series, optional
        基金信息（规模、份额、持有人结构等）
    window : int
        回看窗口（日）
        
    Returns
    -------
    pd.Series
        所有因子值
    """
    factors = {}
    
    # 计算日收益率
    returns = nav_series.pct_change().dropna()
    
    if len(returns) < window:
        print(f'[WARNING] 数据不足，需要至少{window}个交易日')
        return pd.Series()
    
    # 取最近window期
    returns = returns.iloc[-window:]
    nav_window = nav_series.iloc[-window:]
    
    # 1. 收益获取能力
    factors['年化收益率'] = calc_annual_return(nav_window)
    
    # 2. 风险控制能力
    factors['波动率'] = calc_volatility(returns)
    factors['下行风险'] = calc_downside_risk(returns)
    factors['最大回撤'] = calc_max_drawdown(nav_window)
    factors['回撤最大回补天数'] = calc_max_recovery_days(nav_window)
    factors['VaR'] = calc_var(returns)
    
    # 对齐基准收益率
    bench_aligned = benchmark_returns.reindex(returns.index, method='ffill')
    factors['beta'] = calc_beta(returns, bench_aligned)
    
    # 3. 风险调整收益
    factors['夏普比率'] = calc_sharpe_ratio(returns)
    factors['索提诺比率'] = calc_sortino_ratio(returns)
    factors['卡玛比率'] = calc_calmar_ratio(returns, nav_window)
    factors['特雷诺比率'] = calc_treynor_ratio(returns, bench_aligned)
    
    # 4. 牛熊市表现
    bull_ret, bear_ret, bull_win, bear_win = calc_bull_bear_returns(
        returns, bench_aligned, window
    )
    factors['顺境收益率'] = bull_ret
    factors['逆境收益率'] = bear_ret
    factors['顺境战胜市场胜率'] = bull_win
    factors['逆境战胜市场胜率'] = bear_win
    
    # 5. 选股能力
    factors['单因子模型alpha'] = calc_single_factor_alpha(returns, bench_aligned)
    tm_alpha, tm_timing = calc_tm_model(returns, bench_aligned)
    factors['T-M模型alpha'] = tm_alpha
    factors['T-M模型择时'] = tm_timing
    
    hm_alpha, hm_timing = calc_hm_model(returns, bench_aligned)
    factors['H-M模型alpha'] = hm_alpha
    factors['H-M模型择时'] = hm_timing
    
    # 6. 业绩持续性
    factors['Hurst指数'] = calc_hurst_exponent(returns)
    
    # 7. 规模因子（需要fund_info）
    if fund_info is not None:
        factors['基金规模'] = calc_fund_scale(fund_info)
        factors['基金份额'] = calc_fund_shares(fund_info)
        
        manager, inst, personal, avg_shares = calc_holder_structure(fund_info)
        factors['管理人员工持有占比'] = manager
        factors['机构投资者占比'] = inst
        factors['个人投资者占比'] = personal
        factors['户均持有份额'] = avg_shares
        factors['换手率'] = calc_turnover_rate(fund_info)
    
    return pd.Series(factors)
