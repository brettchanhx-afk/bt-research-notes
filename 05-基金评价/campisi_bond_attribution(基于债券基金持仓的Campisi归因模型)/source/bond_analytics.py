# -*- coding: utf-8 -*-
"""
债券分析工具模块

功能：
  - 计算债券久期（Macaulay Duration）
  - 计算修正久期（Modified Duration）
  - 计算凸性（Convexity）
  - 计算债券价格
  - 计算到期收益率（YTM）
"""
import numpy as np
import pandas as pd
from scipy.optimize import newton
from typing import Optional


# ============================================================
# 久期计算
# ============================================================
def macaulay_duration(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 1
) -> float:
    """计算麦考利久期（Macaulay Duration）。
    
    公式：
        D = Σ[t × CF_t / (1+y)^t] / P
    
    Parameters
    ----------
    face_value : float
        债券面值
    coupon_rate : float
        票面利率（年化，如0.05表示5%）
    ytm : float
        到期收益率（年化）
    years_to_maturity : float
        剩余期限（年）
    frequency : int
        付息频率（年付息次数，1=年付，2=半年付）
    
    Returns
    -------
    float
        麦考利久期（年）
    """
    if years_to_maturity <= 0:
        return 0.0
    
    # 每期票息
    coupon = face_value * coupon_rate / frequency
    # 每期收益率
    period_ytm = ytm / frequency
    # 总期数
    n_periods = int(years_to_maturity * frequency)
    
    # 计算现值加权的现金流时间
    weighted_time = 0.0
    pv_cash_flows = 0.0
    
    for t in range(1, n_periods + 1):
        # 第t期的现金流
        if t == n_periods:
            cf = coupon + face_value  # 最后一期包含本金
        else:
            cf = coupon
        
        # 现值
        pv = cf / (1 + period_ytm) ** t
        
        # 时间（年）
        time_years = t / frequency
        
        weighted_time += time_years * pv
        pv_cash_flows += pv
    
    # 麦考利久期
    if pv_cash_flows > 0:
        return weighted_time / pv_cash_flows
    return 0.0


def modified_duration(
    mac_duration: float,
    ytm: float,
    frequency: int = 1
) -> float:
    """计算修正久期（Modified Duration）。
    
    公式：
        MD = D / (1 + y/m)
    
    Parameters
    ----------
    mac_duration : float
        麦考利久期
    ytm : float
        到期收益率（年化）
    frequency : int
        付息频率
    
    Returns
    -------
    float
        修正久期
    """
    return mac_duration / (1 + ytm / frequency)


def effective_duration(
    price_func,
    ytm: float,
    delta_y: float = 0.0001
) -> float:
    """计算有效久期（Effective Duration）。
    
    公式：
        D_eff = (P_- - P_+) / (2 × P_0 × Δy)
    
    Parameters
    ----------
    price_func : callable
        价格函数，接受ytm返回价格
    ytm : float
        当前到期收益率
    delta_y : float
        收益率变动量
    
    Returns
    -------
    float
        有效久期
    """
    p_minus = price_func(ytm - delta_y)
    p_plus = price_func(ytm + delta_y)
    p_0 = price_func(ytm)
    
    return (p_minus - p_plus) / (2 * p_0 * delta_y)


# ============================================================
# 凸性计算
# ============================================================
def convexity(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 1
) -> float:
    """计算凸性（Convexity）。
    
    公式：
        C = Σ[t×(t+1) × CF_t / (1+y)^(t+2)] / P
    
    Parameters
    ----------
    同 macaulay_duration
    
    Returns
    -------
    float
        凸性
    """
    if years_to_maturity <= 0:
        return 0.0
    
    coupon = face_value * coupon_rate / frequency
    period_ytm = ytm / frequency
    n_periods = int(years_to_maturity * frequency)
    
    weighted_time_sq = 0.0
    pv_cash_flows = 0.0
    
    for t in range(1, n_periods + 1):
        if t == n_periods:
            cf = coupon + face_value
        else:
            cf = coupon
        
        pv = cf / (1 + period_ytm) ** t
        time_years = t / frequency
        
        # 凸性公式中的 t*(t+1)/frequency^2
        weighted_time_sq += (t * (t + 1)) / (frequency ** 2) * pv
        pv_cash_flows += pv
    
    if pv_cash_flows > 0:
        # 凸性 = Σ[t(t+1)CF/(1+y)^(t+2)] / P / (1+y)^2
        return weighted_time_sq / pv_cash_flows / (1 + period_ytm) ** 2
    return 0.0


# ============================================================
# 债券价格计算
# ============================================================
def bond_price(
    face_value: float,
    coupon_rate: float,
    ytm: float,
    years_to_maturity: float,
    frequency: int = 1,
    accrued_interest: float = 0.0
) -> float:
    """计算债券全价。
    
    公式：
        P = Σ[C/(1+y)^t] + F/(1+y)^N
    
    Parameters
    ----------
    face_value : float
        面值
    coupon_rate : float
        票面利率（年化）
    ytm : float
        到期收益率（年化）
    years_to_maturity : float
        剩余期限（年）
    frequency : int
        付息频率
    accrued_interest : float
        应计利息
    
    Returns
    -------
    float
        债券全价
    """
    if years_to_maturity <= 0:
        return face_value
    
    coupon = face_value * coupon_rate / frequency
    period_ytm = ytm / frequency
    n_periods = int(years_to_maturity * frequency)
    
    # 票息现值
    pv_coupons = 0.0
    for t in range(1, n_periods + 1):
        pv_coupons += coupon / (1 + period_ytm) ** t
    
    # 本金现值
    pv_principal = face_value / (1 + period_ytm) ** n_periods
    
    return pv_coupons + pv_principal + accrued_interest


def clean_price(full_price: float, accrued_interest: float) -> float:
    """计算净价。
    
    净价 = 全价 - 应计利息
    """
    return full_price - accrued_interest


# ============================================================
# 到期收益率计算
# ============================================================
def yield_to_maturity(
    price: float,
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    frequency: int = 1,
    guess: float = 0.05
) -> float:
    """计算到期收益率（YTM）。
    
    通过牛顿迭代法求解：
        P = Σ[C/(1+y)^t] + F/(1+y)^N
    
    Parameters
    ----------
    price : float
        债券价格（全价）
    face_value : float
        面值
    coupon_rate : float
        票面利率（年化）
    years_to_maturity : float
        剩余期限（年）
    frequency : int
        付息频率
    guess : float
        初始猜测值
    
    Returns
    -------
    float
        到期收益率（年化）
    """
    def price_diff(ytm):
        return bond_price(face_value, coupon_rate, ytm, years_to_maturity, frequency) - price
    
    try:
        ytm = newton(price_diff, guess)
        return max(ytm, 0.0)  # YTM不能为负
    except Exception:
        # 如果牛顿法失败，使用二分法
        return _ytm_bisection(price, face_value, coupon_rate, years_to_maturity, frequency)


def _ytm_bisection(
    price: float,
    face_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    frequency: int = 1,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """二分法求解YTM。"""
    y_low, y_high = 0.0, 1.0
    
    for _ in range(max_iter):
        y_mid = (y_low + y_high) / 2
        p_mid = bond_price(face_value, coupon_rate, y_mid, years_to_maturity, frequency)
        
        if abs(p_mid - price) < tol:
            return y_mid
        
        if p_mid > price:
            y_low = y_mid
        else:
            y_high = y_mid
    
    return (y_low + y_high) / 2


# ============================================================
# 应计利息计算
# ============================================================
def accrued_interest(
    face_value: float,
    coupon_rate: float,
    days_since_last_coupon: int,
    coupon_interval_days: int = 365
) -> float:
    """计算应计利息。
    
    公式：
        AI = 面值 × 票面利率 × 距上次付息天数 / 付息间隔天数
    
    Parameters
    ----------
    face_value : float
        面值
    coupon_rate : float
        票面利率（年化）
    days_since_last_coupon : int
        距上次付息的天数
    coupon_interval_days : int
        付息间隔天数（年付息=365，半年付息=182）
    
    Returns
    -------
    float
        应计利息
    """
    return face_value * coupon_rate * days_since_last_coupon / coupon_interval_days


# ============================================================
# 债券收益率分解
# ============================================================
def decompose_bond_return(
    ytm_start: float,
    ytm_end: float,
    treasury_yield_change: float,
    credit_spread_change: float,
    modified_duration: float,
    holding_period_years: float
) -> dict:
    """分解单只债券收益率（Campisi模型核心）。
    
    公式：
        R = y × dt + (-MD) × dy_treasury + (-MD) × dy_credit
    
    Parameters
    ----------
    ytm_start : float
        期初到期收益率（年化）
    ytm_end : float
        期末到期收益率（年化）
    treasury_yield_change : float
        国债利率变化（年化）
    credit_spread_change : float
        信用利差变化（年化）
    modified_duration : float
        修正久期
    holding_period_years : float
        持有期（年）
    
    Returns
    -------
    dict
        {
            'total_return': 总收益率,
            'coupon_effect': 票息效应,
            'treasury_effect': 国债利率变化效应,
            'credit_effect': 信用利差变化效应,
        }
    """
    # 票息效应：y × dt
    coupon_effect = ytm_start * holding_period_years
    
    # 国债利率变化效应：(-MD) × dy_treasury
    treasury_effect = -modified_duration * treasury_yield_change
    
    # 信用利差变化效应：(-MD) × dy_credit
    credit_effect = -modified_duration * credit_spread_change
    
    # 总收益
    total_return = coupon_effect + treasury_effect + credit_effect
    
    return {
        'total_return': total_return,
        'coupon_effect': coupon_effect,
        'treasury_effect': treasury_effect,
        'credit_effect': credit_effect,
    }
