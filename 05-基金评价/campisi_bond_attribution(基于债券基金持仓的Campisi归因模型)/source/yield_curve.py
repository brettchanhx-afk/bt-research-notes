# -*- coding: utf-8 -*-
"""
收益率曲线处理模块

功能：
  - 收益率曲线插值
  - 期限结构建模
  - 即期利率与远期利率计算
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from typing import Optional, Callable


# ============================================================
# 收益率曲线插值
# ============================================================
class YieldCurve:
    """收益率曲线类。
    
    支持多种插值方法：
      - linear: 线性插值
      - cubic: 三次样条插值
      - hermite: Hermite插值
    """
    
    def __init__(self, terms: np.ndarray, yields: np.ndarray, method: str = 'cubic'):
        """初始化收益率曲线。
        
        Parameters
        ----------
        terms : np.ndarray
            期限数组（年）
        yields : np.ndarray
            收益率数组（%）
        method : str
            插值方法
        """
        self.terms = np.array(terms)
        self.yields = np.array(yields)
        self.method = method
        
        # 构建插值函数
        self._build_interpolator()
    
    def _build_interpolator(self):
        """构建插值函数。"""
        if self.method == 'linear':
            self._interp = interp1d(
                self.terms, self.yields, 
                kind='linear', fill_value='extrapolate'
            )
        elif self.method == 'cubic':
            self._interp = CubicSpline(self.terms, self.yields, extrapolate=True)
        else:
            # 默认使用线性
            self._interp = interp1d(
                self.terms, self.yields,
                kind='linear', fill_value='extrapolate'
            )
    
    def get_yield(self, term: float) -> float:
        """获取特定期限的收益率。
        
        Parameters
        ----------
        term : float
            期限（年）
        
        Returns
        -------
        float
            收益率（%）
        """
        return float(self._interp(term))
    
    def get_yields(self, terms: np.ndarray) -> np.ndarray:
        """批量获取收益率。"""
        return self._interp(terms)
    
    def get_forward_rate(self, term1: float, term2: float) -> float:
        """计算远期利率。
        
        公式：
            (1+r2)^t2 = (1+r1)^t1 × (1+f)^(t2-t1)
            f = [(1+r2)^t2 / (1+r1)^t1]^(1/(t2-t1)) - 1
        
        Parameters
        ----------
        term1, term2 : float
            期限1和期限2（term2 > term1）
        
        Returns
        -------
        float
            远期利率（年化）
        """
        if term2 <= term1:
            return 0.0
        
        r1 = self.get_yield(term1) / 100  # 转为小数
        r2 = self.get_yield(term2) / 100
        
        # 远期利率
        forward = ((1 + r2) ** term2 / (1 + r1) ** term1) ** (1 / (term2 - term1)) - 1
        
        return forward * 100  # 转回百分比
    
    def get_spot_rate(self, term: float) -> float:
        """计算即期利率（简化：假设收益率曲线即为即期利率曲线）。"""
        return self.get_yield(term)


# ============================================================
# 收益率曲线变化计算
# ============================================================
def calculate_yield_change(
    curve_start: YieldCurve,
    curve_end: YieldCurve,
    term: float
) -> float:
    """计算特定期限的收益率变化。
    
    Parameters
    ----------
    curve_start : YieldCurve
        期初收益率曲线
    curve_end : YieldCurve
        期末收益率曲线
    term : float
        期限（年）
    
    Returns
    -------
    float
        收益率变化（%）
    """
    y_start = curve_start.get_yield(term)
    y_end = curve_end.get_yield(term)
    
    return y_end - y_start


def calculate_parallel_shift(
    curve_start: YieldCurve,
    curve_end: YieldCurve
) -> float:
    """计算收益率曲线的平行移动。
    
    取各期限变化的平均值。
    
    Parameters
    ----------
    curve_start, curve_end : YieldCurve
        期初和期末收益率曲线
    
    Returns
    -------
    float
        平行移动量（%）
    """
    # 计算各期限变化
    changes = curve_end.yields - curve_start.yields
    
    return np.mean(changes)


# ============================================================
# 信用利差曲线
# ============================================================
class CreditSpreadCurve:
    """信用利差曲线类。
    
    信用利差 = 同评级债券收益率 - 无风险利率（国开债）
    """
    
    def __init__(
        self,
        risk_free_curve: YieldCurve,
        rating: str,
        spread_basis: float = None
    ):
        """初始化信用利差曲线。
        
        Parameters
        ----------
        risk_free_curve : YieldCurve
            无风险收益率曲线（国开债）
        rating : str
            信用评级
        spread_basis : float
            利差基点（bp），如不提供则使用默认值
        """
        self.risk_free_curve = risk_free_curve
        self.rating = rating
        self.spread_basis = spread_basis or self._get_default_spread(rating)
    
    def _get_default_spread(self, rating: str) -> float:
        """获取默认信用利差（基点）。"""
        spread_map = {
            'AAA': 50,
            'AA': 100,
            'A': 200,
            'BBB': 350,
            'BB': 500,
            'B': 700,
        }
        return spread_map.get(rating, 200)
    
    def get_yield(self, term: float) -> float:
        """获取同评级债券的到期收益率。
        
        Parameters
        ----------
        term : float
            期限（年）
        
        Returns
        -------
        float
            到期收益率（%）
        """
        risk_free_yield = self.risk_free_curve.get_yield(term)
        spread = self.spread_basis / 100  # bp转为%
        
        return risk_free_yield + spread
    
    def get_spread(self, term: float = None) -> float:
        """获取信用利差（bp）。"""
        return self.spread_basis


# ============================================================
# 收益率曲线构造工具
# ============================================================
def bootstrap_yield_curve(
    bond_prices: pd.DataFrame,
    face_values: np.ndarray,
    coupon_rates: np.ndarray,
    maturities: np.ndarray
) -> YieldCurve:
    """通过Bootstrap方法构造收益率曲线。
    
    Parameters
    ----------
    bond_prices : pd.DataFrame
        债券价格数据
    face_values : np.ndarray
        面值数组
    coupon_rates : np.ndarray
        票面利率数组
    maturities : np.ndarray
        到期期限数组
    
    Returns
    -------
    YieldCurve
        构造的收益率曲线
    """
    # 简化实现：假设已有YTM数据
    # 实际Bootstrap需要逐步求解
    
    terms = maturities
    yields = np.zeros(len(terms))
    
    # 这里简化处理，实际需要完整Bootstrap算法
    for i, (price, fv, cr, mat) in enumerate(zip(
        bond_prices['price'].values, face_values, coupon_rates, maturities
    )):
        # 使用简化公式估算YTM
        yields[i] = (fv * cr + (fv - price) / mat) / price * 100
    
    return YieldCurve(terms, yields, method='cubic')


def nelson_siegel_curve(
    terms: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float
) -> np.ndarray:
    """Nelson-Siegel收益率曲线模型。
    
    公式：
        y(t) = β0 + β1*(1-e^(-t/τ))/(t/τ) + β2*[(1-e^(-t/τ))/(t/τ) - e^(-t/τ)]
    
    Parameters
    ----------
    terms : np.ndarray
        期限数组
    beta0, beta1, beta2 : float
        模型参数
    tau : float
        衰减参数
    
    Returns
    -------
    np.ndarray
        收益率数组
    """
    t = terms
    exp_term = np.exp(-t / tau)
    
    # 避免除零
    t_tau = np.where(t > 0, t / tau, 1e-10)
    
    factor1 = (1 - exp_term) / t_tau
    factor2 = factor1 - exp_term
    
    return beta0 + beta1 * factor1 + beta2 * factor2
