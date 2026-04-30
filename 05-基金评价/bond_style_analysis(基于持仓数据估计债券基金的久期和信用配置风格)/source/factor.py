# -*- coding: utf-8 -*-
"""
factor.py - 债券基金风格因子计算模块

严格按照华泰证券研报《基于持仓数据估计债券基金的久期和信用配置风格》(2020-08-21)
实现三大核心因子:
1. 修正麦考利久期 (Modified Duration)
2. 加权平均久期 (风格久期)
3. 加权平均信用评分 (风格信用)

公式来源:
- Macaulay Duration: T * t * w_t,  w_t = CF_t / (P * (1+y)^t)
- Modified Duration: -∂P/P / ∂y = Mac_Duration / (1+y)
- Duration Style: D = Σ(W_i * D_i) / ΣW_i
- Credit Style: C = Σ(W_i * C_i) / ΣW_i

作者: QClaw Agent | 版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


# =============================================================================
# 债券久期计算
# =============================================================================

def calc_macaulay_duration(
    maturities: np.ndarray,
    cash_flows: np.ndarray,
    ytm: float,
) -> float:
    """
    计算麦考利久期 (Macaulay Duration)

    Parameters
    ----------
    maturities : np.ndarray
        各期现金流到期时间 (年)，形状 (n_periods,)
    cash_flows : np.ndarray
        各期现金流金额，形状 (n_periods,)
    ytm : float
        到期收益率 (债券买卖成交双方的应计利息结算)

    Returns
    -------
    float
        麦考利久期 (年)
    """
    if len(maturities) == 0 or len(cash_flows) == 0:
        return 0.0

    maturities = np.asarray(maturities, dtype=float)
    cash_flows = np.asarray(cash_flows, dtype=float)
    ytm = float(ytm)

    # 价格 = Σ CF_t / (1+y)^t
    discount_factors = (1.0 + ytm) ** maturities
    prices = cash_flows / discount_factors
    price_total = np.sum(prices)

    if price_total <= 0 or np.isnan(price_total):
        return 0.0

    # 麦考利久期 = Σ(t * CF_t / (1+y)^t) / P
    weighted_sum = np.sum(maturities * prices)
    return float(weighted_sum / price_total)


def calc_modified_duration(
    maturities: np.ndarray,
    cash_flows: np.ndarray,
    ytm: float,
) -> float:
    """
    计算修正久期 (Modified Duration)

    Modified Duration = Macaulay Duration / (1 + ytm)

    Parameters
    ----------
    maturities : np.ndarray
        各期现金流到期时间
    cash_flows : np.ndarray
        各期现金流金额
    ytm : float
        到期收益率

    Returns
    -------
    float
        修正久期 (年)
    """
    mac_dur = calc_macaulay_duration(maturities, cash_flows, ytm)
    if ytm <= -1.0:  # 防御性检查
        return 0.0
    return mac_dur / (1.0 + ytm)


def calc_bond_duration_simple(
    maturity: float,
    coupon_rate: float = 0.0,
    ytm: float = None,
    freq: int = 2,
) -> float:
    """
    简化的债券久期估算 (适合无完整现金流数据时使用)

    近似公式:
    - 零息债券: Duration = Maturity
    - 附息债券: Duration ≈ Maturity * (1 - (1 + coupon_rate) / ((1 + ytm) * freq)) / freq

    Parameters
    ----------
    maturity : float
        债券剩余期限 (年)
    coupon_rate : float
        年票息率 (如 0.03 表示 3%)
    ytm : float, optional
        到期收益率，默认使用票息率
    freq : int
        年付息次数，默认 2 (半年付)

    Returns
    -------
    float
        修正久期估算值
    """
    if maturity <= 0:
        return 0.0

    ytm = float(ytm) if ytm is not None else float(coupon_rate)
    ytm = max(ytm, 0.0)  # ytm 不能为负

    if coupon_rate == 0.0:
        # 零息债券: Modified Duration = Maturity / (1 + ytm)
        return maturity / (1.0 + ytm)

    # 附息债券修正久期近似
    freq = float(freq)
    mac_dur_approx = maturity * (1.0 - 1.0 / (1.0 + ytm / freq)) + 1.0 / freq
    mod_dur_approx = mac_dur_approx / (1.0 + ytm / freq)

    return float(mod_dur_approx)


# =============================================================================
# 信用评分计算
# =============================================================================

def calc_credit_score(rating: str) -> float:
    """
    根据评级字符串计算信用评分

    评分规则 (来源: 中国人民银行2006年规范):
    - AAA+: 17.0, AAA: 16.5, AA+: 15.5, AA: 15.0, AA-: 14.5
    - A+: 13.5,  A: 13.0,  A-: 12.5
    - BBB+: 11.5, BBB: 11.0, BBB-: 10.5
    - BB+: 9.5,  BB: 9.0,  BB-: 8.5
    - B+: 7.5,   B: 7.0,   B-: 6.5
    - CCC+: 5.5, CCC: 5.0, CCC-: 4.5
    - CC: 3.0,  C: 1.0,  D: 0.0
    """
    from .data_loader import parse_rating
    return parse_rating(rating)


# =============================================================================
# 核心风格因子计算
# =============================================================================

class BondStyleFactor:
    """
    债券基金风格因子计算器

    核心方法 (研报定义):
    1. Duration Style (久期风格) = Σ(W_i * D_i) / ΣW_i
    2. Credit Style (信用风格)   = Σ(W_i * C_i) / ΣW_i

    其中 W_i 为债券 i 的持仓市值权重 (或占净值比)
          D_i 为债券 i 的修正久期
          C_i 为债券 i 的信用评分
    """

    # 久期风格分类阈值
    DURATION_BINS = {
        "short": (0, 3.5),      # 短期: D < 3.5
        "mid": (3.5, 6.0),      # 中期: 3.5 <= D < 6.0
        "long": (6.0, 100),     # 长期: D >= 6.0
    }

    # 信用风格分类阈值
    CREDIT_BINS = {
        "high": (14.0, 17.0),    # 高等级: 评分 >= 14 (AAA~AA)
        "medium": (11.0, 14.0),  # 中等级: 11 <= 评分 < 14 (A~BBB)
        "low": (0.0, 11.0),      # 低等级: 评分 < 11 (BB及以下)
    }

    def __init__(self):
        self.result = {}

    def calc_duration_style(
        self,
        holdings: pd.DataFrame,
        durations: dict = None,
        weight_col: str = "market_value",
    ) -> Tuple[float, str]:
        """
        计算久期风格

        Parameters
        ----------
        holdings : pd.DataFrame
            持仓数据，必须包含 weight_col 列
        durations : dict, optional
            债券代码 -> 修正久期 映射字典
            如 {'019547': 4.5, 'CB114': 6.2}
        weight_col : str
            权重列名，默认 market_value (持仓市值，万元)

        Returns
        -------
        Tuple[float, str]
            (加权平均久期, 风格标签)
            风格标签: 'short'(短期) / 'mid'(中期) / 'long'(长期)
        """
        if durations is None:
            durations = {}

        if holdings.empty or weight_col not in holdings.columns:
            return 0.0, "unknown"

        holdings = holdings.copy()
        holdings["duration"] = holdings["bond_code"].map(
            lambda x: durations.get(x, durations.get(str(x), np.nan))
        )
        holdings = holdings.dropna(subset=["duration", weight_col])

        if holdings.empty:
            return 0.0, "unknown"

        # 加权平均久期
        weights = holdings[weight_col].values
        d_values = holdings["duration"].values

        total_weight = np.sum(weights)
        if total_weight <= 0:
            return 0.0, "unknown"

        weighted_duration = np.sum(weights * d_values) / total_weight

        # 分类
        style = self._classify_duration(weighted_duration)
        self.result["duration_style"] = weighted_duration
        self.result["duration_style_label"] = style

        return float(weighted_duration), style

    def calc_credit_style(
        self,
        holdings: pd.DataFrame,
        ratings: dict = None,
        weight_col: str = "market_value",
    ) -> Tuple[float, str]:
        """
        计算信用风格

        Parameters
        ----------
        holdings : pd.DataFrame
            持仓数据
        ratings : dict, optional
            债券代码 -> 评级字符串 映射字典
        weight_col : str
            权重列名

        Returns
        -------
        Tuple[float, str]
            (加权平均信用评分, 风格标签)
            风格标签: 'high'(高等级) / 'medium'(中等级) / 'low'(低等级)
        """
        if ratings is None:
            ratings = {}

        if holdings.empty or weight_col not in holdings.columns:
            return 0.0, "unknown"

        holdings = holdings.copy()
        holdings["credit_score"] = holdings["bond_code"].map(
            lambda x: ratings.get(x, ratings.get(str(x), 0.0))
        )
        # 如果传入的是字符串评级，转为分数
        if holdings["credit_score"].dtype == object:
            def _map_rating(x):
                r = ratings.get(x, ratings.get(str(x), ""))
                return calc_credit_score(str(r))
            holdings["credit_score"] = holdings["bond_code"].map(_map_rating)

        holdings = holdings.dropna(subset=["credit_score", weight_col])

        if holdings.empty:
            return 0.0, "unknown"

        weights = holdings[weight_col].values
        c_values = holdings["credit_score"].values.astype(float)

        total_weight = np.sum(weights)
        if total_weight <= 0:
            return 0.0, "unknown"

        weighted_credit = np.sum(weights * c_values) / total_weight

        style = self._classify_credit(weighted_credit)
        self.result["credit_style"] = weighted_credit
        self.result["credit_style_label"] = style

        return float(weighted_credit), style

    def calc_combined_style(
        self,
        holdings: pd.DataFrame,
        durations: dict = None,
        ratings: dict = None,
        weight_col: str = "market_value",
    ) -> dict:
        """
        计算久期风格 + 信用风格组合

        Returns
        -------
        dict
            包含: duration_style, duration_label, credit_style, credit_label,
                style_box (久期-信用二维风格箱标签)
        """
        dur_val, dur_label = self.calc_duration_style(holdings, durations, weight_col)
        cred_val, cred_label = self.calc_credit_style(holdings, ratings, weight_col)

        # 研报定义的 3x3 风格箱
        style_box = f"{cred_label}_{dur_label}"

        self.result["duration_style"] = dur_val
        self.result["duration_style_label"] = dur_label
        self.result["credit_style"] = cred_val
        self.result["credit_style_label"] = cred_label
        self.result["style_box"] = style_box

        return self.result.copy()

    @staticmethod
    def _classify_duration(d: float) -> str:
        """将久期数值分类"""
        if d < 3.5:
            return "short"
        elif d < 6.0:
            return "mid"
        else:
            return "long"

    @staticmethod
    def _classify_credit(c: float) -> str:
        """将信用评分分类"""
        if c >= 14.0:
            return "high"
        elif c >= 11.0:
            return "medium"
        else:
            return "low"

    @staticmethod
    def describe_style_box() -> pd.DataFrame:
        """
        返回研报定义的 3x3 风格箱说明

        Returns
        -------
        pd.DataFrame
            风格箱表格
        """
        rows = []
        for credit in ["high", "medium", "low"]:
            for dur in ["long", "mid", "short"]:
                label = f"{credit}_{dur}"
                rows.append({
                    "风格箱标签": label,
                    "信用等级": {"high": "高(AAA~AA)", "medium": "中(A~BBB)", "low": "低(BB及以下)"}[credit],
                    "久期": {"short": "短期(<3.5年)", "mid": "中期(3.5~6年)", "long": "长期(>6年)"}[dur],
                })
        return pd.DataFrame(rows)


# =============================================================================
# 滚动久期跟踪
# =============================================================================

def track_duration_series(
    nav_df: pd.DataFrame,
    holdings_series: List[pd.DataFrame],
    durations_series: List[dict],
    freq: int = 60,
) -> pd.DataFrame:
    """
    滚动跟踪基金久期风格时间序列

    Parameters
    ----------
    nav_df : pd.DataFrame
        净值序列，包含 date, nav 列
    holdings_series : List[pd.DataFrame]
        各报告期持仓列表
    durations_series : List[dict]
        各报告期久期字典列表
    freq : int
        交易日频率，默认 60 个交易日 (~1季度)

    Returns
    -------
    pd.DataFrame
        时间序列，包含 date, duration_style, credit_style, nav
    """
    factor = BondStyleFactor()
    records = []

    for i, (period_holdings, period_durations) in enumerate(
        zip(holdings_series, durations_series)
    ):
        _, dur_label = factor.calc_duration_style(period_holdings, period_durations)

        record = {
            "period_idx": i,
            "n_bonds": len(period_holdings),
            "duration_style": factor.result.get("duration_style", 0.0),
            "duration_label": dur_label,
        }
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# 工具函数
# =============================================================================

def estimate_duration_from_maturity(
    maturity: float,
    coupon_rate: float = 0.0,
    ytm: float = None,
) -> float:
    """
    工具函数: 根据剩余期限估算修正久期

    简化公式:
    - 零息债券: Duration = Maturity / (1 + ytm)
    - 附息债券: Duration ≈ Maturity * (coupon_rate + 1) / (ytm + coupon_rate + 1)

    Parameters
    ----------
    maturity : float
        债券剩余期限 (年)
    coupon_rate : float
        年票息率
    ytm : float, optional
        到期收益率

    Returns
    -------
    float
        估算修正久期
    """
    if maturity <= 0:
        return 0.0
    if ytm is None:
        ytm = coupon_rate
    ytm = max(float(ytm), 0.0)

    if coupon_rate == 0.0:
        return maturity / (1.0 + ytm)

    # 近似公式: Duration ≈ Maturity * (1 + coupon/ytm) / (1 + ytm)
    # 实际采用更精确的久期近似
    mac_dur = maturity * (1.0 + coupon_rate / ytm) / (1.0 + coupon_rate / ytm + 0.5)
    return mac_dur / (1.0 + ytm)
