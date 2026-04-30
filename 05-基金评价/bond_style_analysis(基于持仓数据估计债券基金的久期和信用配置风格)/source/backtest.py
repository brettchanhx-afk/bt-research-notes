# -*- coding: utf-8 -*-
"""
backtest.py - 债券基金风格分析回测模块

基于持仓数据估计债券基金的久期和信用配置风格
支持:
- 单期风格分析
- 多期滚动风格跟踪
- 与业绩基准对比
- 风格稳定性分析 (SDS 漂移指标)

作者: QClaw Agent | 版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


# =============================================================================
# 绩效指标计算
# =============================================================================

def calc_performance_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """
    计算基金绩效指标

    Parameters
    ----------
    returns : pd.Series
        日收益率序列
    periods_per_year : int
        年化交易日数，默认 252

    Returns
    -------
    dict
        包含: annual_return(年化收益率), annual_vol(年化波动率),
              sharpe(夏普比率), max_drawdown(最大回撤),
              win_rate(胜率), calmar(卡尔马比率)
    """
    if returns.empty or len(returns) < 2:
        return _empty_metrics()

    returns = returns.dropna()
    if len(returns) < 2:
        return _empty_metrics()

    # 年化收益率
    cumret = (1.0 + returns).prod() - 1.0
    n_years = len(returns) / periods_per_year
    annual_return = (1.0 + cumret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(periods_per_year)

    # 夏普比率 (无风险利率=0)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    # 最大回撤
    nav = (1.0 + returns).cumprod()
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = (returns > 0).sum() / len(returns)

    # 卡尔马比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "calmar": float(calmar),
        "total_return": float(cumret),
        "n_days": int(len(returns)),
    }


def _empty_metrics() -> dict:
    return {
        "annual_return": 0.0,
        "annual_vol": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "calmar": 0.0,
        "total_return": 0.0,
        "n_days": 0,
    }


# =============================================================================
# 风格漂移指标 (SDS - Style Drift Score)
# =============================================================================

def calc_sds(
    exposure_series: pd.DataFrame,
    lookback: int = 60,
) -> float:
    """
    计算风格漂移指标 (Style Drift Score, Idzorek 2006)

    SDS = sqrt( Σ(var(β_i - β_i_sub)) ) / Σmean(β_i)

    即各风格暴露系数在子区间内的方差均值，再开方除以平均暴露

    Parameters
    ----------
    exposure_series : pd.DataFrame
        风格暴露系数时间序列，列为各风格指数，索引为日期
    lookback : int
        子区间窗口，默认 60 个交易日

    Returns
    -------
    float
        SDS 值，越大表示风格越不稳定
    """
    if len(exposure_series) < lookback * 2:
        return 0.0

    n_sub = len(exposure_series) // lookback
    if n_sub < 2:
        return 0.0

    var_list = []
    mean_list = []

    for col in exposure_series.columns:
        col_vals = exposure_series[col].values
        sub_means = [
            np.mean(col_vals[i * lookback: (i + 1) * lookback])
            for i in range(n_sub)
        ]
        var_list.append(np.var(sub_means))
        mean_list.append(np.mean(col_vals))

    sds = np.sqrt(np.sum(var_list)) / np.sum(mean_list) if np.sum(mean_list) > 0 else 0.0
    return float(sds)


# =============================================================================
# BondBacktestEngine - 回测引擎
# =============================================================================

class BondBacktestEngine:
    """
    债券基金风格回测引擎

    核心功能:
    1. 单期持仓风格分析
    2. 多期滚动风格跟踪
    3. 风格时序与业绩对比
    4. 风格稳定性检测
    """

    def __init__(
        self,
        loader=None,
        factor_engine=None,
    ):
        self.loader = loader
        self.factor = factor_engine
        self.results = {}

    def analyze_single_period(
        self,
        fund_code: str,
        holdings: pd.DataFrame = None,
        durations: dict = None,
        ratings: dict = None,
        weight_col: str = "market_value",
    ) -> dict:
        """
        分析单期持仓风格

        Returns
        -------
        dict
            风格分析结果
        """
        from .factor import BondStyleFactor

        if self.factor is None:
            self.factor = BondStyleFactor()

        if holdings is None and self.loader is not None:
            holdings = self.loader.get_fund_holdings(fund_code)

        if holdings is None or holdings.empty:
            return {"error": "No holdings data available"}

        # 久期风格 + 信用风格
        result = self.factor.calc_combined_style(
            holdings, durations, ratings, weight_col
        )

        # 持仓概况
        result["n_holdings"] = len(holdings)
        result["top_weight"] = float(holdings[weight_col].iloc[0]) if weight_col in holdings.columns else 0.0
        result["total_weight"] = float(holdings[weight_col].sum()) if weight_col in holdings.columns else 0.0

        return result

    def analyze_multi_period(
        self,
        fund_code: str,
        n_periods: int = 4,
        freq: int = 60,
    ) -> pd.DataFrame:
        """
        多期滚动风格分析

        Parameters
        ----------
        fund_code : str
            基金代码
        n_periods : int
            分析期数
        freq : int
            每期交易日数

        Returns
        -------
        pd.DataFrame
            多期风格分析结果
        """
        records = []

        for i in range(n_periods):
            period_label = f"Q{4-i}"  # 最近4期
            try:
                holdings = self.loader.get_fund_holdings(fund_code)
                if holdings.empty:
                    continue

                result = self.analyze_single_period(fund_code, holdings)
                result["period"] = period_label
                records.append(result)
            except Exception as e:
                print(f"[WARN] Period {period_label} analysis failed: {e}")

        return pd.DataFrame(records)

    def style_attribution(
        self,
        nav_df: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> dict:
        """
        风格归因分析

        对比基金收益率与风格指数收益率

        Parameters
        ----------
        nav_df : pd.DataFrame
            净值序列
        benchmark_returns : pd.Series
            基准收益率序列 (债券指数)

        Returns
        -------
        dict
            归因结果
        """
        if nav_df is None or nav_df.empty:
            return {}

        returns = nav_df.get("daily_return", pd.Series())
        if returns.empty:
            return {}

        # 对齐日期
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 30:
            return {}

        fund_ret = returns.loc[common_idx]
        bench_ret = benchmark_returns.loc[common_idx]

        # 计算追踪误差
        active_ret = fund_ret - bench_ret
        tracking_error = active_ret.std() * np.sqrt(252)
        information_ratio = (
            active_ret.mean() / active_ret.std() * np.sqrt(252)
            if active_ret.std() > 0
            else 0.0
        )

        return {
            "tracking_error": float(tracking_error),
            "information_ratio": float(information_ratio),
            "active_return": float(active_ret.mean() * 252),
            "n_common_days": int(len(common_idx)),
        }
