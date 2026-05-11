"""
数据处理工具函数
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


def normalize_factor(factor_values: pd.Series) -> pd.Series:
    """
    因子归一化处理 (Z-score)

    Args:
        factor_values: 因子值

    Returns:
        归一化后的因子值
    """
    mean = factor_values.mean()
    std = factor_values.std()
    if std == 0:
        return pd.Series(0, index=factor_values.index)
    return (factor_values - mean) / std


def rank_factor(factor_values: pd.Series) -> pd.Series:
    """
    因子排序打分 (0-1)

    Args:
        factor_values: 因子值

    Returns:
        排序打分后的因子值
    """
    return factor_values.rank(pct=True)


def calculate_yoy_change(
    df: pd.DataFrame, col: str, periods: int = 52
) -> pd.Series:
    """
    计算同比变化

    Args:
        df: 数据框
        col: 列名
        periods: 周期数 (周频为52周)

    Returns:
        同比变化值
    """
    if len(df) < periods:
        return pd.Series(np.nan, index=df.index)
    return df[col].values - df[col].shift(periods).values


def calculate_qoq_change(
    df: pd.DataFrame, col: str, periods: int = 4
) -> pd.Series:
    """
    计算环比变化

    Args:
        df: 数据框
        col: 列名
        periods: 周期数 (月频为4周)

    Returns:
        环比变化值
    """
    if len(df) < periods:
        return pd.Series(np.nan, index=df.index)
    return df[col].values - df[col].shift(periods).values


def winsorize_factor(
    factor_values: pd.Series, lower: float = 0.02, upper: float = 0.98
) -> pd.Series:
    """
    因子去极值 (分位数法)

    Args:
        factor_values: 因子值
        lower: 下界百分位
        upper: 上界百分位

    Returns:
        去极值后的因子值
    """
    lower_bound = factor_values.quantile(lower)
    upper_bound = factor_values.quantile(upper)
    return factor_values.clip(lower_bound, upper_bound)


def fill_missing_data(
    df: pd.DataFrame, method: str = "ffill"
) -> pd.DataFrame:
    """
    填充缺失数据

    Args:
        df: 数据框
        method: 填充方法 (ffill, bfill, mean)

    Returns:
        填充后的数据框
    """
    if method == "ffill":
        return df.fillna(method="ffill")
    elif method == "bfill":
        return df.fillna(method="bfill")
    elif method == "mean":
        return df.fillna(df.mean())
    else:
        raise ValueError(f"Unsupported fill method: {method}")


def aggregate_by_institution(
    df: pd.DataFrame,
    institution_col: str,
    value_col: str,
    agg_method: str = "sum",
) -> pd.DataFrame:
    """
    按机构类型聚合数据

    Args:
        df: 数据框
        institution_col: 机构列名
        value_col: 值列名
        agg_method: 聚合方法 (sum, mean, weight)

    Returns:
        聚合后的数据框
    """
    if agg_method == "sum":
        return df.groupby([df.index, institution_col])[value_col].sum().unstack()
    elif agg_method == "mean":
        return df.groupby([df.index, institution_col])[value_col].mean().unstack()
    else:
        raise ValueError(f"Unsupported aggregation method: {agg_method}")


def calculate_historical_percentile(
    series: pd.Series, window: int = 60
) -> pd.Series:
    """
    计算历史百分位

    Args:
        series: 数据序列
        window: 窗口大小

    Returns:
        历史百分位序列
    """
    rolling_percentile = series.rolling(window=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    return rolling_percentile


def calculate_correlation_with_lag(
    series1: pd.Series,
    series2: pd.Series,
    lag: int = 0,
) -> float:
    """
    计算带有滞后的相关性

    Args:
        series1: 序列1
        series2: 序列2
        lag: 滞后阶数

    Returns:
        相关系数
    """
    if lag > 0:
        series2 = series2.shift(lag)
    valid_mask = ~(series1.isna() | series2.isna())
    return series1[valid_mask].corr(series2[valid_mask])


def get_portfolio_returns(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    weight_method: str = "equal",
) -> pd.Series:
    """
    计算组合收益

    Args:
        positions: 持仓数据 (行: 日期, 列: 资产)
        returns: 收益数据 (行: 日期, 列: 资产)
        weight_method: 权重方法 (equal, value)

    Returns:
        组合收益序列
    """
    if weight_method == "equal":
        weights = positions.div(positions.sum(axis=1), axis=0).fillna(0)
    else:
        weights = positions.div(positions.sum(axis=1), axis=0).fillna(0)

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    return portfolio_returns


def calculate_cumulative_return(returns: pd.Series) -> pd.Series:
    """
    计算累计收益

    Args:
        returns: 收益序列

    Returns:
        累计收益序列
    """
    return (1 + returns).cumprod() - 1


def calculate_excess_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """
    计算超额收益

    Args:
        portfolio_returns: 组合收益
        benchmark_returns: 基准收益

    Returns:
        超额收益序列
    """
    return portfolio_returns - benchmark_returns


def calculate_annualized_return(
    returns: pd.Series, periods_per_year: int = 52
) -> float:
    """
    计算年化收益

    Args:
        returns: 收益序列
        periods_per_year: 年化周期数

    Returns:
        年化收益
    """
    cumulative_return = (1 + returns).prod()
    n_periods = len(returns)
    years = n_periods / periods_per_year
    if years == 0:
        return 0.0
    return cumulative_return ** (1 / years) - 1


def calculate_annualized_volatility(
    returns: pd.Series, periods_per_year: int = 52
) -> float:
    """
    计算年化波动率

    Args:
        returns: 收益序列
        periods_per_year: 年化周期数

    Returns:
        年化波动率
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 52,
) -> float:
    """
    计算夏普比率

    Args:
        returns: 收益序列
        risk_free_rate: 无风险利率
        periods_per_year: 年化周期数

    Returns:
        夏普比率
    """
    ann_return = calculate_annualized_return(returns, periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    计算最大回撤

    Args:
        cumulative_returns: 累计收益序列

    Returns:
        最大回撤
    """
    wealth_index = 1 + cumulative_returns
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()


def calculate_win_rate(returns: pd.Series) -> float:
    """
    计算调仓胜率

    Args:
        returns: 收益序列

    Returns:
        胜率
    """
    return (returns > 0).sum() / len(returns)
