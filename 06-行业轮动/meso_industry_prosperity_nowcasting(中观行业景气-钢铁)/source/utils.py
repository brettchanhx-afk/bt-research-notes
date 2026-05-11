"""
工具函数模块
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series: pd.Series, significance_level: float = 0.10) -> Tuple[bool, float]:
    """
    使用ADF检验检查时间序列的平稳性

    Parameters:
    -----------
    series : pd.Series
        待检验的时间序列
    significance_level : float
        显著性水平，默认0.10表示10%

    Returns:
    --------
    Tuple[bool, float]
        (是否平稳, p值)
    """
    if len(series) < 5:
        return False, 1.0

    try:
        result = adfuller(series.dropna(), maxlag=4, regression='c')
        p_value = result[1]
        return p_value < significance_level, p_value
    except:
        return False, 1.0


def standardize_series(series: pd.Series) -> pd.Series:
    """
    标准化时间序列（Z-score标准化）

    Parameters:
    -----------
    series : pd.Series
        待标准化的序列

    Returns:
    --------
    pd.Series
        标准化后的序列
    """
    mean = series.mean()
    std = series.std()
    if std < 1e-10:
        return series - mean
    return (series - mean) / std


def calculate_indicator_explanatory(indicator: pd.Series, factor: pd.Series) -> float:
    """
    计算指标对隐含因子的解释度（R²）

    Parameters:
    -----------
    indicator : pd.Series
        指标序列
    factor : pd.Series
        隐含因子序列

    Returns:
    --------
    float
        解释度（R²值）
    """
    valid_idx = indicator.notna() & factor.notna()
    if valid_idx.sum() < 10:
        return 0.0

    ind_vals = indicator[valid_idx]
    fac_vals = factor[valid_idx]

    if fac_vals.std() < 1e-10 or ind_vals.std() < 1e-10:
        return 0.0

    correlation = np.corrcoef(ind_vals, fac_vals)[0, 1]
    return correlation ** 2


def detect_data_gaps(series: pd.Series) -> Tuple[int, float]:
    """
    检测数据缺失情况

    Parameters:
    -----------
    series : pd.Series
        时间序列

    Returns:
    --------
    Tuple[int, float]
        (连续缺失数, 缺失比例)
    """
    total_length = len(series)
    missing_count = series.isna().sum()
    missing_ratio = missing_count / total_length if total_length > 0 else 0

    return missing_count, missing_ratio


def fill_missing_with_interpolation(series: pd.Series, max_consecutive_missing: int = 3) -> pd.Series:
    """
    使用插值法填充缺失值

    Parameters:
    -----------
    series : pd.Series
        待填充的序列
    max_consecutive_missing : int
        最大连续缺失可填充数

    Returns:
    --------
    pd.Series
        填充后的序列
    """
    result = series.copy()

    for _ in range(max_consecutive_missing):
        missing_mask = result.isna()
        if not missing_mask.any():
            break

        result = result.interpolate(method='linear', limit=max_consecutive_missing)

    return result


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series,
                                   window: int = 12) -> pd.Series:
    """
    计算滚动相关系数

    Parameters:
    -----------
    series1 : pd.Series
        第一个序列
    series2 : pd.Series
        第二个序列
    window : int
        滚动窗口大小

    Returns:
    --------
    pd.Series
        滚动相关系数
    """
    combined = pd.DataFrame({'s1': series1, 's2': series2})
    rolling_corr = combined['s1'].rolling(window).corr(combined['s2'])
    return rolling_corr


def calculate_direction_accuracy(predicted: pd.Series, actual: pd.Series) -> float:
    """
    计算方向预测准确率

    Parameters:
    -----------
    predicted : pd.Series
        预测方向（1表示上升，-1表示下降）
    actual : pd.Series
        实际方向

    Returns:
    --------
    float
        准确率
    """
    valid_idx = predicted.notna() & actual.notna() & (predicted != 0) & (actual != 0)
    if valid_idx.sum() == 0:
        return 0.0

    correct = (predicted[valid_idx] == actual[valid_idx]).sum()
    total = valid_idx.sum()
    return correct / total if total > 0 else 0.0


def get_date_range(start_date: str, end_date: str, freq: str = 'M') -> pd.DatetimeIndex:
    """
    获取日期范围

    Parameters:
    -----------
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD
    freq : str
        频率，M表示月度

    Returns:
    --------
    pd.DatetimeIndex
        日期索引
    """
    start = pd.to_datetime(start_date, format='%Y%m%d')
    end = pd.to_datetime(end_date, format='%Y%m%d')
    return pd.date_range(start=start, end=end, freq=freq)
