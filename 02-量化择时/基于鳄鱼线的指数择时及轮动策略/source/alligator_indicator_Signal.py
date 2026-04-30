from typing import List, Tuple, Union, Generator

import numpy as np
import pandas as pd
from talib import MACD, SMA
from .toolkits import get_shift
from .utils import sliding_window

__all__ = [
    "calculate_alligator_indicator",
    "alignment_signal",
    "alligator_classify_rows",
    "get_alligator_signal",
    "calculate_ao",
    "check_continuation_up_or_down",
    "get_ao_indicator_signal",
    "check_classily_top_fractal",
    "check_classily_bottom_fractal",
    "get_fractal_classily",
    "get_fractal_signal",
    "macd_classify_cols",
    "get_macd_signal",
    "get_north_money_signal"
]


#                               鳄鱼线指标计算
def calculate_alligator_indicator(
    close_arr: Union[pd.Series, np.ndarray],
    periods: Tuple[int] = None,
    lag: Tuple[int] = None,
) -> np.ndarray:
   
    if periods is None:
        periods: Tuple[int] = (13, 8, 5)

    if lag is None:
        lag: Tuple[int] = (8, 5, 3)

    if isinstance(close_arr, pd.Series):
        close_arr: np.ndarray = close_arr.values

    max_size: int = max(periods)
    if close_arr.shape[0] < max_size:
        raise ValueError("输入的数据长度小于最大周期。")

    if not isinstance(periods, (list, tuple)):
        raise ValueError("输入的周期不是列表或元组。")

    close_arr: np.ndarray = close_arr.astype(np.float64)
    # 计算鳄鱼线
    alligator_arr: np.ndarray = np.array(
        [get_shift(SMA(close_arr, i), j) for i, j in zip(periods, lag)]
    ).T

    return alligator_arr


def alignment_signal(arr: np.ndarray, alignment_type="bullish") -> np.ndarray:
    
    if alignment_type == "bullish":
        # 多头排列：每列数值比右边的大
        is_aligned: np.ndarray = np.all(np.diff(arr, axis=1) > 0, axis=1)
    elif alignment_type == "bearish":
        # 空头排列：每列数值比右边的小
        is_aligned: np.ndarray = np.all(np.diff(arr, axis=1) < 0, axis=1)
    else:
        raise ValueError("alignment_type must be 'bullish' or 'bearish'")

    # 计算触发信号：今天形成排列且昨天不是
    signal: np.ndarray = np.zeros(len(is_aligned), dtype=bool)
    signal[1:] = is_aligned[1:] & ~is_aligned[:-1]

    return signal



def alligator_classify_rows(alligator_arr: np.ndarray) -> np.ndarray:
    
    # 数组下标[下颚线，牙齿线，上唇线]
    # 检查 x[0] < x[1] < x[2]
    condition1 = alignment_signal(alligator_arr, "bullish")
    # 检查 x[0] > x[1] > x[2]
    condition2 = alignment_signal(alligator_arr, "bearish")

    # 初始化结果数组，所有元素为 np.nan
    result: np.ndarray = np.full(alligator_arr.shape[0], np.nan, dtype=float)
    # 满足 condition1 的行标记为 1
    result[condition1] = 1
    # 满足 condition2 的行标记为 -1
    result[condition2] = -1

    return result


def get_alligator_signal(
    close_df: Union[pd.DataFrame, pd.Series],
    periods: Tuple[int] = None,
    lag: Tuple[int] = None,
    keep_pre_status: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    
    # 无信号（沉睡的鳄鱼）， 维持前一交易日的仓位
    # 故fillna(method='ffill')
    if isinstance(close_df, pd.DataFrame):

        signal: pd.DataFrame = close_df.apply(
            lambda x: alligator_classify_rows(
                calculate_alligator_indicator(x, periods, lag)
            ),
            raw=True,
        )

    elif isinstance(close_df, pd.Series):

        signal: pd.Series = pd.Series(
            alligator_classify_rows(
                calculate_alligator_indicator(close_df, periods, lag)
            ),
            index=close_df.index,
        )

    else:
        raise ValueError("输入的数据不是DataFrame或Series")

    if keep_pre_status:
        return signal.ffill().fillna(0)

    return signal


#                               AO指标计算

def calculate_ao(
    high_df: Union[pd.DataFrame, pd.Series],
    low_df: Union[pd.DataFrame, pd.Series],
    periods: Tuple[int] = (5, 34),
) -> pd.DataFrame:
    
    median_price: Union[pd.DataFrame, pd.Series] = (high_df - low_df) * 0.5

    return (
        median_price.rolling(periods[0]).mean()
        - median_price.rolling(periods[1]).mean()
    )


def check_continuation_up_or_down(arr: np.ndarray) -> int:
   
    if np.all(np.diff(arr) > 0):
        return 1
    elif np.all(np.diff(arr) < 0):
        return -1
    else:
        return np.nan


def get_ao_indicator_signal(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    window: int = 3,
    keep_pre_status: bool = True,
) -> pd.DataFrame:
   
    ao_indicator: pd.DataFrame = calculate_ao(high_df, low_df)

    signal: pd.DataFrame = ao_indicator.rolling(window).apply(
        check_continuation_up_or_down, raw=True
    )
    if keep_pre_status:
        return signal.ffill().fillna(0)
    return signal


#                                   分型指标计算

def check_classily_top_fractal(high_arr: np.ndarray, low_arr: np.ndarray) -> int:
   
    # 无论输入长度
    if high_arr.shape[0] != low_arr.shape[0]:
        raise ValueError("输入的数据长度不一致。")

    # 检查 high_t1 < high_t2 > high_t3 条件
    condition_high = (high_arr[-3, :] < high_arr[-2, :]) & (
        high_arr[-2, :] > high_arr[-1, :]
    )

    # 检查 low_t1 < low_t2 > low_ 条件
    condition_low = (low_arr[-3, :] < low_arr[-2, :]) & (
        low_arr[-2, :] > low_arr[-1, :]
    )

    return (condition_high & condition_low) * 1


def check_classily_bottom_fractal(high_arr: np.ndarray, low_arr: np.ndarray) -> int:
    
    if high_arr.shape[0] != low_arr.shape[0]:
        raise ValueError("输入的数据长度不一致。")

    # 检查 low_t1 > low_t2 < low_t3 条件
    condition_low = (low_arr[-3, :] > low_arr[-2, :]) & (
        low_arr[-2, :] < low_arr[-1, :]
    )
    # 检查 high_t1 > high_t2 < high_t3 条件
    condition_high = (high_arr[-3, :] > high_arr[-2, :]) & (
        high_arr[-2, :] < high_arr[-1, :]
    )

    # 同时满足上述两个条件
    return (condition_low & condition_high) * -1


def get_fractal_classily(
    high_df: pd.DataFrame, low_df: pd.DataFrame, window: int = 3
) -> pd.DataFrame:
   
    if not isinstance(high_df, pd.DataFrame) or not isinstance(low_df, pd.DataFrame):
        raise ValueError("输入的数据不是DataFrame")

    high_df, low_df = high_df.align(low_df)
    data: np.ndarray = np.stack((high_df.values, low_df.values), axis=2)
    datas: Generator = sliding_window(data, window)
    arr = np.array(
        [
            check_classily_top_fractal(arr[:, :, 0], arr[:, :, 1])
            + check_classily_bottom_fractal(arr[:, :, 0], arr[:, :, 1])
            for arr in datas
        ]
    )

    # 如果为1则为顶分型，如果为-1则为底分型
    return pd.DataFrame(arr, index=high_df.index[2:], columns=high_df.columns)


def get_fractal_signal(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    keep_pre_status: bool = True,
) -> pd.DataFrame:
    

    fractal_df: pd.DataFrame = get_fractal_classily(high_df, low_df)

    up_ser: pd.DataFrame = (close_df > close_df.shift(3)) * 1
    down_ser: pd.DataFrame = (close_df < close_df.shift(3)) * -1

    signal: pd.DataFrame = ((up_ser + fractal_df.shift(1)) == 2) * 1 + (
        (down_ser + fractal_df.shift(1)) == -2
    ) * -1
    if keep_pre_status:
        return signal.ffill().fillna(0)
    return signal


def evaluate_signals(row) -> int:
    
    # 检查 Alligator 等于 1 且 AO 或 Fractal 至少有一个等于 1 的情况
    if row[0] == 1 and (row[1] == 1 or row[2] == 1):
        return 1
    # 检查 Alligator、AO 或 Fractal 中任意一个等于 -1 的情况
    elif row[0] == -1 or row[1] == -1 or row[2] == -1:
        return -1
    # 如果以上条件都不满足，返回 0
    else:
        return 0


########                        MACD指标计算

def macd_classify_cols(
    dif: Union[pd.Series, np.ndarray],
    dea: Union[pd.Series, np.ndarray],
    hist: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
   
    # DIF（快线）上穿 DEA（慢线），同时能量柱由绿转红
    DIF_cross_DEA: Union[pd.Series, np.ndarray] = (dif > dea) & (
        get_shift(dif, 1) < get_shift(dea, 1)
    )
    MACD_green_to_red: Union[pd.Series, np.ndarray] = (hist > 0) & (
        get_shift(hist, 1) < 0
    )
    bullish_zero_zone: Union[pd.Series, np.ndarray] = (
        (dif >= 0) & (dea >= 0) & (hist >= 0)
    )

    # 看多
    bullish: Union[pd.Series, np.ndarray] = (
        DIF_cross_DEA & MACD_green_to_red & bullish_zero_zone
    )

    # DIF（快线）下穿 DEA（慢线），同时能量柱由红转绿
    DEA_cross_DIF: Union[pd.Series, np.ndarray] = (dif < dea) & (
        get_shift(dif, 1) > get_shift(dea, 1)
    )
    MACD_red_to_green: Union[pd.Series, np.ndarray] = (hist < 0) & (
        get_shift(hist, 1) > 0
    )
    bearish_zero_zone: Union[pd.Series, np.ndarray] = (dif < 0) & (dea < 0) & (hist < 0)

    # 看空
    bearish: Union[pd.Series, np.ndarray] = (
        DEA_cross_DIF & MACD_red_to_green & bearish_zero_zone
    )

    return bullish.astype(int) - bearish.astype(int)


def get_macd_signal(
    close_df: pd.DataFrame, keep_pre_status: bool = True
) -> pd.DataFrame:
    
    signal: pd.DataFrame = close_df.apply(
        lambda ser: macd_classify_cols(*MACD(ser)), raw=True
    )
    if keep_pre_status:
        return signal.replace(0, np.nan).ffill().fillna(0)
    return signal

########              北向指标计算

def get_north_money_signal(north_money: pd.DataFrame) -> pd.Series:
   
    bottom: pd.Series = north_money["north_money"].rolling(60).quantile(0.2)
    top: pd.Series = north_money["north_money"].rolling(60).quantile(0.8)

    north_signal: pd.Series = pd.Series(
        [np.nan] * len(north_money), index=north_money.index
    )

    north_signal[north_money["north_money"] > top] = 1
    north_signal[north_money["north_money"] < bottom] = -1
    return north_signal.ffill().fillna(0)