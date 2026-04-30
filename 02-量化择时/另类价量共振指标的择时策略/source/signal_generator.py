from collections import namedtuple
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import talib

from .data_loader import query_stock_index_classify, query_sw_classify

DICT = {'sw': query_sw_classify(), 'index': query_stock_index_classify()}



def HMA(price: pd.Series, window: int):
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    return talib.WMA(
        2 * talib.WMA(price, int(window * 0.5)) - talib.WMA(price, window),
        int(np.sqrt(window)))


def calc_volume_momentum(close: pd.Series,
                         volume: pd.Series,
                         bma_window: int = 50,
                         ama_window: int = 100,
                         n_window: int = 3) -> pd.Series:

    close, volume = close.align(volume, axis=0)
    BMA: pd.Series = HMA(close, bma_window)

    price_mom: pd.Series = BMA / BMA.shift(n_window)
    vol_mom: pd.Series = HMA(volume, 5) / HMA(volume, ama_window)

    return price_mom * vol_mom


def get_trendshake_filter(close: pd.Series, fast_window: int, slow_window: int,
                          threshold: Tuple[float]):

    threshold1, threshold2 = threshold
    # 市场划分
    filter_shake: pd.Series = (close.rolling(fast_window).mean() >
                               close.rolling(slow_window).mean())

    return filter_shake.apply(lambda x: np.where(x, threshold1, threshold2))


def get_signal(close: pd.Series, volume: pd.Series, signal_window: Tuple,
               lag_window: int, shake_window: Tuple,
               threshold: Tuple):

    res: namedtuple = namedtuple('res', 'vol_mom,shake_filter,flag')

    bma_window, ama_window = signal_window
    fast_window, slow_window = shake_window
    # 价量共振指标
    price_vol: pd.Series = calc_volume_momentum(close, volume, bma_window,
                                                ama_window, lag_window)
    # 市场划分
    threshold_ser: pd.Series = get_trendshake_filter(close, fast_window,
                                                     slow_window, threshold)

    # 持仓标记 0-空仓 1-持仓
    flag: pd.Series = (price_vol > threshold_ser).astype(int)

    return res(price_vol, threshold_ser, flag)


def bulk_signal(price: pd.DataFrame,
                bma_window: int,
                ama_window: int,
                fast_window: int,
                slow_window: int,
                threshold: Tuple,
                n: int,
                level: str,
                method: str = 'flag') -> pd.Series:
    """批量获取信号标记

    Parameters
    ----------
    price : pd.DataFrame
        index-date columns-OHLCV 及code
    bma_window:int
        bma窗口
    ama_window:int
        ama计算窗口
    fast_window:int
        信号过滤的短周期计算窗口
    slow_window:int
        信号过滤的长周期计算窗口
    threshold:Tuple
        0-threshold1 1-threshold2
    n:int
        bma的滞后期
    level:str
        sw-行业 index-宽基
    method:str
        获取vol_mom,shake_filter,flag
    Returns
    -------
    pd.Series
        MuliIndex level-0 code level-1 date 
        code - sec_name+code
    """

    classify: Dict = DICT[level]

    dic: Dict = {}
    for code, df in price.groupby('code'):
        signal_res: namedtuple = get_signal(df['close'], df['volume'],
                                            (bma_window, ama_window), n,
                                            (fast_window, slow_window),
                                            threshold)
        sec_name: str = classify[code].replace('(申万)', '')
        dic[f'{sec_name}({code})'] = getattr(signal_res, method)

    return pd.concat(dic)


def get_signal_status(flag_ser: pd.Series) -> Tuple:
    """根据持仓标记返回当前信号情况

    Parameters
    ----------
    ser : pd.DataFrame
        MuliIndex level0-sec_name+code level1-date

    Returns
    -------
    Tuple
        信息描述,最近的开仓日期,emoji标志
    """
    # code: str = flag_ser.name
    last_date: Union[pd.Timestamp, Tuple] = flag_ser.index[-1]

    last_date: pd.Timestamp = _check_muliindex(last_date)
    diff_ser: pd.Series = (flag_ser != flag_ser.shift(1))
    diff_id: pd.Series = diff_ser.cumsum()
    # 最近一一期的开仓日期
    hold_flag: pd.Series = diff_id[flag_ser == 1]
    # 表示区间内没有开仓信号
    if hold_flag.empty:
        return f"{last_date.strftime('%Y-%m-%d')} 无开仓信号", None, ':white_circle:'
    last_open_date: Union[pd.Timestamp, Tuple] = hold_flag.idxmax()
    last_open_date: pd.Timestamp = _check_muliindex(last_open_date)

    if flag_ser.iloc[-1]:
        # 如果有信号
        return (
            f"{last_date.strftime('%Y-%m-%d')} 有开仓信号", last_date,
            ':red_circle:'
        ) if last_date == last_open_date else (
            f"{last_date.strftime('%Y-%m-%d')} 当期有持仓(开仓日:{last_open_date.strftime('%Y-%m-%d')})",
            last_open_date, ':o:')

    if flag_ser.iloc[-2]:
        return f"{last_date.strftime('%Y-%m-%d')} 平仓", last_date, ':x:'

    else:

        return f"{last_date.strftime('%Y-%m-%d')} 无开仓信号", None, ':white_circle:'


def _check_muliindex(idx: Tuple) -> pd.Timestamp:

    return idx[1] if isinstance(idx, tuple) else idx
