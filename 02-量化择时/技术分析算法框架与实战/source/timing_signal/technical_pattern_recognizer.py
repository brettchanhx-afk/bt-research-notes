# 引入库
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Callable, Union)
import itertools
import functools
from tqdm.notebook import tqdm
import warnings

from numba import jit
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.stats import ttest_1samp
from scipy.signal import (argrelmin, argrelmax)

import pandas as pd
import numpy as np

from multiprocessing import Pool

import matplotlib as mpl
import mplfinance as mpf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置CPU工作核数
global CPU_WORKER_NUM
CPU_WORKER_NUM = 6


def rolling_windows(a: Union[np.ndarray, pd.Series, pd.DataFrame], window: int) -> np.ndarray:
    
    if window > a.shape[0]:
        raise ValueError(
            "Specified `window` length of {0} exceeds length of"
            " `a`, {1}.".format(window, a.shape[0])
        )
    if isinstance(a, (pd.Series, pd.DataFrame)):
        a = a.values
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    windows = np.squeeze(
        np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    )
    # In cases where window == len(a), we actually want to "unsqueeze" to 2d.
    #     I.e., we still want a "windowed" structure with 1 window.
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def calc_smooth(prices: pd.Series, *, bw: Union[np.ndarray, str] = 'cv_ls', a: float = None, use_array: bool = True) -> Union[pd.Series, np.ndarray]:
    
    if not isinstance(prices, pd.Series):
        raise ValueError('prices必须为pd.Series')

    idx = np.arange(len(prices))

    kr = KernelReg(prices.values, idx,
                   var_type='c', reg_type='ll', bw=bw)

    if a is None:

        f = kr.fit(idx)[0]

    else:

        kr.bw = a * kr.bw  # 论文用的0.3 * h

        f = kr.fit(idx)[0]

    if use_array:

        return f

    else:

        return pd.Series(data=f, index=prices.index)



def find_argrelextrema(prices: Union[pd.Series,pd.DataFrame], *, offset: int = 1,  smooth_func: Callable = calc_smooth, **kw) -> pd.Series:
    
    size = len(prices)

    # TODO:没有考虑d的长度
    if size <= offset:
        
        raise ValueError('price数据长度过小')

    # 计算平滑价格
    
    smooth_arr: np.ndarray = smooth_func(prices, **kw)

    if isinstance(prices,pd.DataFrame):

        prices = prices['close']

    # 请多平滑后的高低点
    local_max = argrelmax(smooth_arr)[0]
    local_min = argrelmin(smooth_arr)[0]

    # 避免max或者min太local
    # 注意这里实在原始数据上找极值

    price_local_max_dt = []  # 储存索引下标

    for i in local_max:

        begin_idx = max(0, i-offset)
        end_idx = min(size, i+offset+1)
        price_local_max_dt.append(prices.iloc[begin_idx:end_idx].idxmax())

    price_local_min_dt = []  # 储存索引下标

    for i in local_min:

        begin_idx = max(0, i-offset)
        end_idx = min(size, i+offset+1)

        price_local_min_dt.append(prices.iloc[begin_idx:end_idx].idxmin())

    idx = (pd.to_datetime(price_local_max_dt + price_local_min_dt)
           .drop_duplicates()
           .sort_values())

    return prices.loc[idx]

# TODO:算法待优化 是否能减少时间复杂度


def find_price_patterns(max_min: pd.Series, save_all: bool = True) -> defaultdict:
    
    if not isinstance(max_min, pd.Series):
        raise ValueError('max_min类型需要为pd.Series')

    patterns = defaultdict(list)  # 储存识别好的 形态信息
    size = len(max_min)

    # 如果max_min小于5则为空
    if size < 5:
        return {}

    arrs: np.ndarray = rolling_windows(max_min.values, 5)  # 平滑并确定好高低点的价格数据
    idxs: np.ndarray = rolling_windows(max_min.index.values, 5)  # 索引

    for idx, arr in zip(idxs, arrs):

        # Head and Shoulders

        if _pattern_HS(arr):
            patterns['头肩顶(HS)'].append([(idx[0], idx[-1]), idx])

        # Inverse Head and Shoulders
        elif _pattern_IHS(arr):
            patterns['头肩底(IHS)'].append([(idx[0], idx[-1]), idx])

        # Broadening Top
        elif _pattern_BTOP(arr):
            patterns['顶部发散(BTOP)'].append([(idx[0], idx[-1]), idx])

        # Broadening Bottom
        elif _pattern_BBOT(arr):
            patterns['底部发散(BBOT)'].append([(idx[0], idx[-1]), idx])

        # Triangle Top
        elif _pattern_TTOP(arr):
            patterns['顶部收敛三角形(TTOP)'].append([(idx[0], idx[-1]), idx])

        # Triangle Bottom
        elif _pattern_TBOP(arr):
            patterns['底部收敛三角形(TBOT)'].append([(idx[0], idx[-1]), idx])

        # Rectangle Top
        elif _pattern_RTOP(arr):

            patterns['顶部矩形(RTOP)'].append([(idx[0], idx[-1]), idx])

        # Rectangle Bottom
        elif _pattern_RBOT(arr):
            patterns['底部矩形(RBOT)'].append([(idx[0], idx[-1]), idx])

        # TODO:双顶(DTOP),双底(DBOP)
        else:
            pass

    # 是否保留所有的形态识别
    if not save_all:
        # 仅保留区间内的
        tmp_dic = {}
        for k, v in patterns.items():
            tmp_dic[k] = v[0]
        patterns = tmp_dic

    return patterns



@jit(nopython=True)
def _pattern_HS(arr: np.ndarray) -> bool:
   
    e1, e2, e3, e4, e5 = arr

    avg1 = np.array([e1, e5]).mean()
    avg2 = np.array([e2, e4]).mean()

    cond1 = (e1 > e2)  # (np.argmax(arr) == 0)
    cond2 = (e3 > e1) and (e3 > e5)
    cond3 = (0.985 * avg1 <= e1 <= avg1 *
             1.015) and (0.985 * avg1 <= e5 <= avg1 * 1.015)
    cond4 = (0.985 * avg2 <= e2 <= avg2 *
             1.015) and (0.985 * avg2 <= e4 <= avg2 * 1.015)

    return np.array([cond1, cond2, cond3, cond4]).all()


@jit(nopython=True)
def _pattern_IHS(arr: np.ndarray) -> bool:
    
    e1, e2, e3, e4, e5 = arr

    avg1 = np.array([e1, e5]).mean()
    avg2 = np.array([e2, e4]).mean()

    cond1 = (e1 < e2)  # (np.argmin(arr) == 0)
    cond2 = (e3 < e1) and (e3 < e5)
    cond3 = (0.985 * avg1 <= e1 <= avg1 *
             1.015) and (0.985 * avg1 <= e5 <= avg1 * 1.015)
    cond4 = (0.985 * avg2 <= e2 <= avg2 *
             1.015) and (0.985 * avg2 <= e4 <= avg2 * 1.015)

    return np.array([cond1, cond2, cond3, cond4]).all()


@jit(nopython=True)
def _pattern_BTOP(arr: np.ndarray) -> bool:
   
    e1, e2, e3, e4, e5 = arr

    cond1 = e1 > e2  # (np.argmax(arr) == 0)
    cond2 = (e1 < e3 < e5)
    cond3 = (e2 > e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_BBOT(arr: np.ndarray) -> bool:
    
    e1, e2, e3, e4, e5 = arr

    cond1 = e1 < e2  # (np.argmin(arr) == 0)
    cond2 = (e1 > e3 > e5)
    cond3 = (e2 < e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_TTOP(arr: np.ndarray) -> bool:
    
    e1, e2, e3, e4, e5 = arr

    cond1 = (np.argmax(arr) == 0)
    cond2 = (e1 > e3 > e5)
    cond3 = (e2 < e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_TBOP(arr: np.ndarray) -> bool:
    
    e1, e2, e3, e4, e5 = arr

    cond1 = (np.argmin(arr) == 0)
    cond2 = (e1 < e3 < e5)
    cond3 = (e2 > e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_RTOP(arr: np.ndarray) -> bool:
    
    e1, e2, e3, e4, e5 = arr

    g1 = np.array([e1, e3, e5])
    g2 = np.array([e2, e4])

    rtop_g1 = np.mean(g1)
    rtop_g2 = np.mean(g2)

    cond1 = (np.argmax(arr) == 0)

    g1_ = np.abs(g1 - rtop_g1) / rtop_g1
    g2_ = np.abs(g2 - rtop_g2) / rtop_g2
    cond2 = np.all(g1_ <= 0.0075)
    cond3 = np.all(g2_ <= 0.0075)
    cond4 = np.min(g1) > np.max(g2)

    return np.array([cond1, cond2, cond3, cond4]).all()


@jit(nopython=True)
def _pattern_RBOT(arr: np.ndarray) -> bool:
    
    e1, e2, e3, e4, e5 = arr

    g1 = np.array([e1, e3, e5])
    g2 = np.array([e2, e4])

    rtop_g1 = np.mean(g1)
    rtop_g2 = np.mean(g2)

    cond1 = (np.argmin(arr) == 0)
    g1_ = np.abs(g1 - rtop_g1) / rtop_g1
    g2_ = np.abs(g2 - rtop_g2) / rtop_g2
    cond2 = np.all(g1_ < 0.0075)
    cond3 = np.all(g2_ < 0.0075)

    cond4 = np.min(g2) > np.max(g1)

    return np.array([cond1, cond2, cond3, cond4]).all()


def get_shorttimeseries_pattern(price:Union[pd.Series,pd.DataFrame],save_all:bool,smooth_func:Callable,**kw)->namedtuple:
    
    Record = namedtuple('Record', 'patterns,points')

    max_min = find_argrelextrema(price, smooth_func=smooth_func,**kw)
    res2 = find_price_patterns(max_min,save_all)
    patterns = defaultdict(list)
    points = defaultdict(list)

    for k,v in res2.items():
        
        if save_all:
            for p1,idx in v:
                
                patterns[k].append(p1)
                points[k].append(idx)

        else:

            p1,idx = v
            patterns[k].append(p1)
            points[k].append(idx)
                
    record = Record(patterns=patterns, points=points)

    return record



"""使用Multiprocessing"""


def _roll_patterns_series(arrs: Tuple[np.ndarray,np.ndarray], offset: int = 1,  smooth_func: Callable = calc_smooth, **kw) -> defaultdict:
   
    # slice_arr, idx_arr, id_num = arrs
    slice_arr, idx_arr = arrs

    close_ser = pd.Series(data=slice_arr, index=idx_arr)

    max_min = find_argrelextrema(close_ser, offset=offset,  smooth_func = calc_smooth, **kw)

    return find_price_patterns(max_min,save_all=True)  # 注意这里保存的所有信息


def rolling_patterns2pool(price: pd.Series, n: int, reset_window: int = None, *, n_workers: int = CPU_WORKER_NUM, **kw) -> namedtuple:
    
    size = len(price)
    
    if reset_window is None:

        reset_window = size - n + 1  # 表示不更新

    if reset_window >= size:

        raise ValueError('reset_window不能大于price长度')

    # 用于储存结果
    Record = namedtuple('Record', 'patterns,points')
    patterns = defaultdict(list)  # 储存识别好的形态
    points = defaultdict(list)  # 出现形态时点

    # 这里拆分整个窗口用于滑动
    idxs: np.ndarray = rolling_windows(price.index.values, n)
    arr: np.ndarray = rolling_windows(price.values, n)
  
    chunk_size = calculate_best_chunk_size(len(idxs), n_workers)

    roll_patterns_series = functools.partial(_roll_patterns_series, **kw)

    with Pool(processes=n_workers) as pool:

        res_tuple: Tuple[Dict] = tuple(
            pool.imap(roll_patterns_series, zip(arr, idxs), chunksize=chunk_size))

    for num, sub_res in enumerate(res_tuple):

        current_pattern = sub_res

        if current_pattern:

            if (num % reset_window == 0) and (num != 0):
                # 当大于更新长度时更新字典

                for k, v in current_pattern.items():

                    point, idx = v[0]
                    patterns[k].append(point)  # 两点为识别出的形态区间
                    points[k].append(idx)  # 形态区间的五点位置

            else:

                # 当不是形态更新节点时 使用首次识别的形态
                keys = patterns.keys()
                # 当窗口滑动时,历史上同一时间出现的形态可能会在多个连续窗口中被识别出来，
                # 为了不重复分析，我们只保留第一次识别到该形态的时点。
                for k, v in current_pattern.items():

                    if k not in keys:
                        point, idx = v[0]
                        patterns[k].append(point)  # 两点为识别出的形态区间
                        points[k].append(idx)  # 形态区间的五点位置
        else:

            continue

    record = Record(patterns=patterns, points=points)

    return record


def calculate_best_chunk_size(data_length: int, n_workers: int) -> int:
    
    chunk_size, extra = divmod(data_length, n_workers * 5)
    if extra:
        chunk_size += 1
    return chunk_size

"""用于画图"""

def plot_patterns_chart(ohlc_data: pd.DataFrame, record_patterns: namedtuple, slice_range: bool = False, subplots: bool = False, ax=None):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    COLORS = ['DeepSkyBlue','DarkOliveGreen', 'Crimson', 'DarkGoldenRod']
    if not record_patterns.patterns:
        raise ValueError('record_patterns为空')

    # 设置蜡烛图风格
    mc = mpf.make_marketcolors(up='r', down='g',
                               wick='i',
                               edge='i',
                               ohlc='i')

    s = mpf.make_mpf_style(marketcolors=mc)

    def _get_slice_price(tline: Union[Dict, np.array]) -> pd.DataFrame:
        """划分区间"""

        if isinstance(tline, dict):

            start_idx = ohlc_data.index.get_loc(tline['tlines'][0][0])
            end_idx = ohlc_data.index.get_loc(tline['tlines'][-1][-1])
            start = max(0, start_idx-25)
            end = min(len(ohlc_data), end_idx+30)

            return ohlc_data.iloc[start:end]

        else:

            start_idx = ohlc_data.index.get_loc(tline[0])
            end_idx = ohlc_data.index.get_loc(tline[-1])
            start = max(0, start_idx-25)
            end = min(len(ohlc_data), end_idx+30)

            return ohlc_data.iloc[start:end]

    # 线段划分标记
    datepairs: List = []
    titles: List = []
    for title, dates in record_patterns.points.items():
        for d in dates:
            #dates = np.sort(np.array(list(record_patterns.point.values())).flatten())
            d = pd.to_datetime(d)
            datepair = [(d1, d2) for d1, d2 in zip(d, d[1:])]
            datepairs.append(datepair)
            titles.append(title)

    tlines = [dict(tlines=datepair, tline_use='close', colors=color, alpha=0.5, linewidths=5) for datepair,
              color in zip(datepairs, itertools.cycle(COLORS)) if datepair is not None]

    # 是否拆分画图
    if subplots:

        length = len(tlines)
        rows = int(np.ceil(length * 0.5))

        if ax is None:
            fig, axes = plt.subplots(rows, 2, figsize=(18, 3 * length))
        else:
            axes = ax

        axes = axes.flatten()

        for ax_i, (title, tline, ax) in enumerate(itertools.zip_longest(titles, tlines, axes)):

            if (ax_i == len(axes)-1) and (length % 2 != 0):

                ax.axis('off')
                break

            ax.set_title(title)

            if slice_range:

                mpf.plot(_get_slice_price(tline), style=s, tlines=tline,
                         type='candle', datetime_format='%Y-%m-%d', ax=ax)

            else:

                mpf.plot(ohlc_data, style=s, tlines=tline,
                         type='candle', datetime_format='%Y-%m-%d', ax=ax)

        plt.subplots_adjust(hspace=0.5)
        return axes

    else:

        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        if slice_range:

            all_dates: np.ndarray = np.array(
                [x for i in record_patterns.points.values() for x in i])
            all_dates = np.sort(np.unique(all_dates.flatten()))
            all_dates = pd.to_datetime(all_dates)

            mpf.plot(_get_slice_price(all_dates), style=s, tlines=tlines,
                     type='candle', datetime_format='%Y-%m-%d', ax=ax)
            return ax

        else:

            mpf.plot(ohlc_data, style=s, tlines=tlines,
                     type='candle', datetime_format='%Y-%m-%d', ax=ax)
        return ax


