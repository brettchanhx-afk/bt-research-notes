from email.policy import default
from typing import (List, Tuple, Dict, Union, Callable, Any)
import math
import warnings

import pandas as pd
import numpy as np

import talib

from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.rolling import (RollingWLS, RollingOLS)


# RSRS计算的类


class RSRS(object):

    def __init__(self, data: pd.DataFrame) -> None:
     
        self.data = data

    def calc_basic_rsrs(self, N: int, method: str, weight: pd.Series = None) -> pd.Series:
        
        func: Dict = {'ols': RollingOLS, 'wls': RollingWLS}

        endog: pd.Series = self.data['high']
        exog: pd.DataFrame = self.data[['low']].copy()
        exog['const'] = 1.

        if (method == 'wls'):

            if weight is None:

                weight = self.data['volume'] / \
                    self.data['volume'].rolling(N).sum()

            mod = func[method](endog, exog, window=N, weights=weight)

        else:
            mod = func[method](endog, exog, window=N)

        self.rolling_res = mod.fit()  # 将回归结果储存在rolling_res中
        self._basic_rsrs = self.rolling_res.params['low']

        return self._basic_rsrs

    def calc_zscore_rsrs(self, N: int, M: int, method: str, weight: pd.Series = None) -> pd.DataFrame:
        
        # 计算基础RSRS
        basic_rsrs: pd.Series = self.calc_basic_rsrs(N, method, weight)

        return (basic_rsrs - basic_rsrs.rolling(M).mean()) / basic_rsrs.rolling(M).std()

    def calc_revise_rsrs(self, N: int, M: int, method: str, weight: pd.Series = None) -> pd.Series:
        
        zscore_rsrs: pd.Series = self.calc_zscore_rsrs(
            N, M, method, weight)  # 计算标准分RSRS
        rsquared: pd.Series = self.rolling_res.rsquared  # 获取R方

        return zscore_rsrs * rsquared

    def calc_right_skewed_rsrs(self, N: int, M: int, method: str, weight: pd.Series = None) -> pd.Series:
       
        revise_rsrs: pd.Series = self.calc_revise_rsrs(
            N, M, method, weight)  # 计算修正标准分RSRS
        return revise_rsrs * self._basic_rsrs

    def calc_insensitivity_rsrs(self, N: int, M: int, method: str, volatility: pd.Series = None, weight: pd.Series = None) -> pd.Series:
       
        if volatility is None:
            ret = self.data['close'].pct_change()
            ret_std = ret.rolling(N).std()
            quantile = ret_std.rolling(M).apply(
                lambda x: x.rank(pct=True)[-1], raw=False) * 2
        else:
            quantile = volatility

        zscore_rsrs: pd.Series = self.calc_zscore_rsrs(
            N, M, method, weight)  # 计算标准分RSRS
        rsquared: pd.Series = self.rolling_res.rsquared

        return zscore_rsrs * rsquared.pow(quantile)


def calc_LLT_MA(price: pd.Series, alpha: float) -> pd.Series:
    
    if not isinstance(price, pd.Series):
        raise ValueError('price必须为pd.Series')

    llt_ser: pd.Series = pd.Series(index=price.index)
    llt_ser[0], llt_ser[1] = price[0], price[1]

    for i, e in enumerate(price.values):

        if i > 1:

            v = (alpha - alpha**2 * 0.25) * e + (alpha ** 2 * 0.5) * price.iloc[i - 1] - (
                alpha - 3 * (alpha**2) * 0.25) * price.iloc[i - 2] + 2 * (
                    1 - alpha) * llt_ser.iloc[i - 1] - (1 - alpha)**2 * llt_ser.iloc[i - 2]

            llt_ser.iloc[i] = v

    return llt_ser


def calc_OLSTL(price: pd.Series, window: int) -> pd.Series:
    
    if not isinstance(price, pd.Series):
        raise ValueError('price必须为pd.Series')

    def _func(arr: np.ndarray) -> float:

        size = len(arr)
        weights = np.arange(1, size+1) - (size + 1) / 3

        avg = weights * arr
        constant = 6 / (size * (size + 1))
        return constant * np.sum(avg)

    return price.rolling(window).apply(_func, raw=True)


def FRAMA(price: pd.Series, window: int, clip: bool = True) -> pd.Series:
   
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    T = int(np.ceil(window * 0.5))
    ser = price.copy()

    # 1.用窗口 W1 内的最高价和最低价计算 N1 = (最高价 – 最低价) / T
    N1 = (ser.rolling(T).max()-ser.rolling(T).min())/T

    # 2.用窗口 W2 内的最高价和最低价计算 N2 = (最高价 – 最低价) / T
    n2_df = ser.shift(T)
    N2 = (n2_df.rolling(T).max()-n2_df.rolling(T).min())/T

    # 3.用窗口 T 内的最高价和最低价计算 N3 = (最高价 – 最低价) / (2T)
    N3 = (ser.rolling(window).max() -
          ser.rolling(window).min()) / window

    # 4.计算分形维数 D = [log(N1+N2) – log(N3)] / log(2)
    D = (np.log10(N1+N2)-np.log10(N3))/np.log10(2)

    # 5.计算指数移动平均的参数 alpha = exp(-4.6(D-1))
    alpha = np.exp(-4.6*(D-1))

    # 设置上线
    if clip:
        alpha = np.clip(alpha, 0.01, 0.2)

    FRAMA = []
    idx = np.argmin(alpha)
    for row, data in enumerate(alpha):
        if row == (idx):
            FRAMA.append(ser.iloc[row])
        elif row > (idx):
            FRAMA.append(data * ser.iloc[row] +
                         (1-data)*FRAMA[-1])
        else:
            FRAMA.append(np.nan)

    FRAMA_se = pd.Series(FRAMA, index=ser.index)

    return FRAMA_se


# 构造HMA
def HMA(price: pd.Series, window: int) -> pd.Series:
    
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    hma = talib.WMA(2 * talib.WMA(price, int(window * 0.5)) -
                    talib.WMA(price, window), int(np.sqrt(window)))

    return hma


def calc_moment(price: pd.Series, cal_m_winodw: int = 20, moment: int = 5, rol_window: int = 90, alpha: Union[float, np.ndarray] = None) -> pd.DataFrame:
    
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    if isinstance(alpha, (float, int)):

        alpha = np.array([alpha])

    # 计算收益率
    pct_chg: pd.Series = price.pct_change()

    # 计算收益率阶距
    moment_ser: pd.Series = pct_chg.rolling(cal_m_winodw).apply(
        stats.moment, kwargs={'moment': moment})

    ema_momentt = pd.concat(
        (moment_ser.ewm(alpha=x, adjust=False).mean() for x in alpha), axis=1)
    ema_momentt.columns = ['{}'.format(round(i, 4)) for i in alpha]

    return ema_momentt


# 相对强弱指标
def calc_RPS(price: pd.Series, window: int=10, default_window: int = 250) -> pd.Series:
    
    if not isinstance(price, pd.Series):
        raise ValueError('price必须为pd.Series')

    size = len(price)
    limit = min(default_window, window)
    if size < limit:

        warnings.warn(
            "price长度低于最低窗口长度%s." % limit
        )

        min_periods = 0

    else:

        min_periods = None

    rps = (price - price.rolling(250, min_periods=min_periods).min()) / (
        price.rolling(250, min_periods=min_periods).max() - price.rolling(250, min_periods=min_periods).min())

    return rps.rolling(window, min_periods=min_periods).mean()

# 强弱 RPS下波动率差值
def calc_volatility_rpc(price:pd.Series,window:int,default_window:int=250)->pd.Series:
    
    rps = calc_RPS(price, window,default_window)
    pct_chg = price.pct_change()
    
    up:np.ndarray = np.where(pct_chg > 0,rps,0)
    down:np.ndarray = np.where(pct_chg <= 0,rps,0)
    
    dif = pd.Series(data=up - down,index=rps.index)
    dif = dif.rolling(window).mean()

    return dif


# 华泰-熊牛线
def calc_ht_bull_bear(price:pd.Series,turnover:pd.Series,window:int)->pd.Series:
   
    if (not isinstance(price,pd.Series)) or (not isinstance(turnover,pd.Series)):
        raise ValueError('price和turnover必须为pd.Series')
        
    pct_chg = price.pct_change()
    vol = pct_chg.rolling(window).std()
    turnover_avg = turnover.rolling(window).mean()

    return turnover_avg / turnover_avg

# 量化投资:策略于技术-熊牛线
def calc_bull_curve(price: pd.Series, alpha:float, n: int, T: int, method:str='bull') -> pd.Series:
   
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    window = n * T # 时间窗口
    epsilon = stats.t.ppf(1 - alpha * 0.5, n)  # 落入牛熊价格区间的置信度为(1-alpha)
    log_ret = np.log(price / price.shift(-1))
    mu = log_ret.rolling(window).mean()
    sigma = log_ret.rolling(window).std()
    close_t = price.shift(T)

    return geometric_mrownian_motion(close_t, mu, sigma, T, epsilon)


def geometric_mrownian_motion(price: pd.Series, mu: pd.Series, sigma: pd.Series, T: int, epsilon: float, method: str = 'bull') -> float:
    
    if method == 'bull':

        return price * np.exp(T * mu + np.sqrt(T) * sigma * epsilon)

    elif method == 'bear':

        return price * np.exp(T * mu - np.sqrt(T) * sigma * epsilon)

    else:

        raise ValueError('method参数仅能为bull或者bear')

"""Hurst指数

from:https://github.com/Mottl/hurst
"""


def __to_inc(x):
    incs = x[1:] - x[:-1]
    return incs


def __to_pct(x):
    pcts = x[1:] / x[:-1] - 1.
    return pcts


def __get_simplified_RS(series, kind):
 
    if kind == 'random_walk':
        incs = __to_inc(series)
        R = max(series) - min(series)  # range in absolute values
        S = np.std(incs, ddof=1)
    elif kind == 'price':
        pcts = __to_pct(series)
        R = max(series) / min(series) - 1.  # range in percent
        S = np.std(pcts, ddof=1)
    elif kind == 'change':
        incs = series
        _series = np.hstack([[0.], np.cumsum(incs)])
        R = max(_series) - min(_series)  # range in absolute values
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due the undefined R/S ratio

    return R / S


def __get_RS(series, kind):
    

    if kind == 'random_walk':
        incs = __to_inc(series)
        mean_inc = (series[-1] - series[0]) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    elif kind == 'price':
        incs = __to_pct(series)
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    elif kind == 'change':
        incs = series
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due undefined R/S

    return R / S


def compute_Hc(series,
               kind="random_walk",
               min_window=10,
               max_window=None,
               simplified=True):
  
    if len(series) < 100:
        raise ValueError("Series length must be greater or equal to 100")

    ndarray_likes = [np.ndarray]
    if "pandas.core.series" in sys.modules.keys():
        ndarray_likes.append(pd.core.series.Series)

    # convert series to numpy array if series is not numpy array or pandas Series
    if type(series) not in ndarray_likes:
        series = np.array(series)

    if "pandas.core.series" in sys.modules.keys() and type(
            series) == pd.core.series.Series:
        if series.isnull().values.any():
            raise ValueError("Series contains NaNs")
        series = series.values  # convert pandas Series to numpy array
    elif np.isnan(np.min(series)):
        raise ValueError("Series contains NaNs")

    if simplified:
        RS_func = __get_simplified_RS
    else:
        RS_func = __get_RS

    err = np.geterr()
    np.seterr(all='raise')

    max_window = max_window or len(series) - 1
    window_sizes = list(
        map(lambda x: int(10**x),
            np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
    window_sizes.append(len(series))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(series), w):
            if (start + w) > len(series):
                break
            _ = RS_func(series[start:start + w], kind)
            if _ != 0:
                rs.append(_)
        RS.append(np.mean(rs))

    A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
    np.seterr(**err)

    c = 10**c
    return H, c, [window_sizes, RS]


def random_walk(length,
                proba=0.5,
                min_lookback=1,
                max_lookback=100,
                cumprod=False):
    
    assert (min_lookback >= 1)
    assert (max_lookback >= min_lookback)

    if max_lookback > length:
        max_lookback = length
        warnings.warn(
            "max_lookback parameter has been set to the length of the random walk series."
        )

    if not cumprod:  # ordinary increments
        series = [0.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback,
                                             min(i - 1, max_lookback) + 1)
                direction = np.sign(series[i - 1] - series[i - 1 - lookback]
                                    ) * np.sign(proba - np.random.uniform())
            series[i] = series[i - 1] + np.fabs(np.random.randn()) * direction
    else:  # percent changes
        series = [1.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback,
                                             min(i - 1, max_lookback) + 1)
                direction = np.sign(series[i - 1] / series[i - 1 - lookback] -
                                    1.) * np.sign(proba - np.random.uniform())
            series[i] = series[i - 1] * np.fabs(1 + np.random.randn() / 1000. *
                                                direction)

    return series
