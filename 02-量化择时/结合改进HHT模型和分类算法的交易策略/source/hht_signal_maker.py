import contextlib
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed,parallel
from PyEMD import EMD
from scipy.signal import hilbert
from tqdm import tqdm
from vmdpy import VMD

from .utils import sliding_window


def calculate_instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    
    analytic_signal: np.ndarray = hilbert(signal)

    return np.angle(analytic_signal)


def decompose_signal(
    signal: np.ndarray, method: str = "EMD", max_imf: int = 9
) -> np.ndarray:
    
    if method == "EMD":
        emd = EMD()
        imfs = emd.emd(signal, max_imf=max_imf)
    elif method == "VMD":
        alpha = 2000  # 惩罚因子
        tau = 0.3  # 噪声容忍度
        K = max_imf  # 模态数量
        DC = 0  # 是否包含直流分量
        init = 1  # 初始化方式
        tol = 1e-6  # 收敛容忍度
        imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    else:
        raise ValueError("Invalid method. Choose 'EMD' or 'VMD'.")
    return imfs



def get_ht_binary_signal(differenced: np.ndarray) -> int:
   
    instantaneous_phase: np.ndarray = calculate_instantaneous_phase(differenced)
    threshold: float = np.pi * 0.5
    return np.where(
        (instantaneous_phase >= -threshold) & (instantaneous_phase <= threshold), 1, 0
    )


def get_ht_signal(
    data: pd.DataFrame, ma_period: int = 60, ht_period: int = 30
) -> pd.DataFrame:
        
    data_: pd.DataFrame = data.copy()
    differenced: pd.Series = data["close"].rolling(ma_period).mean().diff().dropna()
    signal: pd.Series = differenced.rolling(ht_period).apply(
        lambda x: get_ht_binary_signal(x)[-1], raw=True
    )

    data_["binary_signal"] = signal

    return data_


def get_hht_signal(
    data: pd.DataFrame, hht_period: int = 60, imf_index: int = 2, max_imf: int = 9,method:str="EMD"
) -> pd.DataFrame:
    

    data_: pd.DataFrame = data.copy()
    signal:pd.Series = parallel_apply(data["close"], hht_period, imf_index, max_imf,method)
    data_["binary_signal"] = signal
    return data_


def get_hht_binary_signal(
    close: np.ndarray, imf_index: int = 2, max_imf: int = None, method: str = "EMD"
) -> int:
   
    imfs: List[np.ndarray] = decompose_signal(
        close, method=method.upper(), max_imf=max_imf
    )

    # 确保IMF数量足够
    if len(imfs) <= imf_index:
        return [0]  # 或者返回其他默认值

    return get_ht_binary_signal(imfs[imf_index])

def get_last_value(func, *args, **kwargs):
    
    result = func(*args, **kwargs)
    return result[-1]


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tqdm = tqdm_object

        def __call__(self, *args, **kwargs):
            self._tqdm.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def parallel_apply(
    close: pd.Series,window: int=60,imf_index:int=2,max_imf:int=None,method:str="EMD",n_jobs:int=30
) -> pd.Series:
    
    func = partial(get_hht_binary_signal, imf_index=imf_index, max_imf=max_imf, method=method)
    windows = list(sliding_window(close.values, window))
    
    # 使用tqdm_joblib包装进度条
    with tqdm_joblib(tqdm(total=len(windows), desc="Processing")):
        signal = Parallel(n_jobs=n_jobs)(
            delayed(get_last_value)(func, ser)
            for ser in windows
        )

    return pd.Series(signal, index=close.index[window - 1:])
