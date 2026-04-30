from typing import Union

import numba as nb
import numpy as np
import pandas as pd

#用于rolling

@nb.njit
def rolling_windows(arr: np.ndarray, window: int) -> np.ndarray:
    
    shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    windows = np.empty(shape=shape)
    for i in range(shape[0]):
        windows[i] = arr[i : i + window]
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def rolling_frame(
    df: Union[pd.DataFrame, pd.Series, np.array], window: int
) -> np.ndarray:
    
    if window > df.shape[0]:
        raise ValueError(
            "Specified `window` length of {0} exceeds length of"
            " `a`, {1}.".format(window, df.shape[0])
        )

    if isinstance(df, (pd.DataFrame, pd.Series)):
        arr: np.array = df.values

    if arr.ndim == 1:
        arr = arr.copy().reshape(-1, 1)

    return rolling_windows(arr, window)
