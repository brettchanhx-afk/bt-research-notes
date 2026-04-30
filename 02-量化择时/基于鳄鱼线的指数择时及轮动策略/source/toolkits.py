from typing import Tuple

import numpy as np
import pandas as pd



def trans_to_entries_exits(signal: pd.Series) -> Tuple[pd.Series, pd.Series]:
    
    if not isinstance(signal, pd.Series):
        raise ValueError("signal必须是pd.Series对象。")
    entries: pd.Series = signal.apply(lambda x: True if x == 1 else False)
    exits: pd.Series = signal.apply(lambda x: True if x == -1 else False)
    return entries, exits

def get_shift(arr: np.ndarray, periods: int, axis: int = 0) -> np.ndarray:
   
    if arr.shape[axis] < periods:
        raise ValueError("滞后步长大于数组长度。")
    elif periods == 0:
        return arr

    tmp: np.ndarray = np.roll(arr, periods, axis=axis)
    if axis == 0:
        filler = np.nan * np.ones(periods)
    else:
        filler = np.nan * np.ones((periods, arr.shape[1]))

    if periods > 0:
        if axis == 0:
            tmp[:periods] = filler
        else:
            tmp[:, :periods] = filler
    else:
        if axis == 0:
            tmp[periods:] = filler
        else:
            tmp[:, periods:] = filler

    return tmp
