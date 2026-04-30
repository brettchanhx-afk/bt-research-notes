from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from qlib.data.ops import PairRolling

from .chip_distribution import calc_normalization_turnover
from .chip_project_utils import rolling_frame


#历史半衰期换手率筹码分布 
def calc_rc(close_arr: np.ndarray) -> np.ndarray:
    
    if not isinstance(close_arr, np.ndarray):
        raise TypeError("close_arr must be np.ndarray")
    if close_arr.ndim != 1:
        close_arr = close_arr.flatten()
    return np.subtract(1, np.divide(close_arr, close_arr[-1]))


def calc_distribution_of_chips(
    turnover: np.ndarray, close: np.ndarray, N: int
) -> Tuple[np.float64]:
    
    weight: np.ndarray = calc_normalization_turnover(turnover)
    rc: np.ndarray = calc_rc(close)

    # ARC
    arc: np.float64 = np.multiply(weight, rc).sum()

    # VRC
    vrc: np.float64 = N / (N - 1) * (weight * np.square(rc - arc)).sum()

    # SRC
    src: np.float64 = (
        N / (N - 1) * (weight * np.power(rc - arc, 3)).sum() / np.power(vrc, 1.5)
    )

    # KRC
    krc: np.float64 = (
        N / (N - 1) * (weight * np.power(rc - arc, 4)).sum() / np.power(vrc, 2)
    )

    return arc, vrc, src, krc


def calc_roll_cyq(
    turnover: pd.Series, close: pd.Series, N: int, method: str
) -> pd.Series:
   
    method: str = method.upper()
    method_dic: Dict = {"ARC": 0, "VRC": 1, "SRC": 2, "KRC": 3}

    turnover_ls: np.ndarray = rolling_frame(turnover, N)
    close_ls: np.ndarray = rolling_frame(close, N)

    idx: pd.Index = turnover.index[N - 1 :]
    ls: List = [
        calc_distribution_of_chips(left, right, N)[method_dic[method]]
        for left, right in zip(turnover_ls, close_ls)
    ]
    return pd.Series(index=idx, dtype=np.float16, data=ls)


#构建算子
class ARC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(ARC, self).__init__(feature_left, feature_right, N, "Arc")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )

        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)

        return calc_roll_cyq(series_left, series_right, self.N, "ARC")


class VRC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(VRC, self).__init__(feature_left, feature_right, N, "Vrc")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )
        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)
        return calc_roll_cyq(series_left, series_right, self.N, "VRC")


class SRC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(SRC, self).__init__(feature_left, feature_right, N, "Src")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )
        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)
        return calc_roll_cyq(series_left, series_right, self.N, "SRC")


class KRC(PairRolling):
    def __init__(self, feature_left, feature_right, N):

        super(KRC, self).__init__(feature_left, feature_right, N, "Krc")

    def _load_internal(self, instrument, start_index, end_index, *args):

        series_left: pd.Series = self.feature_left.load(
            instrument, start_index, end_index, *args
        )
        series_right: pd.Series = self.feature_right.load(
            instrument, start_index, end_index, *args
        )
        if (series_left.shape[0] < self.N) or (series_right.shape[0] < self.N):
            return pd.Series(dtype=np.float16)
        return calc_roll_cyq(series_left, series_right, self.N, "KRC")
