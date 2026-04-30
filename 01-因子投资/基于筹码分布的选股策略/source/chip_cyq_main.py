from typing import Union
import numpy as np
import pandas as pd
from loguru import logger
from numba import jit

from .chip_distribution import calc_adj_turnover, calc_triang_pdf, calc_uniform_pdf

def calc_curpdf(
    close: float,
    high: float,
    low: float,
    vol: float,
    min_p: float = None,
    max_p: float = None,
    step: float = 0.01,
    method: str = "triang",
) -> np.ndarray:
    
    method: str = method.lower()

    if method == "triang":

        return calc_triang_pdf(close, high, low, vol, min_p, max_p, step)

    elif method == "uniform":

        return calc_uniform_pdf(close, high, low, vol, min_p, max_p, step)

    else:
        raise ValueError("method must be triang or uniform")


@jit(nopython=True)
def calc_cumpdf(curpdf: np.ndarray, turnover: np.ndarray, A: float = 1.0) -> float:
    
    decay: np.ndarray = turnover * A
    diff: np.ndarray = 1 - decay

    mul_array: np.ndarray = (curpdf.T * decay).T
    size: int = len(turnover)
    cumpdf: np.ndarray = np.empty(size)
    for i in range(size):
        cumpdf = cumpdf * diff[i] + mul_array[i] if i else curpdf[i] * decay[i]

    return cumpdf


def calc_dist_chips(
    arr: Union[pd.DataFrame, np.ndarray], method: str, step: float = 0.01
) -> pd.Series:
    
    if isinstance(arr, pd.DataFrame):
        arr: pd.DataFrame = arr[["close", "high", "low", "vol", "turnover_rate"]]
        arr: np.ndarray = arr.values

    method: str = method.lower()
    if method in {"triang", "uniform"}:

        # max_p,min_p可能区间为nan
        max_p: float = np.nanmax(arr[:, 1])
        min_p: float = np.nanmin(arr[:, 2])
        try:
            xs: np.ndarray = np.arange(min_p, max_p + step, step)
        except ValueError as e:
            logger.warning(f"min_p:{min_p}, max_p:{max_p};此段时间可能停牌,请检查")
            raise e

        try:
            curpdf: np.ndarray = np.apply_along_axis(
                lambda x: calc_curpdf(
                    x[0], x[1], x[2], x[3], min_p, max_p, step, method
                ),
                1,
                arr,
            )
        except Exception as e:
            print(min_p, max_p)
            raise e
        cum_vol: np.ndarray = calc_cumpdf(curpdf, arr[:, 4])
        cum_vol: pd.Series = pd.Series(cum_vol, index=xs)

    elif method == "turn_coeff":

        turn_coeff: np.ndarray = calc_adj_turnover(arr[:, 4])
        total_vol: float = arr[:, 3].sum()
        data: pd.DataFrame = pd.Series(
            data=turn_coeff,
            index=arr[:, 1],
        )
        cum_vol: pd.Series = data.groupby(level=0).sum() * total_vol

    return cum_vol


# 筹码分布因子 


class ChipFactor:
    def __init__(
        self,
        close: float,
        cumpdf: pd.Series,
    ) -> None:

        self.close = close
        self.cumpdf = cumpdf  # 过去N日的成交分布

        self.cumpdf.index.names = ["price"]
        self.cumpdf.name = "cumpdf"

    @staticmethod
    def winsorize(cumpdf: pd.Series, scale: int = 3) -> pd.Series:

        std: float = cumpdf.std()
        mean: float = cumpdf.mean()

        return cumpdf.clip(mean - scale * std, mean + scale * std)

    def get_asr(self, lower: float = 0.9, upper: float = 1.1) -> float:
        
        return self.get_winner(upper * self.close) - self.get_winner(lower * self.close)

    def get_cyqk_c(self) -> float:
        
        return self.get_winner(self.close)

    def get_ckdw(self, scale: int = 3) -> float:
        
        winsorize: pd.Series = self.cumpdf

        if scale is not None:
            # 当scale不为None时，对cumpdf进行winsorize
            winsorize: pd.Series = self.winsorize(winsorize, scale)
        # 平均成本
        mean: float = self.get_cost(0.5)
        min_p: float = winsorize.idxmin()
        max_p: float = winsorize.idxmax()

        return (mean - min_p) / (max_p - min_p)

    def get_prp(self) -> float:
        
        # 平均成本
        avg: float = self.get_cost(0.5)

        return self.close / avg - 1

    def get_winner(self, price: float) -> float:
        
        tot_cnt: float = self.cumpdf.sum()  # 总筹码数
        # 累计筹码比例
        acc_cum: pd.Series = self.cumpdf / tot_cnt

        return acc_cum[acc_cum.index <= price].sum()

    def get_cost(self, winner_ratio: float) -> float:
        
        if (winner_ratio < 0) or (winner_ratio > 1):
            raise ValueError("winner_ratio must be in [0,1]")

        tot_cnt: float = self.cumpdf.sum()  # 总筹码数
        # 累计筹码比例
        acc_cum: pd.Series = (self.cumpdf / tot_cnt).cumsum()

        threshold_ser: pd.Series = acc_cum[acc_cum < winner_ratio]
        return np.nan if threshold_ser.empty else threshold_ser.index[-1]
