from typing import Union, Tuple

import numpy as np
import pandas as pd

from .utils import sliding_window

__all__ = ["QRSCreator", "calc_corrcoef", "calc_beta", "calc_zscore"]


def calc_corrcoef(low: np.ndarray, hight: np.ndarray) -> np.ndarray:
    
    if low.shape != hight.shape:
        raise ValueError("low and hight must have the same shape")

    if low.ndim == 2:
        corr_matrix: np.ndarray = np.corrcoef(hight, low, rowvar=False)
        return np.diagonal(corr_matrix, offset=low.shape[1])
    elif low.ndim == 1:
        return np.corrcoef(hight, low)[0, 1]


def calc_beta(low: np.ndarray, hight: np.ndarray) -> np.ndarray:
    
    corr: np.ndarray = calc_corrcoef(low, hight)
    # ddof=0
    return np.std(hight, axis=0) / np.std(low, axis=0) * corr


def calc_zscore(data: np.ndarray) -> np.ndarray:
    
    return (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)


def select_array(arr: np.ndarray, idx: int) -> np.ndarray:
    
    if arr.ndim == 3:
        # 如果是三维数组，返回第一个维度的切片
        return arr[:, :, idx]
    elif arr.ndim == 2:
        # 如果是二维数组，直接返回
        return arr[:, idx]
    else:
        raise ValueError("输入数组必须是二维或三维的")


class QRSCreator:

    def __init__(
        self, low_df: pd.DataFrame | pd.Series, high_df: pd.DataFrame | pd.Series
    ) -> None:

        if not isinstance(low_df, (pd.Series, pd.DataFrame)) or not isinstance(
            high_df, (pd.Series, pd.DataFrame)
        ):
            raise ValueError("low_df and high_df must be pd.Series or pd.DataFrame")

        if low_df.shape != high_df.shape:
            raise ValueError("low_df and high_df must have the same shape")

        # 对齐
        low_df, high_df = low_df.align(high_df)

        self.low_df = low_df
        self.high_df = high_df
        # 转换为numpy数组
        self.low_arr = self.low_df.values
        self.high_arr = self.high_df.values

        # 拼接low,high数组用于后续滚动计算beta,corr等值
        self.data = self._concat_matrix(self.low_arr, self.high_arr)

        # 以下属性用于存储计算结果
        self.beta: Union[pd.Series, pd.DataFrame] = None  # beta
        self.simple_beta: Union[pd.Series, pd.DataFrame] = None  # simple_beta
        self.zscore_beta: Union[pd.Series, pd.DataFrame] = None  # zscore_beta
        self.zscore_simple_beta: Union[pd.Series, pd.DataFrame] = None
        self.regulation: Union[pd.Series, pd.DataFrame] = None  # regulation

    @staticmethod
    def _concat_matrix(low_arr: np.ndarray, high_arr: np.ndarray) -> np.ndarray:
        
        dim: int = low_arr.ndim
        return np.stack([low_arr, high_arr], axis=dim)

    def get_columns(self) -> pd.Index:

        return (
            self.low_df.columns if isinstance(self.low_df, pd.DataFrame) else ["Signal"]
        )

    def get_index(
        self, regression_window: int, zscore_window: int = None
    ) -> pd.DatetimeIndex:
        
        zscore_window: int = zscore_window or 0
        offset: int = 2 if zscore_window else 1
        idx: pd.DatetimeIndex = self.low_df.index[
            regression_window + zscore_window - offset :
        ]

        return idx

    def calc_simple_signal(
        self, regression_window: int = 18, zscore_window: int = 600
    ) -> pd.DataFrame:
        
        # 计算beta不含corr
        beta_matrix: np.ndarray = np.array(
            [
                np.std(select_array(arr, 1), axis=0)
                / np.std(select_array(arr, 0), axis=0)
                for arr in sliding_window(self.data, regression_window)
            ]
        )
        # 计算zscore
        zscore_beta: np.ndarray = np.array(
            [calc_zscore(arr)[-1] for arr in sliding_window(beta_matrix, zscore_window)]
        )

        columns: pd.Index = self.get_columns()
        idx: pd.DatetimeIndex = self.get_index(regression_window, zscore_window)
        # 简单zscore_beta
        self.zscore_simple_beta: pd.DataFrame = pd.DataFrame(
            zscore_beta, index=idx, columns=columns
        )
        # 原始简单beta
        self.simple_beta: pd.DataFrame = pd.DataFrame(
            beta_matrix, index=self.get_index(regression_window), columns=columns
        )
        return self.zscore_simple_beta

    def calc_zscore_beta(
        self, regression_window: int = 18, zscore_window: int = 600
    ) -> Union[pd.Series, pd.DataFrame]:
       
        # 计算beta
        beta_matrix: np.ndarray = np.array(
            [
                calc_beta(select_array(arr, 0), select_array(arr, 1))
                for arr in sliding_window(self.data, regression_window)
            ]
        )
        # 计算zscore
        zscore_beta: np.ndarray = np.array(
            [calc_zscore(arr)[-1] for arr in sliding_window(beta_matrix, zscore_window)]
        )

        columns: pd.Index = self.get_columns()
        idx: pd.DatetimeIndex = self.get_index(regression_window, zscore_window)

        self.zscore_beta: pd.DataFrame = pd.DataFrame(
            zscore_beta, index=idx, columns=columns
        )

        self.beta: pd.DataFrame = pd.DataFrame(
            beta_matrix, index=self.get_index(regression_window), columns=columns
        )
        return self.zscore_beta

    def calc_regulation(
        self, regression_window: int = 18, n: int = 2
    ) -> Union[pd.Series, pd.DataFrame]:
        
        corr_matrix: np.ndarray = np.array(
            [
                calc_corrcoef(select_array(arr, 0), select_array(arr, 1))
                for arr in sliding_window(self.data, regression_window)
            ]
        )

        columns: pd.Index = self.get_columns()
        idx: pd.DatetimeIndex = self.get_index(regression_window)

        self.regulation: pd.DataFrame = pd.DataFrame(
            np.power(corr_matrix, n), index=idx, columns=columns
        )
        return self.regulation

    def calc_regulation_mean(self, window: int) -> Union[pd.Series, pd.DataFrame]:
        
        if self.regulation is None:
            raise ValueError("请先运行calc_regulation方法")

        return self.regulation.rolling(window).mean()

    def fit(
        self,
        regression_window: int = 18,
        zscore_window: int = 600,
        n: int = 2,
        adjust_regulation: bool = False,
        use_simple_beta: bool = False,
    ) -> pd.DataFrame:
        
        if use_simple_beta:
            zscore_beta: pd.DataFrame = self.calc_simple_signal(
                regression_window, zscore_window
            )
        else:
            zscore_beta: pd.DataFrame = self.calc_zscore_beta(
                regression_window, zscore_window
            )

        regulation: pd.DataFrame = self.calc_regulation(regression_window, n).iloc[
            zscore_window - 1 :
        ]

        if adjust_regulation:
            regulation_mean: Union[pd.DataFrame, pd.Series] = self.calc_regulation_mean(
                regression_window
            ).iloc[zscore_window - 1 :]
            regulation: Union[pd.DataFrame, pd.Series] = regulation.div(regulation_mean)

        return zscore_beta * regulation


def test_func(
    low_df: pd.DataFrame, high_df: pd.DataFrame, colNums: int = 1
) -> Tuple[float, float]:
    
    import statsmodels.api as sm

    mod = sm.OLS(
        high_df.iloc[-18:, colNums], sm.add_constant(low_df.iloc[-18:, colNums])
    )
    res = mod.fit()

    return res.params.iloc[-1], res.rsquared


if __name__ == "__main__":

    # TEST
    high_df = pd.DataFrame(
        np.random.uniform(30, 40, (1000, 4)),
        index=pd.date_range(start="2020-01-01", periods=1000),
        columns=["A", "B", "C", "D"],
    )

    low_df = pd.DataFrame(
        np.random.uniform(20, 30, (1000, 4)),
        index=pd.date_range(start="2020-01-01", periods=1000),
        columns=["A", "B", "C", "D"],
    )

    qrs: QRSCreator = QRSCreator(low_df, high_df)
    signal: pd.DataFrame = qrs.fit()

    print("signal:")
    print(signal)

    colNums: int = 1
    print("test:", test_func(low_df, high_df, colNums))
    print("qrs:", (qrs.beta.iloc[-1, colNums], qrs.regulation.iloc[-1, colNums]))
