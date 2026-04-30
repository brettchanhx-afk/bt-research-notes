###
import pandas as pd
from typing import List
from .plotting_utils import calculate_bin_means


def concat_signal_vs_forward_returns(
    signal: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.DataFrame:
    
    return (
        pd.concat({"signal": signal, "forward_returns": forward_returns})
        .stack()
        .unstack(level=[2, 0])
        .sort_index(axis=1)
    )


def concat_ohlc_vs_signal(
    data: pd.DataFrame, signal: pd.DataFrame, target_codes: List[str] = None
) -> pd.DataFrame:
   
    if target_codes is None:
        target_codes: pd.Index = signal.columns

    # 初始化结果列表
    dfs: List[pd.DataFrame] = []
    for code in target_codes:
        # 筛选出目标代码的OHLCV数据
        df: pd.DataFrame = data.query("code==@code")[
            ["code", "open", "high", "low", "close", "volume"]
        ]
        # 对齐信号数据和OHLCV数据
        signal_df, df = signal.align(df, join="left", axis=0)
        # 合并信号数据和OHLCV数据
        ohlcvs: pd.DataFrame = pd.concat(
            (df, signal_df[code].to_frame(name="signal")), axis=1
        )
        # 添加到结果列表
        dfs.append(ohlcvs)
    return pd.concat(dfs)


def calc_signal_bins_corr(
    signal_and_forward_return: pd.DataFrame, step: float = 0.01, threshold: int = None
) -> float:
    
    def _calc_corr(df) -> float:
        df: pd.DataFrame = df.droplevel(0, axis=1)
        test_ser: pd.Series = calculate_bin_means(df, step=step)
        if not threshold:

            test_ser: pd.Series = test_ser.query("count>5")["mean"]
        else:
            test_ser: pd.Series = test_ser["mean"]

        test_ser.index = test_ser.index.map(lambda x: x.mid)
        test_ser: pd.DataFrame = test_ser.to_frame(name="returns").reset_index()
        return test_ser["signal"].corr(test_ser["returns"])

    return signal_and_forward_return.groupby(axis=1, level=0).apply(_calc_corr)
