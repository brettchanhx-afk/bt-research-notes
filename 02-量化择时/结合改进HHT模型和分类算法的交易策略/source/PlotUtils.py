import numpy as np

import pandas as pd
import backtrader as bt
import empyrical as ep
import pyfolio as pf
from typing import Union

__all__ = [
    "get_strategy_return",
    "get_strategy_cumulative_return",
    "trans_minute_to_daily",
    "check_index_tz",
    "calculate_bin_means"
]


def get_strategy_return(
    strat: bt.Cerebro,
) -> pd.Series:
    
    return pd.Series(strat.analyzers.getbyname("time_return").get_analysis())


def trans_minute_to_daily(minutes_close: pd.Series) -> pd.Series:
    
    return minutes_close.resample("D").last().dropna()


def get_strategy_cumulative_return(
    strat: bt.Cerebro,
    starting_value: int = 0,
) -> pd.Series:
    
    return ep.cum_returns(get_strategy_return(strat), starting_value=starting_value)


def check_index_tz(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

def calculate_bin_means(
    df: pd.DataFrame,
    signal_col: str = "signal",
    forward_returns_col: str = "forward_returns",
    step: float = 0.01,
) -> pd.DataFrame:
   
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df 必须是一个 DataFrame")

    signal_ser: pd.Series = df[signal_col]
    bins: np.ndarray = np.arange(signal_ser.min(), signal_ser.max(), step)

    return df.groupby(pd.cut(signal_ser, bins),observed=True)[forward_returns_col].agg(
        ["mean", "count"]
    )

