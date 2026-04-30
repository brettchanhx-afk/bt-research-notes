from typing import Dict, List
import pandas as pd
import pyfolio as pf
from .toolkits import get_strategy_return, trans_minute_to_daily, check_index_tz


__all__ = [
    "show_perf_stats",
    "multi_asset_show_perf_stats",
    "multi_strategy_show_perf_stats",
]


def show_perf_stats(strat, minutes_close: pd.Series = None, return_df: bool = False):
    
    returns: pd.Series = get_strategy_return(strat)

    if minutes_close is not None:
        benchmark_rets: pd.Series = (
            trans_minute_to_daily(minutes_close).pct_change()
            if minutes_close is not None
            else None
        )

        benchmark_rets: pd.DataFrame = check_index_tz(benchmark_rets)
    else:
        benchmark_rets = None
        
    returns: pd.DataFrame = check_index_tz(returns)

    return pf.plotting.show_perf_stats(
        returns, factor_returns=benchmark_rets, return_df=return_df
    )


def multi_asset_show_perf_stats(
    strats: Dict, minutes_close: pd.DataFrame = None
) -> pd.DataFrame:
    
    dfs: List[pd.DataFrame] = []

    for code, strat in strats.items():
        strat_minutes_close: pd.Series = (
            minutes_close.query("code == @code")["close"]
            if minutes_close is not None
            else None
        )
        dfs.append(show_perf_stats(strat, strat_minutes_close, return_df=True))

    return pd.concat(dfs, keys=list(strats.keys()), axis=1)


def multi_strategy_show_perf_stats(
    strats: Dict, minutes_close: pd.DataFrame = None
) -> pd.DataFrame:
    
    dfs: List[pd.DataFrame] = []

    for strategy_name, strat in strats.items():
        code: str = strat.data._name
        strat_minutes_close: pd.Series = (
            minutes_close.query("code == @code")["close"]
            if minutes_close is not None
            else None
        )
        dfs.append(show_perf_stats(strat, strat_minutes_close, return_df=True))

    return pd.concat(dfs, keys=list(strats.keys()), axis=1)
