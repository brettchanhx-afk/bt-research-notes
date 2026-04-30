import datetime as dt
from typing import List, Tuple

import empyrical as ep
import numpy as np
import pandas as pd


def get_max_drawdown_underwater(
        underwater: pd.Series) -> Tuple[dt.datetime, dt.datetime, dt.datetime]:

    valley = underwater.idxmin()  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery


def get_top_drawdowns(returns: pd.Series, top: int = 10) -> List:

    returns = returns.copy()
    df_cum = ep.cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak:recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if ((len(returns) == 0) or (len(underwater) == 0)
                or (np.min(underwater) == 0)):
            break

    return drawdowns


def gen_drawdown_table(returns: pd.Series, top: int = 10) -> pd.DataFrame:

    df_cum: pd.Series = ep.cum_returns(returns, 1.0)
    drawdown_periods: List = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(
        index=list(range(top)),
        columns=[
            "Net drawdown in %",
            "Peak date",
            "Valley date",
            "Recovery date",
            "Valley Duration",
            "End Duration",
            "Duration",
        ],
    )

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):

        df_drawdowns.loc[i, "Valley Duration"] = len(
            pd.date_range(peak, valley, freq="B")
        )
        if pd.isnull(recovery):

            df_drawdowns.loc[i, "End Duration"] = np.nan
            df_drawdowns.loc[i, "Duration"] = np.nan

        else:
            df_drawdowns.loc[i, "End Duration"] = len(
                pd.date_range(valley, recovery, freq="B")
            )

            df_drawdowns.loc[i, "Duration"] = len(
                pd.date_range(peak, recovery, freq="B")
            )
        df_drawdowns.loc[i, "Peak date"] = peak.to_pydatetime().strftime(
            "%Y-%m-%d")
        df_drawdowns.loc[i, "Valley date"] = valley.to_pydatetime().strftime(
            "%Y-%m-%d")
        if isinstance(recovery, float):
            df_drawdowns.loc[i, "Recovery date"] = recovery
        else:
            df_drawdowns.loc[i, "Recovery date"] = recovery.to_pydatetime().strftime(
                "%Y-%m-%d"
            )
        df_drawdowns.loc[i, "Net drawdown in %"] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]
        ) * 100

    df_drawdowns["Peak date"] = pd.to_datetime(df_drawdowns["Peak date"])
    df_drawdowns["Valley date"] = pd.to_datetime(df_drawdowns["Valley date"])
    df_drawdowns["Recovery date"] = pd.to_datetime(
        df_drawdowns["Recovery date"])

    return df_drawdowns
