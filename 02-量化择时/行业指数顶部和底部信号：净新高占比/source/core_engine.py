####
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def find_stage_stock(
    pivot_price: pd.DataFrame,
    window: int,
    method: str = "high",
    offset: int = None,s
) -> pd.DataFrame:
    
    oper: str = {"high": "ge", "low": "le"}[method]
    method: str = {"high": "max", "low": "min"}[method]

    roll_: pd.DataFrame = pivot_price.rolling(window)

    roll_df: pd.DataFrame = getattr(roll_, method)()

    if offset is not None:

        roll_df: pd.DataFrame = roll_df.shift(offset)

    return getattr(pivot_price, oper)(roll_df)


def get_ind_stage_num(pivot_num: pd.DataFrame, sw_cons_dict: Dict) -> pd.DataFrame:
    
    industry_num: pd.DataFrame = pivot_num.copy()
    industry_num.columns = industry_num.columns.map(sw_cons_dict)

    return industry_num.groupby(level=0, axis=1).sum()


def calc_industry_nhnl(
    pivot_price: pd.DataFrame,
    sw_cons_dict: Dict,
    window: int,
    classify_num: pd.DataFrame = None,
    tradition: bool = True,
) -> pd.DataFrame:
    
    if classify_num is None:
        classify_num: pd.Series = pd.Series(Counter(tuple(sw_cons_dict.values())))

    h_field: str = "high"
    l_field: str = "low"
    if tradition:
        h_field, l_field = "close", "close"

    high_num: pd.DataFrame = find_stage_stock(pivot_price[h_field], window, "high", 5)
    low_num: pd.DataFrame = find_stage_stock(pivot_price[l_field], window, "low", 5)

    ind_high: pd.DataFrame = get_ind_stage_num(high_num, sw_cons_dict)
    ind_low: pd.DataFrame = get_ind_stage_num(low_num, sw_cons_dict)

    return (ind_high - ind_low).div(classify_num)


def plot_nhnl_signal(
    price: pd.Series,
    siganl: pd.Series,
    cons_num: int = None,
    title: str = "",
    align: bool = False,
) -> go.Figure:
    
    fig = go.Figure()

    THRESHOLD: Dict = {
        "normal": {"贪婪": 0.3, "乐观": 0.2, "悲观": -0.2, "恐惧": -0.3},
        "other": {"贪婪": 0.4, "乐观": 0.3, "悲观": -0.3, "恐惧": -0.4},
    }

    COLOR: Dict = {
        "贪婪": {"color": "LightSeaGreen"},
        "乐观": {"color": "LightSeaGreen", "dash": "dashdot"},
        "悲观": {"color": "Crimson", "dash": "dashdot"},
        "恐惧": {"color": "Crimson"},
    }

    if align:
        siganl, price = siganl.align(price, join="inner")

    price_ax = go.Scatter(
        x=price.index,
        y=price.values,
        line=dict(color="darkgray"),
        name="close",
    )
    nhnl_ax = go.Scatter(
        x=siganl.index,
        y=siganl.values,
        line=dict(color="DarkSalmon"),
        name="NH-NL",
        yaxis="y2",
    )

    fig.add_trace(price_ax)
    fig.add_trace(nhnl_ax)

    method: str = "normal" if (cons_num > 40 or cons_num is None) else "other"
    threshold_range: Dict = THRESHOLD[method]

    for name, value in threshold_range.items():

        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=np.ones(len(price)) * value,
                line=COLOR[name],
                name=name,
                yaxis="y2",
            )
        )

    fig.update_layout(
        hovermode="x unified",
        yaxis2=dict(
            title="NHNL",
            overlaying="y",
            side="right",
        ),
        title={"text": title},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
