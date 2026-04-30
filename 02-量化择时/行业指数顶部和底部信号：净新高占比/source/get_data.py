###
import json
from typing import Dict

import pandas as pd

__all__ = [
    "sw_cons_name",
    "price",
    "pivot_table",
    "ind_price",
    "pivot_ind_price",
    "classify_num",
]

with open("data/sw_cons_name.json", "r") as f:

    sw_cons_name: Dict = json.load(f)

price: pd.DataFrame = pd.read_csv("data/data.csv", index_col=[0], parse_dates=[0])

pivot_table: pd.DataFrame = pd.pivot_table(
    price, index="trade_date", columns="code", values=["low", "high"]
)
pivot_table.index = pd.to_datetime(pivot_table.index)

ind_price: pd.DataFrame = pd.read_csv(
    "data/industry_price.csv", index_col=[0], parse_dates=[0]
)

pivot_ind_price: pd.DataFrame = pd.pivot_table(
    ind_price, index="trade_date", columns="name", values="close"
)
pivot_ind_price.index = pd.to_datetime(pivot_ind_price.index)

classify_num: pd.DataFrame = pd.read_csv(
    "data/industry_classify_num.csv", index_col=[0], parse_dates=[0]
)
