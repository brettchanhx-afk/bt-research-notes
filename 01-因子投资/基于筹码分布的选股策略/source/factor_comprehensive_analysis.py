from alphalens.utils import quantize_factor
import pandas as pd
from typing import Dict,List

def clean_factor_data(factor_data: pd.DataFrame) -> pd.DataFrame:
    
    clean_factor: pd.DataFrame = factor_data.copy()
    if isinstance(clean_factor.columns,pd.MultiIndex):
        clean_factor.columns = clean_factor.columns.droplevel(0)
        
    clean_factor.index.names = ["date", "assert"]

    return clean_factor


def get_factor_group_returns(
    clean_factor: pd.DataFrame, quantile: int, no_raise: bool = False
) -> pd.DataFrame:
    
    sel_cols: List = [col for col in clean_factor.columns.tolist() if col != "next_ret"]

    returns_dict: Dict = {}
    for col in sel_cols:
        clean_factor[f"{col}_group"] = quantize_factor(
            clean_factor.rename(columns={col: "factor"})[["factor"]],
            quantiles=quantile,
            no_raise=no_raise,
        )
        returns_dict[col] = pd.pivot_table(
            clean_factor.reset_index(),
            index="date",
            columns=f"{col}_group",
            values="next_ret",
        )

    df: pd.DataFrame = pd.concat(returns_dict, axis=1)
    df.index.names = ["date"]
    df.columns.names = ["factor_name", "group"]
    return df



