import datetime as dt
from typing import Dict, Union

import pandas as pd
from dateutil.parser import parse
from IPython.display import display


def trans2strftime(ser: pd.Series, fmt: str = '%Y-%m-%d') -> pd.Series:

    return pd.to_datetime(ser).dt.strftime(fmt)


def format_dt(dt: Union[dt.datetime, dt.date, str],
              fm: str = '%Y-%m-%d') -> str:
    
    return parse(dt).strftime(fm) if isinstance(dt, str) else dt.strftime(fm)


def print_table(table: pd.DataFrame, name: str = None, fmt: str = None):
   
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def get_value_frome_traderanalyzerdict(dic: Dict, *args) -> float:
    
    if len(args) == 1:
        return dic.get(args[0], 0)
    for k in args:

        if res := dic.get(k, None):
            return get_value_frome_traderanalyzerdict(res, *args[1:])

        return 0
