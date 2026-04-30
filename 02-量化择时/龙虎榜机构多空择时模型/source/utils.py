import datetime as dt
from typing import Dict, Union

import pandas as pd
from dateutil.parser import parse
from IPython.display import display


def trans2strftime(ser:pd.Series,fmt:str='%Y-%m-%d')->pd.Series:
    
    return pd.to_datetime(ser).dt.strftime(fmt)

def format_dt(dt: Union[dt.datetime, dt.date, str],
              fm: str = '%Y-%m-%d') -> str:
    
    if isinstance(dt, str):

        return parse(dt).strftime(fm)

    else:

        return dt.strftime(fm)


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
