import pandas as pd
from IPython.display import display


def load_csv(path: str) -> pd.DataFrame:
    
    price = pd.read_csv(path, index_col=[0], parse_dates=['time'])
    price.rename(columns={'time': 'datetime'}, inplace=True)
    price['openinterest'] = 0

    return price


def print_table(table:pd.DataFrame, name:str=None, fmt:str=None):
    
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
