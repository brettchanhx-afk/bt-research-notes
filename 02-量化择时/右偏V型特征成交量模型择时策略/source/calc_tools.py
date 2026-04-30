import numpy as np
import pandas as pd
import talib


def create_signal(data: pd.DataFrame, fast_window: int = 5, slow_window: int = 100, start_dt: str = None, end_dt: str = None, threshold: float = 1.15, a: float = 1.5) -> pd.DataFrame:
    df = data.copy()
    
    df['volume_index']: pd.Series = HMA(
        df['volume'], fast_window) / HMA(df['volume'], slow_window)
    df['forward_returns'] = df['close'].pct_change(5).shift(-5)
    df['threshold_to_long_a'] = threshold
    df['threshold_to_long_b'] = np.power(threshold, -a)
    df['threshold_to_short'] = 1
    return df if (start_dt is None) and (end_dt is None) else df.loc[start_dt:end_dt]


# 构造HMA
def HMA(price: pd.Series, window: int) -> pd.Series:
    
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    return talib.WMA(2 * talib.WMA(price, int(window * 0.5)) - talib.WMA(price, window), int(np.sqrt(window)))
