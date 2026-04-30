import pandas as pd
import numpy as np


def trans_ser2datetime(ser: pd.Series) -> pd.Series:
   
    if not isinstance(ser, pd.Series):

        raise TypeError('ser必须为pd.Series')

    if (ser.dtype != np.dtype('O')) or (ser.dtype != np.dtype('<M8[ns]')):

        ser = pd.to_datetime(ser)

    return ser