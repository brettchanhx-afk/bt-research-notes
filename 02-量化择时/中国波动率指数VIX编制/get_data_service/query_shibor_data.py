import sys

sys.path.append('..')

import functools
import json
import time

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d
from source.calc_tools import YEARS


@functools.lru_cache()
def query_china_shibor_all() -> pd.DataFrame:
    
    t = time.time()
    params = {"_": t}
    res = requests.get("https://cdn.jin10.com/data_center/reports/il_1.json",
                       params=params)
    json_data = res.json()
    temp_df = pd.DataFrame(json_data["values"]).T
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df = temp_df.applymap(lambda x: x[0])
    temp_df = temp_df.astype(float)
    return temp_df


@functools.lru_cache()
def _load_csv() -> pd.DataFrame:
   
    return pd.read_csv(r'data_service/shibor_data/shibor_db.csv',
                       index_col=[0],
                       parse_dates=True,
                       usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])


def get_shibor_data(start: str, end: str) -> pd.DataFrame:
    
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    df2 = query_china_shibor_all()  # 爬虫数据
    df1 = _load_csv()  # 用于补充数据

    df = pd.concat((df1, df2), sort=True).sort_index()
    df = df.loc[~df.index.duplicated(keep='last')]

    return df.loc[start:end]


def get_interpld_shibor(shibor_df: pd.DataFrame) -> pd.DataFrame:
   
    def _interpld_fun(r):
       
        y_vals = r.values / 100

        daily_range = np.arange(1, YEARS)
        periods = [1, 7, 14, 30, 90, 180, 270, 365]

        # 插值三次样条插值法补全利率曲线
        f = interp1d(periods, y_vals, kind='cubic')
        t_ser = pd.Series(data=f(daily_range), index=daily_range)

        return t_ser

    shibor_df = shibor_df.apply(lambda x: _interpld_fun(x), axis=1)

    shibor_df.index = pd.DatetimeIndex(shibor_df.index)

    return shibor_df




