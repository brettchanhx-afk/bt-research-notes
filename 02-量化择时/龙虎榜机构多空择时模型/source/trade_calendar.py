import functools

import pandas as pd

from .tushare_client import TuShare
from .utils import format_dt

my_ts = TuShare()


@functools.lru_cache()
def get_all_trade_days() -> pd.DatetimeIndex:
    
    trade_cal: pd.DataFrame = my_ts.trade_cal(exchange='SSE')

    return pd.to_datetime(trade_cal.query('is_open == 1')['cal_date'].tolist())


def create_cal(trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    
    min_date = trade_dates.min()
    max_date = trade_dates.max()

    dates = pd.date_range(min_date, max_date)
    df = pd.DataFrame(index=dates)

    df['is_tradeday'] = False
    df.loc[trade_dates, 'is_tradeday'] = True

    return df


def get_trade_days(start_date: str = None,
                   end_date: str = None,
                   count: int = None) -> pd.DatetimeIndex:
    
    if (count is not None) and (start_date is not None):
        raise ValueError("不能同时指定 start_date 和 count 两个参数")

    if count is not None:
        count = int(-count)
        start_date = Tdaysoffset(end_date, count)

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)

    if isinstance(end_date, str):

        end_date = pd.to_datetime(end_date)

    start_date = format_dt(start_date)
    end_date = format_dt(end_date)
    all_trade_days = get_all_trade_days()
    idx = all_trade_days.slice_indexer(start_date, end_date)

    return all_trade_days[idx]


def Tdaysoffset(watch_date: str,
                count: int,
                freq: str = 'days') -> pd.Timestamp:
   

    if isinstance(watch_date, str):
        watch_date = pd.to_datetime(watch_date)

    all_trade_days = get_all_trade_days()
    cal_frame = create_cal(trade_dates=all_trade_days)

    holiday = cal_frame.query('not is_tradeday').index
    trade_days = pd.offsets.CustomBusinessDay(weekmask='1' * 7,
                                              holidays=holiday)
    # None时为Days
    if freq == 'days':

        target = watch_date + trade_days * 0 + trade_days * count

    else:
        # 此处需要验证
        ## TODO：验证
        target = watch_date + pd.DateOffset(**{freq: count}) + trade_days

    return target
