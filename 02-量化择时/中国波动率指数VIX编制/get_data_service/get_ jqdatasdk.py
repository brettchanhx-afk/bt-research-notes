import sys

sys.path.append('..')

import datetime as dt
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from jqdata import *
from source.calc_tools import YEARS
from sqlalchemy.sql import func

from .get_data_toolkits import trans_ser2datetime


def get_opt_basic(code: str, start_date: str, end_date: str) -> pd.DataFrame:
   
    opt_basic: pd.DataFrame = opt.run_query(
        query(opt.OPT_CONTRACT_INFO.list_date,
              opt.OPT_CONTRACT_INFO.exercise_date,
              opt.OPT_CONTRACT_INFO.exercise_price,
              opt.OPT_CONTRACT_INFO.contract_type,
              opt.OPT_CONTRACT_INFO.code).filter(
                  opt.OPT_CONTRACT_INFO.underlying_symbol == code,
                  opt.OPT_CONTRACT_INFO.last_trade_date >= start_date,
                  opt.OPT_CONTRACT_INFO.list_date <= end_date))

    return opt_basic


def offset_limit_func(model, fields: Union[List, Tuple], limit: int,
                      *args) -> pd.DataFrame:
    
    total_size: int = model.run_query(query(
        func.count('*')).filter(*args)).iloc[0, 0]
    # print('总数%s' % total_size)
    dfs: List = []

    # 以limit为步长循环offset的参数
    for i in range(0, total_size, limit):

        q = query(*fields).filter(*args).offset(i).limit(limit)  # 自第i条数据之后进行获取
        df: pd.DataFrame = model.run_query(q)
        # print(i, len(df))
        dfs.append(df)

    df: pd.DataFrame = pd.concat(dfs)

    return df


def get_opt_all_price(codes: Union[str, List]) -> pd.DataFrame:
    
    if isinstance(codes, str):

        codes: List = [codes]

    fields: Tuple = tuple(
        getattr(opt.OPT_DAILY_PRICE, field)
        for field in ('date', 'close', 'code'))
    opt_price: pd.DataFrame = offset_limit_func(
        opt, fields, 4000, opt.OPT_DAILY_PRICE.code.in_(codes))

    return opt_price


def calc_maturity(exercise_date: pd.Series,
                  trade_date: pd.Series,
                  days: int = YEARS) -> pd.Series:
  
    exercise_date = trans_ser2datetime(exercise_date)
    trade_date = trans_ser2datetime(trade_date)

    return (exercise_date - trade_date).dt.days / days


def prepare_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    
    # 获取期权基础信息
    opt_basic: pd.DataFrame = get_opt_basic(code, start_date, end_date)
    # 获取期权标的
    code_list: List = opt_basic['code'].unique().tolist()

    # 获取期权标的的所有日线数据
    opt_all_price: pd.DataFrame = get_opt_all_price(code_list)

    # 合并日线数据与基础信息数据
    opt_data: pd.DataFrame = pd.merge(opt_all_price, opt_basic, on='code')

    # 计算T日至到期日的距离
    opt_data['maturity'] = calc_maturity(opt_data['exercise_date'],
                                         opt_data['date'], 365)

    # 获取所需信息
    sel_col = 'date,exercise_date,close,contract_type,exercise_price,maturity'.split(
        ',')

    data = opt_data[sel_col].copy()

    data['contract_type'] = data['contract_type'].map({
        "CO": "call",
        "PO": "put"
    })

    # 范围选择
    data = data.sort_values('date')
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(start_date)
    end_dade = pd.to_datetime(end_date)

    return data.query('date >= @start_date and date <= @end_date')


