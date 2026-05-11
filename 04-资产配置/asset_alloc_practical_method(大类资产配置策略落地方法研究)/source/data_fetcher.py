import pandas as pd
import numpy as np
import tushare as ts
import efinance as ef
import baostock as bs
from datetime import datetime
import sys
sys.path.insert(0, '.')
from config import TUSHARE_TOKEN, TUSHARE_URL, DATA_DIR
import os

def init_tushare():
    pro = ts.pro_api(TUSHARE_TOKEN)
    pro._DataApi__token = TUSHARE_TOKEN
    pro._DataApi__http_url = TUSHARE_URL
    return pro

def init_baostock():
    bs.login()
    return bs

def get_index_daily(symbol, start_date, end_date):
    try:
        df = ef.stock.get_quote_history(symbol, start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df[['日期', '收盘']]
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df['return'] = df['close'].pct_change().fillna(0)
            return df
    except Exception as e:
        print(f"efinance获取指数数据失败: {e}")
    
    try:
        pro = init_tushare()
        df = pro.index_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if not df.empty:
            df = df[['trade_date', 'close']]
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)
            df['return'] = df['close'].pct_change().fillna(0)
            return df
    except Exception as e:
        print(f"tushare获取指数数据失败: {e}")
    
    return None

def get_fund_daily(fund_code, start_date, end_date):
    try:
        df = ef.fund.get_quote_history(fund_code, start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df[['日期', '单位净值']]
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df['return'] = df['close'].pct_change().fillna(0)
            return df
    except Exception as e:
        print(f"efinance获取基金数据失败: {e}")
    
    return None

def get_stock_daily(symbol, start_date, end_date):
    try:
        df = ef.stock.get_quote_history(symbol, start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df[['日期', '收盘']]
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df['return'] = df['close'].pct_change().fillna(0)
            return df
    except Exception as e:
        print(f"efinance获取股票数据失败: {e}")
    
    try:
        pro = init_tushare()
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if not df.empty:
            df = df[['trade_date', 'close']]
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)
            df['return'] = df['close'].pct_change().fillna(0)
            return df
    except Exception as e:
        print(f"tushare获取股票数据失败: {e}")
    
    return None

def get_fund_list_by_type(fund_type='index'):
    fund_list = []
    try:
        pro = init_tushare()
        df = pro.fund_basic(market='E')
        if fund_type == 'index':
            df = df[df['fund_type'] == 'ETF']
        fund_list = df[['ts_code', 'name']].to_dict('records')
    except Exception as e:
        print(f"获取基金列表失败: {e}")
    return fund_list

def get_macro_data(start_date, end_date):
    data = []
    try:
        pro = init_tushare()
        df = pro.cn_m(start_date=start_date, end_date=end_date)
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            data = df.to_dict('records')
    except Exception as e:
        print(f"获取宏观数据失败: {e}")
    return data

def save_data(df, filename):
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"数据已保存到: {filepath}")

def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, encoding='utf-8')
    return None