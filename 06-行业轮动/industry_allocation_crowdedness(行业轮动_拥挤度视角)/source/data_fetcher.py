import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

import tushare as ts
token = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = "http://jiaoch.site"

TUSHARE_AVAILABLE = True

SW_INDUSTRY_CODES = [
    '801010', '801020', '801030', '801040', '801050', '801060', '801070', '801080',
    '801090', '801100', '801110', '801120', '801130', '801140', '801150', '801160',
    '801170', '801180', '801190', '801200', '801210', '801220', '801230', '801710',
    '801720', '801730', '801880', '801890'
]

SW_INDUSTRY_NAMES = {
    '801010': '农林牧渔', '801020': '采掘', '801030': '化工', '801040': '钢铁',
    '801050': '有色金属', '801060': '电子', '801070': '汽车', '801080': '家用电器',
    '801090': '食品饮料', '801100': '纺织服装', '801110': '轻工制造', '801120': '医药生物',
    '801130': '公用事业', '801140': '交通运输', '801150': '房地产', '801160': '商业贸易',
    '801170': '休闲服务', '801180': '银行', '801190': '非银金融', '801200': '建筑装饰',
    '801210': '电气设备', '801220': '国防军工', '801230': '计算机', '801710': '传媒',
    '801720': '通信', '801730': '机械设备', '801880': '家用电器', '801890': '机械设备'
}

TS_CODE_MAP = {
    '农林牧渔': '801010.SI', '采掘': '801020.SI', '化工': '801030.SI', '钢铁': '801040.SI',
    '有色金属': '801050.SI', '电子': '801060.SI', '汽车': '801070.SI', '家用电器': '801080.SI',
    '食品饮料': '801090.SI', '纺织服装': '801100.SI', '轻工制造': '801110.SI', '医药生物': '801120.SI',
    '公用事业': '801130.SI', '交通运输': '801140.SI', '房地产': '801150.SI', '商业贸易': '801160.SI',
    '休闲服务': '801170.SI', '银行': '801180.SI', '非银金融': '801190.SI', '建筑装饰': '801200.SI',
    '电气设备': '801210.SI', '国防军工': '801220.SI', '计算机': '801230.SI', '传媒': '801710.SI',
    '通信': '801720.SI', '机械设备': '801730.SI'
}

def get_industry_index_data_tushare(ts_code, start_date='20180101', end_date='20231231'):
    if not TUSHARE_AVAILABLE:
        return None
    try:
        df = pro.sw_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.set_index('trade_date', inplace=True)
            df.sort_index(inplace=True)
            df.rename(columns={
                'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low',
                'vol': 'volume', 'amount': 'amount'
            }, inplace=True)
            return df
    except Exception as e:
        print(f"  {ts_code} Error: {e}")
    return None

def get_market_index_data_tushare(index_code='000001.SH', start_date='20180101', end_date='20231231'):
    if not TUSHARE_AVAILABLE:
        return None
    try:
        df = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.set_index('trade_date', inplace=True)
            df.sort_index(inplace=True)
            df.rename(columns={
                'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low',
                'vol': 'volume', 'amount': 'amount'
            }, inplace=True)
            return df
    except Exception as e:
        print(f"  Market index {index_code} Error: {e}")
    return None

def get_all_industries_data(start_date='2018-01-01', end_date='2023-12-31', save=True):
    all_data = {}
    start_str = start_date.replace('-', '')
    end_str = end_date.replace('-', '')
    print(f"Fetching SW Industry Index data using tushare...")
    unique_industries = list(set(SW_INDUSTRY_NAMES.values()))
    for i, name in enumerate(unique_industries):
        ts_code = TS_CODE_MAP.get(name)
        if ts_code is None:
            print(f"[{i+1}/{len(unique_industries)}] {name} - No ts_code mapping")
            continue
        try:
            df = get_industry_index_data_tushare(ts_code, start_str, end_str)
            if df is not None and len(df) > 0:
                all_data[name] = df
                print(f"[{i+1}/{len(unique_industries)}] {name} ({ts_code}) Success ({len(df)} records)")
            else:
                print(f"[{i+1}/{len(unique_industries)}] {name} ({ts_code}) No data")
            time.sleep(0.3)
        except Exception as e:
            print(f"[{i+1}/{len(unique_industries)}] {name} ({ts_code}) Failed: {e}")
            continue
    if save and all_data:
        output_path = os.path.join(DATA_DIR, 'industry_data.pkl')
        pd.to_pickle(all_data, output_path)
        print(f"Data saved to {output_path}")
    return all_data

def load_cached_data():
    cache_path = os.path.join(DATA_DIR, 'industry_data.pkl')
    if os.path.exists(cache_path):
        print("Loading cached data...")
        return pd.read_pickle(cache_path)
    return None

def get_market_index_data(index_code='000001.SH', start_date='2018-01-01', end_date='2023-12-31'):
    start_str = start_date.replace('-', '')
    end_str = end_date.replace('-', '')
    df = get_market_index_data_tushare(index_code, start_str, end_str)
    return df

def calculate_returns(price_data, periods=[1, 5, 10, 20, 40]):
    returns = pd.DataFrame(index=price_data.index)
    for p in periods:
        returns[f'ret_{p}d'] = price_data['close'].pct_change(p)
    returns['ret_1d'] = price_data['close'].pct_change()
    return returns

def calculate_turnover(volume_data, amount_data):
    turnover = pd.DataFrame(index=volume_data.index)
    if 'vol' in volume_data.columns and 'amount' in amount_data.columns:
        turnover['turnover'] = amount_data['amount'] / volume_data['vol'] * 100
    elif 'volume' in volume_data.columns and 'amount' in amount_data.columns:
        turnover['turnover'] = amount_data['amount'] / volume_data['volume'] * 100
    if 'turnover' in turnover.columns:
        turnover['turn_ma5'] = turnover['turnover'].rolling(5).mean()
        turnover['turn_ma10'] = turnover['turnover'].rolling(10).mean()
        turnover['turn_ma20'] = turnover['turnover'].rolling(20).mean()
        turnover['turn_ma40'] = turnover['turnover'].rolling(40).mean()
    return turnover

if __name__ == "__main__":
    print("Testing tushare data fetching...")
    test_df = get_industry_index_data_tushare('801180.SI', '20180101', '20180630')
    if test_df is not None:
        print(f"Test data success, {len(test_df)} records")
        print(test_df.tail())
    else:
        print("Test data failed")