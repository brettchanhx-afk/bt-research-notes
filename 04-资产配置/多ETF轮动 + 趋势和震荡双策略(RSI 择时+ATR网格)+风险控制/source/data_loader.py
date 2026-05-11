import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from source.config import DATA_DIR, ETF_POOL, BOND_ETF


class DataLoader:
    def __init__(self):
        self.cache = {}
        self.max_retries = 3
        self.retry_delay = 2

    def _retry_get(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def get_etf_history(self, etf_code, start_date, end_date, adjust='qfq'):
        try:
            import akshare as ak
            df = ak.fund_etf_hist_em(symbol=etf_code, period="daily",
                                      start_date=start_date.replace('-', ''),
                                      end_date=end_date.replace('-', ''),
                                      adjust=adjust)
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'money'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"获取ETF {etf_code} 历史数据失败: {e}")
            return None

    def get_etf_net_value(self, etf_code, start_date, end_date):
        try:
            import akshare as ak
            df = ak.fund_etf_hist_sina(symbol=f"sh{etf_code}" if etf_code.startswith('5') else f"sz{etf_code}")
            if df is not None and not df.empty:
                df = df.rename(columns={'date': 'date', 'net_value': 'net_value'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
                df = df.sort_index()
            return df
        except Exception as e:
            print(f"获取ETF {etf_code} 净值数据失败: {e}")
            return None

    def get_index_history(self, index_code, start_date, end_date):
        try:
            import akshare as ak
            df = ak.index_zh_a_hist(symbol=index_code, period="daily",
                                    start_date=start_date.replace('-', ''),
                                    end_date=end_date.replace('-', ''))
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'money'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"获取指数 {index_code} 历史数据失败: {e}")
            return None

    def load_all_data(self, etf_codes, start_date, end_date, use_cache=True):
        all_data = {}
        cache_file = os.path.join(DATA_DIR, 'etf_data_cache.h5')

        if use_cache and os.path.exists(cache_file):
            try:
                store = pd.HDFStore(cache_file)
                for code in etf_codes:
                    key = f'etf_{code}'
                    if key in store:
                        all_data[code] = store[key]
                store.close()
                print(f"从缓存加载了 {len(all_data)} 个ETF数据")
            except Exception as e:
                print(f"读取缓存失败: {e}")

        codes_to_load = [code for code in etf_codes if code not in all_data]

        if codes_to_load:
            for idx, code in enumerate(codes_to_load):
                print(f"正在加载 {code} ({idx+1}/{len(codes_to_load)})")
                df = self._retry_get(self.get_etf_history, code, start_date, end_date)
                if df is not None and not df.empty:
                    all_data[code] = df
                    print(f"  成功: {code}, 数据量 {len(df)} 条")
                else:
                    print(f"  失败: {code}")

            if use_cache:
                try:
                    store = pd.HDFStore(cache_file)
                    for code in codes_to_load:
                        if code in all_data:
                            store[f'etf_{code}'] = all_data[code]
                    store.close()
                    print(f"缓存已更新")
                except Exception as e:
                    print(f"写入缓存失败: {e}")

        return all_data

    def get_trading_dates(self, start_date, end_date):
        df = self.get_index_history('000001', start_date, end_date)
        if df is not None:
            return df.index.tolist()
        return []

    @staticmethod
    def validate_etf_data(etf_data, min_list_days=300, min_avg_money=2000000):
        valid_codes = {}
        for code, df in etf_data.items():
            if df is None or len(df) < min_list_days:
                continue
            avg_money = df['money'].tail(120).mean() if len(df) >= 120 else df['money'].mean()
            if avg_money >= min_avg_money:
                valid_codes[code] = True
        return valid_codes
