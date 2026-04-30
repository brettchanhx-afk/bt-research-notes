# -*- coding: utf-8 -*-
"""
数据获取模块

功能：
  - 获取债券基金净值数据
  - 获取债券指数数据（国债、信用债、可转债）
  - 构建风险因子收益序列
"""
import os
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TUSHARE_TOKEN, TUSHARE_API_URL, INDEX_CODES


# ============================================================
# Tushare初始化
# ============================================================
def init_tushare():
    """初始化Tushare API"""
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    pro._DataApi__token = TUSHARE_TOKEN
    pro._DataApi__http_url = TUSHARE_API_URL
    return pro


# ============================================================
# 1. 基金净值数据
# ============================================================
def get_fund_nav(
    fund_code: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取基金净值历史数据。
    
    Parameters
    ----------
    fund_code : str
        基金代码
    start_date, end_date : str
        开始/结束日期（YYYY-MM-DD）
    
    Returns
    -------
    pd.DataFrame
        净值数据，包含：
        - date: 日期
        - nav: 单位净值
        - nav_acc: 累计净值
        - daily_return: 日收益率
    """
    # 尝试efinance
    try:
        import efinance as ef
        df = ef.fund.get_quote_history(fund_code, start=start_date, end=end_date)
        
        if df is not None and len(df) > 0:
            df.columns = [str(c).strip() for c in df.columns]
            
            # 列名适配
            date_col = df.columns[0]
            nav_col = df.columns[1]
            
            result = pd.DataFrame({
                'date': pd.to_datetime(df[date_col]),
                'nav': pd.to_numeric(df[nav_col], errors='coerce'),
            })
            result = result.set_index('date').sort_index()
            result['daily_return'] = result['nav'].pct_change()
            
            print(f'  [efinance] 基金{fund_code}净值 {len(result)} 条')
            return result
            
    except Exception as e:
        print(f'  [efinance] 获取净值失败: {e}')
    
    # 备选：tushare
    try:
        pro = init_tushare()
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        
        df = pro.fund_nav(ts_code=f'{fund_code}.OF', start_date=start, end_date=end)
        
        if df is not None and len(df) > 0:
            result = pd.DataFrame({
                'date': pd.to_datetime(df['end_date']),
                'nav': df['unit_nav'].astype(float),
                'nav_acc': df['acc_nav'].astype(float),
            })
            result = result.set_index('date').sort_index()
            result['daily_return'] = result['nav'].pct_change()
            
            print(f'  [tushare] 基金{fund_code}净值 {len(result)} 条')
            return result
            
    except Exception as e:
        print(f'  [tushare] 获取净值失败: {e}')
    
    return pd.DataFrame()


# ============================================================
# 2. 指数数据
# ============================================================
def get_index_data(
    index_code: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取指数历史数据。
    
    Parameters
    ----------
    index_code : str
        指数代码
    start_date, end_date : str
        日期范围
    
    Returns
    -------
    pd.DataFrame
        指数数据，包含日期、收盘价、日收益率
    """
    # 尝试akshare
    try:
        import akshare as ak
        
        # 指数历史行情
        df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
        
        if df is not None and len(df) > 0:
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # 筛选日期范围
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]
            
            df['daily_return'] = df['close'].pct_change()
            
            print(f'  [akshare] 指数{index_code} {len(df)} 条')
            return df[['close', 'daily_return']]
            
    except Exception as e:
        print(f'  [akshare] 指数{index_code}获取失败: {e}')
    
    # 备选：tushare
    try:
        pro = init_tushare()
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        
        df = pro.index_daily(ts_code=f'{index_code}.SH', start_date=start, end_date=end)
        
        if df is not None and len(df) > 0:
            result = pd.DataFrame({
                'date': pd.to_datetime(df['trade_date']),
                'close': df['close'].astype(float),
            })
            result = result.set_index('date').sort_index()
            result['daily_return'] = result['close'].pct_change()
            
            print(f'  [tushare] 指数{index_code} {len(result)} 条')
            return result
            
    except Exception as e:
        print(f'  [tushare] 指数获取失败: {e}')
    
    return pd.DataFrame()


def get_bond_index_data(
    index_type: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取债券指数数据。
    
    Parameters
    ----------
    index_type : str
        指数类型：'treasury', 'corporate_bond', 'convertible', 'cdb'
    start_date, end_date : str
        日期范围
    
    Returns
    -------
    pd.DataFrame
        指数数据
    """
    index_code = INDEX_CODES.get(index_type)
    
    if index_code is None:
        print(f'  [ERROR] 未知的指数类型: {index_type}')
        return pd.DataFrame()
    
    return get_index_data(index_code, start_date, end_date)


# ============================================================
# 3. 批量获取基金数据
# ============================================================
def get_multiple_fund_nav(
    fund_codes: list,
    start_date: str,
    end_date: str
) -> dict:
    """批量获取多只基金净值数据。
    
    Parameters
    ----------
    fund_codes : list
        基金代码列表
    start_date, end_date : str
        日期范围
    
    Returns
    -------
    dict
        {fund_code: DataFrame}
    """
    results = {}
    
    for code in fund_codes:
        try:
            df = get_fund_nav(code, start_date, end_date)
            if len(df) > 0:
                results[code] = df
            time.sleep(0.2)
        except Exception as e:
            print(f'  [ERROR] {code}: {e}')
    
    print(f'\n  成功获取 {len(results)}/{len(fund_codes)} 只基金数据')
    return results


# ============================================================
# 4. 可转债指数数据
# ============================================================
def get_convertible_bond_index(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取可转债指数数据。
    
    中证转债指数（000832）
    """
    try:
        import akshare as ak
        
        # 可转债指数
        df = ak.index_zh_a_hist(symbol="000832", period="daily")
        
        if df is not None and len(df) > 0:
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]
            
            df['daily_return'] = df['close'].pct_change()
            
            print(f'  [akshare] 可转债指数 {len(df)} 条')
            return df[['close', 'daily_return']]
            
    except Exception as e:
        print(f'  [akshare] 可转债指数获取失败: {e}')
    
    # 备选：使用通用指数接口
    return get_bond_index_data('convertible', start_date, end_date)
