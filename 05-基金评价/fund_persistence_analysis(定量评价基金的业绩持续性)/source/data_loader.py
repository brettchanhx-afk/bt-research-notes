# -*- coding: utf-8 -*-
"""
数据获取模块
===================================
支持多数据源：efinance > akshare > baostock > mootdx > yfinance
"""

import pandas as pd
import numpy as np
import warnings
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RISK_FREE_RATE, DATA_SOURCE_PRIORITY

warnings.filterwarnings('ignore')


# ============================================================
# 数据源适配器
# ============================================================

def get_fund_nav(fund_code, start_date=None, end_date=None):
    """
    获取基金净值数据
    
    优先级：efinance > akshare > baostock > mootdx > yfinance
    
    Parameters:
    -----------
    fund_code : str
        基金代码
    start_date : str, optional
        开始日期 (YYYY-MM-DD)
    end_date : str, optional
        结束日期 (YYYY-MM-DD)
        
    Returns:
    --------
    DataFrame : 包含 date, nav, dividend, split_ratio 等列
    """
    errors = []
    
    for source in DATA_SOURCE_PRIORITY:
        try:
            if source == "efinance":
                df = _get_fund_nav_efinance(fund_code, start_date, end_date)
            elif source == "akshare":
                df = _get_fund_nav_akshare(fund_code, start_date, end_date)
            elif source == "baostock":
                df = _get_fund_nav_baostock(fund_code, start_date, end_date)
            elif source == "mootdx":
                df = _get_fund_nav_mootdx(fund_code, start_date, end_date)
            elif source == "yfinance":
                df = _get_fund_nav_yfinance(fund_code, start_date, end_date)
            else:
                continue
            
            if df is not None and len(df) > 0:
                print(f"  [成功] 使用 {source} 获取基金 {fund_code} 数据，共 {len(df)} 条")
                return df
        except Exception as e:
            errors.append(f"{source}: {str(e)}")
            continue
    
    # 所有数据源都失败
    print(f"  [警告] 所有数据源获取失败:")
    for err in errors:
        print(f"    - {err}")
    return None


def _get_fund_nav_efinance(fund_code, start_date, end_date):
    """使用efinance获取基金净值"""
    try:
        import efinance as ef
        
        # 获取基金净值数据 - 使用get_quote_history
        df = ef.fund.get_quote_history(fund_code)
        
        if df is None or len(df) == 0:
            return None
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '单位净值': 'nav',
            '累计净值': 'cum_nav',
            '日涨跌幅': 'daily_return'
        })
        
        # 转换日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # 筛选日期范围
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
        
        # 确保有nav列
        if 'nav' not in df.columns:
            return None
        
        df = df.sort_values('date').reset_index(drop=True)
        return df[['date', 'nav', 'cum_nav']] if 'cum_nav' in df.columns else df[['date', 'nav']]
    
    except Exception as e:
        raise Exception(f"efinance error: {e}")


def _get_fund_nav_akshare(fund_code, start_date, end_date):
    """使用akshare获取基金净值"""
    try:
        import akshare as ak
        
        # 获取基金历史净值 - akshare使用symbol参数
        df = ak.fund_open_fund_info_em(
            symbol=fund_code, 
            indicator='单位净值走势',  # or '累计净值走势'
            period='成立以来'
        )
        
        if df is None or len(df) == 0:
            return None
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '单位净值': 'nav',
            '累计净值': 'cum_nav'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df[['date', 'nav', 'cum_nav']] if 'cum_nav' in df.columns else df[['date', 'nav']]
    
    except Exception as e:
        raise Exception(f"akshare error: {e}")


def _get_fund_nav_baostock(fund_code, start_date, end_date):
    """使用baostock获取基金数据（注意：baostock主要支持股票和部分基金）"""
    try:
        import baostock as bs
        
        # 登录baostock
        bs.login()
        
        # 查询基金净值数据
        rs = bs.query_fund_data(fund_code=f"{fund_code}.OF", start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        if len(data_list) == 0:
            return None
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 重命名列
        df = df.rename(columns={
            'date': 'date',
            'nav': 'nav',
            'navAdjust': 'cum_nav'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df['cum_nav'] = pd.to_numeric(df['cum_nav'], errors='coerce')
        
        df = df.dropna(subset=['nav']).sort_values('date').reset_index(drop=True)
        
        return df[['date', 'nav', 'cum_nav']]
    
    except Exception as e:
        raise Exception(f"baostock error: {e}")


def _get_fund_nav_mootdx(fund_code, start_date, end_date):
    """使用mootdx获取通达信基金数据"""
    try:
        from mootdx import Fund
        
        client = Fund()
        df = client.daily(fund_code)
        
        if df is None or len(df) == 0:
            return None
        
        # mootdx返回的列名通常是: date, open, high, low, close, volume
        df = df.rename(columns={
            'close': 'nav',
            'date': 'date'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        df = df.dropna(subset=['nav']).sort_values('date').reset_index(drop=True)
        
        return df[['date', 'nav']]
    
    except Exception as e:
        raise Exception(f"mootdx error: {e}")


def _get_fund_nav_yfinance(fund_code, start_date, end_date):
    """使用yfinance获取基金数据"""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(fund_code)
        df = ticker.history(start=start_date, end=end_date)
        
        if df is None or len(df) == 0:
            return None
        
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['Date'])
        df['nav'] = df['Close']
        
        return df[['date', 'nav']]
    
    except Exception as e:
        raise Exception(f"yfinance error: {e}")


# ============================================================
# 收益率计算
# ============================================================

def calculate_daily_returns(nav_df):
    """
    从净值数据计算日收益率
    
    Parameters:
    -----------
    nav_df : DataFrame
        包含 date, nav 列的数据框
        
    Returns:
    --------
    DataFrame : 包含 date, daily_return 列
    """
    df = nav_df.copy()
    df = df.sort_values('date')
    df['daily_return'] = df['nav'].pct_change()
    df = df.dropna(subset=['daily_return'])
    return df[['date', 'nav', 'daily_return']]


def calculate_log_returns(nav_df):
    """
    从净值数据计算对数收益率
    
    Parameters:
    -----------
    nav_df : DataFrame
        包含 date, nav 列的数据框
        
    Returns:
    --------
    DataFrame : 包含 date, log_return 列
    """
    df = nav_df.copy()
    df = df.sort_values('date')
    df['log_return'] = np.log(df['nav'] / df['nav'].shift(1))
    df = df.dropna(subset=['log_return'])
    return df[['date', 'nav', 'log_return']]


def calculate_cumulative_returns(returns):
    """
    计算累计收益率
    
    Parameters:
    -----------
    returns : Series
        收益率序列
        
    Returns:
    --------
    Series : 累计收益率
    """
    return (1 + returns).cumprod() - 1


# ============================================================
# 基准数据获取
# ============================================================

def get_benchmark_data(benchmark_code='000300', start_date=None, end_date=None):
    """
    获取基准指数数据（默认沪深300）
    
    Parameters:
    -----------
    benchmark_code : str
        指数代码
    start_date : str
        开始日期
    end_date : str
        结束日期
        
    Returns:
    --------
    DataFrame : 包含 date, close 列
    """
    try:
        import akshare as ak
        
        # 使用akshare获取指数数据
        if benchmark_code == '000300':  # 沪深300
            df = ak.stock_zh_index_daily(symbol=f"sh{benchmark_code}")
        elif benchmark_code.startswith('sh') or benchmark_code.startswith('sz'):
            df = ak.stock_zh_index_daily(symbol=benchmark_code)
        else:
            df = ak.stock_zh_index_daily(symbol=f"sh{benchmark_code}")
        
        if df is None or len(df) == 0:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'close']].rename(columns={'close': 'nav'})
        
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        return df.sort_values('date').reset_index(drop=True)
    
    except Exception as e:
        print(f"  [警告] 获取基准数据失败: {e}")
        return None


def calculate_benchmark_returns(benchmark_code='000300', start_date=None, end_date=None):
    """
    获取基准指数收益率
    
    Returns:
    --------
    DataFrame : 包含 date, benchmark_return 列
    """
    benchmark_df = get_benchmark_data(benchmark_code, start_date, end_date)
    
    if benchmark_df is None:
        return None
    
    benchmark_df = benchmark_df.sort_values('date')
    benchmark_df['benchmark_return'] = benchmark_df['nav'].pct_change()
    benchmark_df = benchmark_df.dropna(subset=['benchmark_return'])
    
    return benchmark_df[['date', 'benchmark_return', 'nav']]


# ============================================================
# 数据保存和加载
# ============================================================

def save_fund_data(fund_code, nav_df, data_dir=DATA_DIR):
    """
    保存基金净值数据到本地
    
    Parameters:
    -----------
    fund_code : str
        基金代码
    nav_df : DataFrame
        净值数据
    data_dir : str
        数据保存目录
    """
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f"{fund_code}_nav.csv")
    nav_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"  [保存] {fund_code} 净值数据已保存至 {file_path}")
    
    return file_path


def load_fund_data(fund_code, data_dir=DATA_DIR):
    """
    从本地加载基金净值数据
    
    Parameters:
    -----------
    fund_code : str
        基金代码
    data_dir : str
        数据目录
        
    Returns:
    --------
    DataFrame : 净值数据
    """
    file_path = os.path.join(data_dir, f"{fund_code}_nav.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    return None


# ============================================================
# 多基金数据获取
# ============================================================

def get_multiple_funds_data(fund_codes, start_date=None, end_date=None, data_dir=DATA_DIR):
    """
    获取多只基金的数据
    
    Parameters:
    -----------
    fund_codes : list
        基金代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    data_dir : str
        数据目录
        
    Returns:
    --------
    dict : {fund_code: nav_df}
    """
    results = {}
    
    for code in fund_codes:
        print(f"\n获取基金 {code} 数据...")
        
        # 尝试从本地加载
        df = load_fund_data(code, data_dir)
        
        # 如果本地没有或数据不完整，从网络获取
        if df is None or len(df) < 30:
            df = get_fund_nav(code, start_date, end_date)
            if df is not None:
                save_fund_data(code, df, data_dir)
        
        if df is not None and len(df) > 0:
            results[code] = df
    
    return results


def merge_funds_returns(funds_data, benchmark_code=None):
    """
    合并多只基金收益率数据
    
    Parameters:
    -----------
    funds_data : dict
        {fund_code: nav_df}
    benchmark_code : str, optional
        基准指数代码
        
    Returns:
    --------
    DataFrame : 包含各基金收益率和基准收益率
    """
    # 合并基金收益率
    dfs = []
    
    for fund_code, nav_df in funds_data.items():
        returns_df = calculate_daily_returns(nav_df)
        returns_df = returns_df.rename(columns={
            'daily_return': f'return_{fund_code}',
            'nav': f'nav_{fund_code}'
        })
        dfs.append(returns_df[['date', f'return_{fund_code}', f'nav_{fund_code}']])
    
    # 合并所有基金
    if len(dfs) > 0:
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='date', how='outer')
    else:
        return None
    
    # 添加基准收益率
    if benchmark_code:
        benchmark_df = calculate_benchmark_returns(benchmark_code)
        if benchmark_df is not None:
            merged_df = pd.merge(merged_df, benchmark_df, on='date', how='left')
    
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    return merged_df
