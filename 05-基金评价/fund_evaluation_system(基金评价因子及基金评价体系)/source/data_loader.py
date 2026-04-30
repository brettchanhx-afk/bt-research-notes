# -*- coding: utf-8 -*-
"""
数据获取模块

数据源优先级：
1. efinance - 基金净值、持仓
2. tushare - 基金信息、持有人结构
3. akshare - 市场指数
"""
import os
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TUSHARE_TOKEN, TUSHARE_API_URL


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
    """
    获取基金净值历史数据
    
    Parameters
    ----------
    fund_code : str
        基金代码
    start_date, end_date : str
        日期范围（YYYY-MM-DD）
        
    Returns
    -------
    pd.DataFrame
        净值数据，包含date, nav, nav_acc
    """
    # 尝试efinance
    try:
        import efinance as ef
        df = ef.fund.get_quote_history(fund_code, beg=start_date, end=end_date)
        
        if df is not None and len(df) > 0:
            # 列名适配
            df.columns = [str(c).strip() for c in df.columns]
            
            result = pd.DataFrame({
                'date': pd.to_datetime(df.iloc[:, 0]),
                'nav': pd.to_numeric(df.iloc[:, 1], errors='coerce'),
                'nav_acc': pd.to_numeric(df.iloc[:, 2], errors='coerce') if df.shape[1] > 2 else np.nan,
            })
            result = result.set_index('date').sort_index()
            result['daily_return'] = result['nav'].pct_change()
            
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
            
            return result
            
    except Exception as e:
        print(f'  [tushare] 获取净值失败: {e}')
    
    return pd.DataFrame()


# ============================================================
# 2. 基金列表
# ============================================================
def get_fund_list(fund_type: str = '股票型') -> pd.DataFrame:
    """
    获取基金列表
    
    Parameters
    ----------
    fund_type : str
        基金类型：股票型、混合型、债券型等
    """
    try:
        import efinance as ef
        df = ef.stock.get_quote_history('基金')
        return df
    except Exception:
        pass
    
    # tushare
    try:
        pro = init_tushare()
        df = pro.fund_basic(market='E')
        
        # 筛选主动偏股型基金
        equity_types = ['股票型', '混合型', '偏股混合型', '灵活配置型']
        df = df[df['fund_type'].isin(equity_types)]
        
        return df[['ts_code', 'fund_name', 'fund_type', 'issue_date', 'm_fee', 'c_fee']]
        
    except Exception as e:
        print(f'  [ERROR] 获取基金列表失败: {e}')
    
    return pd.DataFrame()


# ============================================================
# 3. 基金信息（规模、份额、持有人结构）
# ============================================================
def get_fund_info(fund_code: str) -> pd.Series:
    """
    获取基金详细信息
    
    Returns
    -------
    pd.Series
        包含规模、份额、持有人结构等
    """
    info = {}
    
    try:
        import efinance as ef
        # 基金基本信息
        df = ef.fund.get_fund_codes()
        if df is not None and len(df) > 0:
            fund_row = df[df['基金代码'] == fund_code]
            if len(fund_row) > 0:
                info['fund_name'] = fund_row.iloc[0].get('基金简称', '')
        
    except Exception:
        pass
    
    # tushare获取规模和持有人结构
    try:
        pro = init_tushare()
        
        # 基金规模
        df_share = pro.fund_share(ts_code=f'{fund_code}.OF')
        if df_share is not None and len(df_share) > 0:
            latest = df_share.iloc[0]
            info['shares'] = latest['fd_share']  # 亿份
            info['share_date'] = latest['end_date']
        
        # 持有人结构
        df_holder = pro.fund_hold_struct(ts_code=f'{fund_code}.OF')
        if df_holder is not None and len(df_holder) > 0:
            latest = df_holder.iloc[0]
            info['inst_holding_ratio'] = latest.get('inst_holding_ratio', np.nan) / 100
            info['personal_holding_ratio'] = latest.get('personal_holding_ratio', np.nan) / 100
            info['manager_holding_ratio'] = latest.get('expend_holding_ratio', np.nan) / 100
            
    except Exception as e:
        print(f'  [WARNING] 获取基金信息失败: {e}')
    
    return pd.Series(info)


# ============================================================
# 4. 市场指数数据
# ============================================================
def get_market_index(
    index_code: str = '000001',
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    获取市场指数数据
    
    Parameters
    ----------
    index_code : str
        指数代码：000001=上证指数, 000300=沪深300, 881001=万得全A
    """
    # 尝试akshare
    try:
        import akshare as ak
        
        # 上证指数
        if index_code == '000001':
            df = ak.stock_zh_index_daily(symbol="sh000001")
        elif index_code == '000300':
            df = ak.stock_zh_index_daily(symbol="sh000300")
        else:
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
        
        if df is not None and len(df) > 0:
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # 筛选日期
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df.index >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df.index <= end]
            
            df['daily_return'] = df['close'].pct_change()
            
            return df[['close', 'daily_return']]
            
    except Exception as e:
        print(f'  [akshare] 获取指数失败: {e}')
    
    return pd.DataFrame()


# ============================================================
# 5. 批量获取基金净值
# ============================================================
def get_multiple_fund_nav(
    fund_codes: List[str],
    start_date: str,
    end_date: str,
    max_workers: int = 5
) -> dict:
    """
    批量获取多只基金净值
    
    Returns
    -------
    dict
        {fund_code: DataFrame}
    """
    results = {}
    
    for i, code in enumerate(fund_codes):
        if (i + 1) % 10 == 0:
            print(f'  进度: {i+1}/{len(fund_codes)}')
        
        try:
            df = get_fund_nav(code, start_date, end_date)
            if len(df) > 0:
                results[code] = df
            time.sleep(0.1)
        except Exception as e:
            pass
    
    print(f'\n  成功获取 {len(results)}/{len(fund_codes)} 只基金数据')
    return results


# ============================================================
# 6. 获取同类基金收益率中位数（作为基准）
# ============================================================
def calculate_peer_median_returns(
    fund_navs: dict,
    start_date: str,
    end_date: str
) -> pd.Series:
    """
    计算同类基金收益率中位数
    
    Parameters
    ----------
    fund_navs : dict
        {fund_code: nav_dataframe}
        
    Returns
    -------
    pd.Series
        日度收益率中位数
    """
    if not fund_navs:
        return pd.Series()
    
    # 收集所有基金的收益率
    all_returns = []
    
    for code, nav_df in fund_navs.items():
        if 'daily_return' in nav_df.columns:
            returns = nav_df['daily_return'].dropna()
            returns.name = code
            all_returns.append(returns)
    
    if not all_returns:
        return pd.Series()
    
    # 合并
    returns_df = pd.concat(all_returns, axis=1)
    
    # 计算中位数
    median_returns = returns_df.median(axis=1)
    median_returns.name = 'peer_median'
    
    return median_returns
