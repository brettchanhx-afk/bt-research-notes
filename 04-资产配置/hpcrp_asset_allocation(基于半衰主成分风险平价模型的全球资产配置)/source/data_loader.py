"""
数据加载模块 - 获取全球股票指数数据
优先使用: tushare (A股) + 本地文件 (国际指数)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import os
warnings.filterwarnings('ignore')


# Tushare初始化 (必须使用指定方式)
def init_tushare():
    """初始化tushare - 使用用户指定的配置"""
    import tushare as ts
    
    token = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
    pro = ts.pro_api(token)
    pro._DataApi__token = token
    pro._DataApi__http_url = "http://jiaoch.site"
    
    return pro


def fetch_global_index_data():
    """
    获取全球主要股票指数数据
    
    Returns:
        pd.DataFrame: 指数收益率数据
    """
    
    data_frames = {}
    
    # 1. A股指数 - 使用tushare index_daily接口
    try:
        print("正在通过tushare获取A股指数数据...")
        pro = init_tushare()
        
        indices = [
            ('000300.SH', 'CSI300'),  # 沪深300
            ('000905.SH', 'CSI500'),  # 中证500
            ('000016.SH', 'SSE50'),   # 上证50
            ('399001.SZ', 'SZ100'),   # 深证100
        ]
        
        for ts_code, name in indices:
            try:
                df = pro.index_daily(
                    ts_code=ts_code,
                    start_date='20080101',
                    end_date='20251231'
                )
                
                if df is not None and len(df) > 0:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df = df.sort_values('trade_date')
                    df = df.set_index('trade_date')
                    data_frames[name] = df['close'].astype(float)
                    print(f"  {name}: {len(df)} 条记录")
                time.sleep(0.3)
            except Exception as e:
                print(f"  {name} 获取失败: {e}")
        
        if len(data_frames) >= 2:
            print("A股指数数据获取成功!")
        
    except Exception as e:
        print(f"tushare获取失败: {e}")
    
    # 2. 国际指数 - 从本地Excel读取
    local_file = 'C:/Users/chenh/.qclaw/workspace/hpcrp_asset_allocation/data/global_indices.xlsx'
    
    if os.path.exists(local_file):
        try:
            print("\n正在读取本地国际指数数据...")
            df_intl = pd.read_excel(local_file, index_col=0, parse_dates=True)
            
            # 重命名列以匹配
            df_intl = df_intl.rename(columns={
                'SP500': 'SPX',
                'NDAQ': 'NDAQ',
                'FTSE': 'FTSE',
                'CAC40': 'CAC40',
                'DAX': 'DAX',
                'HSI': 'HSI',
                'HSCEI': 'HSCEI'
            })
            
            for col in df_intl.columns:
                data_frames[col] = df_intl[col].astype(float)
                print(f"  {col}: {len(df_intl)} 条记录")
            
            print("国际指数数据读取成功!")
            
        except Exception as e:
            print(f"读取本地文件失败: {e}")
    else:
        print(f"本地文件不存在: {local_file}")
    
    # 合并所有数据
    if len(data_frames) < 2:
        raise ValueError("未能获取足够的指数数据")
    
    # 构建DataFrame
    all_prices = pd.DataFrame(data_frames)
    
    # 前向填充
    all_prices = all_prices.ffill()
    
    # 去除全是NaN的行
    all_prices = all_prices.dropna(how='all')
    
    # 计算收益率
    returns = all_prices.pct_change().dropna()
    
    # 只保留所有指数都有数据的日期
    returns = returns.dropna()
    
    print(f"\n数据汇总:")
    print(f"  时间范围: {returns.index.min().date()} 到 {returns.index.max().date()}")
    print(f"  交易日数: {len(returns)}")
    print(f"  指数数量: {len(returns.columns)}")
    print(f"  指数列表: {list(returns.columns)}")
    
    return returns


if __name__ == '__main__':
    print("正在获取全球指数数据...")
    data = fetch_global_index_data()
    print(f"\n成功获取 {len(data.columns)} 个指数的数据")