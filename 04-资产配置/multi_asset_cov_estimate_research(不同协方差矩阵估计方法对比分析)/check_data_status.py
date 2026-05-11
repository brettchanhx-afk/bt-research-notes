import sys
sys.path.append('.')
from source.data_fetcher import DataFetcher
import pandas as pd
import numpy as np

data_fetcher = DataFetcher()

print("=== 数据获取现状 ===")
print("成功获取: 沪深300, 中证1000")
print("缺失: 恒生指数, 标普500, 中债指数, 南华商品指数")
print()

try:
    import akshare as ak
    print("尝试使用akshare获取数据...")
    
    try:
        print("获取恒生指数...")
        hs_df = ak.stock_hk_index_daily(symbol="HSI")
        print(f"恒生指数: {len(hs_df)} 条")
    except:
        print("恒生指数获取失败")
    
    try:
        print("获取南华商品指数...")
        nh_df = ak.index_zh_a_hist(symbol="NH0100", period="daily", start_date="20170101", end_date="20231231")
        print(f"南华商品指数: {len(nh_df)} 条")
    except:
        print("南华商品指数获取失败")
        
except ImportError:
    print("akshare未安装，无法使用")

print()
print("=== 请提供以下缺失数据 ===")
print("1. 恒生指数日数据 (trade_date, close)")
print("2. 标普500日数据 (trade_date, close)")
print("3. 中债-国债总财富指数日数据 (trade_date, close)")
print("4. 中债-企业债总财富指数日数据 (trade_date, close)")
print("5. 南华商品指数日数据 (trade_date, close)")
print()
print("请将CSV文件放入 data/ 目录，文件名格式: asset_name.csv")