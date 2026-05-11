import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from source.data_fetcher import DataFetcher

print("=== 数据获取调试 ===")

data_fetcher = DataFetcher()

asset_config = {
    '沪深300': '000300.SH',
    '中证1000': '000852.SH',
}

start_date = '20170101'
end_date = '20231231'

available_data = {}
for asset_name, ts_code in asset_config.items():
    print(f"\n获取 {asset_name} ({ts_code})")
    try:
        df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
        print(f"返回数据: {len(df)} 条")
        if len(df) > 0:
            print(f"列名: {list(df.columns)}")
            print(f"前5行:")
            print(df.head())
            if 'returns' in df.columns:
                available_data[asset_name] = df['returns']
                print(f"returns列长度: {len(df['returns'].dropna())}")
    except Exception as e:
        print(f"错误: {e}")

if available_data:
    returns_df = pd.DataFrame(available_data).dropna()
    print(f"\n合并后数据: {returns_df.shape}")
    print(f"索引类型: {type(returns_df.index)}")
    print(f"索引范围: {returns_df.index.min()} 到 {returns_df.index.max()}")
    
    returns_df.to_csv('output/debug_returns.csv', encoding='utf-8')
    print("数据已保存到 output/debug_returns.csv")
else:
    print("\n没有可用数据")