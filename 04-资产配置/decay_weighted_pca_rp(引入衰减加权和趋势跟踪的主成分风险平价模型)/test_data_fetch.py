"""
Test script to verify real data fetching from tushare
"""
import sys
import tushare as ts
import pandas as pd
import numpy as np

sys.stdout.flush()
print("=" * 80)
print("Tushare 真实数据获取测试")
print("=" * 80)
sys.stdout.flush()

# Initialize tushare API with correct settings
token = '1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb'
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = 'http://jiaoch.site'

print("\nTushare API initialized successfully")
sys.stdout.flush()

# Define assets to fetch
index_codes = ['000300.SH', '000016.SH', '000905.SH']
bond_codes = ['000012.SH', '000013.SH']

all_prices = {}

print('\n正在获取市场数据...')
sys.stdout.flush()

# Fetch stock index data
for code in index_codes:
    try:
        df = pro.index_daily(ts_code=code, start_date='20100101', end_date='20171117')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        df = df.set_index('trade_date')
        name = {'000300.SH': '沪深300', '000016.SH': '上证50', '000905.SH': '中证500'}[code]
        all_prices[name] = df['close']
        print(f'  {name}: {len(df)} 条记录')
        sys.stdout.flush()
    except Exception as e:
        print(f'  获取 {code} 失败: {e}')
        sys.stdout.flush()

# Fetch bond index data (using index_daily interface)
for code in bond_codes:
    try:
        df = pro.index_daily(ts_code=code, start_date='20100101', end_date='20171117')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df = df.set_index('trade_date')
            name = {'000012.SH': '上证国债', '000013.SH': '上证企债'}[code]
            all_prices[name] = df['close']
            print(f'  {name}: {len(df)} 条记录')
            sys.stdout.flush()
        else:
            print(f'  {code} 无债券数据')
            sys.stdout.flush()
    except Exception as e:
        print(f'  获取 {code} 失败: {e}')
        sys.stdout.flush()

# Create price DataFrame
prices = pd.DataFrame(all_prices)
prices = prices.sort_index()
prices = prices.dropna()

# Calculate returns
returns = prices.pct_change().dropna()

asset_names = list(prices.columns)

print(f'\n数据时间范围: {prices.index[0].strftime("%Y-%m-%d")} 至 {prices.index[-1].strftime("%Y-%m-%d")}')
print(f'数据点数量: {len(prices)}')
print(f'资产数量: {len(asset_names)}')
print(f'资产列表: {asset_names}')
sys.stdout.flush()

# Show data statistics
print('\n数据统计:')
print(returns.describe().to_string())
sys.stdout.flush()

print('\n' + "=" * 80)
print("数据获取测试完成!")
print("=" * 80)
sys.stdout.flush()