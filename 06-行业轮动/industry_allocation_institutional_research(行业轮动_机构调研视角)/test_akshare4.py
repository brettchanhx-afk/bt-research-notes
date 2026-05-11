import akshare as ak
import pandas as pd
import os
import time

DATA_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output\data'

print("测试akshare接口...")
print("=" * 60)

print("\n1. 读取股票列表...")
stock_df = pd.read_csv(os.path.join(DATA_DIR, 'stock_basic.csv'))
ts_codes = stock_df[stock_df['ts_code'].str.startswith(('0', '3', '6'))]['ts_code'].tolist()[:10]
print(f"  读取成功: {len(ts_codes)} 只")
print(f"  示例: {ts_codes[:5]}")

print("\n2. 测试获取日线数据 (3只股票)...")
for code in ts_codes[:3]:
    symbol = code.split('.')[0]
    print(f"\n  获取 {symbol}...")
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20100101", end_date="20260505", adjust="qfq")
        print(f"    成功! 记录数: {len(df)}")
    except Exception as e:
        print(f"    失败: {e}")
    time.sleep(0.5)

print("\n" + "=" * 60)
print("测试完成!")