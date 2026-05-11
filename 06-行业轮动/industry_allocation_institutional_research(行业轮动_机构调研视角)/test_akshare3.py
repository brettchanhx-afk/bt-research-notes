import akshare as ak
import pandas as pd
import os
import time

DATA_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output\data'
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("使用akshare提取A股日线数据")
print("=" * 60)

print("\n[1/4] 获取A股股票列表...")
print("  使用 stock_zh_a_spot_em 获取实时行情...")
try:
    stock_df = ak.stock_zh_a_spot_em()
    print(f"  获取成功! 共 {len(stock_df)} 只股票")
    print(stock_df.head())

    ts_codes = stock_df['代码'].tolist()
    print(f"  A股数量: {len(ts_codes)}")
    print(f"  示例: {ts_codes[:5]}")
except Exception as e:
    print(f"  获取失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[2/4] 测试获取单只股票数据...")
try:
    df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20100101", end_date="20260505", adjust="qfq")
    print(f"  成功! 记录数: {len(df)}")
    print(df.head())
except Exception as e:
    print(f"  获取失败: {e}")

print("\n脚本测试完成!")