import akshare as ak
import pandas as pd
import os
import time

print("测试akshare接口...")

print("\n1. 测试 stock_zh_a_hist (单只股票日线)...")
try:
    df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20100101", end_date="20260505", adjust="qfq")
    print(f"  成功! 记录数: {len(df)}")
    print(df.head())
except Exception as e:
    print(f"  失败: {e}")