# -*- coding: utf-8 -*-
import sys, warnings, traceback
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\chenh\.qclaw\workspace\fund_stock_timing\source')
from data_loader import load_fund_nav_efinance, load_benchmark_efinance

print("Testing fund nav...")
try:
    df = load_fund_nav_efinance('021181', '2021-01-01', '2026-04-28')
    print(f"Fund nav result: shape={df.shape}, columns={df.columns.tolist()}")
    print(df.head(3))
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\nTesting benchmark...")
try:
    df2 = load_benchmark_efinance('000300', '2021-01-01', '2026-04-28')
    print(f"Benchmark result: shape={df2.shape}, columns={df2.columns.tolist()}")
    print(df2.head(3))
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
