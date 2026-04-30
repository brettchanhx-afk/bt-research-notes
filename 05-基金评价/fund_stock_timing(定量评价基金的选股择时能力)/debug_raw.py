# -*- coding: utf-8 -*-
import sys, warnings, traceback
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import efinance.fund as ef_fund
import efinance as ef

# Test fund
print("=== FUND TEST ===")
df = ef_fund.get_quote_history('021181')
print(f"Type: {type(df)}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\n{df.dtypes}")
print(df.tail(5).to_string())
print()

# Test benchmark
print("=== BENCHMARK TEST ===")
try:
    df2 = ef.stock.get_quote_history('000300', beg='20210101', end='20260428')
    print(f"Type: {type(df2)}")
    print(f"Shape: {df2.shape}")
    print(f"Columns: {df2.columns.tolist()}")
    print(df2.tail(5).to_string())
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
