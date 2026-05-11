import sys
print("Step 1: Starting program", file=sys.stdout)
sys.stdout.flush()

import pandas as pd
print("Step 2: Imported pandas", file=sys.stdout)
sys.stdout.flush()

import numpy as np
print("Step 3: Imported numpy", file=sys.stdout)
sys.stdout.flush()

import os
print("Step 4: Imported os", file=sys.stdout)
sys.stdout.flush()

# 配置参数
TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_URL = "http://jiaoch.site"

print("Step 5: Config loaded", file=sys.stdout)
sys.stdout.flush()

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Step 6: Output directory created", file=sys.stdout)
sys.stdout.flush()

# 测试数据获取
print("Step 7: Testing data fetch", file=sys.stdout)
sys.stdout.flush()

import tushare as ts
print("Step 8: Imported tushare", file=sys.stdout)
sys.stdout.flush()

try:
    pro = ts.pro_api(TUSHARE_TOKEN)
    pro._DataApi__token = TUSHARE_TOKEN
    pro._DataApi__http_url = TUSHARE_URL
    print("Step 9: Tushare initialized", file=sys.stdout)
    sys.stdout.flush()
    
    df = pro.index_daily(ts_code='000300.SH', start_date='20210101', end_date='20210110')
    print(f"Step 10: Data fetched: {len(df)} rows", file=sys.stdout)
    sys.stdout.flush()
    
    df = df[['trade_date', 'close']]
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)
    print("Step 11: Data processed", file=sys.stdout)
    sys.stdout.flush()
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.stderr.flush()

print("Program completed!", file=sys.stdout)
sys.stdout.flush()