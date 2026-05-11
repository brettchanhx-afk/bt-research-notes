import tushare as ts
import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

token = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = "http://jiaoch.site"

DATA_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output\data'
os.makedirs(DATA_DIR, exist_ok=True)

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 30
REQUEST_DELAY = 1.0

def call_with_retry(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            err_msg = str(e)
            if '并发' in err_msg or 'too many' in err_msg.lower():
                wait_time = INITIAL_RETRY_DELAY * (attempt + 1)
                print(f"    触发并发限制,等待 {wait_time} 秒... (尝试 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                return None, err_msg
    return None, "超过最大重试次数"

print("=" * 60)
print("使用tushare提取2010年至今A股日线数据")
print("=" * 60)

print("\n[1/4] 获取A股股票列表 (带重试机制)...")
stocks, err = call_with_retry(lambda: pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date'))
if err:
    print(f"  获取股票列表失败: {err}")
    exit(1)

stock_list_df = stocks[stocks['ts_code'].str.startswith(('0', '3', '6'))]
ts_codes = stock_list_df['ts_code'].tolist()
print(f"  A股股票数量: {len(ts_codes)}")

print("\n[2/4] 分批提取日线数据...")
print("  注意: API并发限制2个,将自动速率控制")
print("  预计耗时: 约60-90分钟 (取决于网络和API限制)")

all_data = []
failed_codes = []
success_count = 0

start_time = time.time()

for i, code in enumerate(ts_codes):
    df, err = call_with_retry(lambda c=code: ts.pro_bar(ts_code=c, start_date='20100101', end_date='20260505', api=pro, adj='qfq'))

    if df is not None and len(df) > 0:
        all_data.append(df)
        success_count += 1
    elif err:
        failed_codes.append({'code': code, 'error': err})

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        speed = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(ts_codes) - i - 1) / speed / 60 if speed > 0 else 0
        print(f"  进度: {i+1}/{len(ts_codes)} | 成功: {success_count} | 速度: {speed:.1f}个/秒 | 预计剩余: {eta:.1f} 分钟")

    time.sleep(REQUEST_DELAY)

elapsed = time.time() - start_time
print(f"\n  获取完成! 总耗时: {elapsed/60:.1f} 分钟")
print(f"  成功获取: {success_count} 只")
print(f"  获取失败: {len(failed_codes)} 只")

if len(all_data) > 0:
    print("\n[3/4] 合并数据...")
    daily_data = pd.concat(all_data, ignore_index=True)
    daily_data = daily_data.sort_values(['ts_code', 'trade_date'])
    print(f"  总记录数: {len(daily_data)}")

    save_path = os.path.join(DATA_DIR, 'stock_daily_full.csv')
    daily_data.to_csv(save_path, index=False)
    print(f"\n[4/4] 数据已保存: {save_path}")
    print(f"  文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

    print("\n  数据预览:")
    print(daily_data.head())

if len(failed_codes) > 0:
    error_path = os.path.join(DATA_DIR, 'fetch_errors.csv')
    pd.DataFrame(failed_codes).to_csv(error_path, index=False)
    print(f"\n  获取失败的股票已保存: {error_path}")

print("\n" + "=" * 60)
print("数据提取完成!")
print("=" * 60)