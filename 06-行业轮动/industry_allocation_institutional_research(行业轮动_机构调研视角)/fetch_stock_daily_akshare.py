import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output\data'

START_DATE = "20100101"
END_DATE = "20260505"
BATCH_SIZE = 50
REQUEST_DELAY = 0.5

print("=" * 60)
print("使用akshare提取2010年至今A股日线数据")
print("=" * 60)

print("\n[1/4] 读取A股股票列表...")
stock_basic_path = os.path.join(DATA_DIR, 'stock_basic.csv')
if os.path.exists(stock_basic_path):
    stock_df = pd.read_csv(stock_basic_path)
    a_stocks = stock_df[stock_df['ts_code'].str.startswith(('0', '3', '6'))]
    ts_codes = a_stocks['ts_code'].tolist()
    print(f"  从stock_basic.csv读取: {len(ts_codes)} 只股票")
    print(f"  示例: {ts_codes[:5]}")
else:
    print(f"  文件不存在: {stock_basic_path}")
    exit(1)

print("\n[2/4] 分批提取日线数据 (akshare)...")
print("  注意: 无并发限制,可稳定获取")
print("  预计耗时: 约30-60分钟")

all_data = []
failed_codes = []
success_count = 0

start_time = time.time()

for batch_start in range(0, len(ts_codes), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(ts_codes))
    batch_codes = ts_codes[batch_start:batch_end]

    for code in batch_codes:
        symbol = code.split('.')[0]

        for attempt in range(3):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjust="qfq"
                )
                if df is not None and len(df) > 0:
                    df.rename(columns={
                        '日期': 'trade_date',
                        '股票代码': 'ts_code',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'vol',
                        '成交额': 'amount',
                        '涨跌幅': 'pct_chg',
                        '涨跌额': 'change',
                        '换手率': 'turnover'
                    }, inplace=True)
                    df['ts_code'] = code
                    all_data.append(df)
                    success_count += 1
                break
            except Exception as e:
                if attempt == 2:
                    failed_codes.append({'code': code, 'error': str(e)})
                time.sleep(1)

        time.sleep(REQUEST_DELAY)

    elapsed = time.time() - start_time
    speed = batch_end / elapsed if elapsed > 0 else 0
    eta = (len(ts_codes) - batch_end) / speed / 60 if speed > 0 else 0
    print(f"  进度: {batch_end}/{len(ts_codes)} | 成功: {success_count} | 速度: {speed:.1f}个/秒 | 预计剩余: {eta:.1f}分钟")

elapsed = time.time() - start_time
print(f"\n  获取完成! 总耗时: {elapsed/60:.1f} 分钟")
print(f"  成功获取: {success_count} 只")
print(f"  获取失败: {len(failed_codes)} 只")

if len(all_data) > 0:
    print("\n[3/4] 合并数据...")
    daily_data = pd.concat(all_data, ignore_index=True)
    daily_data = daily_data.sort_values(['ts_code', 'trade_date'])
    print(f"  总记录数: {len(daily_data)}")

    cols_to_save = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg', 'change', 'turnover']
    cols_available = [c for c in cols_to_save if c in daily_data.columns]
    daily_data = daily_data[cols_available]

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