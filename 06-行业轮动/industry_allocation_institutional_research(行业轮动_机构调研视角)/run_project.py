import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from source.data_loader import DataLoader
from source.data_preprocessor import DataPreprocessor
from source.backtest import BacktestEngine
from source.strategies.event_driven import EventDrivenStrategy
from source.strategies.regular_stock import RegularStockStrategy
from source.strategies.industry_rotation import IndustryRotationStrategy

OUTPUT_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("机构调研策略 - 数据获取与回测")
print("=" * 60)

dl = DataLoader()
preprocessor = DataPreprocessor()
backtest_engine = BacktestEngine(initial_capital=10000000, commission_rate=0.001)

print("\n[1/5] 获取股票基本信息...")
stock_basic = dl.get_stock_basic_data()
if stock_basic is not None:
    stock_basic.to_csv(os.path.join(DATA_DIR, 'stock_basic.csv'), index=False)
    print(f"  保存股票基本信息: {len(stock_basic)} 只股票")
else:
    print("  获取股票基本信息失败")

print("\n[2/5] 获取中证全指数据...")
index_data = dl.get_index_data(index_code='000985.SH', start_date='20150101', end_date='20210228')
if index_data is not None:
    index_data.to_csv(os.path.join(DATA_DIR, 'index_zh500.csv'))
    print(f"  保存指数数据: {len(index_data)} 条记录")
else:
    print("  获取指数数据失败")

print("\n[3/5] 获取申万行业分类...")
try:
    industry_df = dl.pro.sw_csi()
    if industry_df is not None:
        industry_df.to_csv(os.path.join(DATA_DIR, 'industry_classification.csv'), index=False)
        print(f"  保存行业分类: {len(industry_df)} 条记录")
    else:
        industry_df = stock_basic[['ts_code', 'industry']].dropna()
        industry_df.to_csv(os.path.join(DATA_DIR, 'industry_classification.csv'), index=False)
        print(f"  使用股票行业信息: {len(industry_df)} 条记录")
except Exception as e:
    print(f"  获取行业分类失败: {e}")
    industry_df = stock_basic[['ts_code', 'industry']].dropna()
    industry_df.to_csv(os.path.join(DATA_DIR, 'industry_classification.csv'), index=False)
    print(f"  使用股票行业信息: {len(industry_df)} 条记录")

print("\n[4/5] 获取A股日线数据...")
try:
    stock_list = stock_basic['ts_code'].tolist()[:100]
    all_stock_data = []
    for i, code in enumerate(stock_list):
        try:
            df = dl.get_daily_stock_data(code, '20150101', '20210228')
            if df is not None and len(df) > 0:
                all_stock_data.append(df)
        except Exception as e:
            continue
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(stock_list)}")
    if len(all_stock_data) > 0:
        stock_daily = pd.concat(all_stock_data, ignore_index=True)
        stock_daily.to_csv(os.path.join(DATA_DIR, 'stock_daily.csv'), index=False)
        print(f"  保存股票日线数据: {len(stock_daily)} 条记录")
    else:
        print("  未能获取股票日线数据")
except Exception as e:
    print(f"  获取股票日线数据失败: {e}")

print("\n[5/5] 获取资金流向数据...")
try:
    trade_dates = pd.to_datetime(index_data.index) if index_data is not None else []
    money_flow_samples = []
    for date in trade_dates[:10]:
        date_str = date.strftime('%Y%m%d')
        try:
            mf = dl.get_money_flow(trade_date=date_str)
            if mf is not None and len(mf) > 0:
                money_flow_samples.append(mf)
        except:
            continue
    if len(money_flow_samples) > 0:
        money_flow = pd.concat(money_flow_samples, ignore_index=True)
        money_flow.to_csv(os.path.join(DATA_DIR, 'money_flow.csv'), index=False)
        print(f"  保存资金流向数据: {len(money_flow)} 条记录")
    else:
        print("  未能获取资金流向数据")
except Exception as e:
    print(f"  获取资金流向数据失败: {e}")

print("\n" + "=" * 60)
print("数据获取完成!")
print("=" * 60)

print("\n[提示] 机构调研数据(ASHAREINSTITUTIONALACTIVITY)需要商业终端获取")
print("       tushare免费版暂不支持完整机构调研数据查询")

print("\n" + "=" * 60)
print("开始策略回测...")
print("=" * 60)

if index_data is not None and len(index_data) > 0:
    print("\n回测基准指数数据可用")
    print(f"  数据范围: {index_data.index.min()} 至 {index_data.index.max()}")
    print(f"  数据条数: {len(index_data)}")

if stock_basic is not None:
    print(f"\n股票池数据可用: {len(stock_basic)} 只")

print("\n注意: 由于机构调研数据缺失,无法进行完整策略回测")
print("请补充机构调研数据后运行完整的策略回测")

results_summary = {
    '项目': '机构调研策略复现',
    '数据状态': '基础数据已获取,机构调研数据待补充',
    '指数数据': f'{len(index_data) if index_data is not None else 0}条' if index_data is not None else '获取失败',
    '股票数据': f'{len(stock_basic) if stock_basic is not None else 0}只' if stock_basic is not None else '获取失败',
    '输出路径': OUTPUT_DIR
}

results_df = pd.DataFrame([results_summary])
results_df.to_csv(os.path.join(RESULTS_DIR, 'data_collection_summary.csv'), index=False)
print(f"\n数据收集摘要已保存到: {os.path.join(RESULTS_DIR, 'data_collection_summary.csv')}")

print("\n" + "=" * 60)
print("程序运行完成!")
print("=" * 60)