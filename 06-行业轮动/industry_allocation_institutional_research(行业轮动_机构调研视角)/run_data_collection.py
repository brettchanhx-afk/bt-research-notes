import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from source.data_loader import DataLoader
from source.data_preprocessor import DataPreprocessor

OUTPUT_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("机构调研策略 - 数据获取")
print("=" * 60)

dl = DataLoader()

print("\n[1/4] 获取股票基本信息...")
stock_basic = dl.get_stock_basic_data()
if stock_basic is not None:
    stock_basic.to_csv(os.path.join(DATA_DIR, 'stock_basic.csv'), index=False)
    print(f"  OK: {len(stock_basic)} 只股票")
else:
    print("  FAIL")

print("\n[2/4] 获取中证全指数据...")
index_data = dl.get_index_data(index_code='000985.SH', start_date='20150101', end_date='20210228')
if index_data is not None:
    index_data.to_csv(os.path.join(DATA_DIR, 'index_zh500.csv'))
    print(f"  OK: {len(index_data)} 条记录")
else:
    print("  FAIL")

print("\n[3/4] 获取申万一级行业成分股...")
try:
    sw_industry = dl.pro.sw_csi()
    if sw_industry is not None and len(sw_industry) > 0:
        sw_industry.to_csv(os.path.join(DATA_DIR, 'sw_industry_csi.csv'), index=False)
        print(f"  OK: {len(sw_industry)} 条记录")
    else:
        raise Exception("数据为空")
except Exception as e:
    print(f"  申万行业获取失败，使用股票基本信息中的行业字段")
    if stock_basic is not None:
        industry_df = stock_basic[['ts_code', 'industry']].dropna()
        industry_df.to_csv(os.path.join(DATA_DIR, 'industry_classification.csv'), index=False)
        print(f"  OK: {len(industry_df)} 条记录")

print("\n[4/4] 获取指数成分股...")
try:
    index_components = dl.pro.index_weight(index_code='000985.SH', trade_date='20210228')
    if index_components is not None and len(index_components) > 0:
        index_components.to_csv(os.path.join(DATA_DIR, 'index_components.csv'), index=False)
        print(f"  OK: {len(index_components)} 只成分股")
    else:
        print("  指数成分股权重数据为空")
except Exception as e:
    print(f"  指数成分股获取失败: {e}")

print("\n" + "=" * 60)
print("数据获取完成!")
print("=" * 60)

print("\n保存的数据文件:")
for f in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f}: {size/1024:.1f} KB")

summary_data = {
    '数据项': ['股票基本信息', '中证全指数据', '行业分类', '指数成分股'],
    '记录数': [
        len(stock_basic) if stock_basic is not None else 0,
        len(index_data) if index_data is not None else 0,
        len(sw_industry) if 'sw_industry' in dir() and sw_industry is not None else (len(industry_df) if 'industry_df' in dir() else 0),
        len(index_components) if 'index_components' in dir() and index_components is not None else 0
    ],
    '保存路径': [os.path.join(DATA_DIR, 'stock_basic.csv'),
                 os.path.join(DATA_DIR, 'index_zh500.csv'),
                 os.path.join(DATA_DIR, 'sw_industry_csi.csv') if 'sw_industry' in dir() else os.path.join(DATA_DIR, 'industry_classification.csv'),
                 os.path.join(DATA_DIR, 'index_components.csv')]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(RESULTS_DIR, 'data_collection_summary.csv'), index=False)
print(f"\n摘要已保存: {os.path.join(RESULTS_DIR, 'data_collection_summary.csv')}")

print("\n" + "=" * 60)
print("注意: 机构调研数据(tushare免费版暂不支持)需要商业终端")
print("=" * 60)