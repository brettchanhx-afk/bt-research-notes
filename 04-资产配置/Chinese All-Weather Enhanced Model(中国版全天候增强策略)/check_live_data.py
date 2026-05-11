import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from source.data_loader import DataLoader

print('='*60)
print('检查数据完整性')
print('='*60)

print('\n[1] 配置期望:')
print(f"  开始日期: {config.DATA_CONFIG['start_date']}")
print(f"  结束日期: {config.DATA_CONFIG['end_date']}")
print(f"  使用保存数据: {config.DATA_CONFIG['use_saved_data']}")

print('\n[2] 尝试从网络获取真实数据...')
loader = DataLoader(start_date=config.DATA_CONFIG['start_date'],
                   end_date=config.DATA_CONFIG['end_date'])

try:
    price_data, return_data = loader.load_all_data()

    if price_data is not None and len(price_data) > 0:
        print(f'\n成功获取数据!')
        print(f'  数据行数: {len(price_data)} 行')
        print(f'  实际开始日期: {price_data.index.min()}')
        print(f'  实际结束日期: {price_data.index.max()}')

        print('\n各资产数据情况:')
        for col in price_data.columns:
            missing = price_data[col].isna().sum()
            first_valid = price_data[col].first_valid_index()
            last_valid = price_data[col].last_valid_index()
            if missing > 0:
                print(f'  {col}: 缺失 {missing} 个数据点, 有效区间: {first_valid} ~ {last_valid}')
            else:
                print(f'  {col}: 完整, {first_valid} ~ {last_valid}')
    else:
        print('\n数据获取失败或返回空数据')
except Exception as e:
    print(f'\n获取数据时出错: {e}')
    traceback.print_exc()

print('\n[3] 与配置的时间区间对比:')
config_start = pd.to_datetime(config.DATA_CONFIG['start_date'], format='%Y%m%d')
config_end = pd.to_datetime(config.DATA_CONFIG['end_date'], format='%Y%m%d')

if price_data is not None and len(price_data) > 0:
    actual_start = price_data.index.min()
    actual_end = price_data.index.max()

    print(f'配置期望: {config_start.date()} ~ {config_end.date()}')
    print(f'实际数据: {actual_start.date()} ~ {actual_end.date()}')

    if actual_start > config_start:
        print(f'  [!] 缺少开始区间: {config_start.date()} ~ {actual_start.date()}')
    if actual_end < config_end:
        print(f'  [!] 缺少结束区间: {actual_end.date()} ~ {config_end.date()}')
else:
    print('无法进行对比，数据获取失败')