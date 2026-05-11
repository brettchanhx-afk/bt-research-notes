"""
处理补充数据并运行完整回测

步骤：
1. 读取补充的中债和南华商品指数数据
2. 合并原有数据获取逻辑，优先使用本地数据
3. 运行所有策略回测
4. 输出回测结果
"""

import pandas as pd
import numpy as np
import os

# 读取补充数据
supplement_path = 'data/中债和南华商品指数行情序列.xlsx'
if os.path.exists(supplement_path):
    print(f"读取补充数据: {supplement_path}")
    supplement_df = pd.read_excel(supplement_path)
    print(f"补充数据形状: {supplement_df.shape}")
    
    # 查看数据时间范围
    supplement_df['时间'] = pd.to_datetime(supplement_df['时间'])
    print(f"时间范围: {supplement_df['时间'].min()} 至 {supplement_df['时间'].max()}")
else:
    print(f"警告：补充数据文件不存在: {supplement_path}")
    supplement_df = None

# 提取中债指数
cbce_data = supplement_df[supplement_df['简称'] == '中债综合指数(总值)全价指数'].copy()
cbce_data = cbce_data[['时间', '收盘价(元)']].rename(columns={'时间': 'date', '收盘价(元)': 'CBCE'})
cbce_data = cbce_data.set_index('date').sort_index()
print(f"\n中债综合指数数据: {len(cbce_data)} 条")

# 提取南华商品指数
nhci_data = supplement_df[supplement_df['简称'] == '南华商品'].copy()
nhci_data = nhci_data[['时间', '收盘价(元)']].rename(columns={'时间': 'date', '收盘价(元)': 'NHCI'})
nhci_data = nhci_data.set_index('date').sort_index()
print(f"南华商品指数数据: {len(nhci_data)} 条")

# 保存到data文件夹
cbce_data.to_csv('data/CBCE.csv')
nhci_data.to_csv('data/NHCI.csv')
print("\n补充数据已保存至 data/CBCE.csv 和 data/NHCI.csv")