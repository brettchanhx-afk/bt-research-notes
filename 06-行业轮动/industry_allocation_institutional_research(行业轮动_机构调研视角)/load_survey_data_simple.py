import pandas as pd
import numpy as np
import os

DATA_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\data'
OUTPUT_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output'
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

survey_file = os.path.join(DATA_DIR, '全部A股_当月被调研总次数2010_2026.xlsx')

print("=" * 60)
print("加载机构调研数据")
print("=" * 60)

print("\n读取Excel文件 (可能需要几分钟)...")
df = pd.read_excel(survey_file, engine='openpyxl')
print(f"读取完成! Shape: {df.shape}")

df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])

print(f"\n数据预览:")
print(df.iloc[:5, :6].to_string())

save_path = os.path.join(OUTPUT_DATA_DIR, 'survey_data_raw.csv')
df.to_csv(save_path, index=False)
print(f"\n数据已保存: {save_path}")

print("\n完成!")