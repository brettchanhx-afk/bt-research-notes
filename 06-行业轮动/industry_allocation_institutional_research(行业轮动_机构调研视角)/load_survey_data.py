import pandas as pd
import numpy as np
import os

DATA_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\data'
OUTPUT_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output'
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, 'data')

def load_survey_data_chunks(chunk_size=500):
    survey_file = os.path.join(DATA_DIR, '全部A股_当月被调研总次数2010_2026.xlsx')

    print(f"读取调研数据: {survey_file}")
    print("  使用分块读取模式...")

    cols_df = pd.read_excel(survey_file, nrows=0)
    all_cols = cols_df.columns.tolist()
    print(f"  总列数: {len(all_cols)}")

    date_col = 'Unnamed: 0'
    stock_cols = [c for c in all_cols if c != date_col]

    chunks = []
    for i in range(0, len(stock_cols), chunk_size):
        chunk_cols = [date_col] + stock_cols[i:i+chunk_size]
        chunk_df = pd.read_excel(survey_file, usecols=chunk_cols)
        chunks.append(chunk_df)
        print(f"  读取列块 {i//chunk_size + 1}/{(len(stock_cols) + chunk_size - 1)//chunk_size}")

    df = pd.concat(chunks, axis=1)
    return df

def transform_survey_data(df):
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df = df.T
    df.index.name = 'ts_code'
    df = df.reset_index()

    return df

if __name__ == '__main__':
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("加载机构调研数据")
    print("=" * 60)

    survey_df = load_survey_data_chunks()

    print(f"\n数据形状: {survey_df.shape}")

    survey_df.to_csv(os.path.join(OUTPUT_DATA_DIR, 'survey_data.csv'), index=False)
    print(f"数据已保存: {os.path.join(OUTPUT_DATA_DIR, 'survey_data.csv')}")

    print("\n数据预览:")
    print(survey_df.iloc[:5, :6])