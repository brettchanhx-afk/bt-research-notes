import pandas as pd
import numpy as np
import os

def process_asset_data():
    data_path = 'data/asset_alloc_Data.csv'
    output_path = 'data/processed_asset_data.csv'

    for encoding in ['gbk', 'gb18030', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(data_path, index_col=0, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue

    df.columns = ['沪深300', '新华商品指数', '债券指数', '标普500', '上证50', '纳斯达克100', '中证REITs']

    df = df.replace('--', np.nan)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.index = pd.to_datetime(df.index, errors='coerce')

    valid_dates = df.index[df.index.notna()]
    if len(valid_dates) == 0:
        print("警告: 无法解析日期，请检查数据格式")
        print(f"原始索引示例: {list(df.index[:5])}")
        return None

    df = df.loc[valid_dates]
    df = df.sort_index()

    print("=" * 60)
    print("        补充数据概况")
    print("=" * 60)
    print(f"时间范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"总交易日: {len(df)} 天")
    print()
    print("各资产数据完整性:")
    for col in df.columns:
        valid_count = df[col].notna().sum()
        valid_pct = valid_count / len(df) * 100
        print(f"  {col}: {valid_count} 条 ({valid_pct:.1f}%)")

    df.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n处理后数据已保存到: {output_path}")

    return df

def prepare_backtest_data(start_date='20210101', end_date='20240111'):
    df = pd.read_csv('data/processed_asset_data.csv', index_col=0, encoding='utf-8-sig')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.dropna(subset=[df.columns[0]])

    df = df[(df.index >= start_date) & (df.index <= end_date)]

    print("\n" + "=" * 60)
    print("        回测区间数据")
    print("=" * 60)
    print(f"回测时间: {start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 至 {end_date[:4]}-{end_date[4:6]}-{end_date[6:]}")
    print(f"交易日数: {len(df)} 天")
    print()

    for col in df.columns:
        valid_count = df[col].notna().sum()
        print(f"  {col}: {valid_count} 条有效数据")

    backtest_df = df.dropna()
    print(f"\n所有资产都有数据的交易日: {len(backtest_df)} 天")

    if len(backtest_df) < 100:
        print("警告: 完整数据较少，将使用尽量多的数据回测")

    return df, backtest_df

if __name__ == '__main__':
    df = process_asset_data()
    if df is not None:
        prepare_backtest_data()