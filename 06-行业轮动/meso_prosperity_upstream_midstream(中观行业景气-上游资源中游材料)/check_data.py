import pandas as pd
import os

data_dir = 'd:/Documents/trae_projects/meso_prosperity_upstream_midstream/data'

for enc in ['gbk', 'gb2312', 'gb18030', 'utf-8']:
    try:
        roe_df = pd.read_csv(os.path.join(data_dir, '行业ROE_TTM历史数据.csv'), encoding=enc)
        print('=== 行业ROE_TTM历史数据 ===')
        print('编码:', enc)
        print('列数:', len(roe_df.columns))
        print('行数:', len(roe_df))
        print('日期范围:', roe_df.iloc[0, 0], '到', roe_df.iloc[-1, 0])
        print('列名前10:', list(roe_df.columns)[:10])
        print()
        break
    except Exception as e:
        print(enc, '失败:', str(e)[:60])

industry_files = [
    '石油石化行业中观景气度代理指标.csv',
    '煤炭行业中观景气度代理指标.csv',
    '有色金属行业中观景气度代理指标.csv',
    '钢铁行业中观景气度代理指标.csv',
    '基础化工行业中观景气度代理指标.csv',
    '建材行业中观景气度代理指标.csv'
]

for f in industry_files:
    try:
        df = pd.read_excel(os.path.join(data_dir, f))
        print('=== ' + f.replace('.csv', '') + ' ===')
        print('行数:', len(df), ', 列数:', len(df.columns))
        print('指标数:', len(df.columns) - 1)
        print()
    except Exception as e:
        print(f + ': 读取失败 - ' + str(e)[:100])
