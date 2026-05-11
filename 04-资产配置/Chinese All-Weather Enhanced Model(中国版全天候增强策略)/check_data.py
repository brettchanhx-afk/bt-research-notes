import pandas as pd
import config

print('=== 配置的期望时间区间 ===')
print(f'开始日期: {config.DATA_CONFIG["start_date"]}')
print(f'结束日期: {config.DATA_CONFIG["end_date"]}')
print()

print('=== 实际数据时间区间 ===')
try:
    price_data = pd.read_csv('output/price_data.csv', index_col=0, parse_dates=True)
    print(f'数据行数: {len(price_data)} 行')
    print(f'实际开始日期: {price_data.index.min()}')
    print(f'实际结束日期: {price_data.index.max()}')
    print()
    print('=== 各资产缺失情况 ===')
    for col in price_data.columns:
        missing = price_data[col].isna().sum()
        if missing > 0:
            print(f'{col}: 缺失 {missing} 个数据点')
        else:
            print(f'{col}: 无缺失')
except Exception as e:
    print(f'读取数据失败: {e}')

print()
print('=== 资产列表 ===')
print('配置中期望的资产:')
for category, assets in config.ASSET_POOL.items():
    for code, name in assets.items():
        print(f'  {category}: {code} ({name})')