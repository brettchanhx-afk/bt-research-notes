import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append('.')
from source.data_fetcher import DataFetcher
import pandas as pd

data_fetcher = DataFetcher()

asset_config = {
    '沪深300': '000300.SH',
    '中证1000': '000852.SH',
    '恒生指数': 'HSI.HK',
    '标普500': 'SPX.GI',
    '中债-国债总财富': 'CBA00101.CI',
    '中债-企业债总财富': 'CBA00201.CI',
    '南华商品指数': 'NH0100.NH',
    'COMEX黄金': 'GC00Y.GC',
    'ICE布油': 'CL00Y.NYM',
    '美元指数': 'DX0001.DXY',
}

start_date = '20170101'
end_date = '20231231'

print("=== 数据获取测试 ===")
print(f"时间范围: {start_date} 到 {end_date}")
print()

available_data = {}
missing_assets = []

for asset_name, ts_code in asset_config.items():
    print(f"正在获取: {asset_name} ({ts_code})")
    
    try:
        df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
        
        if len(df) > 0:
            print(f"  OK 成功获取 {len(df)} 条数据")
            available_data[asset_name] = df
        else:
            print(f"  FAIL 获取失败或数据为空")
            missing_assets.append(asset_name)
            
    except Exception as e:
        print(f"  FAIL 获取失败: {str(e)[:50]}")
        missing_assets.append(asset_name)
        
    print()

print("=" * 50)
print(f"成功获取: {len(available_data)} 个资产")
print(f"缺失资产: {len(missing_assets)} 个")

if missing_assets:
    print("\n缺失的资产列表:")
    for asset in missing_assets:
        print(f"  - {asset}")
    print("\n请提供以下数据文件或确认数据源:")
    for asset in missing_assets:
        print(f"  - {asset}.csv (包含 trade_date 和 close 列)")

if available_data:
    print("\n已获取的数据:")
    for name, df in available_data.items():
        print(f"  - {name}: {len(df)} 条记录")
    
    combined_df = pd.DataFrame()
    for name, df in available_data.items():
        if 'returns' in df.columns:
            combined_df[name] = df['returns']
    
    if len(combined_df) > 0:
        output_path = 'data/asset_returns.csv'
        combined_df.to_csv(output_path, encoding='utf-8')
        print(f"\n已保存可用数据到: {output_path}")

print("\n=== 测试完成 ===")