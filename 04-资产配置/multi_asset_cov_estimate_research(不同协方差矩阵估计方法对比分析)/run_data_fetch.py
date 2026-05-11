import sys
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
}

start_date = '20170101'
end_date = '20231231'

results = []

for asset_name, ts_code in asset_config.items():
    try:
        df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
        if len(df) > 0:
            results.append({'asset': asset_name, 'code': ts_code, 'count': len(df), 'status': 'OK'})
        else:
            results.append({'asset': asset_name, 'code': ts_code, 'count': 0, 'status': 'FAIL'})
    except Exception as e:
        results.append({'asset': asset_name, 'code': ts_code, 'count': 0, 'status': 'FAIL', 'error': str(e)[:50]})

result_df = pd.DataFrame(results)
result_df.to_csv('data_fetch_results.csv', index=False, encoding='utf-8')
print('Results saved to data_fetch_results.csv')