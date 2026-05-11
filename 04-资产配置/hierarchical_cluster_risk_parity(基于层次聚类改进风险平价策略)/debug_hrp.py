import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

df = pd.read_csv('data/hierarchical_cluster_10assets_price_2013.csv', index_col=0, header=[0, 1])
df.index = pd.to_datetime(df.index)

column_codes = ['850531.SL', 'STIP.P', '000300.SH', 'HSI.HK', 'SPX.GI',
               'N225.GI', 'BRN0Y.ICE', 'CU00.SHF', 'CBA02001.CB', 'CBA00603.CB']
available_cols = [col for col in column_codes if col in df.columns.get_level_values(0)]
df = df[available_cols].droplevel(1, axis=1)

df = df.replace('--', np.nan).astype(float).dropna(how='all', axis=0)

monthly_prices = df.resample('M').last()
monthly_returns = monthly_prices.pct_change().dropna()

print("Testing HRP on different time periods...")

from source.hrp_strategy import HierarchicalRiskParity

for i in range(100, len(monthly_returns)):
    train_data = monthly_returns.iloc[i-6:i]
    try:
        model = HierarchicalRiskParity(method='hrp')
        weights = model.fit(train_data)
        if i > 140:
            print(f"Period {i}: OK - {list(weights.keys())[:3]}...")
    except Exception as e:
        print(f"Period {i}: FAILED - {type(e).__name__}: {e}")
        if i > 140:
            import traceback
            traceback.print_exc()
            break
