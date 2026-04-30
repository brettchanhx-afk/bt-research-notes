# Simplified test script
import warnings
warnings.filterwarnings('ignore')
import os, sys, numpy as np, pandas as pd

BASE_DIR = r'C:\Users\chenh\.qclaw\workspace\etf_clustering_system'
sys.path.insert(0, BASE_DIR)

print("Testing ETF clustering system...")

# Generate mock data
np.random.seed(42)
index_codes = ['000300.SH', '000016.SH', '399967.SZ', '000688.SH', '399986.SZ']

# Mock constituents
constituents_dict = {}
for idx in index_codes:
    n = 30
    stocks = pd.DataFrame({
        'con_code': [f'{600000+i}.SH' for i in range(n)],
        'weight': np.random.dirichlet(np.ones(n)),
    })
    constituents_dict[idx] = stocks

# Mock returns
dates = pd.date_range('2023-01-01', periods=250, freq='B')
index_returns = pd.DataFrame(
    1 + np.random.randn(250, len(index_codes)) * 0.015,
    index=dates, columns=index_codes
).cumprod()

print("Data generated:")
print(f"  - Indices: {len(constituents_dict)}")
print(f"  - Dates: {len(dates)}")

# Run clustering
from source.clustering import cluster_indices_by_constituents
result = cluster_indices_by_constituents(constituents_dict, index_returns)

print(f"Clustering completed:")
print(f"  - Clusters: {result['n_clusters']}")
print(f"  - Silhouette: {result['silhouette_score']:.4f}")

print("\nSUCCESS!")