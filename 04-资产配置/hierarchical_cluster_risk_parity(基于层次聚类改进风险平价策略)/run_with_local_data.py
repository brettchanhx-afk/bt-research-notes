import sys
sys.path.insert(0, '.')

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Hierarchical Risk Parity Strategy - Using Local Data")
print("=" * 60)

print("\n[Step 1] Loading price data from CSV...")
df = pd.read_csv('data/hierarchical_cluster_10assets_price_2013.csv', index_col=0, header=[0, 1])
df.index = pd.to_datetime(df.index)
df = df.dropna(axis=1, how='all')
df = df.replace('--', np.nan)
df = df.astype(float)
df = df.dropna(how='all')
print(f"Loaded data shape: {df.shape}")
print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

asset_names = ['850531.SL', 'STIP.P', '000300.SH', 'HSI.HK', 'SPX.GI',
               'N225.GI', 'BRN0Y.ICE', 'CU00.SHF', 'CBA02001.CB', 'CBA00603.CB']
available_assets = [col for col in asset_names if col in df.columns]
df = df[available_assets]
print(f"Available assets ({len(available_assets)}): {list(df.columns)}")

print("\n[Step 2] Saving price data...")
price_data = df.copy()
price_data.to_csv(f'{OUTPUT_DIR}/price_data.csv')
print(f"Price data saved to: {OUTPUT_DIR}/price_data.csv")

print("\n[Step 3] Calculating monthly returns...")
monthly_prices = df.resample('M').last()
monthly_returns = monthly_prices.pct_change().dropna()
print(f"Monthly returns shape: {monthly_returns.shape}")

print("\n[Step 4] Calculating correlation matrix...")
corr_matrix = monthly_returns.corr()

plt.figure(figsize=(12, 10))
im = plt.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Asset Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=150)
plt.close()
print("Correlation matrix saved.")

print("\n[Step 5] Hierarchical clustering analysis...")
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))
np.fill_diagonal(distance_matrix, 0)
condensed_dist = squareform(distance_matrix, checks=False)
Z = linkage(condensed_dist, method='single')

plt.figure(figsize=(14, 8))
dendrogram(Z, labels=list(corr_matrix.columns))
plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
plt.xlabel('Assets', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/dendrogram.png', dpi=150)
plt.close()
print("Dendrogram saved.")

print("\n[Step 6] Running strategy backtest...")
from source.hrp_strategy import HierarchicalRiskParity, RiskParity
from source.backtest import run_backtest_from_data

backtest = run_backtest_from_data(price_data, transaction_cost=0.0005, rebalance_freq='monthly')
results = backtest.get_results()

print("\n[Step 7] Plotting NAV curves...")
plt = backtest.plot_nav(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/nav_comparison.png', dpi=150)
plt.close()
print("NAV curve saved.")

print("\n[Step 8] Performance evaluation...")
from source.performance import PerformanceEvaluator

returns_dict = {name: res['returns'] for name, res in results.items()}
evaluator = PerformanceEvaluator(returns_dict)
metrics_df = evaluator.evaluate_all()

print("\n" + "=" * 60)
print("Performance Metrics Summary")
print("=" * 60)
print(metrics_df.to_string(index=False))

plt = evaluator.plot_metrics_comparison(figsize=(16, 12))
plt.savefig(f'{OUTPUT_DIR}/metrics_comparison.png', dpi=150)
plt.close()
print("Metrics comparison saved.")

plt = evaluator.plot_drawdown(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/drawdown_comparison.png', dpi=150)
plt.close()
print("Drawdown comparison saved.")

plt = evaluator.plot_yearly_returns_heatmap(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/yearly_returns_heatmap.png', dpi=150)
plt.close()
print("Yearly returns heatmap saved.")

print("\n[Step 9] Saving backtest results...")
backtest.save_results(filepath=f'{OUTPUT_DIR}/backtest_results.csv')
evaluator.generate_report(output_path=f'{OUTPUT_DIR}/performance_report.csv')

print("\n" + "=" * 60)
print("All outputs generated successfully!")
print("=" * 60)
print("\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  - {f} ({size/1024:.1f} KB)")
