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

print("Generating final outputs...")

print("\nLoading data...")
prices_df = pd.read_csv(f'{OUTPUT_DIR}/price_data.csv', index_col=0, parse_dates=True)
monthly_returns = prices_df.pct_change().dropna()

print("Calculating correlation matrix...")
corr_matrix = monthly_returns.corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Asset Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=150)
plt.close()
print("Correlation matrix saved.")

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

from source.hrp_strategy import RiskParity

all_weights = {}
model = RiskParity()
weights = model.fit(monthly_returns)
all_weights['RiskParity'] = weights

w_array = np.array([weights.get(col, 0) for col in monthly_returns.columns])
returns = (monthly_returns * w_array).sum(axis=1)
portfolio_returns = {'RiskParity': returns}

returns_df = pd.DataFrame(portfolio_returns)

plt.figure(figsize=(14, 8))
cumulative = (1 + returns_df['RiskParity']).cumprod()
plt.plot(cumulative.index, cumulative.values, label='RiskParity', linewidth=2, color='#1f77b4')
plt.title('Risk Parity Strategy NAV', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('NAV', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/nav_comparison.png', dpi=150)
plt.close()
print("NAV curve saved.")

from source.performance import PerformanceEvaluator

evaluator = PerformanceEvaluator(portfolio_returns)
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

yearly_returns = evaluator.get_yearly_returns()
print("\nYearly Returns:")
print((yearly_returns * 100).round(2).to_string())

plt = evaluator.plot_yearly_returns_heatmap(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/yearly_returns_heatmap.png', dpi=150)
plt.close()
print("Yearly returns heatmap saved.")

returns_df.to_csv(f'{OUTPUT_DIR}/backtest_results.csv')
evaluator.generate_report(output_path=f'{OUTPUT_DIR}/performance_report.csv')

print("\n" + "=" * 60)
print("All outputs generated!")
print("=" * 60)
print("\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  - {f} ({size/1024:.1f} KB)")
