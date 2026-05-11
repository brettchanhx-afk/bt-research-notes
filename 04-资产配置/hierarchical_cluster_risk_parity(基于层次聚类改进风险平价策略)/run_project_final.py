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
print("Hierarchical Risk Parity Strategy - Full Backtest")
print("=" * 60)

print("\n[Step 1] Data Loading...")
print("-" * 40)

import tushare as ts
token = '1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb'
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = 'http://jiaoch.site'

assets_config = {
    'CSI300': '000300.SH',
    'HSI': 'HSI.HI',
    'Nikkei225': 'N225.GI',
    'SP500': 'SPX.GI',
    'Gold': 'GC.CMX',
    'CrudeOil': 'B.IPE',
    'Copper': 'CU.SHF',
    'USTBond': 'IEF',
    'CNBond5Y': 'CBA00641.CS',
    'CNCorpAAA': 'CBA04201.CS',
}

price_data = {}
failed_assets = []

for name, code in assets_config.items():
    try:
        print(f"Loading {name} ({code})...", end=" ", flush=True)

        if code.endswith('.SH') or code.endswith('.SZ') or code.endswith('.BI'):
            df = pro.index_daily(ts_code=code, start_date='20070101', end_date='20240930')
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                price_data[name] = df['close']
                print(f"{len(df)} records")
            else:
                failed_assets.append(name)
                print("No data (Tushare)")
        else:
            failed_assets.append(name)
            print(f"Skip (needs other source)")

    except Exception as e:
        failed_assets.append(name)
        print(f"Failed")

print(f"\nSuccessfully loaded {len(price_data)} assets via tushare:")
for name in price_data.keys():
    print(f"  - {name}: {len(price_data[name])} records")

print(f"\nFailed assets ({len(failed_assets)}): {failed_assets}")

print("\n[Step 2] Generating simulated data for missing assets...")
print("-" * 40)

np.random.seed(42)
dates = pd.date_range('2007-01-01', '2024-09-30', freq='B')
n = len(dates)

simulated_assets = {
    'HSI': 0.05,
    'Nikkei225': 0.06,
    'SP500': 0.08,
    'Gold': 0.04,
    'CrudeOil': 0.10,
    'Copper': 0.08,
    'USTBond': 0.02,
}

for name, vol in simulated_assets.items():
    if name not in price_data:
        log_returns = np.random.randn(n) * vol / np.sqrt(252)
        prices = pd.Series(100 * np.exp(np.cumsum(log_returns)), index=dates)
        price_data[name] = prices
        print(f"  Added simulated: {name}")

print("\n[Step 3] Merging data and calculating returns...")
print("-" * 40)
prices_df = pd.DataFrame(price_data)
prices_df = prices_df.dropna()
prices_df = prices_df.resample('M').last()
prices_df = prices_df.dropna()
print(f"Merged price data shape: {prices_df.shape}")
print(f"Time range: {prices_df.index.min().strftime('%Y-%m')} to {prices_df.index.max().strftime('%Y-%m')}")

monthly_returns = prices_df.pct_change().dropna()
print(f"Monthly returns data shape: {monthly_returns.shape}")

print("\n[Step 4] Saving price data...")
print("-" * 40)
prices_df.to_csv(f'{OUTPUT_DIR}/price_data.csv')
print(f"Price data saved to: {OUTPUT_DIR}/price_data.csv")

print("\n[Step 5] Calculating correlation matrix...")
print("-" * 40)
corr_matrix = monthly_returns.corr()
print("\nAsset Correlation Matrix:")
print(corr_matrix.round(2).to_string())

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Asset Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=150)
plt.close()
print(f"\nCorrelation matrix saved to: {OUTPUT_DIR}/correlation_matrix.png")

print("\n[Step 6] Hierarchical clustering analysis...")
print("-" * 40)
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
print(f"Dendrogram saved to: {OUTPUT_DIR}/dendrogram.png")

print("\n[Step 7] Strategy backtest...")
print("-" * 40)

from source.hrp_strategy import HierarchicalRiskParity, RiskParity

all_weights = {}

models = [
    ('HRP', HierarchicalRiskParity(method='hrp')),
    ('Naive_HRP', HierarchicalRiskParity(method='naive')),
    ('Vol_HRP', HierarchicalRiskParity(method='volatility')),
    ('RiskParity', RiskParity()),
]

for name, model in models:
    print(f"Running strategy: {name}...", end=" ", flush=True)
    try:
        weights = model.fit(monthly_returns)
        all_weights[name] = weights
        print("Done")
        print(f"  Weights: {', '.join([f'{k}:{v:.2%}' for k,v in weights.items()])}")
    except Exception as e:
        print(f"Error: {e}")

print("\nCalculating portfolio returns...")
portfolio_returns = {}
for name, weights in all_weights.items():
    w_array = np.array([weights.get(col, 0) for col in monthly_returns.columns])
    returns = (monthly_returns * w_array).sum(axis=1)
    portfolio_returns[name] = returns
    cumulative = (1 + returns).cumprod()
    final_return = (cumulative.iloc[-1] - 1) * 100
    print(f"  {name}: Cumulative return {final_return:.2f}%")

returns_df = pd.DataFrame(portfolio_returns)

print("\n[Step 8] Plotting NAV curves...")
print("-" * 40)
plt.figure(figsize=(14, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, name in enumerate(returns_df.columns):
    cumulative = (1 + returns_df[name]).cumprod()
    plt.plot(cumulative.index, cumulative.values, label=name, linewidth=2, color=colors[i % len(colors)])

plt.title('Strategy NAV Comparison', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('NAV', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/nav_comparison.png', dpi=150)
plt.close()
print(f"NAV curve saved to: {OUTPUT_DIR}/nav_comparison.png")

print("\n[Step 9] Performance evaluation...")
print("-" * 40)
from source.performance import PerformanceEvaluator

evaluator = PerformanceEvaluator(portfolio_returns)
metrics_df = evaluator.evaluate_all()

print("\n" + "=" * 60)
print("Performance Metrics Summary")
print("=" * 60)
print(metrics_df.to_string(index=False))

print("\n[Step 10] Plotting metrics comparison...")
print("-" * 40)
plt = evaluator.plot_metrics_comparison(figsize=(16, 12))
plt.savefig(f'{OUTPUT_DIR}/metrics_comparison.png', dpi=150)
plt.close()
print(f"Metrics comparison saved to: {OUTPUT_DIR}/metrics_comparison.png")

print("\n[Step 11] Plotting drawdown...")
print("-" * 40)
plt = evaluator.plot_drawdown(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/drawdown_comparison.png', dpi=150)
plt.close()
print(f"Drawdown comparison saved to: {OUTPUT_DIR}/drawdown_comparison.png")

print("\n[Step 12] Plotting yearly returns heatmap...")
print("-" * 40)
yearly_returns = evaluator.get_yearly_returns()
print("\nYearly Returns Detail:")
print((yearly_returns * 100).round(2).to_string())

plt = evaluator.plot_yearly_returns_heatmap(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/yearly_returns_heatmap.png', dpi=150)
plt.close()
print(f"\nYearly returns heatmap saved to: {OUTPUT_DIR}/yearly_returns_heatmap.png")

print("\n[Step 13] Saving all results...")
print("-" * 40)
returns_df.to_csv(f'{OUTPUT_DIR}/backtest_results.csv')
print(f"Backtest results saved to: {OUTPUT_DIR}/backtest_results.csv")

evaluator.generate_report(output_path=f'{OUTPUT_DIR}/performance_report.csv')

print("\n" + "=" * 60)
print("Backtest Complete!")
print("=" * 60)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("\nOutput file list:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  - {f} ({size/1024:.1f} KB)")

print("\n" + "=" * 60)
print("Important Note")
print("=" * 60)
print("The following assets could not be obtained via tushare and use simulated data:")
for asset in failed_assets:
    print(f"  - {asset}")
print("\nFor accurate replication, consider:")
print("  1. Wind terminal for complete data")
print("  2. efinance/mootdx for China market data")
print("  3. yfinance for international assets with stable network")
