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

from source import (
    DataLoader,
    HierarchicalRiskParity,
    RiskParity,
    Backtest,
    PerformanceEvaluator
)

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("基于层次聚类改进风险平价策略 - 完整回测")
print("=" * 60)

print("\n[Step 1] 数据加载...")
print("-" * 40)
loader = DataLoader(start_date='20070101', end_date='20240930')
loader.load_all_data()
print(f"\n成功获取 {len(loader.price_data)} 个资产的数据")

for name, data in loader.price_data.items():
    print(f"  - {name}: {len(data)} 条数�?)

print("\n[Step 2] 计算收益�?..")
print("-" * 40)
monthly_returns = loader.get_returns(freq='monthly')
print(f"月收益率数据形状: {monthly_returns.shape}")
print(f"时间范围: {monthly_returns.index.min()} �?{monthly_returns.index.max()}")

print("\n[Step 3] 保存原始数据...")
print("-" * 40)
loader.save_data(filepath=f'{OUTPUT_DIR}/price_data.csv')
print(f"数据已保存至: {OUTPUT_DIR}/price_data.csv")

print("\n[Step 4] 计算相关性矩�?..")
print("-" * 40)
corr_matrix = monthly_returns.corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Asset Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=150)
plt.close()
print(f"相关性矩阵已保存�? {OUTPUT_DIR}/correlation_matrix.png")

print("\n[Step 5] 层次聚类分析...")
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
print(f"聚类树状图已保存�? {OUTPUT_DIR}/dendrogram.png")

print("\n[Step 6] 策略回测...")
print("-" * 40)
backtest = Backtest(
    returns_df=monthly_returns,
    transaction_cost=0.0005,
    rebalance_freq='monthly'
)
results = backtest.run_all_strategies()

print("\n[Step 7] 绘制净值曲�?..")
print("-" * 40)
plt.figure(figsize=(14, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (name, cumulative) in enumerate(backtest.get_cumulative_returns().items()):
    plt.plot(cumulative.index, cumulative.values, label=name, linewidth=2, color=colors[i])

plt.title('Strategy NAV Comparison', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('NAV', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/nav_comparison.png', dpi=150)
plt.close()
print(f"净值曲线已保存�? {OUTPUT_DIR}/nav_comparison.png")

print("\n[Step 8] 绩效评估...")
print("-" * 40)
evaluator = PerformanceEvaluator(backtest.results)
metrics_df = evaluator.evaluate_all()

print("\n===== 绩效指标汇�?=====\n")
print(metrics_df.to_string(index=False))

print("\n[Step 9] 绘制指标对比�?..")
print("-" * 40)
plt = evaluator.plot_metrics_comparison(figsize=(16, 12))
plt.savefig(f'{OUTPUT_DIR}/metrics_comparison.png', dpi=150)
plt.close()
print(f"指标对比图已保存�? {OUTPUT_DIR}/metrics_comparison.png")

print("\n[Step 10] 绘制回撤对比�?..")
print("-" * 40)
plt = evaluator.plot_drawdown(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/drawdown_comparison.png', dpi=150)
plt.close()
print(f"回撤对比图已保存�? {OUTPUT_DIR}/drawdown_comparison.png")

print("\n[Step 11] 绘制年度收益热力�?..")
print("-" * 40)
yearly_returns = evaluator.get_yearly_returns()
print("\n===== 年度收益明细 =====\n")
print((yearly_returns * 100).round(2).to_string())

plt = evaluator.plot_yearly_returns_heatmap(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/yearly_returns_heatmap.png', dpi=150)
plt.close()
print(f"\n年度收益热力图已保存�? {OUTPUT_DIR}/yearly_returns_heatmap.png")

print("\n[Step 12] 绘制资产权重分布...")
print("-" * 40)
weights_history = backtest.get_Weights_history()

for strategy_name in weights_history.keys():
    latest_weights = pd.DataFrame(weights_history[strategy_name]).iloc[-1]

    plt.figure(figsize=(12, 6))
    colors_bar = ['#ff6b6b' if w < 0 else '#4ecdc4' for w in latest_weights.values]
    plt.bar(range(len(latest_weights)), latest_weights.values, color=colors_bar)
    plt.xticks(range(len(latest_weights)), latest_weights.index, rotation=45, ha='right')
    plt.title(f'{strategy_name} - Latest Asset Weights', fontsize=14)
    plt.xlabel('Asset', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/weights_{strategy_name}.png', dpi=150)
    plt.close()
    print(f"权重分布图已保存�? {OUTPUT_DIR}/weights_{strategy_name}.png")

print("\n[Step 13] 保存回测结果...")
print("-" * 40)
backtest.save_results(filepath=f'{OUTPUT_DIR}/backtest_results.csv')

print("\n[Step 14] 生成绩效报告...")
print("-" * 40)
evaluator.generate_report(output_path=f'{OUTPUT_DIR}/performance_report.csv')

print("\n" + "=" * 60)
print("回测完成!")
print("=" * 60)
print(f"\n所有结果已保存�? {OUTPUT_DIR}/")
print("\n输出文件列表:")
for f in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  - {f} ({size/1024:.1f} KB)")
