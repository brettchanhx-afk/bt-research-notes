import sys
sys.path.insert(0, '.')

import os
import warnings
warnings.filterwarnings('ignore')
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("基于层次聚类改进风险平价策略 - 回测")
print("=" * 60)

print("\n[Step 1] 数据加载 (仅使用tushare)...")
print("-" * 40)

import tushare as ts
token = '1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb'
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = 'http://jiaoch.site'

assets_config = [
    ('沪深300', '000300.SH'),
    ('恒生指数', 'HSI.HI'),
    ('日经225', 'N225.GI'),
    ('标普500', 'SPX.GI'),
    ('COMEX黄金', 'GC.CMX'),
    ('ICE布油', 'B.IPE'),
    ('SHFE�?, 'CU.SHF'),
]

price_data = {}

for name, code in assets_config:
    try:
        print(f"获取 {name} ({code})...", end=" ", flush=True)
        df = pro.index_daily(ts_code=code, start_date='20070101', end_date='20240930', timeout=30)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df.set_index('trade_date', inplace=True)
            price_data[name] = df['close']
            print(f"{len(df)} �?)
        else:
            print("无数�?)
    except Exception as e:
        print(f"失败: {str(e)[:50]}")

print(f"\n成功获取 {len(price_data)} 个资产的数据")

if len(price_data) < 3:
    print("错误: 数据不足!")
    sys.exit(1)

prices_df = pd.DataFrame(price_data)
prices_df = prices_df.dropna()
print(f"价格数据形状: {prices_df.shape}")
print(f"时间范围: {prices_df.index.min()} �?{prices_df.index.max()}")

print("\n[Step 2] 计算收益�?..")
print("-" * 40)
monthly_prices = prices_df.resample('M').last()
monthly_returns = monthly_prices.pct_change().dropna()
print(f"月收益率数据形状: {monthly_returns.shape}")

print("\n[Step 3] 层次聚类分析...")
print("-" * 40)
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

corr_matrix = monthly_returns.corr()
print("\n相关性矩�?")
print(corr_matrix.round(2))

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
print(f"\n聚类树状图已保存�? {OUTPUT_DIR}/dendrogram.png")

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

print("\n[Step 4] 策略回测...")
print("-" * 40)

from source.hrp_strategy import HierarchicalRiskParity, RiskParity

all_weights = {}

models = [
    ('层次风险平价(HRP)', HierarchicalRiskParity(method='hrp')),
    ('朴素层次风险平价', HierarchicalRiskParity(method='naive')),
    ('基于波动率HRP', HierarchicalRiskParity(method='volatility')),
    ('传统风险平价', RiskParity()),
]

for name, model in models:
    print(f"运行策略: {name}...", end=" ", flush=True)
    try:
        weights = model.fit(monthly_returns)
        all_weights[name] = weights
        print("完成")
    except Exception as e:
        print(f"错误: {e}")

print("\n计算组合收益...")
portfolio_returns = {}
for name, weights in all_weights.items():
    w_array = np.array([weights.get(col, 0) for col in monthly_returns.columns])
    returns = (monthly_returns * w_array).sum(axis=1)
    portfolio_returns[name] = returns
    cumulative = (1 + returns).cumprod()
    final_return = (cumulative.iloc[-1] - 1) * 100
    print(f"  {name}: 累计收益�?{final_return:.2f}%")

returns_df = pd.DataFrame(portfolio_returns)

print("\n[Step 5] 绘制净值曲�?..")
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
print(f"净值曲线已保存�? {OUTPUT_DIR}/nav_comparison.png")

print("\n[Step 6] 绩效评估...")
print("-" * 40)
from source.performance import PerformanceEvaluator

evaluator = PerformanceEvaluator(portfolio_returns)
metrics_df = evaluator.evaluate_all()

print("\n" + "=" * 60)
print("绩效指标汇�?)
print("=" * 60)
print(metrics_df.to_string(index=False))

print("\n[Step 7] 绘制指标对比�?..")
print("-" * 40)
plt = evaluator.plot_metrics_comparison(figsize=(16, 12))
plt.savefig(f'{OUTPUT_DIR}/metrics_comparison.png', dpi=150)
plt.close()
print(f"指标对比图已保存�? {OUTPUT_DIR}/metrics_comparison.png")

print("\n[Step 8] 绘制回撤对比�?..")
print("-" * 40)
plt = evaluator.plot_drawdown(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/drawdown_comparison.png', dpi=150)
plt.close()
print(f"回撤对比图已保存�? {OUTPUT_DIR}/drawdown_comparison.png")

print("\n[Step 9] 绘制年度收益热力�?..")
print("-" * 40)
yearly_returns = evaluator.get_yearly_returns()
print("\n年度收益明细:")
print((yearly_returns * 100).round(2).to_string())

plt = evaluator.plot_yearly_returns_heatmap(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/yearly_returns_heatmap.png', dpi=150)
plt.close()
print(f"\n年度收益热力图已保存�? {OUTPUT_DIR}/yearly_returns_heatmap.png")

print("\n[Step 10] 保存所有结�?..")
print("-" * 40)
returns_df.to_csv(f'{OUTPUT_DIR}/backtest_results.csv')
print(f"回测结果已保存至: {OUTPUT_DIR}/backtest_results.csv")

prices_df.to_csv(f'{OUTPUT_DIR}/price_data.csv')
print(f"价格数据已保存至: {OUTPUT_DIR}/price_data.csv")

evaluator.generate_report(output_path=f'{OUTPUT_DIR}/performance_report.csv')

print("\n" + "=" * 60)
print("回测完成!")
print("=" * 60)
print(f"\n所有结果已保存�? {OUTPUT_DIR}/")
print("\n输出文件列表:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  - {f} ({size/1024:.1f} KB)")
