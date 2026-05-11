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
print("基于层次聚类改进风险平价策略 - 快速回测测�?)
print("=" * 60)

print("\n[Step 1] 数据加载...")
print("-" * 40)

import tushare as ts
token = '1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb'
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = 'http://jiaoch.site'

assets = {
    '沪深300': '000300.SH',
    '恒生指数': 'HSI.HI',
    '日经225': 'N225.GI',
    '标普500': 'SPX.GI',
    'COMEX黄金': 'GC.CMX',
    'ICE布油': 'B.IPE',
    'SHFE�?: 'CU.SHF',
}

price_data = {}

for name, code in assets.items():
    try:
        print(f"获取 {name} ({code})...", end=" ")
        if code in ['N225.GI', 'SPX.GI', 'GC.CMX', 'B.IPE']:
            import yfinance as yf
            ticker = yf.Ticker(code)
            df = ticker.history(start='2015-01-01', end='2015-12-31')
            if df is not None and len(df) > 0:
                price_data[name] = df['Close']
                print(f"{len(df)} �?)
            else:
                print("无数�?)
        else:
            df = pro.index_daily(ts_code=code, start_date='20150101', end_date='20151231')
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                price_data[name] = df['close']
                print(f"{len(df)} �?)
            else:
                print("无数�?)
    except Exception as e:
        print(f"失败: {e}")

print(f"\n成功获取 {len(price_data)} 个资产的数据")

if len(price_data) < 3:
    print("数据不足，使用模拟数据继�?..")
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2024-09-30', freq='ME')
    n = len(dates)
    for name in ['沪深300', '恒生指数', '日经225', '标普500', 'COMEX黄金']:
        price_data[name] = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)), index=dates)

prices_df = pd.DataFrame(price_data)
prices_df = prices_df.dropna()
print(f"价格数据形状: {prices_df.shape}")

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

def run_strategy(name, model, data):
    print(f"运行策略: {name}")
    try:
        weights = model.fit(data)
        return weights
    except Exception as e:
        print(f"  错误: {e}")
        return None

hrp_model = HierarchicalRiskParity(method='hrp')
naive_model = HierarchicalRiskParity(method='naive')
vol_model = HierarchicalRiskParity(method='volatility')
rp_model = RiskParity()

all_weights = {}
for name, model in [('HRP', hrp_model), ('Naive_HRP', naive_model), ('Vol_HRP', vol_model), ('RiskParity', rp_model)]:
    w = run_strategy(name, model, monthly_returns)
    if w is not None:
        all_weights[name] = w
        print(f"  权重: {w}")

print("\n计算组合收益...")
portfolio_returns = {}
for name, weights in all_weights.items():
    w_array = np.array([weights.get(col, 0) for col in monthly_returns.columns])
    returns = (monthly_returns * w_array).sum(axis=1)
    portfolio_returns[name] = returns
    cumulative = (1 + returns).cumprod()
    print(f"  {name}: 累计收益�?{(cumulative.iloc[-1]-1)*100:.2f}%")

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

print("\n===== 绩效指标汇�?=====\n")
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
print("\n===== 年度收益明细 =====\n")
print((yearly_returns * 100).round(2).to_string())

plt = evaluator.plot_yearly_returns_heatmap(figsize=(14, 8))
plt.savefig(f'{OUTPUT_DIR}/yearly_returns_heatmap.png', dpi=150)
plt.close()
print(f"\n年度收益热力图已保存�? {OUTPUT_DIR}/yearly_returns_heatmap.png")

print("\n[Step 10] 保存结果...")
print("-" * 40)
returns_df.to_csv(f'{OUTPUT_DIR}/backtest_results.csv')
print(f"回测结果已保存至: {OUTPUT_DIR}/backtest_results.csv")

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
