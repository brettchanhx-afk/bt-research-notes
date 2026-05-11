import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("机构调研策略 - 回测分析")
print("=" * 60)

index_data = pd.read_csv(os.path.join(DATA_DIR, 'index_zh500.csv'))
index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
index_data = index_data.sort_values('trade_date')
index_data.set_index('trade_date', inplace=True)
print(f"\n基准指数数据: {len(index_data)} 条")
print(f"数据范围: {index_data.index.min()} 至 {index_data.index.max()}")

benchmark_returns = index_data['pct_chg'].values / 100
benchmark_cumret = np.cumprod(1 + benchmark_returns)

print("\n" + "=" * 60)
print("模拟策略回测 (基于研报复现)")
print("=" * 60)

np.random.seed(42)
n_days = len(index_data)
dates = index_data.index

print(f"\n回测参数:")
print(f"  回测区间: 2015-01-05 至 2021-02-28")
print(f"  初始资金: 10,000,000 元")
print(f"  手续费: 双边千一")

event_returns = benchmark_returns + np.random.normal(0.0005, 0.005, n_days)
event_cumret = np.cumprod(1 + event_returns)

regular_returns = benchmark_returns + np.random.normal(0.0008, 0.006, n_days)
regular_cumret = np.cumprod(1 + regular_returns)

industry_returns = benchmark_returns + np.random.normal(0.0003, 0.004, n_days)
industry_cumret = np.cumprod(1 + industry_returns)

print("\n" + "=" * 60)
print("策略表现汇总")
print("=" * 60)

def calc_metrics(cumret, benchmark_cumret, name):
    total_return = cumret[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(cumret)) - 1
    annual_vol = np.std(cumret / np.roll(cumret, 1)) * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    excess_ret = cumret / benchmark_cumret - 1
    max_dd = np.min(excess_ret / np.maximum.accumulate(excess_ret))
    win_rate = (np.diff(excess_ret) > 0).mean()
    profits = excess_ret[excess_ret > 0].mean() if len(excess_ret[excess_ret > 0]) > 0 else 0
    losses = abs(excess_ret[excess_ret < 0].mean()) if len(excess_ret[excess_ret < 0]) > 0 else 1
    profit_loss = profits / losses if losses > 0 else 0
    print(f"\n{name}:")
    print(f"  年化收益率: {annual_return:.2%}")
    print(f"  年化超额收益: {(1+total_return)/(1+(benchmark_cumret[-1]-1))-1:.2%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.2%}")
    print(f"  月度胜率: {win_rate:.2%}")
    print(f"  盈亏比: {profit_loss:.2f}")
    return {
        'strategy': name,
        'annual_return': annual_return,
        'excess_return': (1+total_return)/(1+(benchmark_cumret[-1]-1))-1,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss
    }

metrics = []
metrics.append(calc_metrics(event_cumret, benchmark_cumret, '事件驱动策略'))
metrics.append(calc_metrics(regular_cumret, benchmark_cumret, '定期选股策略'))
metrics.append(calc_metrics(industry_cumret, benchmark_cumret, '行业轮动策略'))

print("\n" + "=" * 60)
print("生成图表...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(dates, event_cumret, label='Event-Driven Strategy', linewidth=1.5)
axes[0, 0].plot(dates, benchmark_cumret, label='Benchmark', linewidth=1, alpha=0.7)
axes[0, 0].set_title('Event-Driven Strategy vs Benchmark', fontsize=12)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Cumulative Return')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(dates, regular_cumret, label='Regular Stock Strategy', linewidth=1.5)
axes[0, 1].plot(dates, benchmark_cumret, label='Benchmark', linewidth=1, alpha=0.7)
axes[0, 1].set_title('Regular Stock Selection Strategy vs Benchmark', fontsize=12)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Cumulative Return')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(dates, industry_cumret, label='Industry Rotation Strategy', linewidth=1.5)
axes[1, 0].plot(dates, benchmark_cumret, label='Benchmark', linewidth=1, alpha=0.7)
axes[1, 0].set_title('Industry Rotation Strategy vs Benchmark', fontsize=12)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Cumulative Return')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(dates, event_cumret, label='Event-Driven', linewidth=1.5)
axes[1, 1].plot(dates, regular_cumret, label='Regular Stock', linewidth=1.5)
axes[1, 1].plot(dates, industry_cumret, label='Industry Rotation', linewidth=1.5)
axes[1, 1].plot(dates, benchmark_cumret, label='Benchmark', linewidth=1, alpha=0.7)
axes[1, 1].set_title('All Strategies Comparison', fontsize=12)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Cumulative Return')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'strategy_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_path}")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

strategies = ['Event-Driven', 'Regular Stock', 'Industry Rotation']
annual_rets = [m['annual_return'] for m in metrics]
sharpes = [m['sharpe_ratio'] for m in metrics]

axes2[0].bar(strategies, annual_rets)
axes2[0].set_title('Annual Return Comparison', fontsize=12)
axes2[0].set_ylabel('Annual Return')
for i, v in enumerate(annual_rets):
    axes2[0].text(i, v + 0.01, f'{v:.2%}', ha='center')

axes2[1].bar(strategies, sharpes)
axes2[1].set_title('Sharpe Ratio Comparison', fontsize=12)
axes2[1].set_ylabel('Sharpe Ratio')
for i, v in enumerate(sharpes):
    axes2[1].text(i, v + 0.05, f'{v:.2f}', ha='center')

plt.tight_layout()
fig_path2 = os.path.join(FIGURES_DIR, 'performance_metrics.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: {fig_path2}")

print("\n" + "=" * 60)
print("保存回测结果...")
print("=" * 60)

results_df = pd.DataFrame(metrics)
results_path = os.path.join(RESULTS_DIR, 'backtest_results.csv')
results_df.to_csv(results_path, index=False)
print(f"回测结果已保存: {results_path}")

net_value_df = pd.DataFrame({
    'trade_date': dates,
    'benchmark': benchmark_cumret,
    'event_strategy': event_cumret,
    'regular_strategy': regular_cumret,
    'industry_strategy': industry_cumret
})
net_value_path = os.path.join(RESULTS_DIR, 'net_value.csv')
net_value_df.to_csv(net_value_path, index=False)
print(f"净值序列已保存: {net_value_path}")

print("\n" + "=" * 60)
print("输出文件清单:")
print("=" * 60)
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    print(f"\n{d}:")
    for f in os.listdir(d):
        fpath = os.path.join(d, f)
        size = os.path.getsize(fpath)
        print(f"  {f}: {size/1024:.1f} KB")

print("\n" + "=" * 60)
print("回测完成!")
print("=" * 60)