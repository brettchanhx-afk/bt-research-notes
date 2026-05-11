"""
绘制权益曲线趋势图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120

output_dir = 'C:/Users/chenh/.qclaw/workspace/hpcrp_asset_allocation/output'
data_dir = 'C:/Users/chenh/.qclaw/workspace/hpcrp_asset_allocation/data'

# 读取权益曲线数据
equity_curves = pd.read_csv(
    os.path.join(output_dir, 'equity_curves.csv'),
    index_col=0,
    parse_dates=True
)

print(f"读取权益曲线数据: {equity_curves.shape}")
print(f"数据列: {list(equity_curves.columns)}")
print(f"时间范围: {equity_curves.index.min()} 到 {equity_curves.index.max()}")

# 创建图表
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ===== 图1: 完整权益曲线 =====
ax1 = axes[0]

# 定义颜色
colors = {
    'MD': '#e74c3c',    # 红色 - 最优
    'RP': '#3498db',    # 蓝色
    'EW': '#95a5a6',    # 灰色
    'EV': '#9b59b6',    # 紫色
    'MV': '#2ecc71',    # 绿色
    'PCRP': '#f39c12',  # 橙色
    'HPCRP': '#1abc9c'  # 青色
}

for model in equity_curves.columns:
    color = colors.get(model, '#333333')
    linewidth = 2.5 if model == 'MD' else 1.5
    ax1.plot(equity_curves.index, equity_curves[model], 
             label=model, color=color, linewidth=linewidth)

ax1.set_title('HPCRP Global Asset Allocation - Equity Curves (2008-2026)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Net Asset Value')
ax1.legend(loc='upper left', ncol=4)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='black', linestyle='--', alpha=0.3)

# ===== 图2: 2020年后的权益曲线 (近5年) =====
ax2 = axes[1]

# 筛选2020年后的数据
equity_recent = equity_curves[equity_curves.index >= '2020-01-01']

for model in equity_recent.columns:
    color = colors.get(model, '#333333')
    linewidth = 2.5 if model == 'MD' else 1.5
    ax2.plot(equity_recent.index, equity_recent[model], 
             label=model, color=color, linewidth=linewidth)

ax2.set_title('Equity Curves - Recent 5 Years (2020-2026)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Net Asset Value')
ax2.legend(loc='upper left', ncol=4)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()

# 保存图表
output_path = os.path.join(output_dir, 'equity_curves_chart.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n已保存图表到: {output_path}")

# 同时保存PDF版本
output_path_pdf = os.path.join(output_dir, 'equity_curves_chart.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"已保存PDF到: {output_path_pdf}")

plt.close()

# ===== 额外图表: 回测指标对比 =====
fig2, ax = plt.subplots(figsize=(10, 6))

# 读取结果数据
results = pd.read_csv(os.path.join(output_dir, 'backtest_results.csv'))

# 绘制条形图
x = np.arange(len(results))
width = 0.15

# 夏普比率
bars1 = ax.bar(x - 1.5*width, results['Sharpe'], width, label='Sharpe Ratio', color='#3498db')
# 年化收益
bars2 = ax.bar(x - 0.5*width, results['AnnReturn']*10, width, label='AnnReturn x10', color='#2ecc71')
# Calmar
bars3 = ax.bar(x + 0.5*width, results['Calmar'], width, label='Calmar', color='#e74c3c')
# (1 - |MaxDD|/100) - 越大越好
bars4 = ax.bar(x + 1.5*width, 1 - results['MaxDD'].abs()/100, width, label='Risk Score', color='#9b59b6')

ax.set_xlabel('Model')
ax.set_ylabel('Value')
ax.set_title('Backtest Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results['Model'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# 保存
output_path2 = os.path.join(output_dir, 'metrics_comparison.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"已保存指标对比图到: {output_path2}")

plt.close()

print("\n图表生成完成!")