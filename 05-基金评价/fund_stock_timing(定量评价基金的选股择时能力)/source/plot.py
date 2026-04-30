# -*- coding: utf-8 -*-
"""
可视化模块 - 基金选股择时能力定量评价模型
包含：回归结果图、滚动窗口图、业绩归因图、仪表盘

依赖：matplotlib, seaborn
字体：必须在 plt.style.use() 之后设置中文字体（避免被覆盖）
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

# ==================== 中文字体设置（关键：必须在 style.use 之后调用）====================
def setup_chinese_font():
    """设置 matplotlib 中文字体，必须在 plt.style.use() 之后调用。"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120


# ==================== 核心图表函数 ====================
def plot_timing_dashboard(fund_code: str,
                          fund_name: str,
                          model_results: Dict[str, Dict],
                          output_path: str) -> None:
    """
    生成选股择时能力分析仪表盘（四合一图）。

    包含：
        1. 基金 vs 基准超额收益散点图（含回归线）
        2. 三种模型参数对比柱状图
        3. 择时能力指标时序图（滚动窗口）
        4. 业绩归因堆叠面积图

    参数:
        fund_code: 基金代码
        fund_name: 基金名称
        model_results: 三种模型的回归结果字典
        output_path: 输出图片完整路径
    """
    # 关键：先设置 style，再设置字体
    plt.style.use('seaborn-v0_8-whitegrid')
    setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{fund_name} ({fund_code}) 选股择时能力分析\n'
                 'T-M / H-M / C-L Model', fontsize=16, fontweight='bold', y=1.02)

    # ---------- 图1：基金 vs 基准超额收益散点图 ----------
    ax1 = axes[0, 0]
    if 'excess_fund' in model_results and 'excess_bench' in model_results:
        ef = model_results['excess_fund']
        eb = model_results['excess_bench']
        ax1.scatter(eb, ef, alpha=0.4, s=15, color='steelblue', label='Daily Returns')

        # 添加回归线（线性拟合）
        valid = ef.notna() & eb.notna()
        if valid.sum() > 10:
            z = np.polyfit(eb[valid], ef[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(eb[valid].min(), eb[valid].max(), 100)
            ax1.plot(x_line, p(x_line), 'r-', linewidth=2,
                     label=f'Linear Fit (slope={z[0]:.3f})')

        ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax1.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax1.set_xlabel('Market Excess Return (Rm - Rf)', fontsize=10)
        ax1.set_ylabel('Fund Excess Return (Rp - Rf)', fontsize=10)
        ax1.set_title('Fund vs Market Excess Returns', fontsize=11)
        ax1.legend(fontsize=9)

    # ---------- 图2：三种模型参数对比 ----------
    ax2 = axes[0, 1]
    models = ['TM', 'HM', 'CL']
    alpha_vals = [model_results.get(m, {}).get('alpha', 0) for m in models]
    beta2_vals = [model_results.get(m, {}).get('beta2', 0) for m in models]
    timing_flags = [model_results.get(m, {}).get('timing_ability', False) for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, alpha_vals, width, label='Alpha (Stock Ability)',
                    color=['green' if v > 0 else 'red' for v in alpha_vals], alpha=0.8)
    bars2 = ax2.bar(x + width / 2, beta2_vals, width, label='Beta2 (Timing Ability)',
                    color=['blue' if v > 0 else 'orange' for v in beta2_vals], alpha=0.8)

    # 标注显著性
    for i, (a, b, tm) in enumerate(zip(alpha_vals, beta2_vals, timing_flags)):
        marker = '***' if tm else ''
        ax2.annotate(f'{a:.5f}{marker}', xy=(x[i] - width / 2, a),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=8, color='darkgreen')
        ax2.annotate(f'{b:.5f}{marker}', xy=(x[i] + width / 2, b),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=8, color='darkblue')

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{m}\n{"(Timing)" if timing_flags[i] else ""}'
                         for i, m in enumerate(models)])
    ax2.set_ylabel('Coefficient Value', fontsize=10)
    ax2.set_title('Model Coefficients Comparison\n'
                  '(*** = Timing Ability Significant at 5%)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    # ---------- 图3：R-squared 对比 ----------
    ax3 = axes[1, 0]
    r2_vals = [model_results.get(m, {}).get('r_squared', 0) for m in models]
    adj_r2_vals = [model_results.get(m, {}).get('adj_r_squared', 0) for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax3.bar(x - width / 2, r2_vals, width, label='R-squared',
            color='steelblue', alpha=0.8)
    ax3.bar(x + width / 2, adj_r2_vals, width, label='Adj R-squared',
            color='coral', alpha=0.8)

    for i, (r2, r2a) in enumerate(zip(r2_vals, adj_r2_vals)):
        ax3.annotate(f'{r2:.4f}', xy=(x[i] - width / 2, r2),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        ax3.annotate(f'{r2a:.4f}', xy=(x[i] + width / 2, r2a),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_ylabel('R-squared', fontsize=10)
    ax3.set_title('Model Explanatory Power (R-squared)', fontsize=11)
    ax3.legend(fontsize=9)

    # ---------- 图4：择时能力判断汇总表 ----------
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 表格数据
    table_data = []
    for m in models:
        res = model_results.get(m, {})
        if not res:
            continue
        table_data.append([
            m,
            f"{res.get('alpha', 0):.6f}",
            "Yes" if res.get('stock_ability', False) else "No",
            f"{res.get('beta2', 0):.6f}",
            "Yes" if res.get('timing_ability', False) else "No",
            f"{res.get('r_squared', 0):.4f}",
            str(res.get('nobs', 0)),
        ])

    headers = ['Model', 'Alpha', 'Stock?', 'Beta2', 'Timing?', 'R2', 'N']
    col_widths = [0.1, 0.12, 0.1, 0.12, 0.1, 0.1, 0.1]

    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.2, 0.95, 0.65]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 表头样式
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2E4057')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # 高亮显著性行
    for i, m in enumerate(models):
        res = model_results.get(m, {})
        if res.get('timing_ability', False):
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor('#E8F5E9')

    ax4.set_title('Summary Table\n(Green = Timing Ability Confirmed)', fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Dashboard saved: {output_path}")


def plot_rolling_timing(rolling_df: pd.DataFrame,
                       model: str,
                       output_path: str,
                       title: str = "") -> None:
    """
    绘制滚动窗口的择时能力时序图。

    参数:
        rolling_df: 滚动回归结果 DataFrame
        model: 模型名称
        output_path: 输出路径
        title: 图表标题
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    setup_chinese_font()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Alpha 时序
    ax1 = axes[0]
    ax1.plot(rolling_df.index, rolling_df['alpha'], color='green', linewidth=1.5)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.fill_between(rolling_df.index, rolling_df['alpha'],
                     0, where=rolling_df['alpha'] > 0,
                     color='green', alpha=0.2)
    ax1.fill_between(rolling_df.index, rolling_df['alpha'],
                     0, where=rolling_df['alpha'] < 0,
                     color='red', alpha=0.2)
    ax1.set_ylabel('Alpha', fontsize=10)
    ax1.set_title('Stock Selection Ability (Alpha) Over Time', fontsize=11)
    ax1.tick_params(labelbottom=True)  # 关键：共享x轴时强制显示标签

    # Beta2 时序（择时能力）
    ax2 = axes[1]
    ax2.plot(rolling_df.index, rolling_df['beta2'], color='blue', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.fill_between(rolling_df.index, rolling_df['beta2'],
                     0, where=rolling_df['beta2'] > 0,
                     color='blue', alpha=0.2)
    ax2.fill_between(rolling_df.index, rolling_df['beta2'],
                     0, where=rolling_df['beta2'] < 0,
                     color='orange', alpha=0.2)
    ax2.set_ylabel('Beta2', fontsize=10)
    ax2.set_title('Timing Ability (Beta2) Over Time', fontsize=11)
    ax2.tick_params(labelbottom=True)

    # R-squared 时序
    ax3 = axes[2]
    ax3.plot(rolling_df.index, rolling_df['r_squared'], color='purple', linewidth=1.5)
    ax3.set_ylabel('R-squared', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_title('Model Explanatory Power Over Time', fontsize=11)
    ax3.tick_params(labelbottom=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Rolling timing plot saved: {output_path}")


def plot_excess_return_scatter(fund_returns: pd.Series,
                                bench_returns: pd.Series,
                                output_path: str,
                                fund_name: str = "") -> None:
    """
    绘制基金 vs 基准超额收益散点图，并标注回归结果。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 8))

    # 散点图
    ax.scatter(bench_returns, fund_returns, alpha=0.5, s=20,
               c='steelblue', label='Daily Returns')

    # 回归线
    valid = fund_returns.notna() & bench_returns.notna()
    if valid.sum() > 10:
        z = np.polyfit(bench_returns[valid], fund_returns[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(bench_returns[valid].min(), bench_returns[valid].max(), 200)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2,
                label=f'Fit: y = {z[0]:.3f}x + {z[1]:.5f}')

        # 标注四象限
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

        # 填充四象限颜色
        ax.fill_between(x_line, 0, p(x_line), where=p(x_line) > 0,
                        color='green', alpha=0.05)
        ax.fill_between(x_line, 0, p(x_line), where=p(x_line) < 0,
                        color='red', alpha=0.05)

    ax.set_xlabel('Market Excess Return (Rm - Rf)', fontsize=11)
    ax.set_ylabel('Fund Excess Return (Rp - Rf)', fontsize=11)
    ax.set_title(f'{fund_name} - Excess Return Scatter\n'
                 '(T-M / H-M / C-L Basis)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Scatter plot saved: {output_path}")


def plot_attribution_area(decomp_df: pd.DataFrame,
                          output_path: str,
                          title: str = "") -> None:
    """
    绘制业绩归因堆叠面积图。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(14, 6))

    cols = ['alpha', 'beta1_contrib', 'beta2_contrib']
    labels = ['Alpha (Stock)', 'Market Risk (Beta1)', 'Timing (Beta2)']
    colors = ['green', 'steelblue', 'orange']

    # 只绘制存在的列
    plot_cols = [c for c in cols if c in decomp_df.columns]
    plot_labels = [labels[cols.index(c)] for c in plot_cols]
    plot_colors = [colors[cols.index(c)] for c in plot_cols]

    ax.stackplot(decomp_df.index,
                 *[decomp_df[c].values for c in plot_cols],
                 labels=plot_labels,
                 colors=plot_colors,
                 alpha=0.7)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Return', fontsize=11)
    ax.set_title(title or 'Performance Attribution (Cumulative)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Attribution plot saved: {output_path}")


# ==================== 辅助工具 ====================
def ensure_output_dir(output_dir: str) -> None:
    """确保输出目录存在。"""
    os.makedirs(output_dir, exist_ok=True)
