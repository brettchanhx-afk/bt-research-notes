# -*- coding: utf-8 -*-
"""
晨星风格箱 - 可视化模块
严格复现晨星风格箱 3×3 方格图

核心图表:
1. 晨星风格箱 3×3 方格（价值-平衡-成长 × 大盘-中盘-小盘）
2. 净值走势对比图
3. 回撤曲线图
4. 持仓股票在风格箱中的分布散点图
5. 基金风格时序变化图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


def plot_morningstar_stylebox(
    style_result: Dict,
    save_path: str = None,
    fund_name: str = ""
) -> plt.Figure:
    """
    绘制标准晨星风格箱 3×3 方格图
    
    行: 大盘 / 中盘 / 小盘 (从上到下)
    列: 价值 / 平衡 / 成长 (从左到右)
    基金位置用红色圆点标注
    
    Args:
        style_result: analyze_fund_style 的返回结果
        save_path: 保存路径
        fund_name: 基金名称
    
    Returns:
        Figure对象
    """
    fig, ax = plt.subplots(figsize=(8, 9))
    
    # 颜色配置
    colors = {
        '价值': '#2196F3',   # 蓝色
        '平衡': '#4CAF50',   # 绿色
        '成长': '#FF9800',   # 橙色
    }
    size_labels = ['大盘', '中盘', '小盘']
    vg_labels = ['价值', '平衡', '成长']
    
    # 绘制3×3网格
    for i, size in enumerate(size_labels):
        for j, vg in enumerate(vg_labels):
            # 翻转y轴: 大盘在上
            row = 2 - i
            rect = plt.Rectangle(
                (j, row), 1, 1,
                facecolor=colors[vg],
                alpha=0.15,
                edgecolor='#333333',
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            # 添加格子标签
            ax.text(
                j + 0.5, row + 0.5,
                f"{size}\n{vg}",
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='#333333'
            )
    
    # 标注基金位置
    Y = style_result.get('fund_size_score_Y', 150)
    X = style_result.get('fund_vg_score_X', 150)
    
    # 将得分映射到网格坐标
    # Y: [0, 300] → row: [0, 3], 小盘=0, 大盘=3
    # X: [0, 300] → col: [0, 3], 价值=0, 成长=3
    fund_y = min(max(Y / 100.0, 0), 3)  # 翻转: 大盘在上
    fund_x = min(max(X / 100.0, 0), 3)
    
    # 绘制基金定位点
    ax.plot(
        fund_x, fund_y,
        marker='o', markersize=20,
        color='red', alpha=0.8,
        markeredgecolor='white',
        markeredgewidth=2,
        zorder=10
    )
    
    # 基金标签
    style_text = style_result.get('fund_style', '')
    ax.annotate(
        f"{fund_name}\n{style_text}",
        xy=(fund_x, fund_y),
        xytext=(fund_x + 0.3, fund_y + 0.3),
        fontsize=9, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9),
        zorder=11
    )
    
    # 绘制持仓股分布（如果有）
    if 'stock_details' in style_result and isinstance(style_result['stock_details'], pd.DataFrame):
        stocks = style_result['stock_details']
        if 'size_score_y' in stocks.columns and 'vg_score_x' in stocks.columns:
            valid = stocks.dropna(subset=['size_score_y', 'vg_score_x'])
            if not valid.empty:
                stock_ys = np.clip(valid['size_score_y'].values / 100.0, 0, 3)
                stock_xs = np.clip(valid['vg_score_x'].values / 100.0, 0, 3)
                pcts = valid['pct'].values
                
                # 气泡大小按持仓占比
                sizes = pcts * 30 + 20
                
                ax.scatter(
                    stock_xs, stock_ys,
                    s=sizes, c='navy', alpha=0.4,
                    edgecolors='white', linewidth=0.5,
                    zorder=5
                )
    
    # 轴标签
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.1, 3.1)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(vg_labels, fontsize=12, fontweight='bold')
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(reversed(size_labels), fontsize=12, fontweight='bold')
    
    ax.set_xlabel('价值 — 成长 维度', fontsize=12, fontweight='bold')
    ax.set_ylabel('规模 维度', fontsize=12, fontweight='bold')
    ax.set_title(f'晨星风格箱 — {fund_name}', fontsize=14, fontweight='bold')
    
    # 添加得分信息
    info_text = (
        f"规模得分 Y={Y:.1f} ({style_result.get('fund_size_style', '')})\n"
        f"价成得分 X={X:.1f} ({style_result.get('fund_vg_style', '')})"
    )
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[保存] 风格箱图: {save_path}")
    
    return fig


def plot_nav_curve(
    nav: pd.Series,
    benchmark: pd.Series = None,
    fund_name: str = "",
    save_path: str = None
) -> plt.Figure:
    """
    绘制净值走势对比图
    
    Args:
        nav: 基金净值序列
        benchmark: 基准净值序列
        fund_name: 基金名称
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 归一化
    nav_norm = nav / nav.iloc[0] * 1.0
    ax.plot(nav_norm.index, nav_norm.values, label=f'{fund_name}', 
            linewidth=2, color='#1f77b4')
    
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * 1.0
        ax.plot(bench_norm.index, bench_norm.values, label='基准(沪深300)',
                linewidth=1.5, linestyle='--', color='#ff7f0e')
    
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('归一化净值', fontsize=11)
    ax.set_title(f'{fund_name} 净值走势', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_drawdown(
    nav: pd.Series,
    fund_name: str = "",
    save_path: str = None
) -> plt.Figure:
    """绘制回撤曲线"""
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax * 100
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color='#d62728', alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color='#d62728', linewidth=0.8)
    
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('回撤 (%)', fontsize=11)
    ax.set_title(f'{fund_name} 回撤曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_style_history(
    style_history: pd.DataFrame,
    fund_name: str = "",
    save_path: str = None
) -> plt.Figure:
    """
    绘制基金风格时序变化图
    
    Args:
        style_history: 包含 date, Y, X 的DataFrame
        fund_name: 基金名称
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    if 'Y' in style_history.columns:
        axes[0].plot(style_history.index, style_history['Y'], 
                     'o-', color='#1f77b4', linewidth=1.5, markersize=4)
        axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='小/中盘边界')
        axes[0].axhline(y=200, color='gray', linestyle='--', alpha=0.5, label='中/大盘边界')
        axes[0].fill_between(style_history.index, 0, 100, alpha=0.05, color='blue', label='小盘')
        axes[0].fill_between(style_history.index, 100, 200, alpha=0.05, color='green', label='中盘')
        axes[0].fill_between(style_history.index, 200, 300, alpha=0.05, color='red', label='大盘')
        axes[0].set_ylabel('规模得分 Y', fontsize=11)
        axes[0].legend(loc='best', fontsize=8)
    
    if 'X' in style_history.columns:
        axes[1].plot(style_history.index, style_history['X'],
                     'o-', color='#ff7f0e', linewidth=1.5, markersize=4)
        lower = 125  # gamma=0.5
        upper = 175
        axes[1].axhline(y=lower, color='gray', linestyle='--', alpha=0.5, label='价值/平衡边界')
        axes[1].axhline(y=upper, color='gray', linestyle='--', alpha=0.5, label='平衡/成长边界')
        axes[1].fill_between(style_history.index, 0, lower, alpha=0.05, color='blue', label='价值')
        axes[1].fill_between(style_history.index, lower, upper, alpha=0.05, color='green', label='平衡')
        axes[1].fill_between(style_history.index, upper, 300, alpha=0.05, color='red', label='成长')
        axes[1].set_ylabel('价值-成长得分 X', fontsize=11)
        axes[1].legend(loc='best', fontsize=8)
    
    axes[0].set_title(f'{fund_name} 风格时序变化', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('日期', fontsize=11)
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_stock_distribution(
    stock_details: pd.DataFrame,
    mst: float,
    lmt: float,
    vt: float,
    gt: float,
    fund_name: str = "",
    save_path: str = None
) -> plt.Figure:
    """
    绘制持仓股票在风格空间中的分布散点图
    
    X轴: 价值-成长得分
    Y轴: 规模得分
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'size_score_y' not in stock_details.columns or 'vg_score_x' not in stock_details.columns:
        ax.text(0.5, 0.5, '无持仓股风格数据', ha='center', va='center', fontsize=14)
        return fig
    
    valid = stock_details.dropna(subset=['size_score_y', 'vg_score_x'])
    
    if not valid.empty:
        pcts = valid.get('pct', pd.Series([1]*len(valid))).values
        sizes = pcts * 50 + 30
        
        scatter = ax.scatter(
            valid['vg_score_x'], valid['size_score_y'],
            s=sizes, c=pcts if pcts.max() > 0 else 'blue',
            cmap='YlOrRd', alpha=0.7,
            edgecolors='black', linewidth=0.5
        )
        
        # 标注股票名称
        for _, row in valid.iterrows():
            name = row.get('stock_name', row.get('stock_code', ''))
            ax.annotate(
                name, 
                (row['vg_score_x'], row['size_score_y']),
                fontsize=7, alpha=0.8,
                xytext=(5, 5), textcoords='offset points'
            )
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='持仓占比(%)', shrink=0.8)
    
    # 绘制边界线
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=125, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=175, color='gray', linestyle='--', alpha=0.5)
    
    # 区域标签
    ax.text(62.5, 250, '大盘\n价值', ha='center', va='center', fontsize=9, alpha=0.3, color='blue')
    ax.text(150, 250, '大盘\n平衡', ha='center', va='center', fontsize=9, alpha=0.3, color='green')
    ax.text(237.5, 250, '大盘\n成长', ha='center', va='center', fontsize=9, alpha=0.3, color='orange')
    ax.text(62.5, 150, '中盘\n价值', ha='center', va='center', fontsize=9, alpha=0.3, color='blue')
    ax.text(150, 150, '中盘\n平衡', ha='center', va='center', fontsize=9, alpha=0.3, color='green')
    ax.text(237.5, 150, '中盘\n成长', ha='center', va='center', fontsize=9, alpha=0.3, color='orange')
    ax.text(62.5, 50, '小盘\n价值', ha='center', va='center', fontsize=9, alpha=0.3, color='blue')
    ax.text(150, 50, '小盘\n平衡', ha='center', va='center', fontsize=9, alpha=0.3, color='green')
    ax.text(237.5, 50, '小盘\n成长', ha='center', va='center', fontsize=9, alpha=0.3, color='orange')
    
    ax.set_xlabel('价值-成长得分 X', fontsize=12)
    ax.set_ylabel('规模得分 Y', fontsize=12)
    ax.set_title(f'{fund_name} 持仓股风格分布', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_performance_summary(
    perf: Dict,
    fund_name: str = "",
    save_path: str = None
) -> plt.Figure:
    """绘制绩效摘要图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 1. 关键指标
    metrics = {
        '累计收益': f"{perf.get('total_return', 0)*100:.2f}%",
        '年化收益': f"{perf.get('annual_return', 0)*100:.2f}%",
        '夏普比率': f"{perf.get('sharpe_ratio', 0):.2f}",
        '最大回撤': f"{perf.get('max_drawdown', 0)*100:.2f}%",
        '波动率': f"{perf.get('volatility', 0)*100:.2f}%",
    }
    
    axes[0].barh(list(metrics.keys()), [float(v.replace('%','')) for v in metrics.values()],
                 color=['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'])
    axes[0].set_title('关键绩效指标', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('值')
    
    # 2. 月度收益热力图 (简化)
    axes[1].text(0.5, 0.5, '月度收益\n(需要净值数据)', 
                 ha='center', va='center', fontsize=12, alpha=0.5)
    axes[1].set_title('月度收益', fontsize=12, fontweight='bold')
    
    # 3. 风格结论
    axes[2].text(0.5, 0.7, fund_name, ha='center', va='center', 
                 fontsize=14, fontweight='bold')
    axes[2].text(0.5, 0.4, perf.get('fund_style', ''), ha='center', va='center',
                 fontsize=18, fontweight='bold', color='#F44336')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    axes[2].set_title('风格结论', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    print("可视化模块加载成功")
