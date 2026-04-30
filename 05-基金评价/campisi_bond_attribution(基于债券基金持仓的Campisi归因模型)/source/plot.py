# -*- coding: utf-8 -*-
"""
可视化模块

功能：
  - 归因分解饼图
  - 时间序列归因趋势图
  - 债券贡献条形图
  - 久期分布图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Optional, List

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, OUTPUT_DIR, setup_chinese_font

# 设置中文字体
setup_chinese_font()


# ============================================================
# 1. 归因分解饼图
# ============================================================
def plot_attribution_pie(
    summary: Dict,
    title: str = 'Campisi归因分解',
    save_path: Optional[str] = None
):
    """绘制归因分解饼图。
    
    Parameters
    ----------
    summary : Dict
        归因摘要，包含各效应贡献
    title : str
        图表标题
    save_path : str
        保存路径
    """
    # 准备数据
    labels = ['票息效应', '国债利率效应', '信用利差效应']
    values = [
        summary.get('coupon_contrib', 0),
        summary.get('treasury_contrib', 0),
        summary.get('credit_contrib', 0),
    ]
    colors = [COLORS['coupon'], COLORS['treasury'], COLORS['credit']]
    
    # 过滤零值
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if abs(v) > 1e-6]
    if not non_zero:
        print('[WARNING] 所有效应贡献为零，无法绘制饼图')
        return
    
    labels, values, colors = zip(*non_zero)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        np.abs(values),
        labels=labels,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90,
        explode=[0.02] * len(values),
        shadow=True,
    )
    
    # 设置字体
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    # 标题
    total_return = summary.get('total_return', 0)
    ax.set_title(f'{title}\n总收益: {total_return:.4f} ({total_return*100:.2f}%)', 
                 fontsize=14, fontweight='bold')
    
    # 添加图例（显示实际值）
    if abs(total_return) > 1e-6:
        legend_labels = [f'{l}: {v:.4f} ({v/total_return*100:.1f}%)' 
                         for l, v in zip(labels, values)]
    else:
        legend_labels = [f'{l}: {v:.4f}' for l, v in zip(labels, values)]
    ax.legend(wedges, legend_labels, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 2. 时间序列归因趋势图
# ============================================================
def plot_attribution_timeseries(
    df: pd.DataFrame,
    title: str = '滚动归因分析',
    save_path: Optional[str] = None
):
    """绘制时间序列归因趋势图。
    
    Parameters
    ----------
    df : pd.DataFrame
        时间序列归因结果，包含date和各效应贡献
    title : str
        图表标题
    save_path : str
        保存路径
    """
    if len(df) == 0:
        print('[WARNING] 数据为空，无法绘图')
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：累计收益
    ax1 = axes[0]
    dates = df['date'] if 'date' in df.columns else range(len(df))
    
    total = df['total_return'].cumsum()
    coupon = df['coupon_contrib'].cumsum()
    treasury = df['treasury_contrib'].cumsum()
    credit = df['credit_contrib'].cumsum()
    
    ax1.plot(dates, total, label='总收益', color=COLORS['total'], linewidth=2.5)
    ax1.plot(dates, coupon, label='票息效应', color=COLORS['coupon'], linewidth=2)
    ax1.plot(dates, treasury, label='国债利率效应', color=COLORS['treasury'], linewidth=2)
    ax1.plot(dates, credit, label='信用利差效应', color=COLORS['credit'], linewidth=2)
    
    ax1.set_title('累计收益分解', fontsize=13, fontweight='bold')
    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_ylabel('累计收益', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 下图：各期贡献堆叠柱状图
    ax2 = axes[1]
    x = np.arange(len(df))
    width = 0.8
    
    ax2.bar(x, df['coupon_contrib'], width, label='票息效应', color=COLORS['coupon'])
    ax2.bar(x, df['treasury_contrib'], width, bottom=df['coupon_contrib'], 
            label='国债利率效应', color=COLORS['treasury'])
    ax2.bar(x, df['credit_contrib'], width, 
            bottom=df['coupon_contrib'] + df['treasury_contrib'],
            label='信用利差效应', color=COLORS['credit'])
    
    ax2.set_title('各期收益贡献', fontsize=13, fontweight='bold')
    ax2.set_xlabel('期数', fontsize=11)
    ax2.set_ylabel('收益贡献', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 3. 债券贡献条形图
# ============================================================
def plot_bond_contribution(
    df: pd.DataFrame,
    effect: str = 'coupon',
    top_n: int = 15,
    title: str = None,
    save_path: Optional[str] = None
):
    """绘制债券贡献条形图。
    
    Parameters
    ----------
    df : pd.DataFrame
        归因结果，包含各债券贡献
    effect : str
        效应类型：'coupon', 'treasury', 'credit'
    top_n : int
        显示前N只债券
    title : str
        图表标题
    save_path : str
        保存路径
    """
    if len(df) == 0:
        print('[WARNING] 数据为空，无法绘图')
        return
    
    # 列名映射
    col_map = {
        'coupon': 'coupon_effect',
        'treasury': 'treasury_effect',
        'credit': 'credit_effect',
    }
    effect_col = col_map.get(effect, 'coupon_effect')
    
    # 计算加权贡献
    df = df.copy()
    df['contribution'] = df['weight'] * df[effect_col]
    
    # 取贡献最大的前N只
    df_sorted = df.nlargest(top_n, 'contribution')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = [COLORS['positive'] if v >= 0 else COLORS['negative'] 
              for v in df_sorted['contribution']]
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['contribution'], color=colors)
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['bond_code'], fontsize=10)
    ax.set_xlabel('贡献度', fontsize=12)
    
    effect_names = {'coupon': '票息', 'treasury': '国债利率', 'credit': '信用利差'}
    default_title = f'{effect_names.get(effect, effect)}效应 - Top {top_n} 债券贡献'
    ax.set_title(title or default_title, fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, df_sorted['contribution'])):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 4. 久期分布图
# ============================================================
def plot_duration_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """绘制持仓债券久期分布图。
    
    Parameters
    ----------
    df : pd.DataFrame
        归因结果，包含modified_duration
    save_path : str
        保存路径
    """
    if len(df) == 0 or 'modified_duration' not in df.columns:
        print('[WARNING] 数据为空或缺少久期数据')
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：久期直方图
    ax1 = axes[0]
    durations = df['modified_duration'].dropna()
    
    ax1.hist(durations, bins=20, color=COLORS['treasury'], edgecolor='white', alpha=0.7)
    ax1.axvline(durations.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'均值: {durations.mean():.2f}')
    ax1.axvline(durations.median(), color='green', linestyle='--', linewidth=2,
                label=f'中位数: {durations.median():.2f}')
    
    ax1.set_xlabel('修正久期', fontsize=12)
    ax1.set_ylabel('债券数量', fontsize=12)
    ax1.set_title('持仓债券久期分布', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：久期-权重散点图
    ax2 = axes[1]
    
    scatter = ax2.scatter(
        df['modified_duration'], 
        df['weight'] * 100,
        c=df['ytm'] if 'ytm' in df.columns else COLORS['coupon'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.7,
        edgecolors='white'
    )
    
    ax2.set_xlabel('修正久期', fontsize=12)
    ax2.set_ylabel('持仓权重 (%)', fontsize=12)
    ax2.set_title('久期-权重分布', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    if 'ytm' in df.columns:
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('到期收益率 (%)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 5. 综合归因报告图
# ============================================================
def plot_attribution_report(
    summary: Dict,
    results: pd.DataFrame,
    fund_name: str = '债券基金',
    save_path: Optional[str] = None
):
    """绘制综合归因报告图（多子图）。
    
    Parameters
    ----------
    summary : Dict
        归因摘要
    results : pd.DataFrame
        详细归因结果
    fund_name : str
        基金名称
    save_path : str
        保存路径
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. 归因饼图
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['票息', '国债利率', '信用利差']
    values = [
        summary.get('coupon_contrib', 0),
        summary.get('treasury_contrib', 0),
        summary.get('credit_contrib', 0),
    ]
    colors = [COLORS['coupon'], COLORS['treasury'], COLORS['credit']]
    
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if abs(v) > 1e-6]
    if non_zero:
        labels, values, colors = zip(*non_zero)
        ax1.pie(np.abs(values), labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=True)
    ax1.set_title('归因分解', fontsize=12, fontweight='bold')
    
    # 2. 各效应贡献柱状图
    ax2 = fig.add_subplot(gs[0, 1])
    if non_zero:
        x = np.arange(len(values))
        ax2.bar(x, values, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
    else:
        x = np.arange(3)
        ax2.bar(x, [0, 0, 0], color=['gray']*3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['票息', '国债', '信用'])
    ax2.set_ylabel('贡献')
    ax2.set_title('各效应贡献', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 关键指标文本
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    metrics_text = f"""
    {fund_name} - Campisi归因分析
    
    ━━━━━━━━━━━━━━━━━━━━━━━━
    总收益:        {summary.get('total_return', 0):.4f} ({summary.get('total_return', 0)*100:.2f}%)
    
    票息效应:      {summary.get('coupon_contrib', 0):.4f} ({summary.get('coupon_pct', 0):.1f}%)
    国债利率效应:  {summary.get('treasury_contrib', 0):.4f} ({summary.get('treasury_pct', 0):.1f}%)
    信用利差效应:  {summary.get('credit_contrib', 0):.4f} ({summary.get('credit_pct', 0):.1f}%)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━
    持仓债券数:    {summary.get('n_bonds', 0)}
    平均久期:      {summary.get('avg_duration', 0):.2f}
    平均YTM:       {summary.get('avg_ytm', 0)*100:.2f}%
    """
    
    ax3.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax3.transAxes)
    
    # 4. 久期分布
    ax4 = fig.add_subplot(gs[1, 0])
    if 'modified_duration' in results.columns:
        durations = results['modified_duration'].dropna()
        ax4.hist(durations, bins=15, color=COLORS['treasury'], edgecolor='white', alpha=0.7)
        ax4.axvline(durations.mean(), color='red', linestyle='--', label=f'均值: {durations.mean():.2f}')
        ax4.legend(fontsize=9)
    ax4.set_xlabel('修正久期')
    ax4.set_ylabel('数量')
    ax4.set_title('久期分布', fontsize=12, fontweight='bold')
    
    # 5. YTM分布
    ax5 = fig.add_subplot(gs[1, 1])
    if 'ytm' in results.columns:
        ytms = results['ytm'].dropna() * 100
        ax5.hist(ytms, bins=15, color=COLORS['credit'], edgecolor='white', alpha=0.7)
        ax5.axvline(ytms.mean(), color='red', linestyle='--', label=f'均值: {ytms.mean():.2f}%')
        ax5.legend(fontsize=9)
    ax5.set_xlabel('到期收益率 (%)')
    ax5.set_ylabel('数量')
    ax5.set_title('YTM分布', fontsize=12, fontweight='bold')
    
    # 6. 权重-久期散点
    ax6 = fig.add_subplot(gs[1, 2])
    if 'modified_duration' in results.columns and 'weight' in results.columns:
        ax6.scatter(results['modified_duration'], results['weight']*100, 
                    c=COLORS['coupon'], alpha=0.6, s=60)
    ax6.set_xlabel('修正久期')
    ax6.set_ylabel('权重 (%)')
    ax6.set_title('久期-权重分布', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'{fund_name} - Campisi归因分析报告', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()
