# -*- coding: utf-8 -*-
"""
可视化模块
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import setup_chinese_font, COLORS, OUTPUT_DIR

setup_chinese_font()


# ============================================================
# 1. IC时间序列图
# ============================================================
def plot_ic_series(
    ic_series: pd.Series,
    factor_name: str,
    save_path: str = None
):
    """绘制IC时间序列"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # IC时间序列
    ax1 = axes[0]
    ax1.bar(ic_series.index, ic_series.values, 
            color=[COLORS['ic_positive'] if v > 0 else COLORS['ic_negative'] 
                   for v in ic_series.values],
            alpha=0.7, width=20)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=ic_series.mean(), color='red', linestyle='--', 
                label=f'Mean IC: {ic_series.mean():.4f}')
    ax1.set_title(f'{factor_name} - RankIC时间序列', fontsize=14, fontweight='bold')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('RankIC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IC累计图
    ax2 = axes[1]
    ic_cumsum = ic_series.cumsum()
    ax2.plot(ic_cumsum.index, ic_cumsum.values, linewidth=2, color=COLORS['layer1'])
    ax2.fill_between(ic_cumsum.index, 0, ic_cumsum.values, alpha=0.3)
    ax2.set_title('累计IC', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('累计IC')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 2. 分层回测收益图
# ============================================================
def plot_layer_returns(
    layer_returns: pd.DataFrame,
    factor_name: str,
    save_path: str = None
):
    """绘制分层回测累计收益"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = [COLORS['layer1'], COLORS['layer2'], COLORS['layer3']]
    
    for i, col in enumerate(layer_returns.columns):
        color = colors[i] if i < len(colors) else 'gray'
        ax.plot(layer_returns.index, layer_returns[col], 
                label=col, linewidth=2, color=color)
    
    ax.set_title(f'{factor_name} - 分层回测累计收益', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('累计净值')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 3. 因子有效性对比图
# ============================================================
def plot_factor_effectiveness(
    effectiveness_df: pd.DataFrame,
    save_path: str = None
):
    """绘制因子有效性对比"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 按IC均值排序
    df = effectiveness_df.sort_values('mean_ic', ascending=True)
    
    colors = [COLORS['ic_positive'] if v > 0 else COLORS['ic_negative'] 
              for v in df['mean_ic']]
    
    bars = ax.barh(df.index, df['mean_ic'], color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('平均RankIC', fontsize=12)
    ax.set_title('因子有效性对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 4. 雷达图（基金评分）
# ============================================================
def plot_radar_chart(
    scores: Dict[str, float],
    fund_name: str,
    save_path: str = None
):
    """绘制雷达图"""
    categories = list(scores.keys())
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 数据闭合
    values = list(scores.values())
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    
    ax.set_title(f'{fund_name} - 五维评分', fontsize=14, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 5. 动态权重图
# ============================================================
def plot_dynamic_weights(
    weights_df: pd.DataFrame,
    save_path: str = None
):
    """绘制动态权重"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    weights_df.plot.area(ax=ax, alpha=0.7, stacked=True)
    
    ax.set_title('复合因子动态权重', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('权重')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()


# ============================================================
# 6. 因子相关性热力图
# ============================================================
def plot_factor_correlation(
    corr_matrix: pd.DataFrame,
    save_path: str = None
):
    """绘制因子相关性热力图"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(corr_matrix.values, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.index)
    
    # 添加数值标注
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            text = f'{corr_matrix.iloc[i, j]:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=8)
    
    ax.set_title('因子RankIC相关性矩阵', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('相关系数')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    
    plt.close()
