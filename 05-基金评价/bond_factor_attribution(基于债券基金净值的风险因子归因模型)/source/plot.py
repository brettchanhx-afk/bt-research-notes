# -*- coding: utf-8 -*-
"""
可视化模块
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, OUTPUT_DIR, setup_chinese_font

setup_chinese_font()


def plot_factor_exposure(results: dict, title: str = '因子暴露分析', save_path: str = None):
    """绘制因子暴露柱状图"""
    if 'factor_exposures' not in results:
        return
    
    exposures = results['factor_exposures']
    names = list(exposures.keys())
    values = list(exposures.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS.get(n, '#666666') for n in names]
    
    bars = ax.bar(names, values, color=colors, edgecolor='white')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('暴露系数', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rolling_exposure(rolling_results: pd.DataFrame, save_path: str = None):
    """绘制滚动因子暴露时序图"""
    if len(rolling_results) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 提取因子暴露时序
    factor_cols = [c for c in rolling_results.columns if c.endswith('_mean')]
    
    for col in factor_cols:
        factor_name = col.replace('_mean', '')
        ax.plot(rolling_results.index, rolling_results[col], 
                label=factor_name, linewidth=2, color=COLORS.get(factor_name, '#666666'))
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('因子暴露', fontsize=12)
    ax.set_title('滚动因子暴露', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_factor_contribution(contrib_df: pd.DataFrame, save_path: str = None):
    """绘制因子贡献度饼图"""
    if len(contrib_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = contrib_df['factor'].tolist()
    values = contrib_df['contribution'].abs().tolist()
    colors = [COLORS.get(l.lower(), '#666666') for l in labels]
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90)
    
    ax.set_title('因子贡献度分解', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
