# -*- coding: utf-8 -*-
"""
可视化模块
生成 DEPI 分析结果图表：
  - DEPI 时间序列
  - DEPI 分布直方图
  - Top-N 基金柱状图
  - 投入指标相关性热力图
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def setup_chinese_font():
    """配置 matplotlib 中文字体（Windows 解决方块字问题）。
    必须在 import matplotlib.pyplot 之后、plt.style.use 之前调用一次。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    cache_dir = matplotlib.get_cachedir()
    font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
    if os.path.exists(font_cache):
        try:
            os.remove(font_cache)
        except OSError:
            pass

    # 字体配置在 style.use 之后生效
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120


def plot_depi_distribution(depi_df: pd.DataFrame,
                            title: str = 'DEPI分布',
                            save_path: str = None):
    """绘制 DEPI 分布直方图 + 统计信息。
    
    Parameters
    ----------
    depi_df : pd.DataFrame
        包含 DEPI 列的 DataFrame
    title : str
        图表标题
    save_path : str
        保存路径（可选）
    """
    setup_chinese_font()
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：直方图
    depi_vals = depi_df['DEPI'].dropna()
    axes[0].hist(depi_vals, bins=30, edgecolor='white', alpha=0.8, color='steelblue')
    axes[0].axvline(depi_vals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={depi_vals.mean():.3f}')
    axes[0].axvline(depi_vals.median(), color='orange', linestyle='--', linewidth=2, label=f'Median={depi_vals.median():.3f}')
    axes[0].set_xlabel('DEPI', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{title} - Distribution', fontsize=13)
    axes[0].legend()

    # 右图：箱线图
    data_to_plot = [depi_df['DEPI'].dropna()]
    bp = axes[1].boxplot(data_to_plot, vert=True, patch_artist=True,
                          tick_labels=[title])
    bp['boxes'][0].set_facecolor('lightsteelblue')
    axes[1].set_ylabel('DEPI', fontsize=12)
    axes[1].set_title(f'{title} - Boxplot', fontsize=13)

    # 添加统计注释
    stats_text = (
        f'N={len(depi_vals)}\n'
        f'Mean={depi_vals.mean():.3f}\n'
        f'Median={depi_vals.median():.3f}\n'
        f'Std={depi_vals.std():.3f}\n'
        f'Max={depi_vals.max():.3f}\n'
        f'Min={depi_vals.min():.3f}'
    )
    axes[1].text(1.15, depi_vals.median(), stats_text,
                 fontsize=10, va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    plt.show()
    plt.close()


def plot_depi_timeseries(depi_ts_df: pd.DataFrame,
                        top_n: int = 5,
                        title: str = 'DEPI Top-N基金时间序列',
                        save_path: str = None):
    """绘制 Top-N 基金的 DEPI 时间序列折线图。
    
    Parameters
    ----------
    depi_ts_df : pd.DataFrame
        回测输出的时间序列 DataFrame，需包含：基金代码, DEPI, 调仓日期, 区间
    top_n : int
        显示前几名
    save_path : str
        保存路径（可选）
    """
    setup_chinese_font()
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    depi_ts_df['调仓日期'] = pd.to_datetime(depi_ts_df['调仓日期'])
    
    for i, (code, group) in enumerate(depi_ts_df.groupby('基金代码')):
        if i >= top_n:
            break
        group_sorted = group.sort_values('调仓日期')
        ax.plot(group_sorted['调仓日期'], group_sorted['DEPI'],
                marker='o', markersize=4, linewidth=1.5, label=code)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('DEPI', fontsize=12)
    ax.set_title(f'{title}', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.tick_params(labelbottom=True)
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    plt.show()
    plt.close()


def plot_depi_bar_topn(depi_df: pd.DataFrame,
                       top_n: int = 15,
                       title: str = 'DEPI排名前15基金',
                       save_path: str = None):
    """绘制 DEPI Top-N 基金柱状图。
    
    Parameters
    ----------
    depi_df : pd.DataFrame
        DEPI 排名结果
    top_n : int
        显示前几名
    save_path : str
        保存路径
    """
    setup_chinese_font()
    plt.style.use('seaborn-v0_8-whitegrid')
    
    top = depi_df.head(top_n).copy()
    top = top.sort_values('DEPI')
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top)))[::-1]
    bars = ax.barh(top['基金代码'].astype(str), top['DEPI'], color=colors, edgecolor='white')
    ax.set_xlabel('DEPI', fontsize=12)
    ax.set_ylabel('Fund Code', fontsize=12)
    ax.set_title(f'{title}', fontsize=13)
    
    for bar, val in zip(bars, top['DEPI']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    max_depi = top['DEPI'].max()
    ax.set_xlim(0, max_depi * 1.15 if max_depi > 0 else 1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'  [图表] 已保存: {save_path}')
    plt.show()
    plt.close()
