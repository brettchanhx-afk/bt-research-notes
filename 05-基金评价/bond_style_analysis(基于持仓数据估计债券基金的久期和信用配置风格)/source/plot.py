# -*- coding: utf-8 -*-
"""
plot.py - 债券基金风格可视化模块

支持图表:
1. 久期-信用 2D 风格箱
2. 多期风格演变时间线
3. 净值曲线 vs 基准
4. 持仓结构饼图
5. 信用等级分布

作者: QClaw Agent | 版本: 1.0.0
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib import rcParams


# =============================================================================
# Windows 中文字体配置（必须在 plt.style.use 之后调用）
# =============================================================================
def _setup_chinese_font():
    """设置 matplotlib 中文字体，解决 Windows 下中文显示为方格的问题。
    
    关键：plt.style.use() 会覆盖 rcParams 字体配置，
    所以字体设置必须在 style.use() 之后调用。
    """
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    rcParams["axes.unicode_minus"] = False

    # 强制清除字体缓存，确保新配置生效
    cache_dir = matplotlib.get_cachedir()
    font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
    if os.path.exists(font_cache):
        try:
            os.remove(font_cache)
        except OSError:
            pass


# 全局风格 + 字体设置（顺序：先 style，后字体）
plt.style.use("seaborn-v0_8-whitegrid")
_setup_chinese_font()


# =============================================================================
# 风格箱可视化
# =============================================================================

def plot_style_box_2d(
    duration_style: float,
    credit_style: float,
    style_label: str = None,
    title: str = "Bond Fund Style Box",
    save_path: str = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    绘制久期-信用二维风格箱

    X轴: 久期 (短期 < 3.5 < 中期 < 6.0 < 长期)
    Y轴: 信用评分 (低 < 11.0 < 中 < 14.0 < 高)

    Parameters
    ----------
    duration_style : float
        基金久期风格值
    credit_style : float
        基金信用风格值
    style_label : str
        风格标签
    title : str
        图表标题
    save_path : str, optional
        保存路径

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制9个格子
    colors = [
        ["#E8F5E9", "#C8E6C9", "#A5D6A7"],  # 高信用
        ["#FFF9C4", "#FFF176", "#FFEE58"],  # 中信用
        ["#FFCDD2", "#EF9A9A", "#E57373"],  # 低信用
    ]
    credit_labels = ["High (AAA~AA)", "Medium (A~BBB)", "Low (BB below)"]
    duration_labels = ["Short (<3.5Y)", "Medium (3.5~6Y)", "Long (>6Y)"]

    for i, cred_row in enumerate(colors):
        for j, color in enumerate(cred_row):
            rect = plt.Rectangle((j * 4, i * 5), 4, 5, color=color, alpha=0.8, linewidth=0.5)
            ax.add_patch(rect)

    # 标记基金位置
    # 久期: short=2, mid=6, long=10 (居中)
    dur_bins = [0, 3.5, 6.0, 100]
    dur_centers = [2, 6, 10]
    dur_idx = sum(duration_style >= b for b in dur_bins[1:])

    # 信用: low=2.5, medium=7.5, high=12.5
    cred_bins = [0, 11.0, 14.0, 17.0]
    cred_centers = [2.5, 7.5, 12.5]
    cred_idx = sum(credit_style >= b for b in cred_bins[1:])

    x_pos = dur_centers[min(dur_idx, 2)]
    y_pos = cred_centers[min(cred_idx, 2)]

    ax.scatter(x_pos, y_pos, s=300, c="red", marker="*", zorder=10, linewidths=2, edgecolors="white")

    # 添加标签
    label_text = f"Fund: {style_label or 'Unknown'}\nDuration: {duration_style:.2f}Y\nCredit: {credit_style:.2f}"
    ax.annotate(
        label_text,
        (x_pos, y_pos),
        xytext=(x_pos + 1.5, y_pos + 1),
        fontsize=11,
        fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="darkred"),
    )

    # 坐标轴设置
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 15.5)
    ax.set_xlabel("Duration Style (Years)", fontsize=12)
    ax.set_ylabel("Credit Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 添加格子标签
    for j, dl in enumerate(duration_labels):
        ax.text(j * 4 + 2, -0.8, dl, ha="center", fontsize=9, color="gray")
    for i, cl in enumerate(credit_labels):
        ax.text(-1.2, i * 5 + 2.5, cl, ha="right", va="center", fontsize=9, color="gray")

    # 添加分隔线
    ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=6.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=11.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=14.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([2.5, 7.5, 12.5])
    ax.set_yticklabels(["Low (<11)", "Medium (11~14)", "High (>14)"])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] 风格箱图已保存: {save_path}")

    return fig


def plot_style_box_9grid(
    style_records: pd.DataFrame,
    title: str = "Bond Fund Style Box Evolution",
    save_path: str = None,
) -> plt.Figure:
    """
    绘制风格箱9宫格 + 历史轨迹

    Parameters
    ----------
    style_records : pd.DataFrame
        多期风格记录，包含 period, duration_style, credit_style 列
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 当前风格箱
    if len(style_records) > 0:
        latest = style_records.iloc[-1]
        plot_style_box_2d(
            latest["duration_style"],
            latest["credit_style"],
            style_label=latest.get("style_box", ""),
            title="Current Style Box",
            ax=axes[0],
        )

    # 右图: 风格演变时间线
    if "period" in style_records.columns:
        ax2 = axes[1]
        ax2.plot(range(len(style_records)), style_records["duration_style"].values, "b-o", label="Duration", linewidth=2)
        ax2.plot(range(len(style_records)), style_records["credit_style"].values, "r-s", label="Credit Score", linewidth=2)
        ax2.set_xticks(range(len(style_records)))
        ax2.set_xticklabels(style_records["period"].values)
        ax2.set_xlabel("Period")
        ax2.set_ylabel("Score")
        ax2.set_title("Style Evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# 净值曲线图
# =============================================================================

def plot_nav_curve(
    nav_df: pd.DataFrame,
    benchmark_df: pd.DataFrame = None,
    title: str = "Fund NAV vs Benchmark",
    save_path: str = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    绘制净值曲线对比图

    Parameters
    ----------
    nav_df : pd.DataFrame
        净值数据，需包含 date, nav 列
    benchmark_df : pd.DataFrame, optional
        基准净值数据，需包含 date, nav 列
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

    # 上图: 净值曲线
    if "date" in nav_df.columns:
        dates = pd.to_datetime(nav_df["date"])
    else:
        dates = nav_df.index

    nav_col = "nav" if "nav" in nav_df.columns else nav_df.columns[0]
    ax1.plot(dates, nav_df[nav_col].values / nav_df[nav_col].iloc[0], "b-", linewidth=1.5, label="Fund")

    if benchmark_df is not None:
        bench_col = "nav" if "nav" in benchmark_df.columns else benchmark_df.columns[0]
        bench_dates = pd.to_datetime(benchmark_df["date"]) if "date" in benchmark_df.columns else benchmark_df.index
        ax1.plot(bench_dates, benchmark_df[bench_col].values / benchmark_df[bench_col].iloc[0], "gray", linewidth=1, label="Benchmark", linestyle="--")

    ax1.set_ylabel("Normalized NAV")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 下图: 回撤
    nav_vals = nav_df[nav_col].values
    cummax = np.maximum.accumulate(nav_vals)
    drawdown = (nav_vals - cummax) / cummax
    ax2.fill_between(dates, drawdown, 0, color="red", alpha=0.3)
    ax2.plot(dates, drawdown, "r-", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] 净值曲线已保存: {save_path}")

    return fig


# =============================================================================
# 持仓结构图
# =============================================================================

def plot_holdings_pie(
    holdings: pd.DataFrame,
    title: str = "Holdings Structure",
    save_path: str = None,
    top_n: int = 10,
) -> plt.Figure:
    """
    绘制持仓结构饼图

    Parameters
    ----------
    holdings : pd.DataFrame
        持仓数据，需包含 bond_name, market_value 或 pct 列
    top_n : int
        显示前 N 大持仓
    """
    df = holdings.head(top_n).copy()

    if "pct" in df.columns:
        values = df["pct"].values
        labels = df["bond_name"].values if "bond_name" in df.columns else df.index.astype(str)
    elif "market_value" in df.columns:
        values = df["market_value"].values
        labels = df["bond_name"].values if "bond_name" in df.columns else df.index.astype(str)
    else:
        print("[WARN] 持仓数据缺少 pct 或 market_value 列")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3(range(len(labels)))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color("white" if values[np.argmax(labels == autotext.get_text())] > 5 else "black")

    ax.set_title(title, fontsize=13, fontweight="bold")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# 信用评级分布
# =============================================================================

def plot_credit_distribution(
    holdings: pd.DataFrame,
    ratings: dict = None,
    title: str = "Credit Rating Distribution",
    save_path: str = None,
) -> plt.Figure:
    """
    绘制信用评级分布柱状图

    Parameters
    ----------
    holdings : pd.DataFrame
        持仓数据
    ratings : dict, optional
        债券代码 -> 评级字符串 映射
    """
    if ratings is None:
        ratings = {}

    if "bond_code" not in holdings.columns:
        print("[WARN] 持仓数据缺少 bond_code 列")
        return plt.figure()

    holdings = holdings.copy()
    holdings["rating"] = holdings["bond_code"].map(lambda x: ratings.get(x, ratings.get(str(x), "Unknown")))

    rating_counts = holdings["rating"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(rating_counts)), rating_counts.values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(rating_counts)))
    ax.set_xticklabels(rating_counts.index, rotation=45, ha="right")
    ax.set_xlabel("Credit Rating")
    ax.set_ylabel("Number of Bonds")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(val), ha="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
