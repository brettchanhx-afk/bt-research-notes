# -*- coding: utf-8 -*-
"""
visualization.py — Barra 风格分析可视化模块
"""

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class BarraPlotter:
    """Barra 分析图表绘制器。"""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid", dpi: int = 150):
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")
        self.dpi = dpi

    # ── 主报告图 ───────────────────────────────────────────
    def plot_report(self, fund_nav: pd.Series, exposures: pd.Series,
                    rolling: pd.DataFrame, output_path: str = None):
        """
        绘制完整报告图（6 子图布局）。
        """
        fig = plt.figure(figsize=(16, 12))
        gs  = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 基金净值
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(fund_nav.index, fund_nav, color="#1f77b4", lw=2, label="基金净值")
        ax1.set_title("基金累计净值", fontsize=12, fontweight="bold")
        ax1.set_ylabel("累计净值")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 风格暴露度
        ax2 = fig.add_subplot(gs[0, 2])
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in exposures.values]
        ax2.barh(exposures.index, exposures.values, color=colors, alpha=0.7)
        ax2.set_title("风格暴露度", fontsize=12, fontweight="bold")
        ax2.axvline(0, color="black", lw=0.8)
        for i, (name, val) in enumerate(exposures.items()):
            ax2.text(val + 0.01 if val >= 0 else val - 0.08, i, f"{val:.3f}", va="center")

        # 3. 市场暴露时序
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(rolling.index, rolling["market"], color="blue", lw=1.5, label="市场暴露")
        ax3.axhline(exposures["market"], color="blue", ls="--", alpha=0.5, label="平均值")
        ax3.set_title("市场暴露度时序（滚动60日）", fontsize=12, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4-6. 其他因子时序
        factors = [("size", "green", "规模暴露"),
                   ("value", "red", "价值暴露"),
                   ("momentum", "purple", "动量暴露")]
        for idx, (col, color, title) in enumerate(factors):
            ax = fig.add_subplot(gs[2, idx])
            ax.plot(rolling.index, rolling[col], color=color, lw=1.5)
            ax.axhline(exposures[col], color=color, ls="--", alpha=0.5)
            ax.set_title(f"{title}时序", fontsize=11)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Barra 风格分析报告", fontsize=16, fontweight="bold", y=0.995)

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            logger.info(f"图表已保存: {output_path}")
        return fig

    # ── 单因子时序 ─────────────────────────────────────────
    def plot_rolling_exposure(self, rolling: pd.DataFrame, factor: str,
                               output_path: str = None):
        """绘制单个因子的滚动暴露时序图。"""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling.index, rolling[factor], lw=1.5)
        ax.axhline(rolling[factor].mean(), color="red", ls="--", label="均值")
        ax.fill_between(rolling.index, rolling[factor], alpha=0.3)
        ax.set_title(f"{factor.capitalize()} 因子滚动暴露", fontsize=12, fontweight="bold")
        ax.set_ylabel("暴露度")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ── 热力图 ─────────────────────────────────────────────
    def plot_rolling_heatmap(self, rolling: pd.DataFrame, output_path: str = None):
        """绘制滚动暴露热力图。"""
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(rolling[["market", "size", "value", "momentum"]].T,
                    cmap="RdYlGn", center=0, ax=ax,
                    xticklabels=20, cbar_kws={"label": "暴露度"})
        ax.set_title("滚动窗口风格暴露热力图", fontsize=12, fontweight="bold")
        ax.set_xlabel("日期")

        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        return fig
