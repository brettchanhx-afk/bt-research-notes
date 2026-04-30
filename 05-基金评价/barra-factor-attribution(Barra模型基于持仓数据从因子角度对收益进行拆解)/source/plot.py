# -*- coding: utf-8 -*-
"""
plot.py - Barra模型可视化绘图模块

【功能说明】
1. 因子暴露柱状图
2. 因子贡献瀑布图
3. 滚动因子暴露时间序列图
4. 因子显著性热力图
5. Alpha时间序列图
6. 模型拟合度图

【版本】
v1.0  2026-04-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os

# ---- 中文字体配置（Windows） ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# 统一配色方案
COLORS = {
    'positive': '#2ecc71',   # 正向 - 绿色
    'negative': '#e74c3c',   # 负向 - 红色
    'neutral':  '#3498db',   # 中性 - 蓝色
    'alpha':    '#9b59b6',   # Alpha - 紫色
    'benchmark':'#f39c12',   # 基准 - 橙色
    'residual': '#95a5a6',   # 残差 - 灰色
}

FACTOR_COLORS = [
    '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
    '#2980b9', '#27ae60', '#d35400', '#8e44ad', '#2c3e50'
]


class BarraVisualizer:
    """
    Barra模型可视化器

    【使用示例】
        >>> viz = BarraVisualizer()
        >>> viz.plot_factor_exposure(result)
        >>> viz.plot_rolling_exposure(rolling_df)
    """

    def __init__(self, output_dir: str = None):
        """
        【参数】
            output_dir: 图表输出目录，默认项目下output/
        """
        if output_dir is None:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'output'
            )
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # =================================================================
    # 图1: 因子暴露柱状图
    # =================================================================

    def plot_factor_exposure(self,
                             attribution_result,
                             title: str = '基金Barra因子暴露',
                             save_path: str = None,
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制因子暴露柱状图（含显著性标记）

        【参数】
            attribution_result: FactorAttributionResult
            title: 图表标题
            save_path: 保存路径（None则自动生成）
            figsize: 图表大小

        【说明】
            正值用绿色，负值用红色
            *** p<0.001, ** p<0.01, * p<0.05
        """
        fig, ax = plt.subplots(figsize=figsize)

        factor_names = attribution_result.factor_names
        b = attribution_result.b
        t_stats = attribution_result.t_stats
        p_values = attribution_result.p_values

        # 颜色
        colors = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in b]

        # 柱状图
        bars = ax.bar(range(len(factor_names)), b, color=colors, alpha=0.8, edgecolor='white')

        # 显著性标记
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                marker = ''

            y_pos = bar.get_height()
            va = 'bottom' if y_pos >= 0 else 'top'
            offset = 0.05 if y_pos >= 0 else -0.05

            ax.text(i, y_pos + offset, marker, ha='center', va=va, fontsize=12, fontweight='bold')
            ax.text(i, y_pos + offset * 2.5, f'{y_pos:.3f}', ha='center', va=va, fontsize=9)

        # 零线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 标签
        ax.set_xticks(range(len(factor_names)))
        ax.set_xticklabels(factor_names, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('因子暴露系数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 注释
        ax.text(0.98, 0.98, '*** p<0.001  ** p<0.01  * p<0.05',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'factor_exposure.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    # =================================================================
    # 图2: 因子贡献瀑布图
    # =================================================================

    def plot_factor_contribution_waterfall(self,
                                           attribution_result,
                                           title: str = 'Barra因子贡献分解',
                                           save_path: str = None,
                                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制因子贡献瀑布图

        【说明】
            从左到右依次累加各因子贡献，最终达到总收益
        """
        fig, ax = plt.subplots(figsize=figsize)

        contributions = attribution_result.factor_contributions
        alpha = attribution_result.alpha
        residual = attribution_result.residual_contribution

        # 构建瀑布数据
        items = list(contributions.keys()) + ['Alpha', '残差', '总收益']
        values = list(contributions.values()) + [alpha, residual]
        cumulative = np.cumsum(values)
        total = cumulative[-1]
        values.append(total)

        # 绘制
        running_total = 0
        for i, (item, val) in enumerate(zip(items[:-1], values[:-1])):
            color = COLORS['positive'] if val >= 0 else COLORS['negative']
            ax.bar(i, val, bottom=running_total, color=color, alpha=0.8, edgecolor='white')
            y_pos = running_total + val / 2
            ax.text(i, y_pos, f'{val*100:.2f}%', ha='center', va='center', fontsize=9, fontweight='bold')
            running_total += val

        # 总收益柱
        ax.bar(len(items) - 1, total, color=COLORS['neutral'], alpha=0.8, edgecolor='white')
        ax.text(len(items) - 1, total / 2, f'{total*100:.2f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # 连接线
        running_total = 0
        for i, val in enumerate(values[:-1]):
            running_total += val
            if i < len(values) - 2:
                ax.plot([i + 0.4, i + 0.6], [running_total, running_total],
                        color='gray', linestyle='--', linewidth=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(items, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('收益贡献 (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'factor_contribution_waterfall.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    # =================================================================
    # 图3: 滚动因子暴露时间序列图
    # =================================================================

    def plot_rolling_exposure(self,
                              rolling_exposure_df: pd.DataFrame,
                              rolling_significance_df: pd.DataFrame = None,
                              title: str = '滚动因子暴露时序',
                              save_path: str = None,
                              figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        绘制滚动因子暴露时间序列

        【参数】
            rolling_exposure_df: 滚动因子暴露DataFrame (行=日期, 列=因子名)
            rolling_significance_df: 滚动显著性DataFrame
        """
        n_factors = len(rolling_exposure_df.columns)

        fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)

        if n_factors == 1:
            axes = [axes]

        for i, col in enumerate(rolling_exposure_df.columns):
            ax = axes[i]
            data = rolling_exposure_df[col]

            ax.plot(data.index, data.values, color=FACTOR_COLORS[i % len(FACTOR_COLORS)],
                    linewidth=1.5, label=col)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.fill_between(data.index, data.values, 0,
                           where=data.values >= 0, alpha=0.15,
                           color=COLORS['positive'])
            ax.fill_between(data.index, data.values, 0,
                           where=data.values < 0, alpha=0.15,
                           color=COLORS['negative'])

            # 显著性标记
            if rolling_significance_df is not None and col in rolling_significance_df.columns:
                sig_mask = rolling_significance_df[col] == '*'
                if sig_mask.any():
                    sig_dates = rolling_significance_df.index[sig_mask]
                    sig_values = data.reindex(sig_dates)
                    ax.scatter(sig_dates, sig_values, color='red', s=20, zorder=5, marker='*')

            ax.set_ylabel(col, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)

            # 添加均值线
            mean_val = data.mean()
            ax.axhline(y=mean_val, color=FACTOR_COLORS[i % len(FACTOR_COLORS)],
                       linestyle=':', linewidth=1, alpha=0.7)
            ax.text(data.index[-1], mean_val, f' μ={mean_val:.3f}',
                    fontsize=8, va='bottom', color=FACTOR_COLORS[i % len(FACTOR_COLORS)])

        axes[-1].set_xlabel('日期', fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'rolling_exposure.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    # =================================================================
    # 图4: 因子显著性热力图
    # =================================================================

    def plot_significance_heatmap(self,
                                  rolling_significance_df: pd.DataFrame,
                                  title: str = '因子显著性时序',
                                  save_path: str = None,
                                  figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        绘制因子显著性热力图

        【说明】
            深色表示显著，浅色表示不显著
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 转换为数值: '*' → 1, '' → 0
        heatmap_data = (rolling_significance_df == '*').astype(int)

        # 绘制热力图
        im = ax.imshow(heatmap_data.T.values, aspect='auto', cmap='RdYlGn_r',
                       interpolation='nearest', vmin=0, vmax=1)

        # 设置标签
        ax.set_yticks(range(len(heatmap_data.columns)))
        ax.set_yticklabels(heatmap_data.columns, fontsize=10)

        # X轴日期标签（稀疏化）
        n_ticks = min(10, len(heatmap_data))
        tick_positions = np.linspace(0, len(heatmap_data) - 1, n_ticks, dtype=int)
        tick_labels = [str(heatmap_data.index[i].date())[:7]
                       if hasattr(heatmap_data.index[i], 'date')
                       else str(heatmap_data.index[i])[:7]
                       for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['不显著', '显著'])

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'significance_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    # =================================================================
    # 图5: Alpha时间序列图
    # =================================================================

    def plot_alpha_time_series(self,
                                alpha_df: pd.DataFrame,
                                title: str = '滚动Alpha时序',
                                save_path: str = None,
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制Alpha时间序列图

        【参数】
            alpha_df: 包含'alpha'和'alpha_annual'列的DataFrame
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # 子图1: 月度Alpha
        ax1 = axes[0]
        alpha_values = alpha_df['alpha'] * 100
        colors = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in alpha_values]
        ax1.bar(alpha_df.index, alpha_values, color=colors, alpha=0.7, width=20)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('月度Alpha (%)', fontsize=11)
        ax1.set_title('月度Alpha', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 子图2: 年化Alpha
        ax2 = axes[1]
        ax2.plot(alpha_df.index, alpha_df['alpha_annual'] * 100,
                 color=COLORS['alpha'], linewidth=2, label='年化Alpha')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.fill_between(alpha_df.index, alpha_df['alpha_annual'] * 100, 0,
                        where=alpha_df['alpha_annual'] >= 0, alpha=0.15,
                        color=COLORS['positive'])
        ax2.fill_between(alpha_df.index, alpha_df['alpha_annual'] * 100, 0,
                        where=alpha_df['alpha_annual'] < 0, alpha=0.15,
                        color=COLORS['negative'])
        ax2.set_ylabel('年化Alpha (%)', fontsize=11)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.set_title('滚动年化Alpha', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'alpha_time_series.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    # =================================================================
    # 图6: 综合仪表板
    # =================================================================

    def plot_attribution_dashboard(self,
                                   attribution_result,
                                   rolling_exposure_df: pd.DataFrame = None,
                                   alpha_df: pd.DataFrame = None,
                                   fund_code: str = '',
                                   save_path: str = None,
                                   figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
        """
        绘制Barra归因综合仪表板

        【布局】
            左上: 因子暴露柱状图
            右上: 因子贡献饼图
            左下: 滚动因子暴露
            右下: Alpha时序
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # ---- 左上: 因子暴露柱状图 ----
        ax1 = fig.add_subplot(gs[0, 0])

        factor_names = attribution_result.factor_names
        b = attribution_result.b
        p_values = attribution_result.p_values

        colors = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in b]
        bars = ax1.barh(range(len(factor_names)), b, color=colors, alpha=0.8)

        for i, (val, p_val) in enumerate(zip(b, p_values)):
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            ax1.text(val, i, f' {val:.3f}{sig}', va='center',
                    fontsize=9, ha='left' if val >= 0 else 'right')

        ax1.set_yticks(range(len(factor_names)))
        ax1.set_yticklabels(factor_names, fontsize=10)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('因子暴露系数', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # ---- 右上: 因子贡献饼图 ----
        ax2 = fig.add_subplot(gs[0, 1])

        contributions = attribution_result.factor_contributions
        abs_contribs = {k: abs(v) for k, v in contributions.items()}
        abs_contribs['Alpha'] = abs(attribution_result.alpha)
        abs_contribs['残差'] = abs(attribution_result.residual_contribution)

        total_abs = sum(abs_contribs.values())
        if total_abs > 0:
            percentages = {k: v / total_abs * 100 for k, v in abs_contribs.items()}
            sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

            labels = [f'{k}\n{v:.1f}%' for k, v in sorted_items]
            sizes = [v for _, v in sorted_items]
            pie_colors = FACTOR_COLORS[:len(sorted_items)]

            ax2.pie(sizes, labels=labels, colors=pie_colors, autopct='',
                   startangle=90, textprops={'fontsize': 8})

        ax2.set_title('因子贡献占比', fontsize=12, fontweight='bold')

        # ---- 左下: 滚动因子暴露 ----
        ax3 = fig.add_subplot(gs[1, 0])

        if rolling_exposure_df is not None:
            for i, col in enumerate(rolling_exposure_df.columns):
                ax3.plot(rolling_exposure_df.index, rolling_exposure_df[col],
                        color=FACTOR_COLORS[i % len(FACTOR_COLORS)],
                        linewidth=1.2, label=col)

            ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax3.legend(fontsize=8, ncol=3, loc='upper left')
            ax3.set_xlabel('日期', fontsize=10)
            ax3.set_ylabel('暴露系数', fontsize=10)
        ax3.set_title('滚动因子暴露', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # ---- 右下: Alpha时序 ----
        ax4 = fig.add_subplot(gs[1, 1])

        if alpha_df is not None:
            ax4.plot(alpha_df.index, alpha_df['alpha_annual'] * 100,
                    color=COLORS['alpha'], linewidth=2, label='年化Alpha')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.fill_between(alpha_df.index, alpha_df['alpha_annual'] * 100, 0,
                            where=alpha_df['alpha_annual'] >= 0, alpha=0.15,
                            color=COLORS['positive'])
            ax4.fill_between(alpha_df.index, alpha_df['alpha_annual'] * 100, 0,
                            where=alpha_df['alpha_annual'] < 0, alpha=0.15,
                            color=COLORS['negative'])
            ax4.set_xlabel('日期', fontsize=10)
            ax4.set_ylabel('年化Alpha (%)', fontsize=10)
            ax4.legend(fontsize=10)
        ax4.set_title('滚动Alpha', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Barra因子归因仪表板 | {fund_code}',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'attribution_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Barra可视化模块测试\n")

    # 创建模拟结果
    from source.factor import FactorAttributionResult

    result = FactorAttributionResult(
        fund_code='TEST_FUND',
        factor_names=['SIZE', 'VALUE', 'MOMENTUM', 'VOLATILITY', 'QUALITY'],
        b=np.array([1.2, -0.5, 0.8, -0.3, 0.6]),
        t_stats=np.array([3.5, -1.8, 2.5, -0.9, 2.0]),
        p_values=np.array([0.001, 0.08, 0.015, 0.35, 0.05]),
        r_squared=0.72,
        adj_r_squared=0.68,
        factor_contributions={
            'SIZE': 0.024, 'VALUE': -0.008, 'MOMENTUM': 0.015,
            'VOLATILITY': -0.005, 'QUALITY': 0.010
        },
        residual_contribution=0.003,
        alpha=0.004
    )

    viz = BarraVisualizer()

    # 图1: 因子暴露
    fig1 = viz.plot_factor_exposure(result)
    plt.close(fig1)
    print("图1: 因子暴露柱状图 - OK")

    # 图2: 因子贡献瀑布图
    fig2 = viz.plot_factor_contribution_waterfall(result)
    plt.close(fig2)
    print("图2: 因子贡献瀑布图 - OK")

    # 图3: 滚动因子暴露
    dates = pd.date_range('2022-01-01', periods=24, freq='M')
    rolling_df = pd.DataFrame(
        np.random.randn(24, 5) * 0.3 + result.b,
        index=dates, columns=result.factor_names
    )
    sig_df = pd.DataFrame(
        np.random.choice(['', '*'], size=(24, 5), p=[0.4, 0.6]),
        index=dates, columns=result.factor_names
    )
    fig3 = viz.plot_rolling_exposure(rolling_df, sig_df)
    plt.close(fig3)
    print("图3: 滚动因子暴露 - OK")

    # 图4: 显著性热力图
    fig4 = viz.plot_significance_heatmap(sig_df)
    plt.close(fig4)
    print("图4: 显著性热力图 - OK")

    # 图5: Alpha时序
    alpha_df = pd.DataFrame({
        'alpha': np.random.randn(24) * 0.005 + 0.003,
        'alpha_annual': np.random.randn(24) * 0.02 + 0.04
    }, index=dates)
    fig5 = viz.plot_alpha_time_series(alpha_df)
    plt.close(fig5)
    print("图5: Alpha时序 - OK")

    # 图6: 综合仪表板
    fig6 = viz.plot_attribution_dashboard(result, rolling_df, alpha_df, fund_code='TEST')
    plt.close(fig6)
    print("图6: 综合仪表板 - OK")

    print("\n可视化模块测试完成!")
