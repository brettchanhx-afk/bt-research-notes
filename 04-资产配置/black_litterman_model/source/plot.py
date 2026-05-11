"""
plot.py - 可视化模块

提供7张图表:
  1. 累计收益曲线对比
  2. 回撤对比
  3. 权重堆叠面积图
  4. 绩效指标柱状图
  5. 年度收益热力图
  6. 先验/后验收益对比
  7. 资产相关性热力图
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

matplotlib.use('Agg')   # 无头环境
plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# 字体配置 (Windows 中文)
# ============================================================================

def _setup_chinese_font():
    """配置 matplotlib 中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120


_setup_chinese_font()


# ============================================================================
# PlotEngine
# ============================================================================

class PlotEngine:
    """
    回测可视化引擎

    Parameters
    ----------
    backtest_result : BacktestResult or None
    output_dir      : str 图片输出目录
    """

    def __init__(
        self,
        backtest_result,
        output_dir: str = 'output',
    ):
        self.result = backtest_result
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── 工具 ─────────────────────────────────────────────────────────
    def _save(self, fig, name: str):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  保存: {path}")

    def _fmt_pct(self, x, pos=None):
        return f'{x:.0f}%' if x >= 0 else f'{x:.0f}%'

    # ── 图1: 累计收益曲线 ─────────────────────────────────────────────
    def plot_cumulative_returns(self, title: str = None) -> plt.Figure:
        if self.result is None:
            return

        strategies = self.result.strategies
        cum = self.result.cumulative_returns

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, strat in enumerate(strategies):
            if strat in cum:
                s = cum[strat]
                if s is not None and not s.empty:
                    ax.plot(
                        s.index, s.values,
                        label=strat,
                        color=colors[i % len(colors)],
                        linewidth=1.8,
                    )

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(title or '累计收益对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('累计收益率 (%)', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:.0f}%'
        ))
        plt.xticks(rotation=30)
        fig.tight_layout()
        self._save(fig, 'fig1_cumulative_returns.png')
        return fig

    # ── 图2: 回撤曲线 ────────────────────────────────────────────────
    def plot_drawdown(self, title: str = None) -> plt.Figure:
        if self.result is None:
            return

        strategies = self.result.strategies
        colors = ['#1f77b4', '#ff7f0e']

        fig, ax = plt.subplots(figsize=(12, 5))

        for i, strat in enumerate(strategies[:2]):  # 只画前两个
            if strat in self.result.daily_returns:
                r = self.result.daily_returns[strat].dropna() / 100
                cum = (1 + r).cumprod()
                peak = cum.cummax()
                dd = (cum - peak) / peak * 100
                ax.fill_between(
                    dd.index, dd.values, 0,
                    alpha=0.35, label=strat,
                    color=colors[i % len(colors)],
                )
                ax.plot(dd.index, dd.values,
                        color=colors[i % len(colors)], linewidth=0.8)

        ax.set_title(title or '回撤对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('回撤 (%)', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.legend(fontsize=10)
        plt.xticks(rotation=30)
        fig.tight_layout()
        self._save(fig, 'fig2_drawdown.png')
        return fig

    # ── 图3: 权重堆叠面积图 ─────────────────────────────────────────
    def plot_weights(self, strategy: str, title: str = None) -> plt.Figure:
        if self.result is None or strategy not in self.result.weights:
            return

        w_df = self.result.weights[strategy]
        if w_df is None or w_df.empty:
            return

        # 资产颜色映射
        asset_colors = {
            'CSI300':  '#1f77b4',  # 蓝
            'SP500':   '#aec7e8',  # 浅蓝
            'HSI':     '#ff7f0e',  # 橙
            'CR_GOV':  '#2ca02c',  # 绿
            'CR_CORP': '#98df8a',  # 浅绿
            'NHCI':    '#d62728',  # 红
        }
        default_color = '#7f7f7f'

        assets = [c for c in w_df.columns if w_df[c].abs().sum() > 0.01]
        n_assets = len(assets)

        fig, ax = plt.subplots(figsize=(12, 4.5))
        stack_data = w_df[assets].fillna(0) * 100

        color_list = [
            asset_colors.get(a, default_color) for a in assets
        ]
        ax.stackplot(
            stack_data.index,
            [stack_data[a].values for a in assets],
            labels=assets,
            colors=color_list,
            alpha=0.85,
        )
        ax.set_title(title or f'{strategy} 资产配置权重', fontsize=14, fontweight='bold')
        ax.set_ylabel('权重 (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:.0f}%'
        ))
        ax.legend(loc='upper right', fontsize=9, ncol=n_assets)
        plt.xticks(rotation=30)
        fig.tight_layout()
        self._save(fig, f'fig3_weights_{strategy}.png')
        return fig

    # ── 图4: 绩效指标柱状图 ─────────────────────────────────────────
    def plot_stats_bar(self, title: str = None) -> plt.Figure:
        if self.result is None:
            return

        stats_df = self.result.summary_table()
        if stats_df is None or stats_df.empty:
            return

        # 取关键指标
        key_metrics = ['年化收益(%)', '年化波动(%)', '最大回撤(%)', '夏普比率', '收益回撤比']
        available = [m for m in key_metrics if m in stats_df.columns]
        if not available:
            return

        sub = stats_df[available].copy()
        strategies = sub.index.tolist()

        fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4.5))
        if len(available) == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        max_val = sub.values.max()

        for i, (ax, metric) in enumerate(zip(axes, available)):
            bars = ax.bar(
                strategies,
                sub[metric].values,
                color=colors[:len(strategies)],
                alpha=0.85,
                edgecolor='white',
            )
            # 数值标签
            for bar, val in zip(bars, sub[metric].values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + max_val * 0.02,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9,
                )
            ax.set_title(metric, fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=30)
            ax.axhline(0, color='gray', linewidth=0.8)

        fig.suptitle(title or '各策略绩效指标对比', fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        self._save(fig, 'fig4_stats_bar.png')
        return fig

    # ── 图5: 年度收益热力图 ──────────────────────────────────────────
    def plot_yearly_heatmap(self, title: str = None) -> plt.Figure:
        if self.result is None:
            return

        strategies = self.result.strategies
        all_years = set()
        yearly_data = {}
        for strat in strategies:
            if strat in self.result.yearly_stats and self.result.yearly_stats[strat] is not None:
                df = self.result.yearly_stats[strat]
                all_years.update(df.index.tolist())
                yearly_data[strat] = df

        if not yearly_data:
            return

        all_years = sorted(all_years)

        # 构建热力图矩阵
        matrix = {}
        for strat in strategies:
            if strat in yearly_data:
                yd = yearly_data[strat]
                # yd: Series with year as index
                matrix[strat] = {y: float(yd.loc[y]) if y in yd.index else np.nan
                                 for y in all_years}

        # heat_df: rows=年份, cols=策略
        heat_df = pd.DataFrame(matrix, index=all_years)
        # 确保所有策略列都存在
        for strat in strategies:
            if strat not in heat_df.columns:
                heat_df[strat] = np.nan
        heat_df = heat_df[sstrategies]  # 保持顺序

        fig, ax = plt.subplots(figsize=(max(8, len(all_years) * 1.2), max(3, len(strategies) * 1.5)))
        sns.heatmap(
            heat_df.T.fillna(0),
            annot=True, fmt='.1f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': '年度收益率 (%)'},
            linewidths=0.5,
            annot_kws={'fontsize': 9},
        )
        ax.set_title(title or '各策略年度收益热力图 (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('')
        ax.set_xlabel('年份', fontsize=11)
        plt.xticks(rotation=45)
        fig.tight_layout()
        self._save(fig, 'fig5_yearly_heatmap.png')
        return fig

    # ── 图6: 先验/后验收益对比 ──────────────────────────────────────
    def plot_prior_vs_posterior(
        self,
        prior_mu: pd.Series,
        posterior_mu: pd.Series,
        asset_names: list = None,
        title: str = None,
    ) -> plt.Figure:
        if prior_mu is None or posterior_mu is None:
            return

        assets = asset_names or list(prior_mu.index)
        x = np.arange(len(assets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(
            x - width / 2, prior_mu.values, width,
            label='先验均衡收益 Π', color='#2ca02c', alpha=0.85,
        )
        bars2 = ax.bar(
            x + width / 2, posterior_mu.values, width,
            label='后验收益 μ_BL', color='#1f77b4', alpha=0.85,
        )

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, fontsize=10)
        ax.set_ylabel('年化收益率 (%)', fontsize=11)
        ax.set_title(title or 'BL模型: 先验 vs 后验收益', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)

        # 数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.1,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8,
                )

        fig.tight_layout()
        self._save(fig, 'fig6_prior_posterior.png')
        return fig

    # ── 图7: 资产相关性热力图 ───────────────────────────────────────
    def plot_correlation_matrix(
        self,
        daily_returns: pd.DataFrame = None,
        title: str = None,
    ) -> plt.Figure:
        """资产日频收益率相关性热力图"""
        if daily_returns is None:
            return

        # 只保留数值列
        num_df = daily_returns.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return

        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(
            corr,
            mask=mask,
            annot=True, fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            ax=ax,
            linewidths=0.5,
            annot_kws={'fontsize': 10},
            cbar_kws={'shrink': 0.8},
        )
        ax.set_title(title or '资产日频收益率相关性矩阵', fontsize=13, fontweight='bold')
        plt.xticks(rotation=30, fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        fig.tight_layout()
        self._save(fig, 'fig7_correlation_matrix.png')
        return fig
