"""
可视化模块

提供策略回测结果可视化功能
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class BacktestPlotter:
    """
    回测结果可视化
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 100):
        """
        初始化绘图器

        Args:
            figsize: 图形大小
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_cumulative_return(
        self,
        strategy_return: pd.Series,
        benchmark_return: Optional[pd.Series] = None,
        title: str = "Cumulative Return",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制累计收益曲线

        Args:
            strategy_return: 策略累计收益
            benchmark_return: 基准累计收益
            title: 标题
            output_path: 输出路径

        Returns:
            图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        dates = pd.to_datetime(strategy_return.index)
        ax.plot(
            dates,
            strategy_return.values,
            label="Strategy",
            linewidth=2,
            color="#1f77b4",
        )

        if benchmark_return is not None:
            benchmark_aligned = benchmark_return.reindex(strategy_return.index).fillna(0)
            ax.plot(
                dates,
                benchmark_aligned.values,
                label="Benchmark",
                linewidth=1.5,
                color="#ff7f0e",
                linestyle="--",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Return", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_drawdown(
        self,
        cumulative_return: pd.Series,
        title: str = "Drawdown",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制回撤曲线

        Args:
            cumulative_return: 累计收益
            title: 标题
            output_path: 输出路径

        Returns:
            图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        wealth_index = cumulative_return + 1
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        dates = pd.to_datetime(drawdowns.index)
        ax.fill_between(
            dates,
            drawdowns.values * 100,
            0,
            alpha=0.3,
            color="#d62728",
            label="Drawdown",
        )
        ax.plot(dates, drawdowns.values * 100, color="#d62728", linewidth=1)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制收益分布

        Args:
            returns: 收益序列
            title: 标题
            output_path: 输出路径

        Returns:
            图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]), dpi=self.dpi)

        axes[0].hist(returns * 100, bins=50, alpha=0.7, color="#1f77b4", edgecolor="black")
        axes[0].axvline(returns.mean() * 100, color="red", linestyle="--", linewidth=2, label=f"Mean: {returns.mean()*100:.2f}%")
        axes[0].set_title(f"{title} - Histogram", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Return (%)", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].boxplot(returns * 100, vert=True)
        axes[1].set_title(f"{title} - Box Plot", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Return (%)", fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_layer_backtest(
        self,
        layer_returns: Dict[str, pd.Series],
        title: str = "Layer Backtest Results",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制分层回测结果

        Args:
            layer_returns: 各层收益字典
            title: 标题
            output_path: 输出路径

        Returns:
            图形对象
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), dpi=self.dpi)

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

        for i, (layer_name, cum_return) in enumerate(layer_returns.items()):
            if isinstance(cum_return, dict):
                cum_return = cum_return.get("cumulative_return", pd.Series())
            if not isinstance(cum_return, pd.Series) or cum_return.empty:
                continue
            dates = pd.to_datetime(cum_return.index)
            axes[0].plot(dates, cum_return.values * 100, label=layer_name, linewidth=2, color=colors[i % len(colors)])

        axes[0].set_title(f"{title} - Cumulative Returns", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Date", fontsize=10)
        axes[0].set_ylabel("Cumulative Return (%)", fontsize=10)
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

        metrics_data = []
        for layer_name, data in layer_returns.items():
            if isinstance(data, dict):
                metrics = data.get("metrics", {})
                metrics_data.append({"layer": layer_name, **metrics})

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df = metrics_df.set_index("layer")

            x = np.arange(len(metrics_df))
            width = 0.35

            if "annualized_return" in metrics_df.columns:
                bars = axes[1].bar(x, metrics_df["annualized_return"] * 100, width, color=colors[:len(metrics_df)])
                axes[1].set_title(f"{title} - Annualized Returns", fontsize=12, fontweight="bold")
                axes[1].set_xlabel("Layer", fontsize=10)
                axes[1].set_ylabel("Annualized Return (%)", fontsize=10)
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(metrics_df.index)
                axes[1].grid(True, alpha=0.3, axis="y")

                for bar, val in zip(bars, metrics_df["annualized_return"] * 100):
                    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict],
        title: str = "Strategy Comparison",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制策略指标对比

        Args:
            metrics_dict: 指标字典
            title: 标题
            output_path: 输出路径

        Returns:
            图形对象
        """
        comparison_data = []
        for name, metrics in metrics_dict.items():
            row = {"strategy": name}
            row.update(metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.set_index("strategy")

        metric_cols = ["annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"]
        available_cols = [col for col in metric_cols if col in df.columns]

        if not available_cols:
            return plt.Figure()

        fig, axes = plt.subplots(
            2, len(available_cols) // 2 + (len(available_cols) % 2 > 0),
            figsize=(self.figsize[0] * 1.5, self.figsize[1]),
            dpi=self.dpi,
        )
        axes = axes.flatten() if len(available_cols) > 1 else [axes]

        for i, col in enumerate(available_cols):
            values = df[col].values
            strategies = df.index.tolist()

            axes[i].barh(strategies, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(strategies)])
            axes[i].set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
            axes[i].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def save_metrics_table(
        self,
        metrics_dict: Dict[str, Dict],
        output_path: str,
    ) -> None:
        """
        保存指标表格

        Args:
            metrics_dict: 指标字典
            output_path: 输出路径
        """
        comparison_data = []

        for name, metrics in metrics_dict.items():
            row = {"Strategy": name}
            for k, v in metrics.items():
                if isinstance(v, float):
                    row[k.replace("_", " ").title()] = f"{v:.4f}"
                else:
                    row[k.replace("_", " ").title()] = v
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"指标表格已保存至: {output_path}")


def plot_backtest_results(
    strategy_return: pd.Series,
    benchmark_return: Optional[pd.Series] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    便捷函数：绘制回测结果

    Args:
        strategy_return: 策略收益
        benchmark_return: 基准收益
        output_dir: 输出目录

    Returns:
        图形字典
    """
    plotter = BacktestPlotter()
    figures = {}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    cumulative = (1 + strategy_return).cumprod() - 1

    if output_dir:
        fig = plotter.plot_cumulative_return(
            cumulative,
            benchmark_return,
            title="Strategy vs Benchmark Cumulative Return",
            output_path=str(output_dir / "cumulative_return.png"),
        )
    else:
        fig = plotter.plot_cumulative_return(
            cumulative,
            benchmark_return,
            title="Strategy vs Benchmark Cumulative Return",
        )
    figures["cumulative_return"] = fig

    if output_dir:
        fig = plotter.plot_drawdown(
            cumulative,
            title="Strategy Drawdown",
            output_path=str(output_dir / "drawdown.png"),
        )
    else:
        fig = plotter.plot_drawdown(
            cumulative,
            title="Strategy Drawdown",
        )
    figures["drawdown"] = fig

    if output_dir:
        fig = plotter.plot_returns_distribution(
            strategy_return,
            title="Strategy Returns",
            output_path=str(output_dir / "returns_distribution.png"),
        )
    else:
        fig = plotter.plot_returns_distribution(
            strategy_return,
            title="Strategy Returns",
        )
    figures["returns_distribution"] = fig

    return figures
