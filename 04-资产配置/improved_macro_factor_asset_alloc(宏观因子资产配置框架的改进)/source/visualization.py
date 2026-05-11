import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import os

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    plotly_available = True
except ImportError:
    plotly_available = False

from .config import RESULTS_DIR

if matplotlib_available:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir if output_dir else str(RESULTS_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_portfolio_performance(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        title: str = "Portfolio Performance",
        save_path: Optional[str] = None,
        use_plotly: bool = False,
    ) -> None:
        if not matplotlib_available and not plotly_available:
            print("No visualization library available")
            return

        if use_plotly and plotly_available:
            self._plot_portfolio_performance_plotly(
                portfolio_values, benchmark_values, title, save_path
            )
        else:
            self._plot_portfolio_performance_matplotlib(
                portfolio_values, benchmark_values, title, save_path
            )

    def _plot_portfolio_performance_matplotlib(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series],
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))

        portfolio_normalized = portfolio_values / portfolio_values.iloc[0]
        ax.plot(
            portfolio_normalized.index,
            portfolio_normalized.values,
            label="Strategy",
            linewidth=2,
            color="blue",
        )

        if benchmark_values is not None:
            benchmark_normalized = benchmark_values / benchmark_values.iloc[0]
            ax.plot(
                benchmark_normalized.index,
                benchmark_normalized.values,
                label="Benchmark",
                linewidth=2,
                color="gray",
                linestyle="--",
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "portfolio_performance.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {save_path}")

    def _plot_portfolio_performance_plotly(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series],
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig = make_subplots()

        portfolio_normalized = portfolio_values / portfolio_values.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=portfolio_normalized.index,
                y=portfolio_normalized.values,
                mode="lines",
                name="Strategy",
                line=dict(color="blue", width=2),
            )
        )

        if benchmark_values is not None:
            benchmark_normalized = benchmark_values / benchmark_values.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=benchmark_normalized.index,
                    y=benchmark_normalized.values,
                    mode="lines",
                    name="Benchmark",
                    line=dict(color="gray", width=2, dash="dash"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            template="plotly_white",
            hovermode="x unified",
        )

        if save_path is None:
            save_path = os.path.join(self.output_dir, "portfolio_performance.html")

        fig.write_html(save_path)
        print(f"Interactive chart saved to {save_path}")

    def plot_drawdown(
        self,
        portfolio_values: pd.Series,
        title: str = "Portfolio Drawdown",
        save_path: Optional[str] = None,
        use_plotly: bool = False,
    ) -> None:
        if not matplotlib_available and not plotly_available:
            print("No visualization library available")
            return

        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        if use_plotly and plotly_available:
            self._plot_drawdown_plotly(drawdown, title, save_path)
        else:
            self._plot_drawdown_matplotlib(drawdown, title, save_path)

    def _plot_drawdown_matplotlib(
        self,
        drawdown: pd.Series,
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        ax.plot(drawdown.index, drawdown.values, color="red", linewidth=1)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown", fontsize=12)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "drawdown.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {save_path}")

    def _plot_drawdown_plotly(
        self,
        drawdown: pd.Series,
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(color="red", width=1),
                name="Drawdown",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            template="plotly_white",
        )

        if save_path is None:
            save_path = os.path.join(self.output_dir, "drawdown.html")

        fig.write_html(save_path)
        print(f"Interactive chart saved to {save_path}")

    def plot_weights_allocation(
        self,
        weights_history: pd.DataFrame,
        date: Optional[pd.Timestamp] = None,
        title: str = "Portfolio Weights Allocation",
        save_path: Optional[str] = None,
        use_plotly: bool = False,
    ) -> None:
        if not matplotlib_available and not plotly_available:
            print("No visualization library available")
            return

        if date is None:
            latest_weights = weights_history.iloc[-1]
        else:
            latest_weights = weights_history.loc[date]

        if use_plotly and plotly_available:
            self._plot_weights_allocation_plotly(latest_weights, title, save_path)
        else:
            self._plot_weights_allocation_matplotlib(latest_weights, title, save_path)

    def _plot_weights_allocation_matplotlib(
        self,
        weights: pd.Series,
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))

        weights = weights.sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))

        ax.bar(range(len(weights)), weights.values, color=colors)
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels(weights.index, rotation=45, ha="right")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Asset", fontsize=12)
        ax.set_ylabel("Weight", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "weights_allocation.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {save_path}")

    def _plot_weights_allocation_plotly(
        self,
        weights: pd.Series,
        title: str,
        save_path: Optional[str],
    ) -> None:
        weights = weights.sort_values(ascending=False)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=weights.index,
                y=weights.values,
                marker_color=go.bar.Marker(color=list(range(len(weights)))),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Asset",
            yaxis_title="Weight",
            template="plotly_white",
        )

        if save_path is None:
            save_path = os.path.join(self.output_dir, "weights_allocation.html")

        fig.write_html(save_path)
        print(f"Interactive chart saved to {save_path}")

    def plot_factor_exposure(
        self,
        exposure_matrix: pd.DataFrame,
        title: str = "Factor Exposure Matrix",
        save_path: Optional[str] = None,
        use_plotly: bool = False,
    ) -> None:
        if not matplotlib_available and not plotly_available:
            print("No visualization library available")
            return

        if use_plotly and plotly_available:
            self._plot_factor_exposure_plotly(exposure_matrix, title, save_path)
        else:
            self._plot_factor_exposure_matplotlib(exposure_matrix, title, save_path)

    def _plot_factor_exposure_matplotlib(
        self,
        exposure_matrix: pd.DataFrame,
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))

        im = ax.imshow(
            exposure_matrix.values,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-2,
            vmax=2,
        )

        ax.set_xticks(range(len(exposure_matrix.columns)))
        ax.set_xticklabels(exposure_matrix.columns, rotation=45, ha="right")

        ax.set_yticks(range(len(exposure_matrix.index)))
        ax.set_yticklabels(exposure_matrix.index)

        for i in range(len(exposure_matrix.index)):
            for j in range(len(exposure_matrix.columns)):
                value = exposure_matrix.iloc[i, j]
                if not np.isnan(value):
                    color = "white" if abs(value) > 1 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Exposure")

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "factor_exposure.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {save_path}")

    def _plot_factor_exposure_plotly(
        self,
        exposure_matrix: pd.DataFrame,
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig = go.Figure(
            data=go.Heatmap(
                z=exposure_matrix.values,
                x=exposure_matrix.columns,
                y=exposure_matrix.index,
                colorscale="RdBu_r",
                zmid=0,
                text=exposure_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Exposure"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Factor",
            yaxis_title="Asset",
            template="plotly_white",
        )

        if save_path is None:
            save_path = os.path.join(self.output_dir, "factor_exposure.html")

        fig.write_html(save_path)
        print(f"Interactive chart saved to {save_path}")

    def plot_monthly_returns(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns",
        save_path: Optional[str] = None,
        use_plotly: bool = False,
    ) -> None:
        if not matplotlib_available and not plotly_available:
            print("No visualization library available")
            return

        monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

        if use_plotly and plotly_available:
            self._plot_monthly_returns_plotly(monthly_returns, title, save_path)
        else:
            self._plot_monthly_returns_matplotlib(monthly_returns, title, save_path)

    def _plot_monthly_returns_matplotlib(
        self,
        monthly_returns: pd.Series,
        title: str,
        save_path: Optional[str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ["green" if x > 0 else "red" for x in monthly_returns.values]

        ax.bar(range(len(monthly_returns)), monthly_returns.values * 100, color=colors)

        ax.set_xticks(range(len(monthly_returns)))
        ax.set_xticklabels(
            [d.strftime("%Y-%m") for d in monthly_returns.index],
            rotation=45,
            ha="right",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Return (%)", fontsize=12)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "monthly_returns.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved to {save_path}")

    def _plot_monthly_returns_plotly(
        self,
        monthly_returns: pd.Series,
        title: str,
        save_path: Optional[str],
    ) -> None:
        colors = ["green" if x > 0 else "red" for x in monthly_returns.values]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[d.strftime("%Y-%m") for d in monthly_returns.index],
                y=monthly_returns.values * 100,
                marker_color=colors,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Return (%)",
            template="plotly_white",
        )

        if save_path is None:
            save_path = os.path.join(self.output_dir, "monthly_returns.html")

        fig.write_html(save_path)
        print(f"Interactive chart saved to {save_path}")

    def generate_backtest_report(
        self,
        backtest_results: Dict,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        weights_history: Optional[pd.DataFrame] = None,
        exposure_matrix: Optional[pd.DataFrame] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        if save_dir is None:
            save_dir = self.output_dir

        os.makedirs(save_dir, exist_ok=True)

        print("Generating backtest report charts...")

        self.plot_portfolio_performance(
            portfolio_values,
            benchmark_values,
            title="Macro Factor Asset Allocation Strategy Performance",
            save_path=os.path.join(save_dir, "strategy_performance.png"),
        )

        self.plot_drawdown(
            portfolio_values,
            title="Strategy Drawdown",
            save_path=os.path.join(save_dir, "drawdown.png"),
        )

        if weights_history is not None and len(weights_history) > 0:
            self.plot_weights_allocation(
                weights_history,
                title="Latest Portfolio Weights",
                save_path=os.path.join(save_dir, "weights_allocation.png"),
            )

        if exposure_matrix is not None and not exposure_matrix.empty:
            self.plot_factor_exposure(
                exposure_matrix,
                title="Factor Exposure Matrix",
                save_path=os.path.join(save_dir, "factor_exposure.png"),
            )

        if "returns_history" in backtest_results:
            self.plot_monthly_returns(
                backtest_results["returns_history"],
                title="Monthly Returns",
                save_path=os.path.join(save_dir, "monthly_returns.png"),
            )

        print(f"\nAll charts saved to {save_dir}")

        summary_text = f"""
Backtest Performance Summary
=============================
Total Return: {backtest_results.get('total_return', 0) * 100:.2f}%
Annualized Return: {backtest_results.get('annualized_return', 0) * 100:.2f}%
Annualized Volatility: {backtest_results.get('annualized_volatility', 0) * 100:.2f}%
Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}
Max Drawdown: {backtest_results.get('max_drawdown', 0) * 100:.2f}%
Win Rate: {backtest_results.get('win_rate', 0) * 100:.2f}%
Profit/Loss Ratio: {backtest_results.get('profit_loss_ratio', 0):.2f}
Number of Trades: {backtest_results.get('num_trades', 0)}
"""

        summary_path = os.path.join(save_dir, "backtest_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print(f"Summary saved to {summary_path}")
