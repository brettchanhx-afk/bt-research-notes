"""
Plotting module for macro risk parity backtest results.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

from source.config import OUTPUT_DIR, FACTOR_NAMES


def plot_cumulative_returns(results_dict, save_path=None):
    """
    Plot cumulative returns of all strategies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (name, data) in enumerate(results_dict.items()):
        cum_ret = data["cumulative"]
        ax = axes[0]
        ax.plot(cum_ret.index, cum_ret.values, label=name, color=colors[i % len(colors)], linewidth=1.5)

    axes[0].set_title("Cumulative Returns", fontsize=14)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cumulative Return (1 = Starting Value)")
    axes[0].legend(loc="upper left")
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    strategy_names = list(results_dict.keys())
    metrics_df = pd.DataFrame({name: data["metrics"] for name, data in results_dict.items()}).T

    x = np.arange(len(strategy_names))
    width = 0.25

    ann_rets = metrics_df["annualized_return"].values
    ann_vols = metrics_df["annualized_volatility"].values
    max_dds = metrics_df["max_drawdown"].values

    bars1 = axes[1].bar(x - width, ann_rets, width, label="Ann. Return (%)", color="steelblue")
    bars2 = axes[1].bar(x, ann_vols, width, label="Ann. Vol (%)", color="coral")
    bars3 = axes[1].bar(x + width, max_dds, width, label="Max Drawdown (%)", color="seagreen")

    axes[1].set_title("Strategy Performance Metrics", fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategy_names, rotation=15)
    axes[1].legend()

    for bar in bars1:
        height = bar.get_height()
        axes[1].annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved cumulative returns plot to: {save_path}")
    plt.close()


def plot_factor_risk_contribution(frc_dict, save_path=None):
    """
    Plot factor risk contribution over time (box plot or line plot).
    """
    if not frc_dict:
        print("No FRC data available for plotting.")
        return

    n_strategies = len(frc_dict)
    fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 5))

    if n_strategies == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (name, frc_df) in enumerate(frc_dict.items()):
        ax = axes[i]
        frc_df = frc_df.dropna()
        if frc_df.empty:
            continue

        frc_df.plot(kind="box", ax=ax, color=colors[:len(FACTOR_NAMES)], 
                     patch_artist=True, showmeans=True)
        ax.set_title(f"{name}\nFactor Risk Contribution", fontsize=12)
        ax.set_ylabel("FRC")
        ax.set_xticklabels(FACTOR_NAMES, rotation=45)
        ax.axhline(y=1.0/len(FACTOR_NAMES), color="red", linestyle="--", 
                   alpha=0.5, label="Equal risk")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved FRC plot to: {save_path}")
    plt.close()


def plot_factor_risk_contribution_line(frc_dict, save_path=None):
    """
    Plot factor risk contribution over time as stacked area or line chart.
    """
    if not frc_dict:
        return

    n = len(frc_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))

    if n == 1:
        axes = [axes]

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    for i, (name, frc_df) in enumerate(frc_dict.items()):
        ax = axes[i]
        frc_df = frc_df.dropna()
        if frc_df.empty:
            continue

        frc_pct = frc_df * 100
        for j, col in enumerate(frc_pct.columns):
            ax.plot(frc_pct.index, frc_pct[col], label=col, 
                    color=colors[j % len(colors)], linewidth=1.2, alpha=0.8)

        ax.axhline(y=100/len(FACTOR_NAMES), color="gray", linestyle="--", 
                   alpha=0.6, label="Equal distribution")
        ax.set_title(f"{name} - Factor Risk Contribution Over Time", fontsize=12)
        ax.set_ylabel("FRC (%)")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.set_xlabel("")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved FRC line plot to: {save_path}")
    plt.close()


def plot_weight_allocation(weights_dict, save_path=None):
    """
    Plot average asset weight allocation for each strategy.
    """
    if not weights_dict:
        return

    n = len(weights_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    if n == 1:
        axes = [axes]

    colors_bar = plt.cm.tab20(np.linspace(0, 1, 20))

    for i, (name, weight_df) in enumerate(weights_dict.items()):
        ax = axes[i]
        avg_weights = weight_df.mean()
        avg_weights = avg_weights[avg_weights > 0.001]
        colors = [colors_bar[j % len(colors_bar)] for j in range(len(avg_weights))]

        bars = ax.barh(range(len(avg_weights)), avg_weights.values, color=colors)
        ax.set_yticks(range(len(avg_weights)))
        ax.set_yticklabels(avg_weights.index, fontsize=9)
        ax.set_xlabel("Average Weight")
        ax.set_title(f"{name}\nAverage Asset Weights", fontsize=12)
        ax.set_xlim(0, max(avg_weights.values.max() * 1.2, 0.01))

        for bar, val in zip(bars, avg_weights.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f"{val:.2%}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved weight allocation plot to: {save_path}")
    plt.close()


def plot_monthly_returns_heatmap(monthly_returns_dict, save_path=None):
    """
    Plot monthly returns as a heatmap for each strategy.
    """
    if not monthly_returns_dict:
        return

    n = len(monthly_returns_dict)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 10))

    if n == 1:
        axes = [axes]

    for i, (name, ret_series) in enumerate(monthly_returns_dict.items()):
        ax = axes[i]
        df = ret_series.to_frame(name="Return")
        df["Year"] = df.index.year
        df["Month"] = df.index.month
        pivot = df.pivot_table(values="Return", index="Year", columns="Month", aggfunc="first")
        pivot = pivot * 100

        sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0, annot=True, 
                     fmt=".1f", cbar_kws={"label": "Return (%)"}, 
                     annot_kws={"fontsize": 6})
        ax.set_title(f"{name}\nMonthly Returns (%)", fontsize=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved monthly returns heatmap to: {save_path}")
    plt.close()


def plot_all(results, save_dir=None):
    """
    Generate all plots from backtest results.
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR

    os.makedirs(save_dir, exist_ok=True)

    results_dict = {name: data for name, data in results.strategy_results.items()}

    plot_cumulative_returns(
        results_dict,
        save_path=os.path.join(save_dir, "cumulative_returns.png")
    )

    frc_dict = {}
    weights_dict = {}
    monthly_returns_dict = {}

    for name, data in results_dict.items():
        if "frc" in data:
            frc_dict[name] = data["frc"]
        weights_dict[name] = data["weights"]
        monthly_returns_dict[name] = data["returns"]

    if frc_dict:
        plot_factor_risk_contribution(
            frc_dict,
            save_path=os.path.join(save_dir, "frc_boxplot.png")
        )
        plot_factor_risk_contribution_line(
            frc_dict,
            save_path=os.path.join(save_dir, "frc_line.png")
        )

    if weights_dict:
        plot_weight_allocation(
            weights_dict,
            save_path=os.path.join(save_dir, "weight_allocation.png")
        )

    if monthly_returns_dict:
        plot_monthly_returns_heatmap(
            monthly_returns_dict,
            save_path=os.path.join(save_dir, "monthly_returns_heatmap.png")
        )

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    print("Plotting module loaded.")
