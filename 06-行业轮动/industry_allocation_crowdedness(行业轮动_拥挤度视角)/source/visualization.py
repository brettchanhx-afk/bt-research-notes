import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def plot_equity_curves(equity_curves: Dict[str, pd.Series],
                       title: str = "Strategy Equity Curve",
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (14, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, equity in equity_curves.items():
        if equity is not None and len(equity) > 0:
            ax.plot(equity.index, equity.values, label=name, linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Net Value', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()
    return fig, ax

def plot_drawdown_series(drawdowns: Dict[str, pd.Series],
                         title: str = "Drawdown Series",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, dd in drawdowns.items():
        if dd is not None and len(dd) > 0:
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
            ax.plot(dd.index, dd.values, linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown', fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()
    return fig, ax

def plot_returns_distribution(returns: pd.Series,
                                title: str = "Returns Distribution",
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].hist(returns.dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    axes[0].axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.4f}')
    axes[0].set_title('Returns Histogram', fontsize=12)
    axes[0].set_xlabel('Return', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    cumulative_returns = (1 + returns).cumprod()
    axes[1].plot(cumulative_returns.index, cumulative_returns.values, linewidth=1.5)
    axes[1].set_title('Cumulative Returns', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=10)
    axes[1].set_ylabel('Cumulative Return', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()
    return fig, axes

def plot_rolling_metrics(returns: pd.Series,
                        benchmark_returns: Optional[pd.Series] = None,
                        window: int = 60,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (14, 10)):
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    axes[0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5, color='blue')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_title(f'Rolling Sharpe Ratio (Window={window} days)', fontsize=12)
    axes[0].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rolling_mean.index, rolling_mean.values, linewidth=1.5, label='Strategy', color='blue')
    if benchmark_returns is not None:
        bench_rolling = benchmark_returns.rolling(window).mean() * 252
        axes[1].plot(bench_rolling.index, bench_rolling.values, linewidth=1.5, label='Benchmark', color='gray', linestyle='--')
    axes[1].set_title(f'Rolling Annualized Return (Window={window} days)', fontsize=12)
    axes[1].set_ylabel('Return', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(rolling_std.index, rolling_std.values, linewidth=1.5, color='orange')
    axes[2].set_title(f'Rolling Annualized Volatility (Window={window} days)', fontsize=12)
    axes[2].set_ylabel('Volatility', fontsize=10)
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()
    return fig, axes

def plot_crowdedness_heatmap(crowdedness_signals: pd.DataFrame,
                             title: str = "Industry Crowdedness Heatmap",
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (16, 10)):
    signal_sum = crowdedness_signals.sum(axis=1)
    selected_dates = signal_sum.index[::max(1, len(signal_sum) // 100)]
    sampled_signals = crowdedness_signals.loc[selected_dates]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sampled_signals.T.values, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    ax.set_yticks(range(len(sampled_signals.columns)))
    ax.set_yticklabels(sampled_signals.columns, fontsize=7)
    ax.set_xticks(range(0, len(sampled_signals.index), max(1, len(sampled_signals.index) // 10)))
    ax.set_xticklabels([d.strftime('%Y-%m') for d in sampled_signals.index[::max(1, len(sampled_signals.index) // 10)]], rotation=45)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Crowdedness Signal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()
    return fig, ax

def plot_strategy_comparison(results_dict: Dict[str, Dict],
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (14, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    metrics_to_plot = [
        ('annual_return', 'Annual Return', axes[0, 0]),
        ('sharpe_ratio', 'Sharpe Ratio', axes[0, 1]),
        ('max_drawdown', 'Max Drawdown', axes[1, 0]),
        ('win_rate', 'Win Rate', axes[1, 1])
    ]
    for metric_key, metric_label, ax in metrics_to_plot:
        values = []
        names = []
        for name, result in results_dict.items():
            metrics = result.get('metrics', {})
            if metric_key in metrics:
                values.append(metrics[metric_key])
                names.append(name)
        if values:
            colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
            bars = ax.bar(names, values, color=colors, edgecolor='black')
            ax.set_title(metric_label, fontsize=12)
            ax.set_ylabel(metric_label, fontsize=10)
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Strategy Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    plt.show()
    return fig, axes

def create_performance_summary(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    summary_data = []
    for name, result in results_dict.items():
        metrics = result.get('metrics', {})
        row = {
            'Strategy Name': name,
            'Annual Return': f"{metrics.get('annual_return', 0):.2%}",
            'Annual Volatility': f"{metrics.get('annual_volatility', 0):.2%}",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
            'Cumulative Return': f"{metrics.get('cumulative_return', 0):.2%}",
            'Win Rate': f"{metrics.get('win_rate', 0):.2%}"
        }
        if 'annual_excess_return' in metrics:
            row['Annual Excess Return'] = f"{metrics['annual_excess_return']:.2%}"
        summary_data.append(row)
    return pd.DataFrame(summary_data)

if __name__ == "__main__":
    print("Visualization module test...")
    test_returns = pd.Series(np.random.randn(100) * 0.02,
                            index=pd.date_range('2020-01-01', periods=100, freq='D'))
    test_equity = (1 + test_returns).cumprod() * 1000000
    test_drawdown = pd.Series(np.random.uniform(-0.2, 0, 100),
                             index=test_returns.index)
    plot_equity_curves({'Test Strategy': test_equity}, title="Test Strategy Equity Curve")
    plot_drawdown_series({'Test Strategy': test_drawdown}, title="Test Strategy Drawdown")
    print("Visualization module test completed")