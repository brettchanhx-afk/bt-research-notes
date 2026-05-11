import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import matplotlib.font_manager as fm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.config import OUTPUT_DIR

chinese_font_names = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
found_font = None
for font_name in chinese_font_names:
    fonts = [f for f in fm.fontManager.ttflist if font_name in f.name]
    if fonts:
        found_font = fm.FontProperties(fname=fonts[0].fname)
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['font.family'] = 'sans-serif'
        break

plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def get_font():
    return found_font if found_font else fm.FontProperties()


def plot_cumulative_returns(backtest, strategies=None, title='Cumulative Returns',
                           save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(14, 7))

    if strategies is None:
        strategies = list(backtest.portfolio_returns.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for i, strategy in enumerate(strategies):
        if strategy in backtest.portfolio_returns:
            returns = backtest.portfolio_returns[strategy]
            cumulative = (1 + returns).cumprod()
            ax.plot(cumulative.index, cumulative.values, label=strategy,
                   linewidth=1.5, color=colors[i])

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_drawdown(backtest, strategies=None, title='Drawdown Analysis',
                  save_path=None, show=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    if strategies is None:
        strategies = list(backtest.portfolio_returns.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for i, strategy in enumerate(strategies):
        if strategy in backtest.portfolio_returns:
            returns = backtest.portfolio_returns[strategy]
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max

            ax1.fill_between(drawdown.index, drawdown.values, 0,
                           alpha=0.3, color=colors[i], label=strategy)
            ax1.plot(drawdown.index, drawdown.values, linewidth=0.8,
                    color=colors[i])

    ax1.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Drawdown', fontsize=12)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, alpha=0.3)

    drawdown_data = []
    for strategy in strategies:
        if strategy in backtest.portfolio_returns:
            returns = backtest.portfolio_returns[strategy]
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            drawdown_data.append({
                'Strategy': strategy,
                'Max Drawdown': drawdown.min()
            })

    dd_df = pd.DataFrame(drawdown_data)
    dd_df = dd_df.sort_values('Max Drawdown')

    bars = ax2.barh(dd_df['Strategy'], dd_df['Max Drawdown'], color=colors[:len(dd_df)])
    ax2.set_title('Maximum Drawdown by Strategy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Maximum Drawdown', fontsize=12)
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, dd_df['Max Drawdown']):
        ax2.text(val - 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.2%}', va='center', ha='right', fontsize=10, color='white')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_rolling_sharpe(backtest, strategies=None, window=252,
                        title='Rolling Sharpe Ratio', save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(14, 7))

    if strategies is None:
        strategies = list(backtest.portfolio_returns.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for i, strategy in enumerate(strategies):
        if strategy in backtest.portfolio_returns:
            returns = backtest.portfolio_returns[strategy]
            rolling_mean = returns.rolling(window).mean() * 252
            rolling_std = returns.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_mean / rolling_std

            ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                   label=strategy, linewidth=1.2, color=colors[i])

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(f'{title} (Window={window} days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_weights_heatmap(weights_df, title='Portfolio Weights Over Time',
                         save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(14, 8))

    weights_pct = weights_df * 100

    im = ax.imshow(weights_pct.T.values, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    ax.set_yticks(range(len(weights_df.columns)))
    ax.set_yticklabels(weights_df.columns)

    n_ticks = min(10, len(weights_df))
    tick_indices = np.linspace(0, len(weights_df) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([weights_df.index[i].strftime('%Y-%m')
                       for i in tick_indices], rotation=45)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight (%)', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Asset', fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_annual_returns(backtest, strategies=None, title='Annual Returns',
                        save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12, 6))

    if strategies is None:
        strategies = list(backtest.portfolio_returns.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    annual_returns_data = {}
    for strategy in strategies:
        if strategy in backtest.portfolio_returns:
            returns = backtest.portfolio_returns[strategy]
            cumulative = (1 + returns).cumprod()
            annual_returns = cumulative.resample('Y').last().pct_change()
            annual_returns_data[strategy] = annual_returns

    annual_df = pd.DataFrame(annual_returns_data).dropna()
    annual_df.index = annual_df.index.year

    annual_df.plot(kind='bar', ax=ax, width=0.8, color=colors[:len(annual_df.columns)])

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Return', fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc='upper left', frameon=True)
    ax.set_xticklabels(annual_df.index, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_metrics_comparison(backtest, strategies=None, title='Strategy Comparison',
                           save_path=None, show=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if strategies is None:
        strategies = list(backtest.metrics.keys())

    metrics_names = []
    sharpe_values = []
    max_dd_values = []
    annual_ret_values = []
    volatility_values = []

    for strategy in strategies:
        if strategy in backtest.metrics:
            m = backtest.metrics[strategy]
            metrics_names.append(strategy)
            sharpe_values.append(m['sharpe_ratio'])
            max_dd_values.append(m['max_drawdown'])
            annual_ret_values.append(m['annualized_return'])
            volatility_values.append(m['annualized_volatility'])

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_names)))

    axes[0, 0].barh(metrics_names, sharpe_values, color=colors)
    axes[0, 0].set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Sharpe Ratio')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    axes[0, 1].barh(metrics_names, max_dd_values, color=colors)
    axes[0, 1].set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Max Drawdown')
    axes[0, 1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    axes[1, 0].barh(metrics_names, annual_ret_values, color=colors)
    axes[1, 0].set_title('Annualized Return', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Return')
    axes[1, 0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    axes[1, 1].barh(metrics_names, volatility_values, color=colors)
    axes[1, 1].set_title('Annualized Volatility', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Volatility')
    axes[1, 1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_all(backtest, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    print("Generating visualizations...")

    plot_cumulative_returns(
        backtest,
        save_path=os.path.join(output_dir, 'cumulative_returns.png'),
        show=False
    )
    print(f"  - Saved cumulative_returns.png")

    plot_drawdown(
        backtest,
        save_path=os.path.join(output_dir, 'drawdown.png'),
        show=False
    )
    print(f"  - Saved drawdown.png")

    plot_rolling_sharpe(
        backtest,
        window=252,
        save_path=os.path.join(output_dir, 'rolling_sharpe.png'),
        show=False
    )
    print(f"  - Saved rolling_sharpe.png")

    plot_metrics_comparison(
        backtest,
        save_path=os.path.join(output_dir, 'metrics_comparison.png'),
        show=False
    )
    print(f"  - Saved metrics_comparison.png")

    plot_annual_returns(
        backtest,
        save_path=os.path.join(output_dir, 'annual_returns.png'),
        show=False
    )
    print(f"  - Saved annual_returns.png")

    for name, weights_df in backtest.weights_history.items():
        safe_name = name.replace(' ', '_').replace('=', '')
        plot_weights_heatmap(
            weights_df,
            title=f'Portfolio Weights - {name}',
            save_path=os.path.join(output_dir, f'weights_{safe_name}.png'),
            show=False
        )
        print(f"  - Saved weights_{safe_name}.png")

    print("All visualizations saved!")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)

    returns_df = pd.DataFrame(
        np.random.randn(n, 5) * 0.015,
        index=dates,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    )

    from source.backtest import Backtest

    bt = Backtest(returns_df)
    bt.backtest_risk_parity(name='RiskParity')
    bt.backtest_momentum_risk_budget(k=1.0, name='Momentum_k1')

    plot_cumulative_returns(bt, save_path=os.path.join(OUTPUT_DIR, 'test_cumulative.png'))
    plot_metrics_comparison(bt, save_path=os.path.join(OUTPUT_DIR, 'test_metrics.png'))

    print("Test visualizations saved to output directory")
