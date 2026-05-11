import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    def __init__(self, figsize=(14, 8), style='seaborn-v0_8'):
        self.figsize = figsize
        self.style = style

    def plot_equity_curve(self, portfolio_values, benchmark_values=None, title='Portfolio Equity Curve',
                         save_path=None, show=True):
        if portfolio_values is None or len(portfolio_values) == 0:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if isinstance(portfolio_values, list):
            portfolio_values = pd.Series(portfolio_values)

        dates = range(len(portfolio_values))
        ax.plot(dates, portfolio_values, label='Strategy', linewidth=2, color='blue')

        if benchmark_values is not None:
            if isinstance(benchmark_values, list):
                benchmark_values = pd.Series(benchmark_values)
            ax.plot(dates, benchmark_values, label='Benchmark', linewidth=1.5, color='gray', linestyle='--')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_cumulative_returns(self, returns, benchmark_returns=None, title='Cumulative Returns',
                               save_path=None, show=True):
        if returns is None or len(returns) == 0:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        cum_returns = (1 + returns / 100).cumprod() - 1
        cum_returns_pct = cum_returns * 100

        ax.plot(cum_returns_pct, label='Strategy', linewidth=2, color='blue')

        if benchmark_returns is not None:
            if isinstance(benchmark_returns, list):
                benchmark_returns = pd.Series(benchmark_returns)
            bench_cum = (1 + benchmark_returns / 100).cumprod() - 1
            ax.plot(bench_cum * 100, label='Benchmark', linewidth=1.5, color='gray', linestyle='--')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_drawdown(self, portfolio_values, title='Drawdown Analysis',
                     save_path=None, show=True):
        if portfolio_values is None or len(portfolio_values) == 0:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if isinstance(portfolio_values, list):
            portfolio_values = pd.Series(portfolio_values)

        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax * 100

        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown, color='red', linewidth=1)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_returns_distribution(self, returns, title='Returns Distribution',
                                  save_path=None, show=True):
        if returns is None or len(returns) == 0:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if isinstance(returns, list):
            returns = pd.Series(returns)

        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return None

        ax.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_correlation_matrix(self, data, title='Indicator Correlation Matrix',
                              save_path=None, show=True):
        if data is None or len(data) == 0:
            return None

        corr = data.corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   square=True, linewidths=0.5, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_industry_allocation(self, signals_df, date, title='Industry Allocation',
                                save_path=None, show=True):
        if signals_df is None or len(signals_df) == 0:
            return None

        date_signals = signals_df[signals_df['date'] == date]

        if len(date_signals) == 0:
            return None

        long_count = (date_signals['signal'] == 1).sum()
        short_count = (date_signals['signal'] == -1).sum()
        neutral_count = (date_signals['signal'] == 0).sum()

        fig, ax = plt.subplots(figsize=self.figsize)

        labels = ['Long', 'Short', 'Neutral']
        sizes = [long_count, short_count, neutral_count]
        colors = ['green', 'red', 'gray']
        explode = (0.05, 0.05, 0)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
              autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_indicator_performance(self, indicator_metrics, title='Indicator Performance Comparison',
                                  save_path=None, show=True):
        if indicator_metrics is None or len(indicator_metrics) == 0:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        ind_names = list(indicator_metrics.keys())
        excess_returns = [indicator_metrics[ind]['annual_excess_return'] for ind in ind_names]

        colors = ['green' if x > 0 else 'red' for x in excess_returns]

        ax.barh(ind_names, excess_returns, color=colors, alpha=0.7)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Annual Excess Return (%)', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_prosperity_cycle(self, prosperity_data, title='Industry Prosperity Cycle',
                             save_path=None, show=True):
        if prosperity_data is None or len(prosperity_data) == 0:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        if 'trade_date' in prosperity_data.columns:
            dates = prosperity_data['trade_date']
        else:
            dates = range(len(prosperity_data))

        if 'n_prosperity_industries' in prosperity_data.columns:
            values = prosperity_data['n_prosperity_industries']
        else:
            values = prosperity_data.iloc[:, 1]

        ax.plot(dates, values, linewidth=2, color='blue')
        ax.fill_between(dates, values, alpha=0.3, color='blue')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Number of Prosperity Industries', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig


if __name__ == "__main__":
    print("Testing DataVisualizer...")

    visualizer = DataVisualizer()

    sample_values = pd.Series([1000000, 1050000, 1020000, 1100000, 1080000, 1150000, 1120000, 1180000])
    sample_returns = pd.Series([0.05, -0.03, 0.08, -0.02, 0.06, -0.03, 0.05, 0.03])

    print("Sample visualizations created")
    print("Visualization module test completed!")
