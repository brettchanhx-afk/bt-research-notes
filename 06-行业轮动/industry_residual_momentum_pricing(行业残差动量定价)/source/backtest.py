import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class BacktestEngine:
    def __init__(self, initial_capital=1000000.0, fee_rate=0.0):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.results = {}

    def run_backtest(self, prices, signals, benchmark_prices=None, strategy_name='Strategy'):
        common_dates = prices.index.intersection(signals.index)
        prices = prices.loc[common_dates]
        signals = signals.loc[common_dates]

        if benchmark_prices is not None:
            benchmark_common = benchmark_prices.index.intersection(common_dates)
            benchmark_prices = benchmark_prices.loc[benchmark_common]
            benchmark_returns = benchmark_prices.pct_change().fillna(0)
        else:
            benchmark_returns = None

        asset_returns = prices.pct_change().fillna(0)
        strategy_returns = (signals.shift(1) * asset_returns).sum(axis=1)

        turnovers = signals.diff().abs().sum(axis=1)

        if self.fee_rate > 0:
            transaction_costs = turnovers * self.fee_rate
            strategy_returns_net = strategy_returns - transaction_costs
        else:
            strategy_returns_net = strategy_returns

        net_value = (1 + strategy_returns_net).cumprod() * self.initial_capital

        cumulative_strategy = (1 + strategy_returns_net).cumprod()
        cumulative_benchmark = (1 + benchmark_returns).cumprod() if benchmark_returns is not None else None

        excess_returns = strategy_returns_net - benchmark_returns if benchmark_returns is not None else strategy_returns_net

        running_max = net_value.expanding().max()
        drawdown = (net_value - running_max) / running_max
        max_drawdown = drawdown.min()

        annual_return = strategy_returns_net.mean() * 12
        annual_vol = strategy_returns_net.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        results = {
            'net_value': net_value,
            'returns': strategy_returns_net,
            'turnover': turnovers,
            'drawdown': drawdown,
            'cumulative_strategy': cumulative_strategy,
            'cumulative_benchmark': cumulative_benchmark,
            'excess_returns': excess_returns,
            'benchmark_returns': benchmark_returns,
            'prices': prices,
            'signals': signals,
            'metrics': {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'total_return': cumulative_strategy.iloc[-1] - 1 if len(cumulative_strategy) > 0 else 0,
                'annual_turnover': turnovers.mean() * 12
            }
        }

        self.results[strategy_name] = results
        return results

    def calculate_ic(self, factor_values, forward_returns, method='spearman'):
        if method == 'spearman':
            ic_series = factor_values.corrwith(forward_returns, axis=1, method='spearman')
        else:
            ic_series = factor_values.corrwith(forward_returns, axis=1, method='pearson')

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0

        return {
            'ic_series': ic_series,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir
        }

    def calculate_quantile_returns(self, factor_values, returns, n_quantiles=5):
        quantile_results = {}

        for date in factor_values.index:
            if date not in returns.index:
                continue

            factor_row = factor_values.loc[date].dropna()
            return_row = returns.loc[date].dropna()

            common_assets = factor_row.index.intersection(return_row.index)

            if len(common_assets) < n_quantiles:
                continue

            try:
                quantile_labels = pd.qcut(factor_row[common_assets], q=n_quantiles, labels=False, duplicates='drop')
                for q in range(n_quantiles):
                    quantile_assets = quantile_labels[quantile_labels == q].index
                    if len(quantile_assets) > 0:
                        if q not in quantile_results:
                            quantile_results[q] = []
                        quantile_results[q].append(return_row[quantile_assets].mean())
            except Exception as e:
                continue

        quantile_returns_df = pd.DataFrame(quantile_results)
        return quantile_returns_df

    def plot_net_value(self, save_path=None):
        if not self.results:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        for name, result in self.results.items():
            ax.plot(result['net_value'], label=name)

        ax.set_xlabel('Date')
        ax.set_ylabel('Net Value')
        ax.set_title('Strategy Net Value Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_drawdown(self, strategy_name=None, save_path=None):
        if not self.results:
            return

        if strategy_name and strategy_name in self.results:
            results_to_plot = {strategy_name: self.results[strategy_name]}
        else:
            results_to_plot = self.results

        fig, ax = plt.subplots(figsize=(12, 6))

        for name, result in results_to_plot.items():
            ax.fill_between(result['drawdown'].index, result['drawdown'], 0, alpha=0.3, label=name)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_returns_distribution(self, strategy_name=None, save_path=None):
        if not self.results:
            return

        if strategy_name and strategy_name in self.results:
            results_to_plot = {strategy_name: self.results[strategy_name]}
        else:
            results_to_plot = self.results

        fig, axes = plt.subplots(1, len(results_to_plot), figsize=(6*len(results_to_plot), 4))

        if len(results_to_plot) == 1:
            axes = [axes]

        for ax, (name, result) in zip(axes, results_to_plot.items()):
            returns = result['returns'].dropna() * 100
            ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
            ax.set_xlabel('Monthly Return (%)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Returns Distribution')
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def print_performance_summary(self):
        if not self.results:
            return

        print("=" * 80)
        print("Backtest Performance Summary")
        print("=" * 80)

        for name, result in self.results.items():
            metrics = result['metrics']
            print(f"\n{name}:")
            print("-" * 40)
            print(f"  Annual Return:     {metrics['annual_return']*100:.2f}%")
            print(f"  Annual Volatility: {metrics['annual_volatility']*100:.2f}%")
            print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
            print(f"  Calmar Ratio:      {metrics['calmar_ratio']:.2f}")
            print(f"  Total Return:      {metrics['total_return']*100:.2f}%")
            print(f"  Annual Turnover:   {metrics['annual_turnover']:.2f}x")

        print("\n" + "=" * 80)

    def get_results(self, strategy_name=None):
        if strategy_name:
            return self.results.get(strategy_name, None)
        return self.results

    def export_results(self, output_path):
        if not self.results:
            return

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for name, result in self.results.items():
                result['net_value'].to_excel(writer, sheet_name=f'{name}_nav')
                result['returns'].to_excel(writer, sheet_name=f'{name}_returns')

                metrics_df = pd.DataFrame([result['metrics']])
                metrics_df.to_excel(writer, sheet_name=f'{name}_metrics', index=False)