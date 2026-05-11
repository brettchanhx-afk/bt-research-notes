import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class PerformanceEvaluator:
    def __init__(self, returns_dict):
        self.returns_dict = returns_dict
        self.metrics = {}

    def calculate_max_drawdown(self, cumulative_returns):
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_dd = drawdown.min()
        return max_dd

    def calculate_max_drawdown_duration(self, cumulative_returns):
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak

        is_drawdown = drawdown < 0
        if not is_drawdown.any():
            return 0

        drawdown_periods = []
        in_dd = False
        start = None

        for i, (date, dd) in enumerate(drawdown.items()):
            if dd < 0 and not in_dd:
                in_dd = True
                start = i
            elif dd >= 0 and in_dd:
                in_dd = False
                drawdown_periods.append(i - start)

        if in_dd:
            drawdown_periods.append(len(drawdown) - start)

        return max(drawdown_periods) if drawdown_periods else 0

    def calculate_annual_return(self, returns):
        cumulative = (1 + returns).prod()
        years = len(returns) / 12
        return cumulative ** (1 / years) - 1

    def calculate_annual_volatility(self, returns):
        return returns.std() * np.sqrt(12)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        excess_return = returns.mean() * 12 - risk_free_rate
        volatility = returns.std() * np.sqrt(12)
        return excess_return / volatility if volatility != 0 else 0

    def calculate_calmar_ratio(self, returns):
        annual_return = self.calculate_annual_return(returns)
        cumulative = (1 + returns).cumprod()
        max_dd = self.calculate_max_drawdown(cumulative)
        return annual_return / abs(max_dd) if max_dd != 0 else 0

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.0):
        excess_return = returns.mean() * 12 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
        return excess_return / downside_std if downside_std != 0 else 0

    def calculate_win_rate(self, returns):
        return (returns > 0).sum() / len(returns)

    def calculate_win_loss_ratio(self, returns):
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0
        return abs(wins.mean() / losses.mean())

    def evaluate_all(self):
        results = []

        for strategy_name, returns_series in self.returns_dict.items():
            cumulative = (1 + returns_series).cumprod()
            max_dd = self.calculate_max_drawdown(cumulative)
            annual_return = self.calculate_annual_return(returns_series)
            annual_vol = self.calculate_annual_volatility(returns_series)
            sharpe = self.calculate_sharpe_ratio(returns_series)
            calmar = self.calculate_calmar_ratio(returns_series)
            sortino = self.calculate_sortino_ratio(returns_series)
            win_rate = self.calculate_win_rate(returns_series)

            metrics = {
                'Strategy': strategy_name,
                'Annual Return': f"{annual_return * 100:.2f}%",
                'Annual Volatility': f"{annual_vol * 100:.2f}%",
                'Max Drawdown': f"{max_dd * 100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Calmar Ratio': f"{calmar:.2f}",
                'Sortino Ratio': f"{sortino:.2f}",
                'Win Rate': f"{win_rate * 100:.2f}%"
            }

            results.append(metrics)

        self.metrics = pd.DataFrame(results)
        return self.metrics

    def get_yearly_returns(self):
        yearly_results = {}

        for strategy_name, returns_series in self.returns_dict.items():
            yearly_returns = returns_series.resample('A').apply(lambda x: (1 + x).prod() - 1)
            yearly_results[strategy_name] = yearly_returns

        return pd.DataFrame(yearly_results)

    def plot_metrics_comparison(self, figsize=(14, 10)):
        metrics_df = self.evaluate_all()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        strategies = metrics_df['Strategy'].values

        annual_returns = [float(m.rstrip('%')) for m in metrics_df['Annual Return'].values]
        axes[0, 0].barh(strategies, annual_returns, color='#2ecc71')
        axes[0, 0].set_xlabel('Annual Return (%)')
        axes[0, 0].set_title('Annual Return Comparison')
        for i, v in enumerate(annual_returns):
            axes[0, 0].text(v + 0.1, i, f'{v:.2f}%', va='center')

        max_drawdowns = [float(m.rstrip('%')) for m in metrics_df['Max Drawdown'].values]
        axes[0, 1].barh(strategies, max_drawdowns, color='#e74c3c')
        axes[0, 1].set_xlabel('Max Drawdown (%)')
        axes[0, 1].set_title('Max Drawdown Comparison')
        for i, v in enumerate(max_drawdowns):
            axes[0, 1].text(v + 0.1, i, f'{v:.2f}%', va='center')

        sharpe_ratios = [float(m) for m in metrics_df['Sharpe Ratio'].values]
        axes[1, 0].barh(strategies, sharpe_ratios, color='#3498db')
        axes[1, 0].set_xlabel('Sharpe Ratio')
        axes[1, 0].set_title('Sharpe Ratio Comparison')
        for i, v in enumerate(sharpe_ratios):
            axes[1, 0].text(v + 0.1, i, f'{v:.2f}', va='center')

        calmar_ratios = [float(m) for m in metrics_df['Calmar Ratio'].values]
        axes[1, 1].barh(strategies, calmar_ratios, color='#9b59b6')
        axes[1, 1].set_xlabel('Calmar Ratio')
        axes[1, 1].set_title('Calmar Ratio Comparison')
        for i, v in enumerate(calmar_ratios):
            axes[1, 1].text(v + 0.1, i, f'{v:.2f}', va='center')

        plt.tight_layout()
        return plt

    def plot_drawdown(self, figsize=(14, 8)):
        plt.figure(figsize=figsize)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (name, returns) in enumerate(self.returns_dict.items()):
            cumulative = (1 + returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak * 100
            plt.fill_between(drawdown.index, drawdown.values, 0,
                           alpha=0.3, color=colors[i % len(colors)], label=name)
            plt.plot(drawdown.index, drawdown.values,
                    color=colors[i % len(colors)], linewidth=1)

        plt.title('Strategy Drawdown Comparison', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt

    def plot_yearly_returns_heatmap(self, figsize=(14, 8)):
        yearly_df = self.get_yearly_returns()
        yearly_df = yearly_df * 100

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(yearly_df.values.T, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(range(len(yearly_df.index)))
        ax.set_yticks(range(len(yearly_df.columns)))
        ax.set_xticklabels([idx.strftime('%Y') for idx in yearly_df.index])
        ax.set_yticklabels(yearly_df.columns)

        for i in range(len(yearly_df.index)):
            for j in range(len(yearly_df.columns)):
                value = yearly_df.values[i, j]
                color = 'white' if abs(value) > 10 else 'black'
                ax.text(i, j, f'{value:.1f}%', ha='center', va='center',
                       color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label='Annual Return (%)')
        plt.title('Strategy Annual Returns Heatmap', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Strategy', fontsize=12)
        plt.tight_layout()
        return plt

    def generate_report(self, output_path='output/performance_report.csv'):
        metrics_df = self.evaluate_all()
        yearly_df = self.get_yearly_returns()

        yearly_df.to_csv(output_path.replace('.csv', '_yearly.csv'))

        with open(output_path.replace('.csv', '_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Hierarchical Risk Parity Strategy - Performance Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. Performance Metrics Summary\n")
            f.write("-" * 40 + "\n")
            for _, row in metrics_df.iterrows():
                f.write(f"\n{row['Strategy']}:\n")
                f.write(f"  Annual Return: {row['Annual Return']}\n")
                f.write(f"  Annual Volatility: {row['Annual Volatility']}\n")
                f.write(f"  Max Drawdown: {row['Max Drawdown']}\n")
                f.write(f"  Sharpe Ratio: {row['Sharpe Ratio']}\n")
                f.write(f"  Calmar Ratio: {row['Calmar Ratio']}\n")
                f.write(f"  Sortino Ratio: {row['Sortino Ratio']}\n")
                f.write(f"  Win Rate: {row['Win Rate']}\n")

            f.write("\n\n2. Annual Returns Detail\n")
            f.write("-" * 40 + "\n")
            f.write(yearly_df.to_string())

        print(f"\nPerformance report saved to: {output_path}")
        return metrics_df


def compare_with_benchmark(portfolio_returns, benchmark_returns):
    results = {}

    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    results['portfolio_annual_return'] = ((1 + portfolio_returns).prod()) ** (12 / len(portfolio_returns)) - 1
    results['benchmark_annual_return'] = ((1 + benchmark_returns).prod()) ** (12 / len(benchmark_returns)) - 1

    results['portfolio_volatility'] = portfolio_returns.std() * np.sqrt(12)
    results['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(12)

    portfolio_peak = portfolio_cumulative.expanding().max()
    portfolio_dd = (portfolio_cumulative - portfolio_peak) / portfolio_peak
    results['portfolio_max_drawdown'] = portfolio_dd.min()

    benchmark_peak = benchmark_cumulative.expanding().max()
    benchmark_dd = (benchmark_cumulative - benchmark_peak) / benchmark_peak
    results['benchmark_max_drawdown'] = benchmark_dd.min()

    results['portfolio_sharpe'] = results['portfolio_annual_return'] / results['portfolio_volatility']
    results['benchmark_sharpe'] = results['benchmark_annual_return'] / results['benchmark_volatility']

    return pd.DataFrame([results], index=['Metrics'])
