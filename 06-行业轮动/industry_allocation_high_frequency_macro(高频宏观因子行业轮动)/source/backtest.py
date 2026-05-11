import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from source.allocation import DavisDoubleHitStrategy, MacroRiskAllocator

class BacktestEngine:
    def __init__(self, initial_capital=1000000, rebalance_freq='monthly'):
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.portfolio_value = initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []

    def calculate_returns(self, prices, weights):
        if len(prices) < 2:
            return 0
        returns = (prices.iloc[-1] / prices.iloc[0]) - 1
        return returns

    def rebalance(self, target_weights, current_prices):
        total_value = self.portfolio_value
        new_positions = {}

        for asset, weight in target_weights.items():
            if asset in current_prices.index:
                new_positions[asset] = total_value * weight / current_prices[asset]

        self.positions = new_positions
        return new_positions

    def update_portfolio_value(self, current_prices):
        total_value = 0
        for asset, shares in self.positions.items():
            if asset in current_prices.index:
                total_value += shares * current_prices[asset]
        self.portfolio_value = total_value
        return total_value

    def run_backtest(self, prices_df, weights_df, benchmark_prices=None):
        if prices_df.empty or weights_df.empty:
            print("Error: Empty price or weights data")
            return None

        common_dates = prices_df.index.intersection(weights_df.index)
        if len(common_dates) == 0:
            print("Error: No common dates between prices and weights")
            return None

        prices_aligned = prices_df.loc[common_dates]
        weights_aligned = weights_df.loc[common_dates]

        portfolio_values = [self.initial_capital]
        dates = [common_dates[0]]

        for i in range(1, len(common_dates)):
            date = common_dates[i]
            prev_date = common_dates[i-1]

            date_prices = prices_aligned.loc[:date].iloc[-1]
            prev_prices = prices_aligned.loc[:prev_date].iloc[-1]

            if self.rebalance_freq == 'monthly':
                if date.month != prev_date.month:
                    self.positions = {}
                    for col in weights_aligned.columns:
                        if not pd.isna(weights_aligned.loc[date, col]):
                            self.positions[col] = self.portfolio_value * weights_aligned.loc[date, col] / date_prices[col]

            if self.positions:
                new_value = sum(
                    self.positions.get(col, 0) * date_prices[col]
                    for col in self.positions.keys()
                    if col in date_prices.index
                )
            else:
                new_value = portfolio_values[-1] * (1 + (date_prices.mean() / prev_prices.mean() - 1))

            portfolio_values.append(new_value)
            dates.append(date)

        results = pd.DataFrame({
            'portfolio_value': portfolio_values
        }, index=dates)

        results['returns'] = results['portfolio_value'].pct_change()
        results['cumulative_returns'] = (1 + results['returns']).cumprod() - 1

        if benchmark_prices is not None:
            benchmark_aligned = benchmark_prices.loc[common_dates]
            benchmark_values = (benchmark_aligned / benchmark_aligned.iloc[0]) * self.initial_capital
            results['benchmark_value'] = benchmark_values.values
            results['benchmark_returns'] = results['benchmark_value'].pct_change()
            results['excess_returns'] = results['returns'] - results['benchmark_returns']
            results['cumulative_excess'] = (1 + results['excess_returns']).cumprod() - 1

        return results

    def calculate_metrics(self, backtest_results):
        if backtest_results is None or backtest_results.empty:
            return None

        returns = backtest_results['returns'].dropna()
        cum_returns = backtest_results['cumulative_returns'].iloc[-1]

        annual_return = (1 + cum_returns) ** (52 / len(returns)) - 1 if len(returns) > 0 else 0
        annual_volatility = returns.std() * np.sqrt(52)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        cumulative_returns_series = 1 + backtest_results['returns'].dropna()
        running_max = cumulative_returns_series.cummax()
        drawdown = (cumulative_returns_series - running_max) / running_max
        max_drawdown = drawdown.min()

        if 'benchmark_returns' in backtest_results.columns:
            tracking_error = backtest_results['excess_returns'].std() * np.sqrt(52)
            information_ratio = (annual_return - (1 + backtest_results['cumulative_returns'].iloc[-1] if 'benchmark_returns' in backtest_results.columns else annual_return)) / tracking_error if tracking_error > 0 else 0
        else:
            tracking_error = 0
            information_ratio = 0

        metrics = {
            'total_return': cum_returns,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }

        return metrics

    def plot_results(self, backtest_results, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if backtest_results is not None and not backtest_results.empty:
            backtest_results['portfolio_value'].plot(ax=axes[0, 0], label='Portfolio')
            if 'benchmark_value' in backtest_results.columns:
                backtest_results['benchmark_value'].plot(ax=axes[0, 0], label='Benchmark')
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].legend()

            backtest_results['cumulative_returns'].plot(ax=axes[0, 1], label='Portfolio')
            if 'cumulative_excess' in backtest_results.columns:
                backtest_results['cumulative_excess'].plot(ax=axes[0, 1], label='Excess Returns')
            axes[0, 1].set_title('Cumulative Returns')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Return')
            axes[0, 1].legend()

            cumulative_returns_series = 1 + backtest_results['returns'].dropna()
            running_max = cumulative_returns_series.cummax()
            drawdown = (cumulative_returns_series - running_max) / running_max
            drawdown.plot(ax=axes[1, 0], color='red')
            axes[1, 0].set_title('Drawdown')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Drawdown')

            returns = backtest_results['returns'].dropna()
            axes[1, 1].hist(returns, bins=50, alpha=0.75, edgecolor='black')
            axes[1, 1].set_title('Return Distribution')
            axes[1, 1].set_xlabel('Return')
            axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")

        return fig

class DavisDoubleHitBacktest(BacktestEngine):
    def __init__(self, initial_capital=1000000, rebalance_freq='quarterly'):
        super().__init__(initial_capital, rebalance_freq)
        self.strategy = DavisDoubleHitStrategy()

    def prepare_data(self, factor_returns, industry_returns):
        factor_returns_weekly = factor_returns.resample('W-FRI').last()
        industry_returns_weekly = industry_returns.resample('W-FRI').last()

        return factor_returns_weekly, industry_returns_weekly

    def run_strategy_backtest(self, factor_returns, industry_returns, top_n=10):
        factor_weekly, industry_weekly = self.prepare_data(factor_returns, industry_returns)

        if factor_weekly.empty or industry_weekly.empty:
            print("Error: Empty data after preprocessing")
            return None

        selection_history = []
        weights_history = []

        for i in range(self.strategy.lookback_window, len(factor_weekly)):
            window_factors = factor_weekly.iloc[:i]
            current_factors = factor_weekly.iloc[i]

            delta_g_pred = self.strategy.fit_macro_delta_g_mapping(
                window_factors,
                industry_weekly.iloc[:i].apply(lambda x: x.diff(4).dropna())
            )

            delta_pb_pred = self.strategy.fit_macro_delta_pb_mapping(
                window_factors,
                np.log(industry_weekly.iloc[:i] / industry_weekly.iloc[:i].shift(4))
            )

            if delta_g_pred is not None and delta_pb_pred is not None:
                try:
                    composite = self.strategy.calculate_composite_factor(delta_g_pred, delta_pb_pred)
                    top_industries = self.strategy.select_top_industries(composite, top_n)

                    selection_history.append({
                        'date': factor_weekly.index[i],
                        'selected_industries': top_industries
                    })

                    equal_weight = 1.0 / len(top_industries) if len(top_industries) > 0 else 0
                    weights = {ind: equal_weight for ind in top_industries}
                    weights_history.append({
                        'date': factor_weekly.index[i],
                        'weights': weights
                    })
                except Exception as e:
                    pass

        if not selection_history:
            print("Error: No valid selections made during backtest")
            return None

        selection_df = pd.DataFrame(selection_history).set_index('date')
        weights_df = pd.DataFrame([w['weights'] for w in weights_history], index=[w['date'] for w in weights_history])

        backtest_results = self.run_backtest(industry_weekly, weights_df)

        return {
            'results': backtest_results,
            'selection_history': selection_df,
            'weights_history': weights_df
        }

def run_full_backtest(config):
    from source.data_fetcher import DataFetcher
    from source.factors import HighFrequencyMacroFactors

    fetcher = DataFetcher()
    factor_calculator = HighFrequencyMacroFactors(fetcher)
    backtest_engine = DavisDoubleHitBacktest()

    print("Loading data...")
    industry_data = fetcher.load_data("industry_indices")

    print("Running backtest...")
    results = backtest_engine.run_strategy_backtest(
        factor_returns=pd.DataFrame(),
        industry_returns=industry_data.set_index('trade_date')['close'] if 'trade_date' in industry_data.columns else industry_data['close']
    )

    if results and results['results'] is not None:
        metrics = backtest_engine.calculate_metrics(results['results'])
        print("\n=== Backtest Results ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

        output_dir = config.get('OUTPUT_DIR', 'output')
        os.makedirs(output_dir, exist_ok=True)
        backtest_engine.plot_results(results['results'], save_path=os.path.join(output_dir, 'backtest_results.png'))

        results['results'].to_csv(os.path.join(output_dir, 'backtest_values.csv'))

        return results

    return None

if __name__ == "__main__":
    print("Backtest Module initialized successfully!")
