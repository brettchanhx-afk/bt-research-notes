import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .hrp_strategy import HierarchicalRiskParity, RiskParity
import warnings
warnings.filterwarnings('ignore')


class Backtest:
    def __init__(self, returns_df, transaction_cost=0.0005, rebalance_freq='monthly'):
        self.returns_df = returns_df.dropna()
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        self.results = {}
        self.weights_history = {}

    def _get_rebalance_dates(self):
        if self.rebalance_freq == 'monthly':
            return self.returns_df.resample('M').last().index
        elif self.rebalance_freq == 'quarterly':
            return self.returns_df.resample('QE').last().index
        else:
            return self.returns_df.resample('M').last().index

    def _calculate_turnover(self, weights_current, weights_previous):
        return np.sum(np.abs(weights_current - weights_previous))

    def run_strategy(self, strategy_name, strategy_func, **kwargs):
        print(f"\nRunning strategy: {strategy_name}")
        print("-" * 40)

        returns = self.returns_df.copy()
        rebalance_dates = self._get_rebalance_dates()

        if len(rebalance_dates) < 12:
            print(f"  Warning: Not enough data for backtest")
            return None

        weights_history = {}
        portfolio_returns = []
        dates = []

        window_size = 6
        prev_weights = None

        rebalance_dates = list(rebalance_dates)

        for i in range(window_size, len(rebalance_dates)):
            date = rebalance_dates[i]

            train_start = i - window_size
            train_data = returns.loc[rebalance_dates[train_start]:date].iloc[:-1]

            if len(train_data) < window_size or train_data.isna().all().any():
                continue

            try:
                train_clean = train_data.dropna(axis=1)
                if len(train_clean.columns) < 2:
                    continue
                weights = strategy_func(train_clean, **kwargs)
            except Exception as e:
                print(f"  Warning: {date.strftime('%Y-%m')} weight calc failed: {e}")
                weights = prev_weights if prev_weights is not None else np.ones(len(returns.columns)) / len(returns.columns)

            weights = np.array(weights)
            weights = np.nan_to_num(weights, nan=1.0/len(weights))
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()

            weights_df = pd.Series(weights, index=returns.columns)
            weights_history[date] = weights_df

            if prev_weights is not None:
                turnover = self._calculate_turnover(weights, prev_weights.values)
                cost = turnover * self.transaction_cost
            else:
                cost = 0

            prev_weights = weights_df

            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                period_returns = returns.loc[date:next_date].iloc[1:]
            else:
                period_returns = returns.loc[date:].iloc[1:]

            if len(period_returns) > 0 and not period_returns.isna().all().all():
                clean_mask = ~(period_returns.isna().any(axis=1) | weights_df.isna().any())
                clean_returns = period_returns[clean_mask]

                if len(clean_returns) > 0:
                    strat_ret = (clean_returns.values * weights_df.values).sum(axis=1)
                    strat_ret = strat_ret - cost
                    portfolio_returns.extend(strat_ret.tolist())
                    dates.extend(clean_returns.index.tolist())

        if len(portfolio_returns) == 0:
            print(f"  Warning: No return data generated!")
            return None

        portfolio_returns_series = pd.Series(portfolio_returns, index=dates)
        cumulative_returns = (1 + portfolio_returns_series).cumprod()

        self.results[strategy_name] = {
            'returns': portfolio_returns_series,
            'cumulative_returns': cumulative_returns,
            'weights_history': weights_history
        }
        self.weights_history[strategy_name] = weights_history

        final_return = (cumulative_returns.iloc[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0
        print(f"  Backtest complete! Cumulative return: {final_return:.2f}%")

        return portfolio_returns_series

    def run_all_strategies(self):
        def hrp_func(data, **kwargs):
            model = HierarchicalRiskParity(method='hrp')
            return list(model.fit(data).values())

        def naive_hrp_func(data, **kwargs):
            model = HierarchicalRiskParity(method='naive')
            return list(model.fit(data).values())

        def vol_hrp_func(data, **kwargs):
            model = HierarchicalRiskParity(method='volatility')
            return list(model.fit(data).values())

        def rp_func(data, **kwargs):
            model = RiskParity()
            return list(model.fit(data).values())

        self.run_strategy('HRP', hrp_func)
        self.run_strategy('Naive_HRP', naive_hrp_func)
        self.run_strategy('Vol_HRP', vol_hrp_func)
        self.run_strategy('RiskParity', rp_func)

        return self.results

    def get_results(self):
        return self.results

    def get_cumulative_returns(self):
        return {name: res['cumulative_returns'] for name, res in self.results.items()}

    def get_weights_history(self):
        return self.weights_history

    def plot_nav(self, figsize=(14, 8)):
        plt.figure(figsize=figsize)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (name, res) in enumerate(self.results.items()):
            nav = res['cumulative_returns']
            plt.plot(nav.index, nav.values, label=name, linewidth=2, color=colors[i % len(colors)])

        plt.title('Strategy NAV Comparison', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('NAV', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt

    def plot_weights_heatmap(self, strategy_name, date=None, figsize=(12, 6)):
        if strategy_name not in self.weights_history:
            print(f"Strategy {strategy_name} not found!")
            return None

        weights_df = pd.DataFrame(self.weights_history[strategy_name]).T

        if date is None:
            date = weights_df.index[-1]

        if date not in weights_df.index:
            date = weights_df.index[-1]

        weights = weights_df.loc[date]

        plt.figure(figsize=figsize)
        colors = ['#ff6b6b' if w < 0 else '#4ecdc4' for w in weights.values]
        plt.bar(range(len(weights)), weights.values, color=colors)
        plt.xticks(range(len(weights)), weights.index, rotation=45, ha='right')
        plt.title(f'{strategy_name} - Asset Weight Distribution ({date.strftime("%Y-%m")})', fontsize=14)
        plt.xlabel('Asset', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return plt

    def save_results(self, filepath='output/backtest_results.csv'):
        results_df = pd.DataFrame()
        for name, res in self.results.items():
            cumulative = res['cumulative_returns']
            temp_df = pd.DataFrame({
                f'{name}_NAV': cumulative,
                f'{name}_DailyReturn': res['returns']
            })
            results_df = pd.concat([results_df, temp_df], axis=1)

        results_df.to_csv(filepath)
        print(f"\nBacktest results saved to: {filepath}")
        return results_df


def run_backtest_from_data(price_data, transaction_cost=0.0005, rebalance_freq='monthly'):
    prices_df = pd.DataFrame(price_data)
    prices_df = prices_df.dropna()

    monthly_prices = prices_df.resample('M').last()
    returns = monthly_prices.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 12:
        raise ValueError("Not enough data for backtest")

    backtest = Backtest(returns, transaction_cost=transaction_cost, rebalance_freq=rebalance_freq)
    backtest.run_all_strategies()

    return backtest
