import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.risk_parity import RiskParity
from source.momentum_risk_budget import MomentumRiskBudget, MomentumRiskBudgetStrategy
from source.hierarchical_risk_parity import HierarchicalRiskParity, HierarchicalMomentumRiskBudget, HierarchicalMomentumSumBudget
from source.config import (
    RISK_PARITY_PARAMS, MOMENTUM_PARAMS, BACKTEST_PARAMS, ASSETS
)


class Backtest:
    def __init__(self, returns_df, transaction_cost=0.0005):
        self.returns_df = returns_df
        self.transaction_cost = transaction_cost
        self.weights_history = {}
        self.portfolio_returns = {}
        self.metrics = {}

    def backtest_risk_parity(self, lookback_days=126, name='RiskParity'):
        daily_returns = self.returns_df.copy()
        rebalance_dates = self._get_rebalance_dates('monthly')

        rp = RiskParity(lookback_days=lookback_days)

        weights_list = []
        portfolio_ret_list = []
        current_weights = None
        dates_list = []

        for i, date in enumerate(daily_returns.index):
            if date not in rebalance_dates and current_weights is not None:
                ret = daily_returns.loc[date].values
                portfolio_ret = np.dot(current_weights, ret)
                portfolio_ret_list.append(portfolio_ret)
                weights_list.append(current_weights)
                dates_list.append(date)
                continue

            lookback_end = date
            lookback_start = lookback_end - pd.Timedelta(days=lookback_days * 2)

            subset = daily_returns.loc[lookback_start:lookback_end].dropna()

            if len(subset) < 30:
                if current_weights is not None:
                    portfolio_ret = np.dot(current_weights, daily_returns.loc[date].values)
                    portfolio_ret_list.append(portfolio_ret)
                    weights_list.append(current_weights)
                    dates_list.append(date)
                continue

            weights = rp.calculate_weights(subset)

            if current_weights is not None:
                turnover = np.sum(np.abs(weights.values - current_weights))
                tc = turnover * self.transaction_cost
                ret = daily_returns.loc[date].values - tc
                portfolio_ret = np.dot(weights.values, ret)
            else:
                portfolio_ret = np.dot(weights.values, daily_returns.loc[date].values)

            current_weights = weights.values

            portfolio_ret_list.append(portfolio_ret)
            weights_list.append(weights.values)
            dates_list.append(date)

        portfolio_returns = pd.Series(portfolio_ret_list, index=dates_list)
        weights_df = pd.DataFrame(weights_list, index=dates_list, columns=self.returns_df.columns)

        self.weights_history[name] = weights_df
        self.portfolio_returns[name] = portfolio_returns
        self.metrics[name] = self._calculate_metrics(portfolio_returns)

        return portfolio_returns, weights_df

    def backtest_momentum_risk_budget(self, k=1.0, lookback_days=126, name=None):
        if name is None:
            name = f'MomentumRiskBudget_k{k}'

        daily_returns = self.returns_df.copy()
        rebalance_dates = self._get_rebalance_dates('monthly')

        mrb = MomentumRiskBudget(k=k)

        weights_list = []
        portfolio_ret_list = []
        current_weights = None
        dates_list = []

        for i, date in enumerate(daily_returns.index):
            if date not in rebalance_dates and current_weights is not None:
                ret = daily_returns.loc[date].values
                portfolio_ret = np.dot(current_weights, ret)
                portfolio_ret_list.append(portfolio_ret)
                weights_list.append(current_weights)
                dates_list.append(date)
                continue

            try:
                weights = mrb.calculate_weights(daily_returns, date, cov_lookback_days=lookback_days)
            except Exception as e:
                if current_weights is not None:
                    portfolio_ret = np.dot(current_weights, daily_returns.loc[date].values)
                    portfolio_ret_list.append(portfolio_ret)
                    weights_list.append(current_weights)
                    dates_list.append(date)
                continue

            if current_weights is not None:
                turnover = np.sum(np.abs(weights.values - current_weights))
                tc = turnover * self.transaction_cost
                ret = daily_returns.loc[date].values - tc
                portfolio_ret = np.dot(weights.values, ret)
            else:
                portfolio_ret = np.dot(weights.values, daily_returns.loc[date].values)

            current_weights = weights.values

            portfolio_ret_list.append(portfolio_ret)
            weights_list.append(weights.values)
            dates_list.append(date)

        portfolio_returns = pd.Series(portfolio_ret_list, index=dates_list)
        weights_df = pd.DataFrame(weights_list, index=dates_list, columns=self.returns_df.columns)

        self.weights_history[name] = weights_df
        self.portfolio_returns[name] = portfolio_returns
        self.metrics[name] = self._calculate_metrics(portfolio_returns)

        return portfolio_returns, weights_df

    def backtest_hierarchical_momentum(self, k=1.0, n_clusters=3, method='avg',
                                       lookback_days=126, name=None):
        if name is None:
            name = f'HierarchicalMomentum_{method}_k{k}'

        daily_returns = self.returns_df.copy()
        rebalance_dates = self._get_rebalance_dates('monthly')

        if method == 'avg':
            hmrb = HierarchicalMomentumRiskBudget(k=k, n_clusters=n_clusters)
        else:
            hmrb = HierarchicalMomentumSumBudget(k=k, n_clusters=n_clusters)

        weights_list = []
        portfolio_ret_list = []
        current_weights = None
        dates_list = []

        for i, date in enumerate(daily_returns.index):
            if date not in rebalance_dates and current_weights is not None:
                ret = daily_returns.loc[date].values
                portfolio_ret = np.dot(current_weights, ret)
                portfolio_ret_list.append(portfolio_ret)
                weights_list.append(current_weights)
                dates_list.append(date)
                continue

            predicted_sharpe = {}
            for col in daily_returns.columns:
                predicted_sharpe[col] = np.random.randn() * 0.5

            try:
                weights = hmrb.calculate_weights(
                    daily_returns, predicted_sharpe, date,
                    cov_lookback_days=lookback_days
                )
            except Exception as e:
                if current_weights is not None:
                    portfolio_ret = np.dot(current_weights, daily_returns.loc[date].values)
                    portfolio_ret_list.append(portfolio_ret)
                    weights_list.append(current_weights)
                    dates_list.append(date)
                continue

            if current_weights is not None:
                turnover = np.sum(np.abs(weights.values - current_weights))
                tc = turnover * self.transaction_cost
                ret = daily_returns.loc[date].values - tc
                portfolio_ret = np.dot(weights.values, ret)
            else:
                portfolio_ret = np.dot(weights.values, daily_returns.loc[date].values)

            current_weights = weights.values

            portfolio_ret_list.append(portfolio_ret)
            weights_list.append(weights.values)
            dates_list.append(date)

        portfolio_returns = pd.Series(portfolio_ret_list, index=dates_list)
        weights_df = pd.DataFrame(weights_list, index=dates_list, columns=self.returns_df.columns)

        self.weights_history[name] = weights_df
        self.portfolio_returns[name] = portfolio_returns
        self.metrics[name] = self._calculate_metrics(portfolio_returns)

        return portfolio_returns, weights_df

    def run_all_strategies(self):
        print("Running Risk Parity backtest...")
        self.backtest_risk_parity(name='RiskParity')

        for k in MOMENTUM_PARAMS['k_values']:
            print(f"Running Momentum Risk Budget (k={k}) backtest...")
            self.backtest_momentum_risk_budget(k=k)

        print("Running Hierarchical Momentum (avg) backtest...")
        self.backtest_hierarchical_momentum(k=1.0, n_clusters=3, method='avg',
                                           name='HierarchicalMomentum_avg_k1')

        print("Running Hierarchical Momentum (sum) backtest...")
        self.backtest_hierarchical_momentum(k=1.0, n_clusters=3, method='sum',
                                           name='HierarchicalMomentum_sum_k1')

    def _get_rebalance_dates(self, frequency='monthly'):
        if frequency == 'monthly':
            return self.returns_df.resample('M').last().index
        elif frequency == 'weekly':
            return self.returns_df.resample('W').last().index
        else:
            return self.returns_df.index

    def _calculate_metrics(self, returns_series, periods_per_year=252):
        if len(returns_series) == 0:
            return {}

        cumulative_returns = (1 + returns_series).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        n_years = len(returns_series) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        annualized_volatility = returns_series.std() * np.sqrt(periods_per_year)

        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        positive_days = (returns_series > 0).sum()
        total_days = len(returns_series)
        win_rate = positive_days / total_days if total_days > 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'cumulative_returns': cumulative_returns,
            'returns': returns_series
        }

        return metrics

    def get_metrics_summary(self):
        summary = []
        for name, metrics in self.metrics.items():
            summary.append({
                'Strategy': name,
                'Annualized Return': f"{metrics['annualized_return']:.2%}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Annualized Volatility': f"{metrics['annualized_volatility']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Calmar Ratio': f"{metrics['calmar_ratio']:.2f}",
                'Win Rate': f"{metrics['win_rate']:.2%}"
            })
        return pd.DataFrame(summary)

    def get_all_returns(self):
        returns_df = pd.DataFrame(self.portfolio_returns)
        return returns_df

    def get_all_weights(self):
        return self.weights_history


def run_full_backtest(daily_returns_df, start_date=None, end_date=None):
    bt = Backtest(daily_returns_df, transaction_cost=RISK_PARITY_PARAMS['transaction_cost'])

    if start_date:
        daily_returns_df = daily_returns_df[daily_returns_df.index >= start_date]
    if end_date:
        daily_returns_df = daily_returns_df[daily_returns_df.index <= end_date]

    bt.run_all_strategies()

    return bt


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)

    returns_df = pd.DataFrame(
        np.random.randn(n, 5) * 0.015,
        index=dates,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    )

    print("Running backtest on sample data...")
    bt = Backtest(returns_df)

    bt.backtest_risk_parity()
    bt.backtest_momentum_risk_budget(k=1.0)

    print("\nBacktest Results:")
    print(bt.get_metrics_summary())
