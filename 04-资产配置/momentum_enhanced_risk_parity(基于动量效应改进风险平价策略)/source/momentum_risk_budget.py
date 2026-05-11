import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.config import MOMENTUM_PARAMS, ASSETS


class MomentumRiskBudget:
    def __init__(self, k=1.0, lookback_l1=None, lookback_l2=None):
        self.k = k
        self.lookback_l1 = lookback_l1 if lookback_l1 else MOMENTUM_PARAMS['sharpe_lookback_l1']
        self.lookback_l2 = lookback_l2 if lookback_l2 else MOMENTUM_PARAMS['sharpe_lookback_l2']
        self.weights = None
        self.predicted_sharpe = None

    def predict_sharpe_ratio(self, returns_df, asset_name, current_date, lookback_days=None):
        if asset_name not in self.lookback_l1 or lookback_days is None:
            lookback_days = self.lookback_l1.get(asset_name, 60)

        lookback_l1_days = self.lookback_l1.get(asset_name, 60)
        lookback_l2_days = self.lookback_l2.get(asset_name, 20)

        end_date = current_date
        start_date_l1 = current_date - pd.Timedelta(days=lookback_l1_days * 2)
        start_date_l2 = current_date - pd.Timedelta(days=lookback_l2_days * 2)

        subset_l1 = returns_df.loc[start_date_l1:end_date, asset_name].dropna()
        subset_l2 = returns_df.loc[start_date_l2:end_date, asset_name].dropna()

        if len(subset_l1) < 20 or len(subset_l2) < 10:
            return 0.0

        return_mean = subset_l1.mean() * 252
        return_std = subset_l1.std() * np.sqrt(252)

        if return_std == 0:
            return 0.0

        sharpe = return_mean / return_std

        sharpe = np.clip(sharpe, -3.5, 3.5)

        return sharpe

    def calculate_risk_budget(self, predicted_sharpe_dict):
        budgets = {}
        sharpe_values = np.array(list(predicted_sharpe_dict.values()))
        sharpe_keys = list(predicted_sharpe_dict.keys())

        for i, key in enumerate(sharpe_keys):
            ir_i = sharpe_values[i]
            ek = np.exp(self.k * ir_i)
            b = (1 + np.log(ek + 1)) ** 2
            budgets[key] = b

        total = sum(budgets.values())
        for key in budgets:
            budgets[key] = budgets[key] / total

        return budgets

    def risk_parity_with_budget(self, cov_matrix, risk_budgets, asset_names):
        n_assets = len(asset_names)
        cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix

        def risk_contrib(weights):
            w = np.array(weights)
            port_var = np.dot(w.T, np.dot(cov, w))
            marginal = np.dot(cov, w)
            contrib = w * marginal / np.sqrt(port_var + 1e-10)
            return contrib

        def objective(weights):
            w = np.array(weights)
            rc = risk_contrib(w)
            target_rc = np.array([risk_budgets.get(name, 1/n_assets) for name in asset_names])
            return np.sum((rc - target_rc) ** 2)

        initial = np.ones(n_assets) / n_assets
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        result = minimize(objective, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-10})

        if result.success:
            return pd.Series(result.x, index=asset_names)
        else:
            print(f"Optimization warning: {result.message}")
            return pd.Series(initial, index=asset_names)

    def calculate_weights(self, returns_df, current_date, cov_lookback_days=126):
        asset_names = returns_df.columns.tolist()

        cov_start = current_date - pd.Timedelta(days=cov_lookback_days * 2)
        cov_subset = returns_df.loc[cov_start:current_date, asset_names].dropna()

        if len(cov_subset) < 30:
            cov_subset = returns_df[asset_names].dropna()
            if len(cov_subset) < 30:
                return pd.Series(1/len(asset_names), index=asset_names)

        cov_matrix = cov_subset.cov()

        predicted_sharpe = {}
        for asset in asset_names:
            predicted_sharpe[asset] = self.predict_sharpe_ratio(returns_df, asset, current_date)

        self.predicted_sharpe = predicted_sharpe

        risk_budgets = self.calculate_risk_budget(predicted_sharpe)

        weights = self.risk_parity_with_budget(cov_matrix, risk_budgets, asset_names)

        self.weights = weights
        return weights

    def get_weights(self):
        return self.weights

    def get_predicted_sharpe(self):
        return self.predicted_sharpe


class MomentumRiskBudgetStrategy:
    def __init__(self, k_values=None, lookback_l1=None, lookback_l2=None):
        self.k_values = k_values if k_values else MOMENTUM_PARAMS['k_values']
        self.lookback_l1 = lookback_l1 if lookback_l1 else MOMENTUM_PARAMS['sharpe_lookback_l1']
        self.lookback_l2 = lookback_l2 if lookback_l2 else MOMENTUM_PARAMS['sharpe_lookback_l2']
        self.strategies = {}
        self.history = {}

        for k in self.k_values:
            self.strategies[k] = MomentumRiskBudget(
                k=k,
                lookback_l1=self.lookback_l1,
                lookback_l2=self.lookback_l2
            )
            self.history[k] = {'weights': [], 'dates': [], 'sharpe': []}

    def calculate_all_weights(self, returns_df, current_date):
        for k, strategy in self.strategies.items():
            weights = strategy.calculate_weights(returns_df, current_date)
            self.history[k]['weights'].append(weights)
            self.history[k]['dates'].append(current_date)
            self.history[k]['sharpe'].append(strategy.get_predicted_sharpe())

    def get_weights_history(self, k):
        if k not in self.history:
            return None
        history = self.history[k]
        return pd.DataFrame(history['weights'], index=history['dates'])

    def get_all_results(self):
        results = {}
        for k, history in self.history.items():
            if len(history['weights']) > 0:
                weights_df = pd.DataFrame(history['weights'], index=history['dates'])
                results[k] = {
                    'weights': weights_df,
                    'sharpe_history': history['sharpe']
                }
        return results


class HierarchicalMomentumRiskBudget:
    def __init__(self, k=1.0, n_clusters=3):
        self.k = k
        self.n_clusters = n_clusters
        self.momentum_strategy = MomentumRiskBudget(k=k)
        self.cluster_labels = None

    def _cluster_assets(self, returns_df):
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        corr_matrix = returns_df.corr()
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        dist_condensed = squareform(dist_matrix.values, checks=False)

        linkage_matrix = linkage(dist_condensed, method='ward')
        self.cluster_labels = fcluster(linkage_matrix, t=self.n_clusters, criterion='maxclust')

        return self.cluster_labels

    def calculate_weights(self, returns_df, current_date, cov_lookback_days=126):
        asset_names = returns_df.columns.tolist()
        clusters = self._cluster_assets(returns_df)

        cov_start = current_date - pd.Timedelta(days=cov_lookback_days * 2)
        cov_subset = returns_df.loc[cov_start:current_date, asset_names].dropna()

        if len(cov_subset) < 30:
            cov_subset = returns_df[asset_names].dropna()

        cov_matrix = cov_subset.cov()

        predicted_sharpe = {}
        for asset in asset_names:
            predicted_sharpe[asset] = self.momentum_strategy.predict_sharpe_ratio(
                returns_df, asset, current_date
            )

        risk_budgets = self.momentum_strategy.calculate_risk_budget(predicted_sharpe)

        n_clusters = len(np.unique(clusters))
        cluster_budget = 1.0 / n_clusters

        final_weights = pd.Series(0.0, index=asset_names)

        for cluster_id in np.unique(clusters):
            cluster_idx = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_assets = [asset_names[i] for i in cluster_idx]
            cluster_budgets = {a: risk_budgets[a] * cluster_budget * n_clusters
                             for a in cluster_assets}

            cluster_returns = returns_df[cluster_assets].loc[cov_start:current_date].dropna()
            cluster_cov = cov_matrix.loc[cluster_assets, cluster_assets]

            if len(cluster_assets) == 1:
                cluster_weights = pd.Series({cluster_assets[0]: 1.0})
            else:
                cluster_weights = self.momentum_strategy.risk_parity_with_budget(
                    cluster_cov, cluster_budgets, cluster_assets
                )

            for asset in cluster_assets:
                final_weights[asset] = cluster_weights.get(asset, 0) * cluster_budget

        final_weights = final_weights / final_weights.sum()
        self.weights = final_weights

        return final_weights


def calculate_strategy_returns(returns_df, weights_df, transaction_cost=0.0005):
    n_periods = len(returns_df)
    portfolio_returns = []
    cumulative_returns = [1.0]
    current_weights = None

    for i in range(n_periods):
        period_return = returns_df.iloc[i].values

        if current_weights is not None:
            new_weights = weights_df.iloc[i].values
            turnover = np.sum(np.abs(new_weights - current_weights))
            tc = turnover * transaction_cost
            period_return = period_return - tc
            current_weights = new_weights
        else:
            current_weights = weights_df.iloc[i].values

        portfolio_return = np.dot(current_weights, period_return)
        portfolio_returns.append(portfolio_return)
        cumulative_returns.append(cumulative_returns[-1] * (1 + portfolio_return))

    return np.array(portfolio_returns), np.array(cumulative_returns[1:])


def calculate_performance_metrics(returns, periods_per_year=12):
    returns_series = pd.Series(returns)

    cumulative_returns = (1 + returns_series).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

    annualized_return = (1 + total_return) ** (periods_per_year / max(len(returns_series), 1)) - 1

    annualized_volatility = returns_series.std() * np.sqrt(periods_per_year)

    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'total_return': total_return,
        'cumulative_returns': cumulative_returns
    }

    return metrics


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)

    returns_df = pd.DataFrame(
        np.random.randn(n, 5) * 0.015,
        index=dates,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    )

    print("Testing Momentum Risk Budget Strategy...")

    strategy = MomentumRiskBudget(k=1.0)
    current_date = dates[-1]
    weights = strategy.calculate_weights(returns_df, current_date)
    print(f"Weights:\n{weights}")
    print(f"Predicted Sharpe:\n{strategy.get_predicted_sharpe()}")

    print("\nTesting multiple k values...")
    multi_strategy = MomentumRiskBudgetStrategy(k_values=[0.5, 1.0, 1.5])

    for date in dates[-5:]:
        multi_strategy.calculate_all_weights(returns_df, date)

    results = multi_strategy.get_all_results()
    for k, result in results.items():
        print(f"\nk={k}: {len(result['weights'])} weight records")
