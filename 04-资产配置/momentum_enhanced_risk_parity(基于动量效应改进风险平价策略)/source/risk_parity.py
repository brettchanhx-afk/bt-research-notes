import numpy as np
import pandas as pd
from scipy.optimize import minimize

class RiskParity:
    def __init__(self, lookback_days=126):
        self.lookback_days = lookback_days
        self.weights = None
        self.cov_matrix = None

    def calculate_covariance(self, returns_df):
        return returns_df.cov()

    def risk_contribution(self, weights, cov_matrix):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_risk = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_risk / np.sqrt(portfolio_variance)
        return risk_contrib

    def risk_parity_objective(self, weights, cov_matrix, target_risk):
        current_risk = self.risk_contribution(weights, cov_matrix)
        return np.sum((current_risk - target_risk) ** 2)

    def calculate_weights(self, returns_df, asset_names=None):
        n_assets = returns_df.shape[1]
        if asset_names is None:
            asset_names = returns_df.columns.tolist()

        self.cov_matrix = self.calculate_covariance(returns_df)
        cov = self.cov_matrix.values

        initial_weights = np.ones(n_assets) / n_assets

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(
            self.risk_parity_objective,
            initial_weights,
            args=(cov, np.ones(n_assets) / n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if result.success:
            self.weights = pd.Series(result.x, index=asset_names)
            return self.weights
        else:
            print(f"Optimization failed: {result.message}")
            return pd.Series(initial_weights, index=asset_names)

    def calculate_portfolio_return(self, weights, next_returns):
        return np.dot(weights, next_returns)

    def get_weights(self):
        return self.weights


class HierarchicalRiskParity:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.weights = None
        self.cluster_labels = None
        self.linkage_matrix = None

    def _compute_correlation_distance(self, returns_df):
        corr_matrix = returns_df.corr()
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        return dist_matrix.values

    def _cluster_assets(self, returns_df, method='ward'):
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        dist_matrix = self._compute_correlation_distance(returns_df)
        dist_condensed = squareform(dist_matrix, checks=False)

        self.linkage_matrix = linkage(dist_condensed, method=method)
        self.cluster_labels = fcluster(self.linkage_matrix, t=self.n_clusters, criterion='maxclust')

        return self.cluster_labels

    def _get_cluster_covariance(self, returns_df, cov_matrix, clusters):
        cluster_cov = {}
        for cluster_id in np.unique(clusters):
            cluster_idx = np.where(clusters == cluster_id)[0]
            cluster_cov[cluster_id] = cov_matrix[np.ix_(cluster_idx, cluster_idx)]
        return cluster_cov

    def _allocate_risk_to_clusters(self, total_risk, cov_matrix, clusters):
        n_clusters = len(np.unique(clusters))
        risk_allocation = total_risk / n_clusters
        return {i: risk_allocation for i in range(1, n_clusters + 1)}

    def _within_cluster_allocation(self, cluster_returns, cluster_cov, target_risk):
        n_assets = len(cluster_returns)
        if n_assets == 1:
            return np.array([1.0])

        def risk_contrib(weights):
            portfolio_var = np.dot(weights.T, np.dot(cluster_cov, weights))
            marginal = np.dot(cluster_cov, weights)
            contrib = weights * marginal / np.sqrt(portfolio_var + 1e-10)
            return contrib

        def objective(w):
            rc = risk_contrib(w)
            return np.sum((rc - target_risk / n_assets) ** 2)

        initial = np.ones(n_assets) / n_assets
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        result = minimize(objective, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500})

        return result.x if result.success else initial

    def calculate_weights(self, returns_df, asset_names=None):
        if asset_names is None:
            asset_names = returns_df.columns.tolist()

        cov_matrix = returns_df.cov().values
        clusters = self._cluster_assets(returns_df)

        n_assets = returns_df.shape[1]
        total_risk = np.sum(np.sqrt(np.diag(cov_matrix)))
        target_risk = total_risk

        cluster_risk_budget = self._allocate_risk_to_clusters(target_risk, cov_matrix, clusters)

        final_weights = np.zeros(n_assets)

        for cluster_id in np.unique(clusters):
            cluster_idx = np.where(clusters == cluster_id)[0]
            cluster_returns = returns_df.iloc[:, cluster_idx]
            cluster_cov = cov_matrix[np.ix_(cluster_idx, cluster_idx)]

            cluster_target = cluster_risk_budget[cluster_id]
            cluster_weights = self._within_cluster_allocation(
                cluster_returns, cluster_cov, cluster_target
            )

            final_weights[cluster_idx] = cluster_weights

        self.weights = pd.Series(final_weights, index=asset_names)
        self.weights = self.weights / self.weights.sum()

        return self.weights

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_weights(self):
        return self.weights


def calculate_portfolio_metrics(returns, weights, periods_per_year=12):
    returns_series = pd.Series(returns)

    cumulative_returns = (1 + returns_series).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

    annualized_return = (1 + total_return) ** (periods_per_year / len(returns_series)) - 1 if len(returns_series) > 0 else 0

    annualized_volatility = returns_series.std() * np.sqrt(periods_per_year)

    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'cumulative_returns': cumulative_returns
    }

    return metrics


def calculate_period_metrics(returns_df, weights_df, transaction_cost=0.0005):
    n_periods = len(returns_df)
    portfolio_returns = []
    cumulative_returns = [1.0]
    current_weights = None

    for i in range(n_periods):
        period_return = returns_df.iloc[i].values

        if current_weights is not None:
            new_weights = weights_df.iloc[i].values
            turnover = np.sum(np.abs(new_weights - current_weights))
            tc = turnover * transaction_cost / 2
            period_return = period_return - tc
            current_weights = new_weights
        else:
            current_weights = weights_df.iloc[i].values

        portfolio_return = np.dot(current_weights, period_return)
        portfolio_returns.append(portfolio_return)
        cumulative_returns.append(cumulative_returns[-1] * (1 + portfolio_return))

    portfolio_returns = np.array(portfolio_returns)
    cumulative_returns = np.array(cumulative_returns[1:])

    return portfolio_returns, cumulative_returns


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
    n = len(dates)

    returns_df = pd.DataFrame(
        np.random.randn(n, 5) * 0.02,
        index=dates,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    )

    print("Testing Risk Parity...")
    rp = RiskParity(lookback_days=126)
    weights = rp.calculate_weights(returns_df)
    print(f"Risk Parity Weights:\n{weights}")

    risk_contrib = rp.risk_contribution(weights.values, rp.cov_matrix.values)
    print(f"\nRisk Contributions:\n{risk_contrib}")

    print("\nTesting Hierarchical Risk Parity...")
    hrp = HierarchicalRiskParity(n_clusters=2)
    hrp_weights = hrp.calculate_weights(returns_df)
    print(f"HRP Weights:\n{hrp_weights}")
    print(f"Cluster Labels: {hrp.get_cluster_labels()}")
