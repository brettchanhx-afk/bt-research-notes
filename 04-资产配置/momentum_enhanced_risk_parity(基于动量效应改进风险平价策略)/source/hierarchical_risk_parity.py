import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize


class HierarchicalRiskParity:
    def __init__(self, n_clusters=3, method='ward'):
        self.n_clusters = n_clusters
        self.method = method
        self.cluster_labels = None
        self.linkage_matrix = None
        self.weights = None

    def _compute_distance_matrix(self, returns_df):
        corr_matrix = returns_df.corr()
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        return dist_matrix.values

    def _cluster_assets(self, returns_df):
        dist_matrix = self._compute_distance_matrix(returns_df)
        dist_condensed = squareform(dist_matrix, checks=False)

        self.linkage_matrix = linkage(dist_condensed, method=self.method)
        self.cluster_labels = fcluster(self.linkage_matrix, t=self.n_clusters, criterion='maxclust')

        return self.cluster_labels

    def _within_cluster_risk_parity(self, cluster_returns, cluster_cov):
        n_assets = len(cluster_returns)
        if n_assets == 1:
            return np.array([1.0])

        def risk_contribution(weights):
            w = np.array(weights)
            port_var = np.dot(w.T, np.dot(cluster_cov, w))
            marginal = np.dot(cluster_cov, w)
            contrib = w * marginal / (np.sqrt(port_var) + 1e-10)
            return contrib

        def objective(weights):
            w = np.array(weights)
            rc = risk_contribution(w)
            target_rc = np.ones(n_assets) / n_assets
            return np.sum((rc - target_rc) ** 2)

        initial = np.ones(n_assets) / n_assets
        bounds = tuple((0.001, 1) for _ in range(n_assets))
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

        n_clusters = len(np.unique(clusters))
        cluster_risk_budget = 1.0 / n_clusters

        final_weights = np.zeros(len(asset_names))

        for cluster_id in np.unique(clusters):
            cluster_idx = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_assets = [asset_names[i] for i in cluster_idx]

            cluster_returns = returns_df.iloc[:, cluster_idx].dropna()
            cluster_cov = cov_matrix[np.ix_(cluster_idx, cluster_idx)]

            cluster_weights = self._within_cluster_risk_parity(cluster_returns, cluster_cov)

            for i, idx in enumerate(cluster_idx):
                final_weights[idx] = cluster_weights[i] * cluster_risk_budget

        final_weights = final_weights / final_weights.sum()
        self.weights = pd.Series(final_weights, index=asset_names)

        return self.weights

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_weights(self):
        return self.weights


class HierarchicalMomentumRiskBudget:
    def __init__(self, k=1.0, n_clusters=3):
        self.k = k
        self.n_clusters = n_clusters
        self.cluster_labels = None
        self.weights = None

    def _compute_distance_matrix(self, returns_df):
        corr_matrix = returns_df.corr()
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        return dist_matrix.values

    def _cluster_assets(self, returns_df):
        dist_matrix = self._compute_distance_matrix(returns_df)
        dist_condensed = squareform(dist_matrix, checks=False)

        linkage_matrix = linkage(dist_condensed, method='ward')
        self.cluster_labels = fcluster(linkage_matrix, t=self.n_clusters, criterion='maxclust')

        return self.cluster_labels

    def _calculate_budget_from_sharpe(self, predicted_sharpe, k):
        budgets = {}
        sharpe_values = np.array(list(predicted_sharpe.values()))
        sharpe_keys = list(predicted_sharpe.keys())

        for i, key in enumerate(sharpe_keys):
            ir_i = sharpe_values[i]
            ek = np.exp(k * ir_i)
            b = (1 + np.log(ek + 1)) ** 2
            budgets[key] = b

        total = sum(budgets.values())
        for key in budgets:
            budgets[key] = budgets[key] / total

        return budgets

    def _within_cluster_momentum_budget(self, cluster_returns, cluster_cov, cluster_budgets, k):
        n_assets = len(cluster_returns.columns)
        asset_names = cluster_returns.columns.tolist()

        if n_assets == 1:
            return pd.Series({asset_names[0]: 1.0})

        def risk_contribution(weights):
            w = np.array(weights)
            port_var = np.dot(w.T, np.dot(cluster_cov, w))
            marginal = np.dot(cluster_cov, w)
            contrib = w * marginal / (np.sqrt(port_var) + 1e-10)
            return contrib

        def objective(weights):
            w = np.array(weights)
            rc = risk_contribution(w)
            target_rc = np.array([cluster_budgets.get(name, 1/n_assets) for name in asset_names])
            return np.sum((rc - target_rc) ** 2)

        initial = np.ones(n_assets) / n_assets
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        result = minimize(objective, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500})

        if result.success:
            return pd.Series(result.x, index=asset_names)
        return pd.Series(initial, index=asset_names)

    def calculate_weights(self, returns_df, predicted_sharpe, current_date,
                         cov_lookback_days=126, asset_names=None):
        if asset_names is None:
            asset_names = returns_df.columns.tolist()

        clusters = self._cluster_assets(returns_df)

        cov_start = current_date - pd.Timedelta(days=cov_lookback_days * 2)
        cov_subset = returns_df.loc[cov_start:current_date, asset_names].dropna()

        if len(cov_subset) < 30:
            cov_subset = returns_df[asset_names].dropna()

        cov_matrix = cov_subset.cov()

        global_budgets = self._calculate_budget_from_sharpe(predicted_sharpe, self.k)

        n_clusters = len(np.unique(clusters))
        cluster_weight = 1.0 / n_clusters

        final_weights = pd.Series(0.0, index=asset_names)

        for cluster_id in np.unique(clusters):
            cluster_idx = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_assets = [asset_names[i] for i in cluster_idx]

            cluster_budgets = {a: global_budgets[a] * cluster_weight * n_clusters
                             for a in cluster_assets}

            cluster_returns = returns_df[cluster_assets].loc[cov_start:current_date].dropna()
            cluster_cov = cov_matrix.loc[cluster_assets, cluster_assets]

            cluster_weights = self._within_cluster_momentum_budget(
                cluster_returns, cluster_cov.values, cluster_budgets, self.k
            )

            for asset in cluster_assets:
                final_weights[asset] = cluster_weights.get(asset, 0) * cluster_weight

        final_weights = final_weights / final_weights.sum()
        self.weights = final_weights

        return final_weights

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_weights(self):
        return self.weights


class HierarchicalMomentumSumBudget(HierarchicalMomentumRiskBudget):
    def calculate_weights(self, returns_df, predicted_sharpe, current_date,
                         cov_lookback_days=126, asset_names=None):
        if asset_names is None:
            asset_names = returns_df.columns.tolist()

        clusters = self._cluster_assets(returns_df)

        cov_start = current_date - pd.Timedelta(days=cov_lookback_days * 2)
        cov_subset = returns_df.loc[cov_start:current_date, asset_names].dropna()

        if len(cov_subset) < 30:
            cov_subset = returns_df[asset_names].dropna()

        cov_matrix = cov_subset.cov()
        global_budgets = self._calculate_budget_from_sharpe(predicted_sharpe, self.k)

        n_clusters = len(np.unique(clusters))

        final_weights = pd.Series(0.0, index=asset_names)

        for cluster_id in np.unique(clusters):
            cluster_idx = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_assets = [asset_names[i] for i in cluster_idx]

            cluster_budget_sum = sum(global_budgets.get(a, 0) for a in cluster_assets)
            cluster_budgets = {a: global_budgets.get(a, 0) / cluster_budget_sum
                             for a in cluster_assets if cluster_budget_sum > 0}

            cluster_returns = returns_df[cluster_assets].loc[cov_start:current_date].dropna()
            cluster_cov = cov_matrix.loc[cluster_assets, cluster_assets]

            cluster_weights = self._within_cluster_momentum_budget(
                cluster_returns, cluster_cov.values, cluster_budgets, self.k
            )

            for asset in cluster_assets:
                final_weights[asset] = cluster_weights.get(asset, 0)

        final_weights = final_weights / final_weights.sum()
        self.weights = final_weights

        return final_weights


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)

    returns_df = pd.DataFrame(
        np.random.randn(n, 5) * 0.015,
        index=dates,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    )

    print("Testing Hierarchical Risk Parity...")
    hrp = HierarchicalRiskParity(n_clusters=2)
    weights = hrp.calculate_weights(returns_df)
    print(f"HRP Weights:\n{weights}")
    print(f"Cluster Labels: {hrp.get_cluster_labels()}")

    print("\nTesting Hierarchical Momentum Risk Budget...")
    predicted_sharpe = {f'Asset{i}': np.random.randn() * 0.5 for i in range(1, 6)}
    hmrb = HierarchicalMomentumRiskBudget(k=1.0, n_clusters=2)
    weights = hmrb.calculate_weights(returns_df, predicted_sharpe, dates[-1])
    print(f"HM RB Weights:\n{weights}")
