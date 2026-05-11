import numpy as np
import pandas as pd
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')


class PortfolioBuilder:
    def __init__(self):
        self.weights = None

    def minimum_variance_portfolio(self, cov_matrix, allow_short=False, weight_bounds=None):
        n_assets = cov_matrix.shape[0]

        if weight_bounds is None:
            if allow_short:
                weight_bounds = (-1, 1)
            else:
                weight_bounds = (0, 1)

        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if allow_short:
            initial_weights = np.ones(n_assets) / n_assets
        else:
            initial_weights = np.ones(n_assets) / n_assets

        result = optimize.minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=tuple([weight_bounds] * n_assets),
            constraints=constraints,
            options={'maxiter': 5000, 'ftol': 1e-15}
        )

        if result.success:
            self.weights = result.x
            return result.x
        else:
            # 优化失败时使用解析解
            try:
                inv_cov = np.linalg.inv(cov_matrix)
                ones = np.ones(n_assets)
                analytical_weights = inv_cov @ ones / (ones @ inv_cov @ ones)
                # 如果解析解在约束范围内，使用解析解
                if allow_short or all(w >= 0 for w in analytical_weights):
                    return analytical_weights / analytical_weights.sum()
            except:
                pass
            return np.ones(n_assets) / n_assets

    def target_volatility_portfolio(self, cov_matrix, target_volatility, allow_short=False, weight_bounds=None):
        n_assets = cov_matrix.shape[0]

        if weight_bounds is None:
            if allow_short:
                weight_bounds = (-1, 1)
            else:
                weight_bounds = (0, 1)

        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(w @ cov_matrix @ w) - target_volatility}
        ]

        if allow_short:
            initial_weights = np.ones(n_assets) / n_assets
        else:
            initial_weights = np.ones(n_assets) / n_assets

        result = optimize.minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=tuple([weight_bounds] * n_assets),
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            self.weights = result.x
            return result.x

        def objective_with_vol_penalty(weights):
            vol = np.sqrt(weights @ cov_matrix @ weights)
            return weights @ cov_matrix @ weights + 1000 * (vol - target_volatility) ** 2

        result = optimize.minimize(
            objective_with_vol_penalty,
            initial_weights,
            method='SLSQP',
            bounds=tuple([weight_bounds] * n_assets),
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
            options={'maxiter': 1000}
        )

        if result.success:
            self.weights = result.x
            return result.x

        return np.ones(n_assets) / n_assets

    def risk_parity_portfolio(self, cov_matrix, risk_budget=None, allow_short=False, weight_bounds=None):
        n_assets = cov_matrix.shape[0]

        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets

        if weight_bounds is None:
            if allow_short:
                weight_bounds = (-1, 1)
            else:
                weight_bounds = (0.01, 1)

        def risk_contribution(weights):
            portfolio_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_var
            return risk_contrib

        def risk_parity_objective(weights):
            rc = risk_contribution(weights)
            target_rc = risk_budget * np.sum(rc)
            return np.sum((rc - target_rc) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        initial_weights = np.ones(n_assets) / n_assets

        result = optimize.minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=tuple([weight_bounds] * n_assets),
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            self.weights = result.x
            return result.x
        else:
            return np.ones(n_assets) / n_assets

    def equal_weight_portfolio(self, n_assets):
        self.weights = np.ones(n_assets) / n_assets
        return self.weights

    def maximum_sharpe_portfolio(self, expected_returns, cov_matrix, risk_free_rate=0.0, allow_short=False, weight_bounds=None):
        n_assets = len(expected_returns)

        if weight_bounds is None:
            if allow_short:
                weight_bounds = (-1, 1)
            else:
                weight_bounds = (0, 1)

        def negative_sharpe(weights):
            port_return = weights @ expected_returns
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol == 0:
                return -np.inf
            return -(port_return - risk_free_rate) / port_vol

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        initial_weights = np.ones(n_assets) / n_assets

        result = optimize.minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=tuple([weight_bounds] * n_assets),
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            self.weights = result.x
            return result.x
        else:
            return np.ones(n_assets) / n_assets

    def hierarchical_risk_parity(self, returns, cov_matrix=None):
        if cov_matrix is None:
            cov_matrix = returns.cov().values

        n_assets = cov_matrix.shape[0]

        def get_quasi_diag(cov_matrix):
            distances = np.zeros((n_assets, n_assets))
            for i in range(n_assets):
                for j in range(n_assets):
                    distances[i, j] = np.sqrt(cov_matrix[i, i] + cov_matrix[j, j] - 2 * cov_matrix[i, j])
            return distances

        def hierarchical_clustering(distances):
            clusters = [[i] for i in range(n_assets)]
            while len(clusters) > 1:
                min_dist = np.inf
                merge_idx = None
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        cluster_i = clusters[i]
                        cluster_j = clusters[j]
                        avg_dist = np.mean([distances[a, b] for a in cluster_i for b in cluster_j])
                        if avg_dist < min_dist:
                            min_dist = avg_dist
                            merge_idx = (i, j)

                i, j = merge_idx
                clusters[i] = clusters[i] + clusters[j]
                clusters.pop(j)

            return clusters[0]

        distances = get_quasi_diag(cov_matrix)
        ordered_indices = hierarchical_clustering(distances)

        def bisection(values, target):
            left, right = 0, len(values) - 1
            while left < right:
                mid = (left + right) // 2
                if values[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left

        weights = np.zeros(n_assets)
        ordered_cov = cov_matrix[np.ix_(ordered_indices, ordered_indices)]

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                weights[ordered_indices[i]] += 1.0 / ordered_cov[i, j]
                weights[ordered_indices[j]] += 1.0 / ordered_cov[i, j]

        weights = weights / np.sum(weights)
        self.weights = weights
        return weights

    def get_weights(self):
        return self.weights