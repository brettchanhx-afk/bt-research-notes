import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')


class HierarchicalRiskParity:
    def __init__(self, method='hrp'):
        self.method = method
        self.weights = None
        self.asset_order = None
        self.cluster_tree = None

    def _compute_distance_matrix(self, corr_matrix):
        if isinstance(corr_matrix, pd.DataFrame):
            corr_matrix = corr_matrix.values
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix

    def _hierarchical_clustering(self, distance_matrix):
        if isinstance(distance_matrix, pd.DataFrame):
            distance_matrix = distance_matrix.values
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='single')
        return Z

    def _quasi_diagonalization(self, corr_matrix, Z, asset_names):
        sorted_indices = self._get_sorted_indices(Z)
        sorted_asset_names = [asset_names[i] for i in sorted_indices]
        sorted_corr_matrix = corr_matrix.iloc[sorted_indices, sorted_indices]
        return sorted_corr_matrix, sorted_asset_names, sorted_indices

    def _get_sorted_indices(self, Z):
        return list(leaves_list(Z))

    def _naive_bisection(self, sorted_indices, cov_matrix):
        def bisect_recursive(indices):
            if len(indices) == 1:
                return {indices[0]: 1.0}

            mid = len(indices) // 2
            left = indices[:mid]
            right = indices[mid:]

            left_cov = cov_matrix.iloc[left, left].values
            right_cov = cov_matrix.iloc[right, right].values

            left_var = np.sum(left_cov) / (len(left) ** 2)
            right_var = np.sum(right_cov) / (len(right) ** 2)

            total_var = left_var + right_var
            left_weight = right_var / total_var
            right_weight = left_var / total_var

            left_weights = bisect_recursive(left)
            right_weights = bisect_recursive(right)

            result = {}
            for k, v in left_weights.items():
                result[k] = v * left_weight
            for k, v in right_weights.items():
                result[k] = v * right_weight

            return result

        weights_dict = bisect_recursive(sorted_indices)
        weights = np.array([weights_dict[i] for i in range(len(sorted_indices))])
        return weights / weights.sum()

    def _volatility_based_bisection(self, sorted_indices, cov_matrix):
        def bisect_recursive(indices):
            if len(indices) == 1:
                return {indices[0]: 1.0}

            mid = len(indices) // 2
            left = indices[:mid]
            right = indices[mid:]

            left_vol = np.sqrt(np.trace(cov_matrix.iloc[left, left]) / len(left))
            right_vol = np.sqrt(np.trace(cov_matrix.iloc[right, right]) / len(right))

            total_vol = left_vol + right_vol
            left_weight = right_vol / total_vol
            right_weight = left_vol / total_vol

            left_weights = bisect_recursive(left)
            right_weights = bisect_recursive(right)

            result = {}
            for k, v in left_weights.items():
                result[k] = v * left_weight
            for k, v in right_weights.items():
                result[k] = v * right_weight

            return result

        weights_dict = bisect_recursive(sorted_indices)
        weights = np.array([weights_dict[i] for i in range(len(sorted_indices))])
        return weights / weights.sum()

    def _recursive_bisection(self, sorted_indices, cov_matrix):
        def bisect_recursive(indices):
            if len(indices) == 1:
                return {indices[0]: 1.0}

            mid = len(indices) // 2
            left = indices[:mid]
            right = indices[mid:]

            left_cov = cov_matrix.iloc[left, left].values
            right_cov = cov_matrix.iloc[right, right].values

            left_var = np.sum(left_cov) / (len(left) ** 2)
            right_var = np.sum(right_cov) / (len(right) ** 2)

            total_var = left_var + right_var
            left_weight = right_var / total_var
            right_weight = left_var / total_var

            left_weights = bisect_recursive(left)
            right_weights = bisect_recursive(right)

            result = {}
            for k, v in left_weights.items():
                result[k] = v * left_weight
            for k, v in right_weights.items():
                result[k] = v * right_weight

            return result

        weights_dict = bisect_recursive(sorted_indices)
        weights = np.array([weights_dict[i] for i in range(len(sorted_indices))])
        return weights / weights.sum()

    def fit(self, returns_df):
        corr_matrix = returns_df.corr()
        cov_matrix = returns_df.cov()
        asset_names = list(returns_df.columns)

        distance_matrix = self._compute_distance_matrix(corr_matrix)
        Z = self._hierarchical_clustering(distance_matrix)
        self.cluster_tree = Z

        sorted_corr, sorted_asset_names, sorted_indices = self._quasi_diagonalization(
            corr_matrix, Z, asset_names
        )
        self.asset_order = sorted_asset_names

        sorted_cov = cov_matrix.iloc[sorted_indices, sorted_indices]

        if self.method == 'hrp':
            self.weights = self._recursive_bisection(list(range(len(sorted_indices))), sorted_cov)
        elif self.method == 'naive':
            self.weights = self._naive_bisection(list(range(len(sorted_indices))), sorted_cov)
        elif self.method == 'volatility':
            self.weights = self._volatility_based_bisection(list(range(len(sorted_indices))), sorted_cov)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        final_weights = np.zeros(len(asset_names))
        for i, idx in enumerate(sorted_indices):
            final_weights[idx] = self.weights[i]

        self.weights = final_weights
        weight_dict = dict(zip(asset_names, self.weights))
        return weight_dict

    def get_weights(self):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return dict(zip(self.asset_order if hasattr(self, 'asset_order') else range(len(self.weights)), self.weights))


class RiskParity:
    def __init__(self):
        self.weights = None

    def fit(self, returns_df):
        cov_matrix = returns_df.cov()
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vol / inv_vol.sum()
        self.weights = weights
        return dict(zip(returns_df.columns, weights))


def compute_portfolio_returns(weights, returns_df):
    return (returns_df * weights).sum(axis=1)


def compute_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights @ cov_matrix @ weights)


def get_dendrogram_data(Z, asset_names):
    return dendrogram(Z, labels=asset_names, no_plot=True)


def plot_dendrogram(Z, asset_names, figsize=(12, 6)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    dendrogram(Z, labels=asset_names)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Assets')
    plt.ylabel('Distance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt
