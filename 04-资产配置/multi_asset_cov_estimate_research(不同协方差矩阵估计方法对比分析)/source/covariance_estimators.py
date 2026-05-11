import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')


class CovarianceEstimator:
    def __init__(self):
        self.estimates = {}

    def sample_covariance(self, returns, lookback=None):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 2:
            return np.zeros((data.shape[1], data.shape[1]))

        cov_matrix = data.cov().values
        return cov_matrix

    def ledoit_wolf_constant_variance(self, returns, lookback=None):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 10:
            return self.sample_covariance(returns, lookback)

        try:
            n_assets = data.shape[1]
            sample_cov = data.cov().values
            mu = np.trace(sample_cov) / n_assets
            target = mu * np.eye(n_assets)

            shrinkage = LedoitWolf().fit(data)
            alpha = min(0.5, max(0, (1 - shrinkage.shrinkage_)))

            cov_estimate = alpha * target + (1 - alpha) * sample_cov
            return cov_estimate
        except:
            return self.sample_covariance(returns, lookback)

    def ledoit_wolf_single_factor(self, returns, lookback=None):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 10:
            return self.sample_covariance(returns, lookback)

        try:
            n_assets = data.shape[1]
            sample_cov = data.cov().values

            equal_weighted_portfolio = np.ones(n_assets) / n_assets
            market_returns = data.values @ equal_weighted_portfolio

            X = data.values
            y = market_returns
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            beta = beta.reshape(-1, 1)

            market_var = np.var(market_returns)
            residual_var = np.var(y - X @ beta)
            F = market_var * np.outer(beta, beta) + residual_var * np.eye(n_assets)

            D = np.diag(sample_cov)
            D_half = np.sqrt(D)
            D_inv_half = np.linalg.inv(D_half)
            corr_matrix = D_inv_half @ sample_cov @ D_inv_half

            shrinkage = LedoitWolf().fit(data)
            alpha = min(0.5, max(0, (1 - shrinkage.shrinkage_)))

            cov_estimate = alpha * F + (1 - alpha) * sample_cov
            return cov_estimate
        except:
            return self.sample_covariance(returns, lookback)

    def ledoit_wolf_constant_correlation(self, returns, lookback=None):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 10:
            return self.sample_covariance(returns, lookback)

        try:
            n_assets = data.shape[1]
            sample_cov = data.cov().values

            std = np.sqrt(np.diag(sample_cov))
            corr_matrix = sample_cov / np.outer(std, std)
            np.fill_diagonal(corr_matrix, 1.0)

            upper_tri_indices = np.triu_indices(n_assets, k=1)
            avg_correlation = np.mean(corr_matrix[upper_tri_indices])

            target = np.outer(std, std) * avg_correlation
            np.fill_diagonal(target, sample_cov)

            shrinkage = LedoitWolf().fit(data)
            alpha = min(0.5, max(0, (1 - shrinkage.shrinkage_)))

            cov_estimate = alpha * target + (1 - alpha) * sample_cov
            return cov_estimate
        except:
            return self.sample_covariance(returns, lookback)

    def random_matrix_filtering(self, returns, lookback=None):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 10:
            return self.sample_covariance(returns, lookback)

        try:
            n_assets = data.shape[1]
            n_samples = len(data)

            if n_assets > n_samples:
                return self.sample_covariance(returns, lookback)

            sample_cov = data.cov().values

            std = np.sqrt(np.diag(sample_cov))
            corr_matrix = sample_cov / np.outer(std, std)

            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)

            Q = n_samples / n_assets
            lambda_max = (1 + np.sqrt(1 / Q)) ** 2
            lambda_min = (1 - np.sqrt(1 / Q)) ** 2

            threshold = lambda_max

            keep_mask = eigenvalues > threshold
            filtered_eigenvalues = np.where(keep_mask, eigenvalues, np.mean(eigenvalues[~keep_mask]))

            adjusted_corr = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

            diag_mask = np.eye(n_assets, dtype=bool)
            adjusted_corr[diag_mask] = 1.0

            cov_estimate = np.outer(std, std) * adjusted_corr
            cov_estimate[diag_mask] = sample_cov[diag_mask]

            return cov_estimate
        except:
            return self.sample_covariance(returns, lookback)

    def risk_metrics_ewma(self, returns, lookback=None, lambda_param=0.94):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 2:
            return self.sample_covariance(returns, lookback)

        try:
            n_assets = data.shape[1]
            returns_array = data.values

            ewma_cov = np.cov(returns_array)

            for t in range(1, len(returns_array)):
                epsilon = returns_array[t].reshape(-1, 1) @ returns_array[t].reshape(1, -1)
                ewma_cov = lambda_param * ewma_cov + (1 - lambda_param) * epsilon

            return ewma_cov
        except:
            return self.sample_covariance(returns, lookback)

    def vec_garch(self, returns, lookback=None):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 30:
            return self.sample_covariance(returns, lookback)

        try:
            from arch import arch_model

            n_assets = data.shape[1]
            std_results = []

            for i in range(n_assets):
                asset_returns = data.iloc[:, i].values
                asset_returns = asset_returns[~np.isnan(asset_returns)]

                if len(asset_returns) < 30:
                    std_results.append(np.std(asset_returns))
                    continue

                try:
                    model = arch_model(asset_returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
                    result = model.fit(disp='off', show_warning=False)
                    forecast = result.forecast(horizon=1)
                    std_results.append(np.sqrt(forecast.variance.values[-1, 0]) / 100)
                except:
                    std_results.append(np.std(asset_returns))

            base_cov = self.sample_covariance(data)
            garch_std = np.array(std_results).reshape(-1, 1)

            garch_cov = base_cov.copy()
            for i in range(n_assets):
                for j in range(n_assets):
                    if i != j:
                        std_product = garch_std[i] * garch_std[j]
                        if std_product > 0:
                            corr = base_cov[i, j] / (np.sqrt(base_cov[i, i]) * np.sqrt(base_cov[j, j]))
                            garch_cov[i, j] = corr * std_product

            np.fill_diagonal(garch_cov, np.array(garch_std).flatten() ** 2)

            return garch_cov
        except Exception as e:
            print(f"VEC-GARCH error: {e}")
            return self.sample_covariance(returns, lookback)

    def ccc_garch(self, returns, lookback=None):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 30:
            return self.sample_covariance(returns, lookback)

        try:
            from arch import arch_model

            n_assets = data.shape[1]
            std_results = []
            base_corr = data.corr().values

            for i in range(n_assets):
                asset_returns = data.iloc[:, i].values
                asset_returns = asset_returns[~np.isnan(asset_returns)]

                if len(asset_returns) < 30:
                    std_results.append(np.std(asset_returns))
                    continue

                try:
                    model = arch_model(asset_returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
                    result = model.fit(disp='off', show_warning=False)
                    forecast = result.forecast(horizon=1)
                    std_results.append(np.sqrt(forecast.variance.values[-1, 0]) / 100)
                except:
                    std_results.append(np.std(asset_returns))

            garch_std = np.array(std_results).reshape(-1)
            D = np.diag(garch_std)
            corr_matrix = np.where(np.eye(n_assets) == 1, 1.0, base_corr)

            cov_estimate = D @ corr_matrix @ D
            return cov_estimate
        except Exception as e:
            print(f"CCC-GARCH error: {e}")
            return self.sample_covariance(returns, lookback)

    def dcc_garch(self, returns, lookback=None, theta1=0.05, theta2=0.95):
        if lookback is not None:
            data = returns.tail(lookback)
        else:
            data = returns

        if len(data) < 30:
            return self.sample_covariance(returns, lookback)

        try:
            from arch import arch_model

            n_assets = data.shape[1]
            returns_array = data.values

            standardized_residuals = np.zeros_like(returns_array)

            for i in range(n_assets):
                asset_returns = returns_array[:, i]
                asset_returns = asset_returns[~np.isnan(asset_returns)]

                if len(asset_returns) < 30:
                    standardized_residuals[:, i] = 1.0
                    continue

                try:
                    model = arch_model(asset_returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
                    result = model.fit(disp='off', show_warning=False)
                    forecast = result.forecast(horizon=1)
                    conditional_std = np.sqrt(result.conditional_volatility / 100)
                    standardized_residuals[:, i] = asset_returns / np.where(conditional_std > 0, conditional_std, 1.0)
                except:
                    standardized_residuals[:, i] = 1.0

            valid_mask = ~np.isnan(standardized_residuals).any(axis=1)
            valid_residuals = standardized_residuals[valid_mask]

            if len(valid_residuals) < 10:
                return self.sample_covariance(returns, lookback)

            Q_bar = np.cov(valid_residuals.T)
            Q_t = Q_bar.copy()

            for t in range(len(valid_residuals)):
                epsilon = valid_residuals[t].reshape(-1, 1)
                Q_t = (1 - theta1 - theta2) * Q_bar + theta1 * (epsilon @ epsilon.T) + theta2 * Q_t

            D_t = np.diag(np.sqrt(np.diag(Q_t)))
            R_t = np.linalg.inv(D_t) @ Q_t @ np.linalg.inv(D_t)
            R_t = np.where(np.eye(n_assets) == 1, 1.0, R_t)

            garch_std = []
            for i in range(n_assets):
                asset_returns = returns_array[:, i]
                asset_returns = asset_returns[~np.isnan(asset_returns)]

                if len(asset_returns) < 30:
                    garch_std.append(np.std(asset_returns))
                    continue

                try:
                    model = arch_model(asset_returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
                    result = model.fit(disp='off', show_warning=False)
                    forecast = result.forecast(horizon=1)
                    garch_std.append(np.sqrt(forecast.variance.values[-1, 0]) / 100)
                except:
                    garch_std.append(np.std(asset_returns))

            D = np.diag(garch_std)
            cov_estimate = D @ R_t @ D

            return cov_estimate
        except Exception as e:
            print(f"DCC-GARCH error: {e}")
            return self.sample_covariance(returns, lookback)

    def get_covariance(self, returns, method='sample_cov', lookback=None, **kwargs):
        methods = {
            'sample_cov': self.sample_covariance,
            'ledoit_wolf_constant_variance': self.ledoit_wolf_constant_variance,
            'ledoit_wolf_single_factor': self.ledoit_wolf_single_factor,
            'ledoit_wolf_constant_correlation': self.ledoit_wolf_constant_correlation,
            'random_matrix': self.random_matrix_filtering,
            'risk_metrics': lambda r, lb: self.risk_metrics_ewma(r, lb, lambda_param=kwargs.get('lambda', 0.94)),
            'vec_garch': self.vec_garch,
            'ccc_garch': self.ccc_garch,
            'dcc_garch': self.dcc_garch,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}")

        return methods[method](returns, lookback)

    def compute_rmse(self, estimated_cov, true_cov):
        n = estimated_cov.shape[0]
        diff = estimated_cov - true_cov
        rmse = np.sqrt(np.sum(diff ** 2) / (n * n))
        return rmse

    def ensure_positive_definite(self, cov_matrix, shrinkage_factor=None):
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        if shrinkage_factor is None:
            min_eigenvalue = np.min(eigenvalues)
            if min_eigenvalue <= 0:
                shrinkage_factor = abs(min_eigenvalue) + 1e-8
            else:
                return cov_matrix

        eigenvalues = np.maximum(eigenvalues, shrinkage_factor)
        adjusted_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return (adjusted_cov + adjusted_cov.T) / 2