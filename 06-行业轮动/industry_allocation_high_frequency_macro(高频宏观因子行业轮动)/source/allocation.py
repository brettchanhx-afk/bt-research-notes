import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config.settings import FACTOR_CONFIG

class MacroRiskAllocator:
    def __init__(self, lambda_param=0.5):
        self.lambda_param = lambda_param
        self.factor_names = list(FACTOR_CONFIG.keys())
        self.exposure_matrix = None
        self.benchmark_weights = None

    def fit_exposure_matrix(self, factor_returns, asset_returns, rolling_window=52):
        if factor_returns.empty or asset_returns.empty:
            return None

        common_dates = factor_returns.index.intersection(asset_returns.index)
        if len(common_dates) < rolling_window * 2:
            return None

        exposures_list = []
        dates = []

        for i in range(rolling_window, len(common_dates) - rolling_window, rolling_window):
            start_idx = i
            end_idx = min(i + rolling_window, len(common_dates))

            X = factor_returns.iloc[start_idx:end_idx].values
            dates.append(common_dates[end_idx])

            for col in asset_returns.columns:
                y = asset_returns[col].iloc[start_idx:end_idx].values
                try:
                    X_with_const = np.column_stack([np.ones(len(X)), X])
                    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    exposures_list.append({
                        'date': common_dates[end_idx],
                        'asset': col,
                        **{f'beta_{j}': beta[j+1] for j in range(len(beta)-1)}
                    })
                except:
                    pass

        if not exposures_list:
            return None

        exposure_df = pd.DataFrame(exposures_list)
        self.exposure_matrix = exposure_df.groupby('asset')[[f'beta_{j}' for j in range(len(self.factor_names))]].mean()
        self.exposure_matrix.columns = self.factor_names

        return self.exposure_matrix

    def set_benchmark_weights(self, n_assets, equal_weight=True):
        if equal_weight:
            self.benchmark_weights = np.ones(n_assets) / n_assets
        else:
            self.benchmark_weights = np.ones(n_assets) / n_assets
        return self.benchmark_weights

    def optimize_allocation(self, macro_views, asset_covariance=None, lambda_param=None):
        if self.exposure_matrix is None:
            print("Error: Exposure matrix not fitted yet")
            return None

        if self.benchmark_weights is None:
            print("Error: Benchmark weights not set yet")
            return None

        n_assets = len(self.benchmark_weights)
        lambda_p = lambda_param if lambda_param is not None else self.lambda_param

        views = np.array([macro_views.get(f, 0) for f in self.factor_names])

        E0 = self.benchmark_weights @ self.exposure_matrix.values

        def objective(delta_w):
            if asset_covariance is not None:
                tracking_error = delta_w @ asset_covariance @ delta_w
            else:
                tracking_error = np.sum(delta_w ** 2)

            exposure_change = (delta_w @ self.exposure_matrix.values)
            target_change = np.abs(E0.values) * views
            exposure_deviation = np.sum((exposure_change - target_change) ** 2)

            return lambda_p * tracking_error + (1 - lambda_p) * exposure_deviation

        constraints = [
            {'type': 'eq', 'fun': lambda dw: np.sum(dw)}
        ]

        bounds = [(-w0, 1 - w0) for w0 in self.benchmark_weights]

        x0 = np.zeros(n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimal_delta_w = result.x
            optimal_weights = self.benchmark_weights + optimal_delta_w
            return {
                'weights': optimal_weights,
                'delta_w': optimal_delta_w,
                'success': True
            }
        else:
            return {
                'weights': self.benchmark_weights,
                'delta_w': np.zeros(n_assets),
                'success': False
            }

    def get_recommended_weights(self, macro_views, asset_covariance=None):
        result = self.optimize_allocation(macro_views, asset_covariance)
        if result and result['success']:
            return result['weights']
        return self.benchmark_weights

class DavisDoubleHitStrategy:
    def __init__(self, lookback_window=52):
        self.lookback_window = lookback_window
        self.factor_names = list(FACTOR_CONFIG.keys())

    def calculate_delta_g(self, roe_ttm_series):
        if roe_ttm_series.empty:
            return pd.Series(dtype=float)

        delta_g = roe_ttm_series.diff(4)
        return delta_g.dropna()

    def calculate_delta_pb(self, price_series, book_value_series):
        if price_series.empty or book_value_series.empty:
            return pd.Series(dtype=float)

        pb = price_series / book_value_series
        delta_pb = np.log(pb).diff(4)
        return delta_pb.dropna()

    def fit_macro_delta_g_mapping(self, factor_returns, delta_g_series):
        common_dates = factor_returns.index.intersection(delta_g_series.index)
        if len(common_dates) < self.lookback_window:
            return None

        X = factor_returns.loc[common_dates]
        y = delta_g_series.loc[common_dates]

        X_aligned, y_aligned = X.align(y, join='inner')

        if len(X_aligned) < self.lookback_window:
            return None

        results = []
        for i in range(self.lookback_window, len(X_aligned)):
            X_train = X_aligned.iloc[i-self.lookback_window:i]
            y_train = y_aligned.iloc[i-self.lookback_window:i]

            try:
                X_with_const = np.column_stack([np.ones(len(X_train)), X_train.values])
                beta = np.linalg.lstsq(X_with_const, y_train.values, rcond=None)[0]
                residuals = y_train.values - X_with_const @ beta
                r_squared = 1 - (residuals**2).sum() / ((y_train - y_train.mean())**2).sum()

                X_test = np.ones(1 + len(self.factor_names))
                X_test[1:] = X_aligned.iloc[i].values
                y_pred = X_test @ beta

                results.append({
                    'date': X_aligned.index[i],
                    'y_pred': y_pred,
                    'r_squared': r_squared,
                    'beta': beta[1:]
                })
            except Exception as e:
                pass

        return pd.DataFrame(results) if results else None

    def fit_macro_delta_pb_mapping(self, factor_returns, delta_pb_series):
        common_dates = factor_returns.index.intersection(delta_pb_series.index)
        if len(common_dates) < self.lookback_window:
            return None

        X = factor_returns.loc[common_dates]
        y = delta_pb_series.loc[common_dates]

        X_aligned, y_aligned = X.align(y, join='inner')

        if len(X_aligned) < self.lookback_window:
            return None

        results = []
        for i in range(self.lookback_window, len(X_aligned)):
            X_train = X_aligned.iloc[i-self.lookback_window:i]
            y_train = y_aligned.iloc[i-self.lookback_window:i]

            try:
                X_with_const = np.column_stack([np.ones(len(X_train)), X_train.values])
                beta = np.linalg.lstsq(X_with_const, y_train.values, rcond=None)[0]
                residuals = y_train.values - X_with_const @ beta
                r_squared = 1 - (residuals**2).sum() / ((y_train - y_train.mean())**2).sum()

                X_test = np.ones(1 + len(self.factor_names))
                X_test[1:] = X_aligned.iloc[i].values
                y_pred = X_test @ beta

                results.append({
                    'date': X_aligned.index[i],
                    'y_pred': y_pred,
                    'r_squared': r_squared,
                    'beta': beta[1:]
                })
            except Exception as e:
                pass

        return pd.DataFrame(results) if results else None

    def calculate_composite_factor(self, delta_g_pred, delta_pb_pred):
        combined = pd.DataFrame({
            'delta_g': delta_g_pred.set_index('date')['y_pred'] if 'y_pred' in delta_g_pred.columns else delta_g_pred,
            'delta_pb': delta_pb_pred.set_index('date')['y_pred'] if 'y_pred' in delta_pb_pred.columns else delta_pb_pred
        })

        combined = combined.dropna()

        composite = combined['delta_g'] * combined['delta_pb']

        return composite.sort_values(ascending=False)

    def select_top_industries(self, composite_factor, top_n=10):
        return composite_factor.head(top_n).index.tolist()

def calculate_asset_covariance(returns_df, lookback=52):
    if returns_df.empty or len(returns_df) < lookback:
        return returns_df.cov().values

    cov_matrix = returns_df.tail(lookback).cov().values
    return cov_matrix

if __name__ == "__main__":
    print("Macro Risk Allocation Module initialized successfully!")
    print("Factor names:", list(FACTOR_CONFIG.keys()))
