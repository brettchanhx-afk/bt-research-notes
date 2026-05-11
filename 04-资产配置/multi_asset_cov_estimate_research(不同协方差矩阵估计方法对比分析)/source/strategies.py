import numpy as np
import pandas as pd
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')


class BlackLittermanStrategy:
    def __init__(self, risk_aversion=10):
        self.risk_aversion = risk_aversion
        self.weights = None

    def compute_implied_returns(self, market_cap_weights, cov_matrix):
        implied_returns = self.risk_aversion * cov_matrix @ market_cap_weights
        return implied_returns

    def black_litterman(
        self,
        prior_returns,
        cov_matrix,
        views_pick_matrix,
        views_returns,
        views_confidence=None,
        tau=0.025
    ):
        if views_confidence is None:
            views_confidence = np.eye(len(views_returns))

        P = views_pick_matrix
        Q = views_returns.reshape(-1, 1)
        Omega = views_confidence

        tau_cov = tau * cov_matrix

        middle_matrix = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)

        posterior_returns = prior_returns + middle_matrix @ P.T @ np.linalg.inv(Omega) @ (Q - P @ prior_returns.reshape(-1, 1))

        posterior_cov = cov_matrix + middle_matrix

        return posterior_returns.flatten(), posterior_cov

    def optimize_portfolio(self, expected_returns, cov_matrix, allow_short=False, weight_bounds=None):
        n_assets = len(expected_returns)

        if weight_bounds is None:
            if allow_short:
                weight_bounds = (-0.5, 0.5)
            else:
                weight_bounds = (0, 1)

        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

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
        else:
            return np.ones(n_assets) / n_assets

    def run_strategy(
        self,
        returns,
        cov_estimator,
        market_cap_weights,
        lookback_period=252 * 5,
        method='sample_cov',
        views_pick_matrix=None,
        views_returns=None,
        tau=0.0185,
        allow_short=False
    ):
        if len(returns) < lookback_period:
            print(f"Insufficient data for BL strategy")
            return None

        historical_data = returns.tail(lookback_period)

        cov_matrix = cov_estimator.get_covariance(historical_data, method=method)
        cov_matrix = cov_estimator.ensure_positive_definite(cov_matrix)

        prior_returns = historical_data.mean().values * 252

        if views_pick_matrix is not None and views_returns is not None:
            posterior_returns, posterior_cov = self.black_litterman(
                prior_returns,
                cov_matrix,
                views_pick_matrix,
                views_returns,
                tau=tau
            )
            expected_returns = posterior_returns
            cov_matrix = posterior_cov
        else:
            expected_returns = prior_returns

        weights = self.optimize_portfolio(expected_returns, cov_matrix, allow_short=allow_short)

        self.weights = weights
        return weights

    def run_rolling_bl_strategy(
        self,
        returns,
        cov_estimator,
        market_cap_weights,
        lookback_period=252 * 5,
        rebalance_freq='monthly',
        method='sample_cov',
        allow_short=False,
        stock_weight_limit=0.10,
        turnover_limit=0.60
    ):
        if rebalance_freq == 'monthly':
            rebalance_dates = self._get_monthly_dates(returns)
        else:
            rebalance_dates = self._get_monthly_dates(returns)

        start_idx = lookback_period
        if start_idx >= len(rebalance_dates):
            start_idx = lookback_period
        rebalance_dates = rebalance_dates[start_idx:]

        portfolio_values = []
        weights_history = []
        prev_weights = None

        for date in rebalance_dates:
            date_idx = returns.index.get_loc(date) if date in returns.index else None
            if date_idx is None:
                continue

            historical_data = returns.iloc[:date_idx]
            if len(historical_data) < lookback_period:
                continue

            lookback_data = historical_data.tail(lookback_period)

            try:
                cov_matrix = cov_estimator.get_covariance(lookback_data, method=method)
                cov_matrix = cov_estimator.ensure_positive_definite(cov_matrix)

                prior_returns = lookback_data.mean().values * 252

                views_pick_matrix = np.array([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]
                ])
                views_returns = np.array([
                    lookback_data.iloc[:, 0].tail(20).mean() * 252,
                    lookback_data.iloc[:, 1].tail(20).mean() * 252
                ])

                posterior_returns, posterior_cov = self.black_litterman(
                    prior_returns,
                    cov_matrix,
                    views_pick_matrix,
                    views_returns,
                    tau=0.0185
                )

                weights = self.optimize_portfolio(posterior_returns, posterior_cov, allow_short=allow_short)

                stock_weight = sum(weights[:3])
                if stock_weight > 0.20:
                    scale = 0.20 / stock_weight
                    weights[:3] *= scale
                    weights[3:] *= (1 - scale) / sum(weights[3:]) if sum(weights[3:]) > 0 else 0

            except Exception as e:
                print(f"Error at {date}: {e}")
                weights = np.ones(returns.shape[1]) / returns.shape[1]

            if prev_weights is not None:
                turnover = np.sum(np.abs(weights - prev_weights))
                if turnover > turnover_limit:
                    weights = prev_weights + (weights - prev_weights) * (turnover_limit / turnover)

            weights_history.append(weights)
            prev_weights = weights.copy()

            period_returns = returns.iloc[date_idx:min(date_idx + 22, len(returns))]
            if len(period_returns) > 0:
                portfolio_return = period_returns @ weights
                daily_returns = portfolio_return
                portfolio_values.append({
                    'date': date,
                    'weights': weights.copy(),
                    'daily_returns': daily_returns,
                    'period_return': portfolio_return.sum()
                })

        if len(portfolio_values) == 0:
            return None

        result_df = pd.DataFrame(portfolio_values)
        result_df.set_index('date', inplace=True)

        return result_df, weights_history

    def _get_monthly_dates(self, returns):
        monthly_idx = returns.resample('M').indices
        return sorted(list(monthly_idx.keys()))


class RiskParityStrategy:
    def __init__(self):
        self.weights = None

    def risk_parity_portfolio(self, cov_matrix, risk_budget=None, allow_short=False):
        n_assets = cov_matrix.shape[0]

        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets

        weight_bounds = (0.01, 1) if not allow_short else (-1, 1)

        def risk_contribution(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            portfolio_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_var
            return risk_contrib

        def risk_parity_objective(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            rc = risk_contribution(weights)
            target_rc = risk_budget * np.sum(rc)
            return np.sum((rc - target_rc) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}]

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
            weights = np.abs(result.x)
            self.weights = weights / np.sum(weights)
            return self.weights
        else:
            self.weights = np.ones(n_assets) / n_assets
            return self.weights

    def run_rolling_risk_parity(
        self,
        returns,
        cov_estimator,
        lookback_period=126,
        rebalance_freq='monthly',
        method='sample_cov',
        allow_short=False
    ):
        if rebalance_freq == 'monthly':
            rebalance_dates = self._get_monthly_dates(returns)
        else:
            rebalance_dates = self._get_monthly_dates(returns)

        start_idx = lookback_period
        if start_idx >= len(rebalance_dates):
            start_idx = lookback_period
        rebalance_dates = rebalance_dates[start_idx:]

        portfolio_values = []
        weights_history = []
        prev_weights = None

        for date in rebalance_dates:
            date_idx = returns.index.get_loc(date) if date in returns.index else None
            if date_idx is None:
                continue

            historical_data = returns.iloc[:date_idx]
            if len(historical_data) < lookback_period:
                continue

            lookback_data = historical_data.tail(lookback_period)

            try:
                cov_matrix = cov_estimator.get_covariance(lookback_data, method=method)
                cov_matrix = cov_estimator.ensure_positive_definite(cov_matrix)

                weights = self.risk_parity_portfolio(cov_matrix, allow_short=allow_short)

            except Exception as e:
                print(f"Error at {date}: {e}")
                weights = np.ones(returns.shape[1]) / returns.shape[1]

            weights_history.append(weights)

            period_returns = returns.iloc[date_idx:min(date_idx + 22, len(returns))]
            if len(period_returns) > 0:
                portfolio_return = period_returns @ weights
                portfolio_values.append({
                    'date': date,
                    'weights': weights.copy(),
                    'daily_returns': portfolio_return,
                    'period_return': portfolio_return.sum()
                })

        if len(portfolio_values) == 0:
            return None

        result_df = pd.DataFrame(portfolio_values)
        result_df.set_index('date', inplace=True)

        return result_df, weights_history

    def _get_monthly_dates(self, returns):
        monthly_idx = returns.resample('M').indices
        return sorted(list(monthly_idx.keys()))


class BLMultiMethodBacktest:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital

    def run_backtest(
        self,
        returns,
        cov_estimator,
        market_cap_weights,
        methods=['sample_cov'],
        lookback_period=252 * 5,
        rebalance_freq='monthly',
        allow_short=False,
        stock_weight_limit=0.10,
        turnover_limit=0.60
    ):
        results = {}

        for method in methods:
            print(f"Running BL backtest for method: {method}")

            bl_strategy = BlackLittermanStrategy(risk_aversion=10)

            result_df, weights_history = bl_strategy.run_rolling_bl_strategy(
                returns=returns,
                cov_estimator=cov_estimator,
                market_cap_weights=market_cap_weights,
                lookback_period=lookback_period,
                rebalance_freq=rebalance_freq,
                method=method,
                allow_short=allow_short,
                stock_weight_limit=stock_weight_limit,
                turnover_limit=turnover_limit
            )

            if result_df is not None and len(result_df) > 0:
                cumulative_returns = (1 + result_df['daily_returns'].apply(lambda x: pd.Series(x) if isinstance(x, pd.Series) else pd.Series([x]))).cumprod()
                initial_value = self.initial_capital
                portfolio_values = initial_value * cumulative_returns

                if isinstance(portfolio_values, pd.DataFrame):
                    portfolio_values = portfolio_values.iloc[:, 0]

                result_df['portfolio_value'] = portfolio_values.fillna(initial_value)

                results[method] = {
                    'portfolio_values': result_df['portfolio_value'],
                    'weights_history': weights_history,
                    'daily_returns': result_df['daily_returns']
                }

        return results


class RiskParityMultiMethodBacktest:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital

    def run_backtest(
        self,
        returns,
        cov_estimator,
        methods=['sample_cov'],
        lookback_period=126,
        rebalance_freq='monthly',
        allow_short=False
    ):
        results = {}

        for method in methods:
            print(f"Running Risk Parity backtest for method: {method}")

            rp_strategy = RiskParityStrategy()

            result_df, weights_history = rp_strategy.run_rolling_risk_parity(
                returns=returns,
                cov_estimator=cov_estimator,
                lookback_period=lookback_period,
                rebalance_freq=rebalance_freq,
                method=method,
                allow_short=allow_short
            )

            if result_df is not None and len(result_df) > 0:
                daily_returns_series = []
                for dr in result_df['daily_returns']:
                    if isinstance(dr, pd.Series):
                        daily_returns_series.extend(dr.values.flatten().tolist())
                    else:
                        daily_returns_series.append(dr)

                daily_returns_array = np.array(daily_returns_series)
                daily_returns_array = daily_returns_array[~np.isnan(daily_returns_array)]

                if len(daily_returns_array) > 0:
                    portfolio_values = self.initial_capital * np.cumprod(np.concatenate([[1], 1 + daily_returns_array]))

                    results[method] = {
                        'portfolio_values': pd.Series(portfolio_values),
                        'weights_history': weights_history,
                        'daily_returns': pd.Series(daily_returns_array)
                    }

        return results