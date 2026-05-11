import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
from scipy.optimize import minimize
from scipy.linalg import inv

from .config import (
    PORTFOLIO_CONFIG,
    ASSETS_CONFIG,
)

warnings.filterwarnings("ignore")


class PortfolioOptimizer:
    def __init__(
        self,
        risk_parity_method: str = "ewma_volatility",
        volatility_window: int = 60,
        max_leverage: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        self.risk_parity_method = risk_parity_method
        self.volatility_window = volatility_window
        self.max_leverage = max_leverage
        self.min_weight = min_weight
        self.max_weight = max_weight

    def calculate_volatility(
        self, returns: pd.DataFrame, method: str = "ewma_volatility"
    ) -> pd.Series:
        if method == "ewma_volatility":
            span = self.volatility_window
            volatility = returns.ewm(span=span).std()
            volatility = volatility.iloc[-1]
        elif method == "historical_volatility":
            volatility = returns.std() * np.sqrt(252)
        elif method == "rolling_volatility":
            volatility = returns.rolling(self.volatility_window).std().iloc[-1] * np.sqrt(252)
        else:
            volatility = returns.std() * np.sqrt(252)

        return volatility

    def calculate_covariance_matrix(
        self, returns: pd.DataFrame, method: str = "ewma_covariance"
    ) -> pd.DataFrame:
        if method == "ewma_covariance":
            span = self.volatility_window
            cov = returns.ewm(span=span).cov()
            cov = cov.iloc[-returns.shape[1]:]
        elif method == "historical_covariance":
            cov = returns.cov() * 252
        elif method == "shrunk_covariance":
            sample_cov = returns.cov() * 252
            corr = returns.corr()
            diag = np.diag(sample_cov)
            shrinkage = 0.2
            shrunk_cov = shrinkage * np.diag(diag) + (1 - shrinkage) * sample_cov
            cov = pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)
        else:
            cov = returns.cov() * 252

        return cov

    def risk_parity_weights(
        self,
        returns: pd.DataFrame,
        factor_exposure: Optional[pd.DataFrame] = None,
        factor_target: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        volatility = self.calculate_volatility(returns)

        inv_vol = 1 / volatility
        inv_vol = inv_vol.replace([np.inf, -np.inf], 0)
        weights = inv_vol / inv_vol.sum()

        if factor_exposure is not None and factor_target is not None:
            weights = self._apply_factor_constraint(
                weights, factor_exposure, factor_target, returns.columns
            )

        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()

        return weights

    def _apply_factor_constraint(
        self,
        base_weights: np.ndarray,
        factor_exposure: pd.DataFrame,
        factor_target: Dict[str, float],
        asset_names: pd.Index,
    ) -> np.ndarray:
        exposure_matrix = factor_exposure.loc[asset_names]

        target_exposure = np.array([
            factor_target.get(col, 0) for col in exposure_matrix.columns
        ])

        current_exposure = exposure_matrix.values.T @ base_weights

        exposure_diff = target_exposure - current_exposure

        adjustment = exposure_diff * 0.1

        adjusted_weights = base_weights + exposure_matrix.values @ adjustment

        adjusted_weights = np.clip(adjusted_weights, self.min_weight, self.max_weight * 0.5)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        return adjusted_weights

    def minimum_variance_weights(
        self,
        returns: pd.DataFrame,
        factor_exposure: Optional[pd.DataFrame] = None,
        factor_target: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        cov_matrix = self.calculate_covariance_matrix(returns)

        n_assets = len(returns.columns)

        def portfolio_variance(weights):
            return weights @ cov_matrix.values @ weights

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return result.x
        else:
            return initial_weights

    def risk_parity_with_factor_constraint(
        self,
        returns: pd.DataFrame,
        factor_exposure: pd.DataFrame,
        factor_target: Dict[str, float],
        factor_std: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        n_assets = len(returns.columns)

        cov_matrix = self.calculate_covariance_matrix(returns)
        volatility = self.calculate_volatility(returns)

        exposure_matrix = factor_exposure.loc[returns.columns]

        if factor_std is None:
            factor_std = {col: 1.0 for col in exposure_matrix.columns}

        def objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)

            factor_contribution = exposure_matrix.values.T @ (weights * volatility.values)
            factor_contribution = factor_contribution / (factor_std.get(col, 1.0) for col in exposure_matrix.columns)

            risk_contribution = weights * (cov_matrix.values @ weights) / portfolio_vol

            target_risk = portfolio_vol / n_assets

            risk_diff = risk_contribution - target_risk

            factor_diff = factor_contribution - np.array([
                factor_target.get(col, 0) for col in exposure_matrix.columns
            ])

            return np.sum(risk_diff**2) + 0.1 * np.sum(factor_diff**2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        initial_weights = self.risk_parity_weights(returns)

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        if result.success:
            return result.x
        else:
            return initial_weights

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = "risk_parity",
        factor_exposure: Optional[pd.DataFrame] = None,
        factor_target: Optional[Dict[str, float]] = None,
        factor_std: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        if method == "risk_parity":
            weights = self.risk_parity_weights(returns, factor_exposure, factor_target)
        elif method == "minimum_variance":
            weights = self.minimum_variance_weights(returns, factor_exposure, factor_target)
        elif method == "risk_parity_factor":
            weights = self.risk_parity_with_factor_constraint(
                returns, factor_exposure, factor_target, factor_std
            )
        else:
            weights = self.risk_parity_weights(returns)

        weights_series = pd.Series(weights, index=returns.columns)
        return weights_series

    def calculate_portfolio_returns(
        self,
        weights: pd.Series,
        asset_returns: pd.DataFrame,
    ) -> pd.Series:
        aligned_weights = weights.reindex(asset_returns.columns).fillna(0)
        portfolio_returns = (asset_returns * aligned_weights).sum(axis=1)
        return portfolio_returns

    def calculate_factor_exposure_of_portfolio(
        self,
        weights: pd.Series,
        factor_exposure: pd.DataFrame,
    ) -> pd.Series:
        aligned_weights = weights.reindex(factor_exposure.index).fillna(0)
        portfolio_exposure = factor_exposure.mul(aligned_weights, axis=0).sum(axis=0)
        return portfolio_exposure

    def rebalance_portfolio(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        turnover_limit: float = 0.5,
    ) -> pd.Series:
        weight_diff = (target_weights - current_weights).abs().sum()

        if weight_diff > turnover_limit:
            adjustment = weight_diff / turnover_limit
            adjusted_target = current_weights + (target_weights - current_weights) / adjustment
        else:
            adjusted_target = target_weights

        return adjusted_target
