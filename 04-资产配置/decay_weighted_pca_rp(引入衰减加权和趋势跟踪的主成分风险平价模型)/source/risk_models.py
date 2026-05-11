"""
Risk model module for calculating risk metrics and covariance matrices.
Implements standard risk parity and related models.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class RiskModel:
    """
    Base class for risk models.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize RiskModel with returns data.

        Args:
            returns: DataFrame of asset returns
        """
        self.returns = returns
        self.cov_matrix = None
        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)

    def calculate_covariance(self) -> np.ndarray:
        """
        Calculate covariance matrix from returns.

        Returns:
            Covariance matrix
        """
        self.cov_matrix = self.returns.cov().values
        return self.cov_matrix

    def calculate_volatility(self) -> np.ndarray:
        """
        Calculate asset volatilities from covariance matrix.

        Returns:
            Array of volatilities
        """
        if self.cov_matrix is None:
            self.calculate_covariance()

        return np.sqrt(np.diag(self.cov_matrix))

    def calculate_correlation(self) -> np.ndarray:
        """
        Calculate correlation matrix from covariance matrix.

        Returns:
            Correlation matrix
        """
        if self.cov_matrix is None:
            self.calculate_covariance()

        std_vec = np.sqrt(np.diag(self.cov_matrix))
        std_matrix = np.outer(std_vec, std_vec)

        return self.cov_matrix / std_matrix


class StandardRiskParity(RiskModel):
    """
    Standard Risk Parity (RP) model.
    Allocates weights such that each asset contributes equally to portfolio risk.
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize Standard Risk Parity model.

        Args:
            returns: DataFrame of asset returns
        """
        super().__init__(returns)

    def calculate_weights(self) -> np.ndarray:
        """
        Calculate risk parity weights.

        Returns:
            Array of optimal weights
        """
        if self.cov_matrix is None:
            self.calculate_covariance()

        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            target_rc = port_vol / self.n_assets
            return np.sum((risk_contrib - target_rc) ** 2)

        n = self.n_assets
        initial_weights = np.ones(n) / n

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]

        result = minimize(
            risk_contribution,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n)]
        )

        if result.success:
            return result.x
        else:
            logger.warning("Optimization did not converge, using equal weights")
            return initial_weights

    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate risk contributions for given weights.

        Args:
            weights: Portfolio weights

        Returns:
            Array of risk contributions
        """
        if self.cov_matrix is None:
            self.calculate_covariance()

        port_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        marginal_contrib = np.dot(self.cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / port_vol

        return risk_contrib


class RiskParitywithCVaR(RiskModel):
    """
    Risk Parity model with Conditional Value at Risk (CVaR) constraint.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        cvar_confidence: float = 0.95
    ):
        """
        Initialize Risk Parity with CVaR model.

        Args:
            returns: DataFrame of asset returns
            cvar_confidence: Confidence level for CVaR
        """
        super().__init__(returns)
        self.cvar_confidence = cvar_confidence

    def calculate_cvar(self, weights: np.ndarray) -> float:
        """
        Calculate Conditional Value at Risk.

        Args:
            weights: Portfolio weights

        Returns:
            CVaR value
        """
        portfolio_returns = np.dot(self.returns.values, weights)
        var_threshold = np.percentile(portfolio_returns, (1 - self.cvar_confidence) * 100)
        cvar = -np.mean(portfolio_returns[portfolio_returns <= var_threshold])

        return cvar

    def calculate_weights(self) -> np.ndarray:
        """
        Calculate risk parity weights with CVaR constraint.

        Returns:
            Array of optimal weights
        """
        if self.cov_matrix is None:
            self.calculate_covariance()

        def objective(weights):
            return self.calculate_cvar(weights)

        n = self.n_assets
        initial_weights = np.ones(n) / n

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n)]
        )

        if result.success:
            return result.x
        else:
            logger.warning("Optimization did not converge, using equal weights")
            return initial_weights


class BlackLitterman(RiskModel):
    """
    Black-Litterman model for expected returns estimation.
    Combines market equilibrium with investor views.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_caps: np.ndarray,
        risk_aversion: float = 2.5,
        view_confidence: float = 0.5
    ):
        """
        Initialize Black-Litterman model.

        Args:
            returns: DataFrame of asset returns
            market_caps: Array of market capitalizations
            risk_aversion: Risk aversion parameter
            view_confidence: Confidence in views
        """
        super().__init__(returns)
        self.market_caps = market_caps
        self.risk_aversion = risk_aversion
        self.view_confidence = view_confidence
        self.pi = None
        self.omega = None
        self.expected_returns = None

    def calculate_market_implied_returns(self) -> np.ndarray:
        """
        Calculate market equilibrium returns.

        Returns:
            Array of implied returns
        """
        if self.cov_matrix is None:
            self.calculate_covariance()

        weights = self.market_caps / np.sum(self.market_caps)
        self.pi = self.risk_aversion * np.dot(self.cov_matrix, weights)

        return self.pi

    def update_with_views(
        self,
        view_matrix: np.ndarray,
        view_returns: np.ndarray
    ) -> np.ndarray:
        """
        Update expected returns with investor views.

        Args:
            view_matrix: Matrix of view mappings
            view_returns: Array of view returns

        Returns:
            Updated expected returns
        """
        if self.pi is None:
            self.calculate_market_implied_returns()

        self.omega = np.diag(np.diag(view_matrix @ self.cov_matrix @ view_matrix.T))

        view_adj = np.linalg.inv(
            np.linalg.inv(self.omega) + view_matrix.T @ np.linalg.inv(self.cov_matrix) @ view_matrix
        ) @ (
            np.linalg.inv(self.omega) @ view_returns +
            np.linalg.inv(self.cov_matrix) @ self.pi
        )

        self.expected_returns = self.view_confidence * view_adj + (1 - self.view_confidence) * self.pi

        return self.expected_returns

    def calculate_weights(self) -> np.ndarray:
        """
        Calculate optimal weights based on expected returns.

        Returns:
            Array of optimal weights
        """
        if self.expected_returns is None:
            if self.pi is None:
                self.calculate_market_implied_returns()
            self.expected_returns = self.pi

        if self.cov_matrix is None:
            self.calculate_covariance()

        weights = np.linalg.solve(
            self.risk_aversion * self.cov_matrix,
            self.expected_returns
        )

        weights = weights / np.sum(weights)

        return weights


class RiskModelFactory:
    """
    Factory for creating risk model instances.
    """

    @staticmethod
    def create_model(
        model_type: str,
        returns: pd.DataFrame,
        **kwargs
    ) -> RiskModel:
        """
        Create a risk model instance.

        Args:
            model_type: Type of risk model ('rp', 'cvar_rp', 'black_litterman')
            returns: DataFrame of asset returns
            **kwargs: Additional model-specific parameters

        Returns:
            RiskModel instance
        """
        model_type = model_type.lower()

        if model_type == 'rp':
            return StandardRiskParity(returns)
        elif model_type == 'cvar_rp':
            return RiskParitywithCVaR(returns, **kwargs)
        elif model_type == 'black_litterman':
            return BlackLitterman(returns, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def calculate_diversification_ratio(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """
    Calculate diversification ratio.

    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix

    Returns:
        Diversification ratio
    """
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))

    return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0


def calculate_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate risk contributions for each asset.

    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix

    Returns:
        Array of risk contributions
    """
    port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / port_vol

    return risk_contrib


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2017-01-01', periods=100, freq='D')
    assets = ['Asset_A', 'Asset_B', 'Asset_C']

    returns_data = np.random.randn(100, 3) * 0.02
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)

    rp_model = StandardRiskParity(returns)
    weights = rp_model.calculate_weights()

    print("Risk Parity Weights:")
    for asset, weight in zip(assets, weights):
        print(f"  {asset}: {weight:.4f}")

    risk_contrib = rp_model.calculate_risk_contributions(weights)
    print("\nRisk Contributions:")
    for asset, rc in zip(assets, risk_contrib):
        print(f"  {asset}: {rc:.4f}")