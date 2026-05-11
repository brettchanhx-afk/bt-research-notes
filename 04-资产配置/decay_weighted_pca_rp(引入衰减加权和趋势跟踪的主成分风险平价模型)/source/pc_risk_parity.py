"""
Principal Components Risk Parity (PCRP) model module.
Implements PCRP with optional decay weighting and trend following enhancements.
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv, eigh
from scipy.optimize import minimize
import logging
from typing import Optional, Tuple, Dict, Union

from .decay_weighting import DecayWeighting, create_decay_weighting
from .trend_following import TrendFollowing, create_trend_following
from .risk_models import calculate_risk_contribution

logger = logging.getLogger(__name__)


class PrincipalComponentsRiskParity:
    """
    Principal Components Risk Parity (PCRP) model.
    Uses PCA to transform correlated assets into uncorrelated principal components,
    then applies risk parity on the principal components.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        use_decay_weighting: bool = False,
        decay_params: Optional[Dict] = None,
        n_components: Optional[int] = None
    ):
        """
        Initialize PCRP model.

        Args:
            returns: DataFrame of asset returns
            use_decay_weighting: Whether to use decay weighting
            decay_params: Parameters for decay weighting
            n_components: Number of principal components to use (None for all)
        """
        self.returns = returns
        self.use_decay_weighting = use_decay_weighting
        self.decay_params = decay_params

        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)
        self.n_components = n_components or self.n_assets

        self.cov_matrix = None
        self.principal_components = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transform_matrix = None
        self.weights = None

        self._prepare_data()

    def _prepare_data(self):
        """Prepare covariance matrix and PCA decomposition."""
        if self.use_decay_weighting:
            decay_weighting = create_decay_weighting(self.returns, self.decay_params)
            self.cov_matrix = decay_weighting.calculate_decay_weighted_covariance()
        else:
            self.cov_matrix = self.returns.cov().values * 252

        self._compute_pca()

    def _compute_pca(self):
        """Compute PCA decomposition of covariance matrix."""
        eigenvalues, eigenvectors = eigh(self.cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues = eigenvalues[:self.n_components]
        self.eigenvectors = eigenvectors[:, :self.n_components]

        self.principal_components = self.returns.values @ self.eigenvectors

        self.transform_matrix = self.eigenvectors

    def calculate_pc_weights(self) -> np.ndarray:
        """
        Calculate weights for principal components using risk parity.

        Returns:
            Array of principal component weights
        """
        pc_cov = np.diag(self.eigenvalues)

        def risk_contribution(weights):
            weighted_cov = np.sum(weights * self.eigenvalues)
            marginal_contrib = self.eigenvalues * weights
            risk_contrib = weights * marginal_contrib / weighted_cov
            target_rc = weighted_cov / self.n_components
            return np.sum((risk_contrib - target_rc) ** 2)

        n = self.n_components
        initial_weights = np.ones(n) / n

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
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
            logger.warning("PC weight optimization did not converge")
            return initial_weights

    def transform_to_asset_weights(
        self,
        pc_weights: np.ndarray
    ) -> np.ndarray:
        """
        Transform principal component weights to asset weights.

        Args:
            pc_weights: Weights for principal components

        Returns:
            Array of asset weights
        """
        asset_weights = np.dot(self.eigenvectors, pc_weights * self.eigenvalues)

        asset_weights = asset_weights / np.sum(np.abs(asset_weights))

        return asset_weights

    def calculate_weights(self) -> np.ndarray:
        """
        Calculate final asset weights.

        Returns:
            Array of optimal asset weights
        """
        pc_weights = self.calculate_pc_weights()

        self.weights = self.transform_to_asset_weights(pc_weights)

        return self.weights

    def get_risk_contributions(self) -> np.ndarray:
        """
        Get risk contributions for each asset.

        Returns:
            Array of risk contributions
        """
        if self.weights is None:
            self.calculate_weights()

        return calculate_risk_contribution(self.weights, self.cov_matrix)

    def filter_assets(
        self,
        correlation_threshold: float = 0.95,
        min_explained_variance: float = 0.95
    ) -> np.ndarray:
        """
        Filter out highly correlated assets based on PCA.

        Args:
            correlation_threshold: Threshold for correlation filtering
            min_explained_variance: Minimum explained variance to retain

        Returns:
            Array of asset indices to keep
        """
        cum_variance = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
        n_keep = np.searchsorted(cum_variance, min_explained_variance) + 1

        if n_keep < self.n_assets:
            logger.info(f"Reducing from {self.n_assets} to {n_keep} components "
                       f"to explain {min_explained_variance:.1%} variance")

        return np.arange(min(n_keep, self.n_assets))


class PCRPwithTrendFollowing(PrincipalComponentsRiskParity):
    """
    PCRP model enhanced with trend following for expected return estimation.
    Adjusts weights based on trend signals.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        use_decay_weighting: bool = False,
        decay_params: Optional[Dict] = None,
        n_components: Optional[int] = None,
        trend_method: str = 'ma_crossover',
        trend_params: Optional[Dict] = None
    ):
        """
        Initialize PCRP with trend following.

        Args:
            returns: DataFrame of asset returns
            prices: DataFrame of asset prices
            use_decay_weighting: Whether to use decay weighting
            decay_params: Parameters for decay weighting
            n_components: Number of principal components
            trend_method: Method for trend following
            trend_params: Parameters for trend following
        """
        self.prices = prices
        self.trend_method = trend_method
        self.trend_params = trend_params

        super().__init__(
            returns,
            use_decay_weighting,
            decay_params,
            n_components
        )

        self.trend_signals = None
        self.adjusted_weights = None

    def calculate_trend_signals(self) -> pd.DataFrame:
        """
        Calculate trend signals using trend following.

        Returns:
            DataFrame of trend signals
        """
        trend_follower = create_trend_following(
            self.prices,
            method=self.trend_method,
            params=self.trend_params
        )

        self.trend_signals = trend_follower.get_expected_returns()

        return self.trend_signals

    def calculate_adjusted_weights(
        self,
        trend_weight: float = 0.3
    ) -> np.ndarray:
        """
        Calculate weights adjusted by trend signals.

        Args:
            trend_weight: Weight given to trend signals

        Returns:
            Array of adjusted weights
        """
        if self.weights is None:
            self.calculate_weights()

        if self.trend_signals is None:
            self.calculate_trend_signals()

        latest_signal = self.trend_signals.iloc[-1].values

        base_weights = self.weights.copy()
        trend_signal = latest_signal * trend_weight

        adjusted = base_weights * (1 + trend_signal)

        adjusted = np.clip(adjusted, 0, 1)
        adjusted = adjusted / np.sum(adjusted)

        self.adjusted_weights = adjusted

        return adjusted


class PCRPwithDecayWeighting(PrincipalComponentsRiskParity):
    """
    PCRP model with decay weighting for covariance estimation.
    Uses decay weighting for both volatility and correlation estimation.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        decay_params: Optional[Dict] = None,
        n_components: Optional[int] = None,
        adaptive: bool = False
    ):
        """
        Initialize PCRP with decay weighting.

        Args:
            returns: DataFrame of asset returns
            decay_params: Parameters for decay weighting
            n_components: Number of principal components
            adaptive: Whether to use adaptive decay weighting
        """
        self.adaptive = adaptive

        super().__init__(
            returns,
            use_decay_weighting=True,
            decay_params=decay_params,
            n_components=n_components
        )


class WDCPCRP:
    """
    Weighted Decay-weighted Correlation PCRP (WDC-PCRP) model.
    Combines decay weighting for covariance estimation with trend following.
    This is the full model from the research report.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        decay_params: Optional[Dict] = None,
        trend_params: Optional[Dict] = None,
        n_components: Optional[int] = None
    ):
        """
        Initialize WDC-PCRP model.

        Args:
            returns: DataFrame of asset returns
            prices: DataFrame of asset prices
            decay_params: Parameters for decay weighting
            trend_params: Parameters for trend following
            n_components: Number of principal components
        """
        self.returns = returns
        self.prices = prices
        self.decay_params = decay_params
        self.trend_params = trend_params
        self.n_components = n_components or returns.shape[1]

        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)

        self.cov_matrix = None
        self.weights = None
        self.expected_returns = None

        self._prepare_components()

    def _prepare_components(self):
        """Prepare decay weighting and trend following components."""
        self.decay_weighting = create_decay_weighting(
            self.returns,
            params=self.decay_params,
            adaptive=False
        )

        self.cov_matrix = self.decay_weighting.calculate_decay_weighted_covariance()

        self.trend_following = create_trend_following(
            self.prices,
            method='ma_crossover',
            params=self.trend_params
        )

    def calculate_expected_risk(self) -> np.ndarray:
        """
        Calculate expected risk using decay-weighted covariance.

        Returns:
            Covariance matrix
        """
        return self.cov_matrix

    def calculate_expected_return(self) -> np.ndarray:
        """
        Calculate expected returns using trend following.

        Returns:
            Array of expected returns
        """
        self.expected_returns = self.trend_following.get_expected_returns().iloc[-1].values

        return self.expected_returns

    def calculate_weights(self) -> np.ndarray:
        """
        Calculate portfolio weights using WDC-PCRP model.

        Returns:
            Array of optimal weights
        """
        eigenvalues, eigenvectors = eigh(self.cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]

        def risk_contribution(weights):
            weighted_cov = np.sum(weights * eigenvalues)
            marginal_contrib = eigenvalues * weights
            risk_contrib = weights * marginal_contrib / weighted_cov
            target_rc = weighted_cov / self.n_components
            return np.sum((risk_contrib - target_rc) ** 2)

        n = self.n_components
        initial_weights = np.ones(n) / n

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        result = minimize(
            risk_contribution,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n)]
        )

        pc_weights = result.x if result.success else initial_weights

        asset_weights = np.dot(eigenvectors, pc_weights * eigenvalues)

        expected_return = self.calculate_expected_return()
        if expected_return is not None:
            return_direction = np.sign(expected_return)
            return_direction[return_direction == 0] = 1
            asset_weights = asset_weights * return_direction

        asset_weights = np.clip(asset_weights, 0, 1)
        asset_weights = asset_weights / np.sum(asset_weights)

        self.weights = asset_weights

        return self.weights

    def get_portfolio_metrics(self) -> Dict:
        """
        Get portfolio risk and return metrics.

        Returns:
            Dict with portfolio metrics
        """
        if self.weights is None:
            self.calculate_weights()

        portfolio_vol = np.sqrt(np.dot(self.weights, np.dot(self.cov_matrix, self.weights)))

        if self.expected_returns is not None:
            portfolio_return = np.dot(self.weights, self.expected_returns)
        else:
            portfolio_return = 0

        risk_contrib = calculate_risk_contribution(self.weights, self.cov_matrix)

        return {
            'weights': self.weights,
            'portfolio_volatility': portfolio_vol,
            'portfolio_return': portfolio_return,
            'risk_contributions': risk_contrib,
            'asset_names': self.asset_names
        }


def create_pcrp_model(
    returns: pd.DataFrame,
    model_type: str = 'standard',
    prices: Optional[pd.DataFrame] = None,
    **kwargs
) -> Union[PrincipalComponentsRiskParity, PCRPwithTrendFollowing, WDCPCRP]:
    """
    Factory function to create PCRP model instances.

    Args:
        returns: DataFrame of asset returns
        model_type: Type of PCRP model
        prices: DataFrame of asset prices (required for trend following models)
        **kwargs: Additional model parameters

    Returns:
        PCRP model instance
    """
    model_type = model_type.lower()

    if model_type == 'standard':
        return PrincipalComponentsRiskParity(returns, **kwargs)
    elif model_type == 'decay':
        return PCRPwithDecayWeighting(returns, **kwargs)
    elif model_type == 'tf':
        if prices is None:
            raise ValueError("prices required for trend following models")
        return PCRPwithTrendFollowing(returns, prices, **kwargs)
    elif model_type == 'wdc':
        if prices is None:
            raise ValueError("prices required for WDC-PCRP model")
        return WDCPCRP(returns, prices, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2017-01-01', periods=100, freq='D')
    assets = ['Asset_A', 'Asset_B', 'Asset_C']

    returns_data = np.random.randn(100, 3) * 0.02
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)

    prices_data = 100 + np.cumsum(returns_data, axis=0)
    prices = pd.DataFrame(prices_data, index=dates, columns=assets)

    pcrp = PrincipalComponentsRiskParity(returns)
    weights = pcrp.calculate_weights()

    print("PCRP Weights:")
    for asset, weight in zip(assets, weights):
        print(f"  {asset}: {weight:.4f}")

    risk_contrib = pcrp.get_risk_contributions()
    print("\nRisk Contributions:")
    for asset, rc in zip(assets, risk_contrib):
        print(f"  {asset}: {rc:.4f}")