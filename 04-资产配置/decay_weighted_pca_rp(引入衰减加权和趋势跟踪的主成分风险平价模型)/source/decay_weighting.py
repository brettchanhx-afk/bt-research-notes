"""
Decay weighting module for expected risk estimation.
Implements exponential decay weighting for volatility and correlation estimation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict

from .config import DECAY_WEIGHTING_PARAMS

logger = logging.getLogger(__name__)


class DecayWeighting:
    """
    Decay weighting for risk estimation.
    Uses exponential decay to give more weight to recent observations.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        volatility_half_life: int = 30,
        correlation_half_life: int = 60
    ):
        """
        Initialize DecayWeighting with returns data.

        Args:
            returns: DataFrame of asset returns
            volatility_half_life: Half-life for volatility decay (days)
            correlation_half_life: Half-life for correlation decay (days)
        """
        self.returns = returns
        self.volatility_half_life = volatility_half_life
        self.correlation_half_life = correlation_half_life

        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)

        self._volatility_weights = None
        self._correlation_weights = None

    def generate_decay_weights(self, n_periods: int, half_life: int) -> np.ndarray:
        """
        Generate exponentially decaying weights.

        Args:
            n_periods: Number of periods
            half_life: Half-life for decay

        Returns:
            Array of decay weights
        """
        decay_rate = np.log(2) / half_life
        time_indices = np.arange(n_periods - 1, -1, -1)
        weights = np.exp(-decay_rate * time_indices)

        return weights / weights.sum()

    def calculate_volatility_weights(self) -> np.ndarray:
        """
        Calculate weights for volatility estimation.

        Returns:
            Array of volatility weights
        """
        n_periods = len(self.returns)
        self._volatility_weights = self.generate_decay_weights(
            n_periods,
            self.volatility_half_life
        )

        return self._volatility_weights

    def calculate_correlation_weights(self) -> np.ndarray:
        """
        Calculate weights for correlation estimation.

        Returns:
            Array of correlation weights
        """
        n_periods = len(self.returns)
        self._correlation_weights = self.generate_decay_weights(
            n_periods,
            self.correlation_half_life
        )

        return self._correlation_weights

    def calculate_decay_weighted_volatility(
        self,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate decay-weighted volatility for each asset.

        Args:
            lookback: Number of periods to use (None for all)

        Returns:
            Series of weighted volatilities
        """
        if self._volatility_weights is None:
            self.calculate_volatility_weights()

        if lookback is not None:
            returns_subset = self.returns.tail(lookback)
            weights_subset = self._volatility_weights[-lookback:]
            weights_subset = weights_subset / weights_subset.sum()
        else:
            returns_subset = self.returns
            weights_subset = self._volatility_weights

        squared_returns = returns_subset ** 2
        weighted_variance = (squared_returns.T * weights_subset).sum()

        weighted_volatility = np.sqrt(weighted_variance * 252)

        return weighted_volatility

    def calculate_decay_weighted_covariance(
        self,
        lookback: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate decay-weighted covariance matrix.

        Args:
            lookback: Number of periods to use (None for all)

        Returns:
            Covariance matrix
        """
        if self._correlation_weights is None:
            self.calculate_correlation_weights()

        if lookback is not None:
            returns_subset = self.returns.tail(lookback)
            weights_subset = self._correlation_weights[-lookback:]
            weights_subset = weights_subset / weights_subset.sum()
        else:
            returns_subset = self.returns
            weights_subset = self._correlation_weights

        centered_returns = returns_subset - returns_subset.mean()
        weighted_cov = np.zeros((self.n_assets, self.n_assets))

        for i in range(self.n_assets):
            for j in range(self.n_assets):
                weighted_cov[i, j] = np.sum(
                    weights_subset *
                    centered_returns.iloc[:, i].values *
                    centered_returns.iloc[:, j].values
                )

        return weighted_cov * 252

    def calculate_decay_weighted_correlation(
        self,
        lookback: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate decay-weighted correlation matrix.

        Args:
            lookback: Number of periods to use (None for all)

        Returns:
            Correlation matrix
        """
        cov_matrix = self.calculate_decay_weighted_covariance(lookback)

        std_vec = np.sqrt(np.diag(cov_matrix))
        std_matrix = np.outer(std_vec, std_vec)

        correlation_matrix = cov_matrix / std_matrix

        np.fill_diagonal(correlation_matrix, 1.0)

        return correlation_matrix

    def get_expected_risk(self) -> Dict[str, float]:
        """
        Get expected risk estimates using decay weighting.

        Returns:
            Dict with volatility and correlation estimates
        """
        volatility = self.calculate_decay_weighted_volatility()
        correlation = self.calculate_decay_weighted_correlation()

        return {
            'volatility': volatility.to_dict(),
            'correlation': pd.DataFrame(
                correlation,
                index=self.asset_names,
                columns=self.asset_names
            )
        }


class AdaptiveDecayWeighting(DecayWeighting):
    """
    Adaptive decay weighting that adjusts half-life based on market conditions.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        base_volatility_half_life: int = 30,
        base_correlation_half_life: int = 60,
        volatility_scaling: float = 1.0,
        correlation_scaling: float = 1.0
    ):
        """
        Initialize AdaptiveDecayWeighting.

        Args:
            returns: DataFrame of asset returns
            base_volatility_half_life: Base half-life for volatility
            base_correlation_half_life: Base half-life for correlation
            volatility_scaling: Scaling factor for volatility half-life
            correlation_scaling: Scaling factor for correlation half-life
        """
        super().__init__(
            returns,
            base_volatility_half_life,
            base_correlation_half_life
        )

        self.base_volatility_half_life = base_volatility_half_life
        self.base_correlation_half_life = base_correlation_half_life
        self.volatility_scaling = volatility_scaling
        self.correlation_scaling = correlation_scaling

    def calculate_adaptive_volatility_half_life(self) -> int:
        """
        Calculate adaptive half-life for volatility based on volatility regime.

        Returns:
            Adaptive half-life
        """
        recent_vol = self.returns.tail(20).std()
        long_vol = self.returns.tail(60).std()

        vol_ratio = recent_vol / long_vol

        avg_ratio = np.mean(vol_ratio)

        if avg_ratio > 1.5:
            scaling = 0.5
        elif avg_ratio > 1.2:
            scaling = 0.75
        elif avg_ratio < 0.5:
            scaling = 1.5
        elif avg_ratio < 0.8:
            scaling = 1.25
        else:
            scaling = 1.0

        adaptive_half_life = int(
            self.base_volatility_half_life * scaling * self.volatility_scaling
        )

        return max(5, adaptive_half_life)

    def calculate_adaptive_correlation_half_life(self) -> int:
        """
        Calculate adaptive half-life for correlation based on correlation stability.

        Returns:
            Adaptive half-life
        """
        recent_corr = self.returns.tail(20).corr()
        long_corr = self.returns.tail(60).corr()

        corr_diff = np.abs(recent_corr - long_corr).mean()

        if corr_diff > 0.3:
            scaling = 0.5
        elif corr_diff > 0.2:
            scaling = 0.75
        elif corr_diff < 0.1:
            scaling = 1.5
        elif corr_diff < 0.15:
            scaling = 1.25
        else:
            scaling = 1.0

        adaptive_half_life = int(
            self.base_correlation_half_life * scaling * self.correlation_scaling
        )

        return max(10, adaptive_half_life)

    def calculate_adaptive_volatility_weights(self) -> np.ndarray:
        """
        Calculate adaptive weights for volatility estimation.

        Returns:
            Array of adaptive volatility weights
        """
        adaptive_half_life = self.calculate_adaptive_volatility_half_life()

        n_periods = len(self.returns)
        return self.generate_decay_weights(n_periods, adaptive_half_life)

    def calculate_adaptive_correlation_weights(self) -> np.ndarray:
        """
        Calculate adaptive weights for correlation estimation.

        Returns:
            Array of adaptive correlation weights
        """
        adaptive_half_life = self.calculate_adaptive_correlation_half_life()

        n_periods = len(self.returns)
        return self.generate_decay_weights(n_periods, adaptive_half_life)


def create_decay_weighting(
    returns: pd.DataFrame,
    params: Optional[Dict] = None,
    adaptive: bool = False
) -> DecayWeighting:
    """
    Factory function to create DecayWeighting instance.

    Args:
        returns: DataFrame of asset returns
        params: Dict with half-life parameters
        adaptive: Whether to use adaptive decay weighting

    Returns:
        DecayWeighting instance
    """
    if params is None:
        params = DECAY_WEIGHTING_PARAMS

    if adaptive:
        return AdaptiveDecayWeighting(
            returns,
            volatility_half_life=params.get('volatility_half_life', 30),
            correlation_half_life=params.get('correlation_half_life', 60)
        )
    else:
        return DecayWeighting(
            returns,
            volatility_half_life=params.get('volatility_half_life', 30),
            correlation_half_life=params.get('correlation_half_life', 60)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2017-01-01', periods=100, freq='D')
    assets = ['Asset_A', 'Asset_B', 'Asset_C']

    returns_data = np.random.randn(100, 3) * 0.02
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)

    dw = DecayWeighting(returns, volatility_half_life=30, correlation_half_life=60)

    volatility = dw.calculate_decay_weighted_volatility()
    print("Decay-Weighted Volatility:")
    print(volatility)

    correlation = dw.calculate_decay_weighted_correlation()
    print("\nDecay-Weighted Correlation Matrix:")
    print(pd.DataFrame(
        correlation,
        index=assets,
        columns=assets
    ))