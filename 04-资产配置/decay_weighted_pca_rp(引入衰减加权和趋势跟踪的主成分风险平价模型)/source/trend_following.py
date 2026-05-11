"""
Trend following module for expected return estimation.
Implements various trend following methods to estimate expected asset走势.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Union

from .config import TF_PARAMS

logger = logging.getLogger(__name__)


class TrendFollowing:
    """
    Trend following base class for expected return estimation.
    Assumes market trends persist - buy assets in uptrend, sell/avoid assets in downtrend.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        short_ma: int = 5,
        long_ma: int = 20,
        lookback_period: int = 20
    ):
        """
        Initialize TrendFollowing with price data.

        Args:
            prices: DataFrame of asset prices
            short_ma: Short moving average period
            long_ma: Long moving average period
            lookback_period: Period for trend calculation
        """
        self.prices = prices
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.lookback_period = lookback_period

        self.asset_names = prices.columns.tolist()
        self.n_assets = len(self.asset_names)

        self._signals = None
        self._expected_returns = None

    def calculate_moving_averages(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate short and long moving averages.

        Returns:
            Tuple of (short_ma, long_ma) DataFrames
        """
        short_ma = self.prices.rolling(window=self.short_ma).mean()
        long_ma = self.prices.rolling(window=self.long_ma).mean()

        return short_ma, long_ma

    def calculate_ma_crossover_signal(self) -> pd.DataFrame:
        """
        Calculate moving average crossover signal.
        Signal = 1 if short_ma > long_ma (bullish), -1 if short_ma < long_ma (bearish), 0 otherwise.

        Returns:
            DataFrame of signals
        """
        short_ma, long_ma = self.calculate_moving_averages()

        signal = pd.DataFrame(0, index=self.prices.index, columns=self.asset_names)

        signal[short_ma > long_ma] = 1
        signal[short_ma < long_ma] = -1

        self._signals = signal

        return signal

    def calculate_momentum_signal(self) -> pd.DataFrame:
        """
        Calculate momentum-based signal.
        Signal based on price change over lookback period.

        Returns:
            DataFrame of signals
        """
        signal = pd.DataFrame(0, index=self.prices.index, columns=self.asset_names)

        price_change = self.prices.pct_change(self.lookback_period)

        signal[price_change > 0.02] = 1
        signal[price_change < -0.02] = -1

        self._signals = signal

        return signal

    def calculate_trend_strength(self) -> pd.DataFrame:
        """
        Calculate trend strength indicator.
        Ratio of current price to moving average.

        Returns:
            DataFrame of trend strengths
        """
        short_ma, long_ma = self.calculate_moving_averages()

        avg_ma = (short_ma + long_ma) / 2

        trend_strength = (self.prices - avg_ma) / avg_ma

        return trend_strength

    def get_expected_returns(
        self,
        method: str = 'ma_crossover'
    ) -> pd.DataFrame:
        """
        Get expected returns based on trend following.

        Args:
            method: Method to use ('ma_crossover', 'momentum', 'trend_strength')

        Returns:
            DataFrame of expected returns
        """
        if method == 'ma_crossover':
            signals = self.calculate_ma_crossover_signal()
        elif method == 'momentum':
            signals = self.calculate_momentum_signal()
        elif method == 'trend_strength':
            trend_strength = self.calculate_trend_strength()
            signals = np.sign(trend_strength)
        else:
            raise ValueError(f"Unknown method: {method}")

        expected_returns = signals * 0.01

        self._expected_returns = expected_returns

        return expected_returns


class DualMovingAverageCrossover(TrendFollowing):
    """
    Dual Moving Average Crossover (DMAC) trend following strategy.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        short_ma: int = 5,
        long_ma: int = 20,
        lookback_period: int = 20
    ):
        """
        Initialize DMAC strategy.

        Args:
            prices: DataFrame of asset prices
            short_ma: Short MA period
            long_ma: Long MA period
            lookback_period: Lookback period for signals
        """
        super().__init__(prices, short_ma, long_ma, lookback_period)

    def generate_position_signals(self) -> pd.DataFrame:
        """
        Generate position signals based on DMAC.
        1 = Long, 0 = Neutral, -1 = Short

        Returns:
            DataFrame of position signals
        """
        return self.calculate_ma_crossover_signal()

    def calculate_position_size(self) -> pd.DataFrame:
        """
        Calculate position sizes based on trend strength.

        Returns:
            DataFrame of position sizes
        """
        signals = self.generate_position_signals()
        trend_strength = self.calculate_trend_strength()

        position_size = signals * np.abs(trend_strength)
        position_size = position_size.clip(-1, 1)

        return position_size


class MovingAverageBands(TrendFollowing):
    """
    Moving Average Bands trend following strategy.
    Uses multiple moving averages to determine trend regime.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        short_ma: int = 5,
        medium_ma: int = 20,
        long_ma: int = 60,
        lookback_period: int = 20
    ):
        """
        Initialize MAB strategy.

        Args:
            prices: DataFrame of asset prices
            short_ma: Short MA period
            medium_ma: Medium MA period
            long_ma: Long MA period
            lookback_period: Lookback period
        """
        super().__init__(prices, short_ma, long_ma, lookback_period)
        self.medium_ma = medium_ma

    def calculate_ma_bands(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate three moving average bands.

        Returns:
            Tuple of (short_ma, medium_ma, long_ma) DataFrames
        """
        short = self.prices.rolling(window=self.short_ma).mean()
        medium = self.prices.rolling(window=self.medium_ma).mean()
        long = self.prices.rolling(window=self.long_ma).mean()

        return short, medium, long

    def generate_trend_regime_signals(self) -> pd.DataFrame:
        """
        Generate trend regime signals.
        2 = Strong Uptrend (prices > short > medium > long)
        1 = Weak Uptrend (prices > long)
        0 = Neutral
        -1 = Weak Downtrend (prices < long)
        -2 = Strong Downtrend (prices < short < medium < long)

        Returns:
            DataFrame of trend regime signals
        """
        short, medium, long = self.calculate_ma_bands()

        signal = pd.DataFrame(0, index=self.prices.index, columns=self.asset_names)

        strong_uptrend = (self.prices > short) & (short > medium) & (medium > long)
        weak_uptrend = self.prices > long
        strong_downtrend = (self.prices < short) & (short < medium) & (medium < long)
        weak_downtrend = self.prices < long

        signal[strong_uptrend] = 2
        signal[weak_uptrend & ~strong_uptrend] = 1
        signal[weak_downtrend & ~strong_downtrend] = -1
        signal[strong_downtrend] = -2

        return signal

    def calculate_position_from_regime(
        self,
        long_allocation: float = 0.5,
        neutral_allocation: float = 0.0,
        short_allocation: float = -0.5
    ) -> pd.DataFrame:
        """
        Calculate position sizes from trend regime.

        Args:
            long_allocation: Max allocation for long positions
            neutral_allocation: Allocation for neutral regime
            short_allocation: Max allocation for short positions

        Returns:
            DataFrame of position sizes
        """
        regime_signals = self.generate_trend_regime_signals()

        position = pd.DataFrame(0, index=self.prices.index, columns=self.asset_names)

        position[regime_signals == 2] = long_allocation
        position[regime_signals == 1] = long_allocation * 0.5
        position[regime_signals == -1] = short_allocation * 0.5
        position[regime_signals == -2] = short_allocation

        return position


class RSITrendFollowing(TrendFollowing):
    """
    RSI-based trend following strategy.
    Combines RSI overbought/oversold with trend direction.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        rsi_period: int = 14,
        short_ma: int = 5,
        long_ma: int = 20,
        lookback_period: int = 20
    ):
        """
        Initialize RSI trend following strategy.

        Args:
            prices: DataFrame of asset prices
            rsi_period: RSI calculation period
            short_ma: Short MA period
            long_ma: Long MA period
            lookback_period: Lookback period
        """
        super().__init__(prices, short_ma, long_ma, lookback_period)
        self.rsi_period = rsi_period

    def calculate_rsi(self) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.

        Returns:
            DataFrame of RSI values
        """
        delta = self.prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_rsi_signals(self) -> pd.DataFrame:
        """
        Generate signals based on RSI combined with trend.

        Returns:
            DataFrame of signals
        """
        rsi = self.calculate_rsi()
        ma_signals = self.calculate_ma_crossover_signal()

        signal = pd.DataFrame(0, index=self.prices.index, columns=self.asset_names)

        oversold_bullish = (rsi < 30) & (ma_signals == 1)
        overbought_bearish = (rsi > 70) & (ma_signals == -1)

        signal[oversold_bullish] = 1
        signal[overbought_bearish] = -1
        signal[(ma_signals == 1) & (rsi >= 30) & (rsi <= 70)] = 0.5
        signal[(ma_signals == -1) & (rsi >= 30) & (rsi <= 70)] = -0.5

        return signal


def create_trend_following(
    prices: pd.DataFrame,
    method: str = 'ma_crossover',
    params: Optional[Dict] = None
) -> TrendFollowing:
    """
    Factory function to create TrendFollowing instance.

    Args:
        prices: DataFrame of asset prices
        method: Trend following method
        params: Dict with method parameters

    Returns:
        TrendFollowing instance
    """
    if params is None:
        params = TF_PARAMS

    if method == 'ma_crossover':
        return TrendFollowing(
            prices,
            short_ma=params.get('short_ma', 5),
            long_ma=params.get('long_ma', 20),
            lookback_period=params.get('lookback_period', 20)
        )
    elif method == 'dmac':
        return DualMovingAverageCrossover(
            prices,
            short_ma=params.get('short_ma', 5),
            long_ma=params.get('long_ma', 20),
            lookback_period=params.get('lookback_period', 20)
        )
    elif method == 'mab':
        return MovingAverageBands(
            prices,
            short_ma=params.get('short_ma', 5),
            medium_ma=params.get('medium_ma', 20),
            long_ma=params.get('long_ma', 60),
            lookback_period=params.get('lookback_period', 20)
        )
    elif method == 'rsi':
        return RSITrendFollowing(
            prices,
            rsi_period=params.get('rsi_period', 14),
            short_ma=params.get('short_ma', 5),
            long_ma=params.get('long_ma', 20),
            lookback_period=params.get('lookback_period', 20)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2017-01-01', periods=100, freq='D')
    assets = ['Asset_A', 'Asset_B', 'Asset_C']

    prices_data = 100 + np.cumsum(np.random.randn(100, 3) * 2, axis=0)
    prices = pd.DataFrame(prices_data, index=dates, columns=assets)

    tf = TrendFollowing(prices, short_ma=5, long_ma=20)

    signals = tf.calculate_ma_crossover_signal()
    print("MA Crossover Signals:")
    print(signals.tail(10))

    expected_returns = tf.get_expected_returns(method='ma_crossover')
    print("\nExpected Returns:")
    print(expected_returns.tail(10))