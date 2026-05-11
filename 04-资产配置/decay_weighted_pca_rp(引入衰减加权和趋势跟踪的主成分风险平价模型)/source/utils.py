"""
Utility functions for the decay-weighted PCA risk parity project.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def setup_logging(log_file=None, level=logging.INFO):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=level, format=log_format)


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns from prices."""
    return prices.pct_change().dropna(how='all')


def calculate_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_correlation_matrix(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate rolling correlation matrix."""
    return returns.rolling(window=window).corr()


def exponential_decay_weights(n_periods: int, half_life: int) -> np.ndarray:
    """Generate exponentially decaying weights."""
    decay_rate = np.log(2) / half_life
    weights = np.exp(-decay_rate * np.arange(n_periods - 1, -1, -1))
    return weights / weights.sum()


def calculate_decay_weighted_covariance(
    returns: pd.DataFrame,
    half_life: int = 30
) -> np.ndarray:
    """
    Calculate covariance matrix with exponential decay weighting.

    Args:
        returns: DataFrame of asset returns
        half_life: Half-life for exponential decay

    Returns:
        Covariance matrix
    """
    n_assets = returns.shape[1]
    n_periods = len(returns)

    weights = exponential_decay_weights(n_periods, half_life)

    centered_returns = returns - returns.mean()
    weighted_returns = centered_returns.multiply(weights, axis=0)

    cov_matrix = np.dot(weighted_returns.T, centered_returns) / weights.sum()

    return cov_matrix


def calculate_decay_weighted_correlation(
    returns: pd.DataFrame,
    half_life: int = 60
) -> np.ndarray:
    """
    Calculate correlation matrix with exponential decay weighting.

    Args:
        returns: DataFrame of asset returns
        half_life: Half-life for exponential decay

    Returns:
        Correlation matrix
    """
    cov_matrix = calculate_decay_weighted_covariance(returns, half_life)

    std_vec = np.sqrt(np.diag(cov_matrix))
    std_matrix = np.outer(std_vec, std_vec)

    corr_matrix = cov_matrix / std_matrix

    return corr_matrix


def validate_data(free_data: pd.DataFrame, risk_data: pd.DataFrame) -> bool:
    """
    Validate that free_data and risk_data have compatible structure.

    Args:
        free_data: Data without risk-free rate adjustment
        risk_data: Data with risk-free rate adjustment

    Returns:
        True if data is valid

    Raises:
        ValueError: If data is not compatible
    """
    if free_data.empty or risk_data.empty:
        raise ValueError("Data cannot be empty")

    if not free_data.index.equals(risk_data.index):
        raise ValueError("Data indices do not match")

    if not free_data.columns.equals(risk_data.columns):
        raise ValueError("Data columns do not match")

    return True


def resample_to_monthly(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to monthly data."""
    return daily_data.resample('M').last()


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 12
) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_calmar_ratio(
    annual_return: float,
    max_drawdown: float
) -> float:
    """Calculate Calmar ratio."""
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0


def format_date(date_str: str) -> str:
    """Format date string to YYYYMMDD."""
    if isinstance(date_str, datetime):
        return date_str.strftime('%Y%m%d')
    return date_str


def get_date_range(start_date: str, end_date: str) -> tuple:
    """Parse and validate date range."""
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    if start >= end:
        raise ValueError("Start date must be before end date")

    return start, end


def create_output_filename(prefix: str, extension: str = 'csv') -> str:
    """Create timestamped output filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"