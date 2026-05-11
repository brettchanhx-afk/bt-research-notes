import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .config import (
    ASSETS_CONFIG,
    FACTOR_CONFIG,
    REGRESSION_CONFIG,
)

warnings.filterwarnings("ignore")


class FactorExposure:
    def __init__(
        self,
        window_years: int = 5,
        half_life_years: int = 1,
        min_window_years: int = 3,
    ):
        self.window_years = window_years
        self.half_life_years = half_life_years
        self.min_window_years = min_window_years

    def calculate_exponential_weights(
        self, n_periods: int, half_life_periods: int
    ) -> np.ndarray:
        decay_rate = np.log(2) / half_life_periods
        t = np.arange(n_periods)
        weights = np.exp(-decay_rate * (n_periods - 1 - t))
        return weights / weights.sum()

    def calculate_factor_exposure_single_period(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
        use_weights: bool = True,
    ) -> np.ndarray:
        aligned_asset, aligned_factors = asset_returns.align(
            factor_returns, join="inner"
        )

        if len(aligned_asset) < 36:
            return np.full(factor_returns.shape[1], np.nan)

        valid_mask = ~(aligned_asset.isna() | aligned_factors.isna().any(axis=1))
        if valid_mask.sum() < 36:
            return np.full(factor_returns.shape[1], np.nan)

        clean_asset = aligned_asset[valid_mask]
        clean_factors = aligned_factors[valid_mask]

        if use_weights and len(clean_asset) > 0:
            n_periods = len(clean_asset)
            half_life_months = self.half_life_years * 12
            weights = self.calculate_exponential_weights(n_periods, half_life_months)
        else:
            weights = np.ones(len(clean_asset)) / len(clean_asset)

        scaler = StandardScaler()
        scaled_factors = scaler.fit_transform(clean_factors)

        model = LinearRegression()
        try:
            model.fit(scaled_factors, clean_asset, sample_weight=weights)
            exposures = model.coef_
        except Exception:
            model = LinearRegression()
            model.fit(scaled_factors, clean_asset)
            exposures = model.coef_

        return exposures

    def calculate_factor_exposure_series(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        if rebalance_dates is None:
            rebalance_dates = factor_returns.resample("M").last().index

        exposure_series = {}

        for date in rebalance_dates:
            window_end = date
            window_start = date - pd.DateOffset(years=self.window_years)

            window_factors = factor_returns[
                (factor_returns.index >= window_start) & (factor_returns.index <= window_end)
            ]
            window_asset = asset_returns[
                (asset_returns.index >= window_start) & (asset_returns.index <= window_end)
            ]

            if len(window_factors) < self.min_window_years * 12:
                continue

            exposure = self.calculate_factor_exposure_single_period(
                window_asset, window_factors
            )

            if not np.any(np.isnan(exposure)):
                exposure_series[date] = exposure

        if exposure_series:
            exposure_df = pd.DataFrame(
                exposure_series,
                index=factor_returns.columns
            ).T
            return exposure_df
        else:
            return pd.DataFrame()

    def calculate_all_exposures(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, pd.DataFrame]:
        all_exposures = {}

        for asset_name in asset_returns.columns:
            asset_ret = asset_returns[asset_name]

            exposure_series = self.calculate_factor_exposure_series(
                asset_ret, factor_returns, rebalance_dates
            )

            if len(exposure_series) > 0:
                all_exposures[asset_name] = exposure_series

        return all_exposures

    def get_latest_exposure_matrix(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if date is None:
            date = factor_returns.index.max()

        exposures = self.calculate_all_exposures(
            asset_returns, factor_returns,
            rebalance_dates=pd.DatetimeIndex([date])
        )

        if not exposures:
            return pd.DataFrame()

        latest_exposures = {}
        for asset_name, exposure_df in exposures.items():
            if len(exposure_df) > 0:
                latest_exposures[asset_name] = exposure_df.iloc[-1]

        if latest_exposures:
            exposure_matrix = pd.DataFrame(latest_exposures).T
            exposure_matrix.columns = factor_returns.columns
            return exposure_matrix
        else:
            return pd.DataFrame()

    def calculate_historical_exposure(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        rebalance_freq: str = "M",
    ) -> pd.DataFrame:
        rebalance_dates = factor_returns.resample(rebalance_freq).last().index

        all_exposures = self.calculate_all_exposures(
            asset_returns, factor_returns, rebalance_dates
        )

        if not all_exposures:
            return pd.DataFrame()

        combined_exposures = []
        for asset_name, exposure_df in all_exposures.items():
            temp_df = exposure_df.copy()
            temp_df["asset"] = asset_name
            combined_exposures.append(temp_df)

        if combined_exposures:
            return pd.concat(combined_exposures)
        else:
            return pd.DataFrame()

    def calculate_r_squared(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
        exposure_vector: np.ndarray,
    ) -> float:
        aligned_asset, aligned_factors = asset_returns.align(
            factor_returns, join="inner"
        )

        valid_mask = ~(aligned_asset.isna() | aligned_factors.isna().any(axis=1))
        if valid_mask.sum() < 36:
            return np.nan

        clean_asset = aligned_asset[valid_mask]
        clean_factors = aligned_factors[valid_mask]

        predicted = clean_factors @ exposure_vector
        ss_res = np.sum((clean_asset - predicted) ** 2)
        ss_tot = np.sum((clean_asset - clean_asset.mean()) ** 2)

        if ss_tot == 0:
            return 0.0

        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    def get_exposure_matrix_with_r_squared(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        exposure_matrix = self.get_latest_exposure_matrix(
            asset_returns, factor_returns, date
        )

        if exposure_matrix.empty:
            return pd.DataFrame()

        r_squared_dict = {}
        for asset_name in exposure_matrix.index:
            asset_ret = asset_returns[asset_name]
            exposure = exposure_matrix.loc[asset_name].values

            r_squared = self.calculate_r_squared(asset_ret, factor_returns, exposure)
            r_squared_dict[asset_name] = r_squared

        exposure_matrix["R_Square"] = pd.Series(r_squared_dict)

        return exposure_matrix
