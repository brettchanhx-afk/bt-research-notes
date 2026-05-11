"""
Macro factor computation module.
Uses the provided high-frequency macro factor portfolio data
to construct mimicking portfolio factor returns.
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

from source.config import (
    MACRO_FACTOR_COLS,
    FACTOR_NAMES,
    FACTOR_COLS_IN_HIGHFREQ,
    BACKTEST_START,
    BACKTEST_END,
)


def build_mimicking_factors(hf_factors_df):
    """
    Build macro factor returns from high-frequency mimicking portfolio data.

    Parameters:
    -----------
    hf_factors_df : pd.DataFrame
        Daily high-frequency factor mimicking portfolio data

    Returns:
    --------
    factor_returns : pd.DataFrame
        Daily factor returns for each of the 6 macro factors
    """
    factor_returns_dict = {}

    for factor_name, config in MACRO_FACTOR_COLS.items():
        mimicking = config["mimicking"]
        factor_ret = None

        for asset_col, cfg in mimicking.items():
            if asset_col not in hf_factors_df.columns:
                print(f"  WARNING: {asset_col} not found in high-freq data for {factor_name}")
                continue

            series = hf_factors_df[asset_col].dropna()
            if factor_ret is None:
                factor_ret = series * cfg["weight"] * cfg.get("direction", 1)
            else:
                factor_ret = factor_ret + series * cfg["weight"] * cfg.get("direction", 1)

        if factor_ret is not None:
            factor_ret = factor_ret.dropna()
            factor_ret = factor_ret.replace([np.inf, -np.inf], np.nan).dropna()
            factor_returns_dict[factor_name] = factor_ret
            print(f"  {factor_name}: built from mimicking portfolio ({len(factor_ret)} obs)")
        else:
            print(f"  WARNING: {factor_name} - no data available")

    result = pd.DataFrame(factor_returns_dict)
    result = result.sort_index()
    result = result.dropna(how="all")

    start_dt = pd.to_datetime(BACKTEST_START)
    end_dt = pd.to_datetime(BACKTEST_END)
    result = result[(result.index >= start_dt) & (result.index <= end_dt)]

    for fname in FACTOR_NAMES:
        if fname not in result.columns:
            result[fname] = 0.0

    return result[FACTOR_NAMES]


def build_factor_returns_from_prices(asset_prices, hf_factors_df):
    """
    Build factor returns from asset prices using mimicking portfolios.
    First compute asset returns, then combine them to build factors.
    """
    asset_rets = asset_prices.pct_change().dropna()
    asset_rets = asset_rets.replace([np.inf, -np.inf], np.nan)

    factor_returns_dict = {}

    mimicking_from_assets = {
        "增长因子": {
            "南华商品": {"weight": 0.61, "direction": 1},
            "沪深300": {"weight": 0.24, "direction": 1},
            "中证500": {"weight": 0.15, "direction": 1},
        },
        "通胀因子": {
            "布伦特原油": {"weight": 0.35, "direction": 1},
            "南华商品": {"weight": 0.22, "direction": 1},
            "中债企业债": {"weight": 0.43, "direction": 1},
        },
        "利率因子": {
            "中债国债": {"weight": 1.0, "direction": -1},
        },
        "信用因子": {
            "中债企业债": {"weight": 1.0, "direction": 1},
            "中债国债": {"weight": 1.0, "direction": -1},
        },
        "汇率因子": {
            "布伦特原油": {"weight": 1.0, "direction": 1},
            "沪深300": {"weight": 1.0, "direction": -1},
        },
        "流动性因子": {
            "沪深300": {"weight": 1.0, "direction": 1},
            "中证500": {"weight": 1.0, "direction": -1},
        },
    }

    for factor_name, config in mimicking_from_assets.items():
        factor_ret = None
        for asset_name, cfg in config.items():
            if asset_name not in asset_rets.columns:
                continue
            series = asset_rets[asset_name].dropna()
            if factor_ret is None:
                factor_ret = series * cfg["weight"] * cfg.get("direction", 1)
            else:
                factor_ret = factor_ret + series * cfg["weight"] * cfg.get("direction", 1)

        if factor_ret is not None:
            factor_ret = factor_ret.dropna()
            factor_ret = factor_ret.replace([np.inf, -np.inf], np.nan).dropna()
            factor_returns_dict[factor_name] = factor_ret
            print(f"  {factor_name}: built from asset mimicking portfolio ({len(factor_ret)} obs)")
        else:
            print(f"  WARNING: {factor_name} - no data")

    result = pd.DataFrame(factor_returns_dict)
    result = result.sort_index()

    start_dt = pd.to_datetime(BACKTEST_START)
    end_dt = pd.to_datetime(BACKTEST_END)
    result = result[(result.index >= start_dt) & (result.index <= end_dt)]

    for fname in FACTOR_NAMES:
        if fname not in result.columns:
            result[fname] = 0.0

    return result[FACTOR_NAMES]


def compute_factor_exposures(asset_returns, factor_returns, window=36):
    """
    Compute rolling factor exposure matrix (B) for each asset to each factor.
    Uses rolling OLS regression.
    """
    n_assets = asset_returns.shape[1]
    n_factors = factor_returns.columns.tolist()
    asset_names = asset_returns.columns.tolist()

    all_betas = []

    for t in range(len(asset_returns)):
        lookback_data = min(window, t + 1)
        y_window = asset_returns.iloc[t-lookback_data+1:t+1].values
        x_window = factor_returns.iloc[t-lookback_data+1:t+1].values

        X = np.column_stack([np.ones(len(x_window)), x_window])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_window, rcond=None)
            betas = coef[1:].T
        except Exception:
            betas = np.zeros((n_assets, len(n_factors)))

        all_betas.append(betas)

    betas_array = np.array(all_betas)

    betas_dict = {}
    for i, aname in enumerate(asset_names):
        for j, fname in enumerate(n_factors):
            betas_dict[(aname, fname)] = betas_array[:, i, j]

    return betas_dict, betas_array


def compute_factor_covariance_rolling(factor_returns, window=36):
    """
    Compute rolling factor covariance matrices.
    """
    covs = []
    dates = factor_returns.index.tolist()

    for i in range(len(factor_returns)):
        lookback = min(window, i + 1)
        window_data = factor_returns.iloc[i-lookback+1:i+1]
        cov = window_data.cov().values
        covs.append(cov)

    return np.array(covs), dates


if __name__ == "__main__":
    from source.data_loader import load_asset_prices, load_high_freq_macro_factors

    print("Testing macro factors module...")
    hf_factors = load_high_freq_macro_factors()
    asset_prices = load_asset_prices()

    factor_rets = build_mimicking_factors(hf_factors)
    print(f"\nFactor returns shape: {factor_rets.shape}")
    print(factor_rets.head())
