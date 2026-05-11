"""
Data loader module - loads provided CSV data files.
"""
import warnings
import os
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from source.config import (
    DATA_DIR,
    BACKTEST_START,
    BACKTEST_END,
    ASSET_COLS_RAW,
    ASSET_NAME_MAP,
)


def _try_encodings(filepath, **kwargs):
    for enc in ["gbk", "utf-8", "gb18030", "latin1"]:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Could not read {filepath} with any encoding")


def load_asset_prices():
    """
    Load the 6 asset daily prices from seven_assets_price_2013_PCA.csv.
    Returns a DataFrame with columns renamed to Chinese names.
    """
    cache_path = os.path.join(DATA_DIR, "asset_prices_clean.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
            print(f"[DataLoader] Loaded asset prices from cache.")
            return df

    filepath = os.path.join(DATA_DIR, "seven_assets_price_2013_PCA.csv")
    df = _try_encodings(filepath, header=None)

    raw_codes = df.iloc[0, 1:].tolist()
    date_col = df.iloc[2:, 0]
    price_data = df.iloc[2:, 1:]

    price_data.columns = raw_codes
    price_data.index = pd.to_datetime(date_col, errors="coerce")
    price_data.index.name = "date"

    price_data = price_data.apply(pd.to_numeric, errors="coerce")
    price_data = price_data.dropna(how="all")
    price_data = price_data.sort_index()

    rename_map = {v: k for k, v in ASSET_NAME_MAP.items() if k in ASSET_COLS_RAW}
    actual_rename = {ASSET_NAME_MAP.get(c, c): c for c in price_data.columns}

    price_data.columns = [ASSET_NAME_MAP.get(c, c) for c in price_data.columns]

    start_dt = pd.to_datetime(BACKTEST_START)
    end_dt = pd.to_datetime(BACKTEST_END)
    price_data = price_data[(price_data.index >= start_dt) & (price_data.index <= end_dt)]

    with open(cache_path, "wb") as f:
        pickle.dump(price_data, f)
    print(f"[DataLoader] Asset prices loaded: {price_data.shape}, date range {price_data.index[0]} to {price_data.index[-1]}")

    return price_data


def load_high_freq_macro_factors():
    """
    Load high-frequency macro factor mimicking portfolio data.
    Returns a clean DataFrame with parsed dates and numeric values.
    """
    cache_path = os.path.join(DATA_DIR, "high_freq_factors_clean.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
            print(f"[DataLoader] Loaded high-frequency factors from cache.")
            return df

    filepath = os.path.join(DATA_DIR, "high_frequency_macro_factor_portfolio.csv")
    df = _try_encodings(filepath, index_col=0)

    df = df[~df.index.isna()]
    valid_mask = ~df.index.str.contains("数据来源|NaN|指标名称", na=False)
    df = df[valid_mask]

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"[DataLoader] High-freq factors loaded: {df.shape}, date range {df.index[0]} to {df.index[-1]}")

    return df


def load_original_macro_factors():
    """
    Load original (monthly) macro factor data.
    """
    cache_path = os.path.join(DATA_DIR, "original_macro_factors_clean.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
            print(f"[DataLoader] Loaded original macro factors from cache.")
            return df

    filepath = os.path.join(DATA_DIR, "original_macro_factor_2013.csv")
    df = _try_encodings(filepath, index_col=0)

    df = df[~df.index.isna()]
    valid_mask = ~df.index.str.contains("数据来源|NaN|指标名称", na=False)
    df = df[valid_mask]

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"[DataLoader] Original macro factors loaded: {df.shape}, date range {df.index[0]} to {df.index[-1]}")

    return df


def calculate_returns(prices, freq="D"):
    """
    Calculate returns from prices.
    freq: 'D' for daily, 'ME' for month-end
    """
    if freq == "ME":
        prices = prices.resample("M").last()
    rets = prices.pct_change()
    rets = rets.dropna(how="all")
    return rets


def resample_to_monthly(df):
    """Resample daily data to month-end."""
    return df.resample("M").last()


if __name__ == "__main__":
    print("Testing data loader...")
    prices = load_asset_prices()
    print(f"Prices shape: {prices.shape}")
    print(prices.head())
    print()

    hf_factors = load_high_freq_macro_factors()
    print(f"HF factors shape: {hf_factors.shape}")
    print(hf_factors.head())
    print()

    orig_factors = load_original_macro_factors()
    print(f"Original factors shape: {orig_factors.shape}")
    print(orig_factors.head())
