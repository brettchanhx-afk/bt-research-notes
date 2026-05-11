import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime


def get_trade_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    import tushare as ts
    from source.config import TUSHARE_TOKEN, TUSHARE_API_URL

    token = TUSHARE_TOKEN
    pro = ts.pro_api(token)
    pro._DataApi__token = token
    pro._DataApi__http_url = TUSHARE_API_URL

    df = pro.trade_cal(start_date=start_date, end_date=end_date)
    df = df[df['is_open'] == 1]
    return pd.to_datetime(df['cal_date'])


def get_month_end_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    return dates


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    result = np.full_like(a, fill_value, dtype=float)
    mask = b != 0
    result[mask] = a[mask] / b[mask]
    return result


def winsorize(data: np.ndarray, lower_quantile: float = 0.25, upper_quantile: float = 0.75) -> np.ndarray:
    q_low = np.nanpercentile(data, lower_quantile * 100)
    q_high = np.nanpercentile(data, upper_quantile * 100)
    result = np.clip(data, q_low, q_high)
    return result


def interpolate_missing(data: pd.Series) -> pd.Series:
    data = data.copy()
    data = data.interpolate(method='linear', limit_direction='both')
    return data


def calculate_yoy(data: pd.Series) -> pd.Series:
    result = data.pct_change(periods=12) * 100
    return result


def calculate_qoq(data: pd.Series) -> pd.Series:
    result = data.pct_change(periods=1) * 100
    return result


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0:
        return np.zeros_like(data)
    return (data - mean) / std


def rolling_correlation(x: pd.Series, y: pd.Series, window: int = 12) -> pd.Series:
    return x.rolling(window).corr(y)


def time_diff_alignment(x: pd.Series, y: pd.Series, max_lag: int = 4) -> Tuple[int, float]:
    best_lag = 0
    best_corr = 0.0

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x_aligned = x.iloc[lag:].values
            y_aligned = y.iloc[:len(x)-lag].values
        elif lag < 0:
            x_aligned = x.iloc[:lag].values
            y_aligned = y.iloc[-lag:].values
        else:
            x_aligned = x.values
            y_aligned = y.values

        valid_mask = ~(np.isnan(x_aligned) | np.isnan(y_aligned))
        if valid_mask.sum() < 10:
            continue

        corr = np.corrcoef(x_aligned[valid_mask], y_aligned[valid_mask])[0, 1]
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    return dtw_matrix[n, m]


def normalize_date_format(date_str: str) -> str:
    if isinstance(date_str, str):
        for fmt in ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']:
            try:
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except:
                continue
    return str(date_str)


def save_parquet(data: pd.DataFrame, filepath: str) -> None:
    data.to_parquet(filepath, index=True)


def load_parquet(filepath: str) -> pd.DataFrame:
    return pd.read_parquet(filepath)


def ensure_dir(directory: str) -> None:
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
