import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def save_to_csv(data, filepath, index=True):
    if data is None or len(data) == 0:
        return False

    ensure_dir(os.path.dirname(filepath))

    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=index)
        return True
    return False


def save_to_json(data, filepath):
    if data is None:
        return False

    ensure_dir(os.path.dirname(filepath))

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)

    return True


def save_to_pickle(data, filepath):
    if data is None:
        return False

    ensure_dir(os.path.dirname(filepath))

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    return True


def load_from_pickle(filepath):
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def load_from_csv(filepath, parse_dates=None):
    if not os.path.exists(filepath):
        return None

    if parse_dates:
        data = pd.read_csv(filepath, parse_dates=parse_dates)
    else:
        data = pd.read_csv(filepath)

    return data


def format_number(num, decimals=2):
    if num is None:
        return 'N/A'

    if isinstance(num, (int, float)):
        return f'{num:,.{decimals}f}'

    return str(num)


def format_percent(num, decimals=2):
    if num is None:
        return 'N/A'

    if isinstance(num, (int, float)):
        return f'{num:.{decimals}f}%'

    return str(num)


def calculate_date_range(start_date, end_date, freq='M'):
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return dates.tolist()


def get_trading_dates(start_date, end_date):
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    trading_dates = [d for d in all_dates if d.weekday() < 5]

    return trading_dates


def merge_dataframes(dfs, on, how='outer'):
    if not dfs:
        return pd.DataFrame()

    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=on, how=how)

    return result


def resample_data(data, freq='M', agg_method='last'):
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if 'trade_date' in data.columns:
        data = data.set_index('trade_date')

    if agg_method == 'last':
        resampled = data.resample(freq).last()
    elif agg_method == 'mean':
        resampled = data.resample(freq).mean()
    elif agg_method == 'sum':
        resampled = data.resample(freq).sum()

    return resampled.reset_index()


def remove_outliers(data, columns=None, n_std=3):
    if data is None or len(data) == 0:
        return data

    df = data.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()

            df = df[np.abs(df[col] - mean) <= n_std * std]

    return df


def normalize_data(data, columns=None, method='zscore'):
    if data is None or len(data) == 0:
        return data

    df = data.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df[col] = (df[col] - mean) / std
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)

    return df


def print_progress(current, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()


def log_message(message, level='INFO'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] [{level}] {message}')


def create_cache_key(*args):
    cache_key = '_'.join([str(arg) for arg in args])
    return cache_key


class DataCache:
    def __init__(self, cache_dir='.cache'):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)

    def get(self, key):
        filepath = os.path.join(self.cache_dir, f'{key}.pkl')
        return load_from_pickle(filepath)

    def set(self, key, data):
        filepath = os.path.join(self.cache_dir, f'{key}.pkl')
        return save_to_pickle(data, filepath)

    def has(self, key):
        filepath = os.path.join(self.cache_dir, f'{key}.pkl')
        return os.path.exists(filepath)

    def clear(self):
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    print("Testing utils module...")

    cache = DataCache(cache_dir='.test_cache')
    cache.set('test_key', {'data': [1, 2, 3]})
    result = cache.get('test_key')
    print(f"Cache test result: {result}")

    print("Utils module test completed!")
