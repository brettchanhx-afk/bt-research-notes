"""
宏观指标预处理模块
包括：变频、缺失值填充、季节性调整、HP滤波等
"""
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class MacroPreprocessor:
    def __init__(self):
        self.processed_data = {}

    def standardize_frequency(self, df, date_col='trade_date', value_col='value',
                             target_freq='M'):
        """
        将数据统一到目标频率（月频）
        对于日频数据，取月末值
        """
        if df.empty:
            return df

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        if target_freq == 'M':
            result = df[value_col].resample('M').last()
        elif target_freq == 'Q':
            result = df[value_col].resample('Q').last()
        else:
            result = df[value_col].resample(target_freq).last()

        result = result.dropna()
        return result.reset_index().rename(columns={value_col: f'{value_col}_{target_freq}'})

    def fill_missing_values(self, series, method='linear', max_fill_ratio=0.1):
        """
        缺失值填充
        方法: linear, ffill, bfill, interpolate
        """
        if len(series) == 0:
            return series

        series = series.copy()
        missing_count = series.isna().sum()
        total_count = len(series)

        if missing_count / total_count > max_fill_ratio:
            print(f"警告: 缺失值比例 {missing_count/total_count:.2%} 超过阈值 {max_fill_ratio:.2%}")

        if method == 'linear':
            series = series.interpolate(method='linear')
        elif method == 'ffill':
            series = series.fillna(method='ffill')
        elif method == 'bfill':
            series = series.fillna(method='bfill')
        elif method == 'interpolate':
            valid_idx = series.dropna().index
            if len(valid_idx) > 2:
                f = interp1d(
                    valid_idx.astype(np.int64),
                    series.loc[valid_idx].values.astype(np.float64),
                    kind='linear',
                    fill_value='extrapolate'
                )
                series = pd.Series(
                    f(series.index.astype(np.int64)),
                    index=series.index
                )

        series = series.fillna(method='bfill').fillna(method='ffill')
        return series

    def calculate_yoy(self, series, date_col='trade_date'):
        """
        计算同比增长率
        """
        if len(series) == 0:
            return series

        series = series.copy()
        series[date_col] = pd.to_datetime(series[date_col])
        series = series.sort_values(date_col)
        series = series.set_index(date_col)

        yoy = series.pct_change(periods=12) * 100
        return yoy.reset_index()

    def hp_filter(self, series, lamb=129600):
        """
        HP滤波分解趋势和周期
        lamb: 平滑参数，月度数据通常使用129600
        """
        if len(series) < 24:
            print(f"数据长度不足，无法进行HP滤波: {len(series)}")
            return series, series

        series = series.copy()
        series = series.dropna()

        n = len(series)
        dates = np.arange(n)

        B = np.zeros(n)
        B[0] = 1
        B[1] = -2
        B[2] = 1

        B2 = np.zeros(n)
        B2[0] = 1
        B2[1] = -1

        for i in range(3, n):
            B[i] = 1
            B2[i] = -1

        I = np.eye(n)
        B_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(max(0, i-2), min(n, i+3)):
                if i == j:
                    B_mat[i, j] = 1
                elif abs(i - j) == 1:
                    B_mat[i, j] = -2
                elif abs(i - j) == 2:
                    B_mat[i, j] = 1

        I_kron_I = np.kron(I, I)
        B_kron_B = np.kron(B_mat, B_mat)

        A = I_kron_I + lamb * B_kron_B

        try:
            trend = np.linalg.solve(A, series.values.flatten())
        except:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import spsolve
            A_sparse = csr_matrix(A)
            trend = spsolve(A_sparse, series.values.flatten())

        cycle = series.values.flatten() - trend

        trend_series = pd.Series(trend, index=series.index)
        cycle_series = pd.Series(cycle, index=series.index)

        return trend_series, cycle_series

    def seasonal_adjustment_x11(self, series, period=12):
        """
        X-11季节调整方法（简化版）
        使用移动平均提取季节因子
        """
        if len(series) < 2 * period:
            print(f"数据长度不足，无法进行季节性调整: {len(series)}")
            return series

        series = series.copy()
        series = series.dropna()

        n = len(series)
        detrended = np.zeros(n)

        for i in range(n):
            start_idx = max(0, i - period)
            end_idx = min(n, i + period + 1)
            window = series.iloc[start_idx:end_idx].values
            if len(window) > 0:
                detrended[i] = series.iloc[i] - np.mean(window)

        seasonal = np.zeros(n)
        for m in range(period):
            month_values = []
            for i in range(m, n, period):
                if i < n:
                    month_values.append(detrended[i])

            if len(month_values) > 0:
                seasonal[m::period] = np.mean(month_values)

        seasonally_adjusted = series.values - seasonal

        return pd.Series(seasonally_adjusted, index=series.index)

    def check_stationarity(self, series):
        """
        简单的平稳性检验（基于自相关）
        如果自相关系数衰减缓慢，可能非平稳
        """
        if len(series) < 20:
            return False

        series = series.dropna()
        autocorr = [series.autocorr(lag=i) for i in range(1, min(13, len(series)//2))]

        if abs(autocorr[0]) > 0.9 and all(abs(a) > 0.5 for a in autocorr[:3]):
            return False
        return True

    def preprocess_indicator(self, df, date_col='trade_date', value_col='value',
                            fill_method='linear', apply_hp_filter=True,
                            calculate_yoy_flag=True):
        """
        完整的指标预处理流程
        """
        if df.empty:
            return pd.DataFrame()

        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])
        result = result.sort_values(date_col)

        result = self.standardize_frequency(result, date_col, value_col, 'M')

        if result.empty or f'{value_col}_M' not in result.columns:
            return pd.DataFrame()

        value_series = result[f'{value_col}_M']

        value_series = self.fill_missing_values(value_series, method=fill_method)

        if calculate_yoy_flag and len(value_series) > 12:
            yoy_values = value_series.pct_change(periods=12) * 100
            yoy_values = self.fill_missing_values(yoy_values, method='linear')
        else:
            yoy_values = value_series

        if apply_hp_filter and len(yoy_values) > 24:
            trend, cycle = self.hp_filter(yoy_values)
            processed_series = cycle
        else:
            processed_series = yoy_values

        processed_df = pd.DataFrame({
            'trade_date': result['trade_date'].values,
            'original': value_series.values,
            'yoy': yoy_values.values if calculate_yoy_flag else value_series.values,
            'processed': processed_series.values
        })

        return processed_df

    def merge_to_frequency(self, df_list, date_col='trade_date', freq='M'):
        """
        将多个指标合并到统一频率
        """
        merged = None

        for df in df_list:
            if df.empty:
                continue

            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)

            if merged is None:
                merged = df
            else:
                merged = merged.join(df, how='outer')

        if merged is not None:
            merged = merged.reset_index()
            merged[date_col] = pd.to_datetime(merged[date_col])
            merged = merged.sort_values(date_col)

        return merged

    def detect_outliers(self, series, n_std=3):
        """
        使用标准差方法检测异常值
        """
        if len(series) < 12:
            return pd.Series([False] * len(series), index=series.index)

        mean = series.mean()
        std = series.std()

        outliers = np.abs(series - mean) > n_std * std
        return pd.Series(outliers.values, index=series.index)

    def winsorize(self, series, lower=0.01, upper=0.99):
        """
        去极值处理
        """
        if len(series) < 10:
            return series

        lower_val = series.quantile(lower)
        upper_val = series.quantile(upper)

        return series.clip(lower=lower_val, upper=upper_val)

    def normalize_zscore(self, series):
        """
        Z-score标准化
        """
        if len(series) < 2:
            return series

        mean = series.mean()
        std = series.std()

        if std == 0:
            return series - mean

        return (series - mean) / std
