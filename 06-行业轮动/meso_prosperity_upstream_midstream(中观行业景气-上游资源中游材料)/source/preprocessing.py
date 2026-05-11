"""
数据预处理模块
处理指标数据的季节性、趋势和口径转换

参考研报: 华泰证券-中观行业景气度：Nowcasting初探 (2021-09-26)
         华泰证券-行业配置策略：投资时钟视角 (2021-07-06)
"""

import pandas as pd
import numpy as np
from scipy import signal
from typing import Optional, Tuple, List, Dict
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class IndicatorPreprocessor:
    """指标预处理器"""

    def __init__(self):
        self.processing_log = []

    def check_stationarity(self, series: pd.Series, significance_level: float = 0.1) -> Tuple[bool, float]:
        """
        检验序列平稳性

        使用ADF检验

        Parameters:
        -----------
        series : pd.Series
            时间序列
        significance_level : float
            显著性水平

        Returns:
        --------
        is_stationary : bool
            是否平稳
        p_value : float
            p值
        """
        if len(series) < 10:
            return False, 1.0

        series_clean = series.dropna()

        if len(series_clean) < 10:
            return False, 1.0

        try:
            result = adfuller(series_clean, maxlag=12, regression='c')
            p_value = result[1]
            is_stationary = p_value < significance_level
            return is_stationary, p_value
        except:
            return False, 1.0

    def remove_trend(self, series: pd.Series, method: str = 'linear') -> pd.Series:
        """
        去除趋势

        Parameters:
        -----------
        series : pd.Series
            原始序列
        method : str
            趋势去除方法，'linear'或'mean'

        Returns:
        --------
        pd.Series
            去趋势后的序列
        """
        if len(series) < 6:
            return series - series.mean()

        if method == 'linear':
            x = np.arange(len(series))
            y = series.values

            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 3:
                return series - series.mean()

            coeffs = np.polyfit(x[valid_mask], y[valid_mask], 1)
            trend = np.polyval(coeffs, x)
            detrended = series.values - trend

            return pd.Series(detrended, index=series.index)

        elif method == 'mean':
            rolling_mean = series.rolling(window=12, min_periods=1).mean()
            return series - rolling_mean

        return series

    def handle_seasonality(self, series: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        处理季节性

        使用移动平均去除季节性

        Parameters:
        -----------
        series : pd.Series
            原始序列
        period : int, optional
            季节周期，默认12个月

        Returns:
        --------
        pd.Series
            去季节性后的序列
        """
        if period is None:
            period = 12 if len(series) >= 24 else 4

        if len(series) < period * 2:
            return series

        try:
            ma = series.rolling(window=period, center=True, min_periods=1).mean()
            deseasonalized = series - ma

            return deseasonalized
        except:
            return series

    def transform_to_yoy(self, series: pd.Series) -> pd.Series:
        """
        转换为同比序列

        Parameters:
        -----------
        series : pd.Series
            原始序列

        Returns:
        --------
        pd.Series
            同比序列
        """
        if len(series) < 13:
            return series

        yoy = series.pct_change(periods=12) * 100

        return yoy

    def transform_to_mom(self, series: pd.Series) -> pd.Series:
        """
        转换为环比序列

        Parameters:
        -----------
        series : pd.Series
            原始序列

        Returns:
        --------
        pd.Series
            环比序列
        """
        mom = series.pct_change() * 100

        return mom

    def handle_outliers(self, series: pd.Series, n_std: float = 3.0) -> pd.Series:
        """
        处理异常值

        使用n倍标准差方法

        Parameters:
        -----------
        series : pd.Series
            原始序列
        n_std : float
            标准差倍数

        Returns:
        --------
        pd.Series
            处理后的序列
        """
        if len(series) < 10:
            return series

        mean_val = series.mean()
        std_val = series.std()

        if std_val == 0:
            return series

        upper_bound = mean_val + n_std * std_val
        lower_bound = mean_val - n_std * std_val

        cleaned = series.copy()
        cleaned = cleaned.clip(lower=lower_bound, upper=upper_bound)

        return cleaned

    def fill_missing_values(self, series: pd.Series, method: str = 'interpolate') -> pd.Series:
        """
        填充缺失值

        Parameters:
        -----------
        series : pd.Series
            原始序列
        method : str
            填充方法，'interpolate', 'forward', 'backward', 'mean'

        Returns:
        --------
        pd.Series
            填充后的序列
        """
        if not series.isna().any():
            return series

        if method == 'interpolate':
            return series.interpolate(method='linear')

        elif method == 'forward':
            return series.fillna(method='ffill')

        elif method == 'backward':
            return series.fillna(method='bfill')

        elif method == 'mean':
            return series.fillna(series.mean())

        return series

    def standardize(self, series: pd.Series) -> pd.Series:
        """
        标准化（Z-score）

        Parameters:
        -----------
        series : pd.Series
            原始序列

        Returns:
        --------
        pd.Series
            标准化后的序列
        """
        mean_val = series.mean()
        std_val = series.std()

        if std_val == 0:
            return series - mean_val

        return (series - mean_val) / std_val

    def normalize(self, series: pd.Series, min_val: float = 0, max_val: float = 1) -> pd.Series:
        """
        归一化到指定范围

        Parameters:
        -----------
        series : pd.Series
            原始序列
        min_val : float
            最小值
        max_val : float
            最大值

        Returns:
        --------
        pd.Series
            归一化后的序列
        """
        series_min = series.min()
        series_max = series.max()

        if series_max == series_min:
            return pd.Series(min_val, index=series.index)

        normalized = (series - series_min) / (series_max - series_min)
        normalized = normalized * (max_val - min_val) + min_val

        return normalized

    def process_indicator(self, series: pd.Series,
                          remove_trend: bool = True,
                          handle_outliers: bool = True,
                          fill_missing: bool = True,
                          standardize_result: bool = False) -> pd.Series:
        """
        综合处理指标

        Parameters:
        -----------
        series : pd.Series
            原始序列
        remove_trend : bool
            是否去除趋势
        handle_outliers : bool
            是否处理异常值
        fill_missing : bool
            是否填充缺失值
        standardize_result : bool
            是否标准化结果

        Returns:
        --------
        pd.Series
            处理后的序列
        """
        result = series.copy()

        if fill_missing:
            result = self.fill_missing_values(result)

        if handle_outliers:
            result = self.handle_outliers(result)

        if remove_trend:
            result = self.remove_trend(result)

        if standardize_result:
            result = self.standardize(result)

        return result


class IndicatorSelector:
    """指标选择器"""

    def __init__(self, min_explained_variance: float = 0.2,
                 max_stationarity_pvalue: float = 0.1,
                 min_indicators: int = 15):
        """
        Parameters:
        -----------
        min_explained_variance : float
            隐含因子对指标的最小解释度
        max_stationarity_pvalue : float
            指标序列平稳性的最大p值
        min_indicators : int
            最少选用的指标数目
        """
        self.min_explained_variance = min_explained_variance
        self.max_stationarity_pvalue = max_stationarity_pvalue
        self.min_indicators = min_indicators
        self.preprocessor = IndicatorPreprocessor()

    def evaluate_indicator(self, indicator: pd.Series,
                         factor: pd.Series) -> Tuple[float, bool, dict]:
        """
        评估单个指标

        Parameters:
        -----------
        indicator : pd.Series
            代理指标
        factor : pd.Series
            隐含因子

        Returns:
        --------
        explained_variance : float
            解释度
        is_stationary : bool
            是否平稳
        details : dict
            详细评估结果
        """
        common_idx = indicator.index.intersection(factor.index)

        if len(common_idx) < 10:
            return 0.0, False, {}

        indicator_aligned = indicator.loc[common_idx]
        factor_aligned = factor.loc[common_idx]

        is_stationary, p_value = self.preprocessor.check_stationarity(
            indicator_aligned, self.max_stationarity_pvalue
        )

        try:
            from sklearn.linear_model import LinearRegression

            X = factor_aligned.values.reshape(-1, 1)
            y = indicator_aligned.values

            reg = LinearRegression()
            reg.fit(X, y)

            y_pred = reg.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)

            explained_var = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        except:
            explained_var = 0.0

        details = {
            'p_value': p_value,
            'explained_variance': explained_var,
            'coefficient': reg.coef_[0] if 'reg' in dir() else 0
        }

        return explained_var, is_stationary, details

    def select_indicators(self, indicators: Dict[str, pd.Series],
                         factor: pd.Series) -> List[str]:
        """
        选择最优指标

        Parameters:
        -----------
        indicators : Dict[str, pd.Series]
            候选指标字典
        factor : pd.Series
            隐含因子

        Returns:
        --------
        List[str]
            选中的指标名称列表
        """
        selected = []
        rejected = []

        for name, indicator in indicators.items():
            explained_var, is_stationary, details = self.evaluate_indicator(
                indicator, factor
            )

            if explained_var >= self.min_explained_variance:
                selected.append((name, explained_var, details))
            else:
                rejected.append((name, explained_var, is_stationary))

        selected.sort(key=lambda x: x[1], reverse=True)

        if len(selected) < self.min_indicators:
            additional = [x for x in rejected if x[2]]
            additional.sort(key=lambda x: x[1], reverse=True)

            needed = self.min_indicators - len(selected)
            selected.extend(additional[:needed])

        return [x[0] for x in selected]


def detect_frequency(series: pd.Series) -> int:
    """
    检测数据频率

    Parameters:
    -----------
    series : pd.Series
        时间序列

    Returns:
    --------
    frequency : int
        估计的频率（天数）
    """
    if len(series) < 3:
        return 30

    diffs = series.index.to_series().diff().dt.days.dropna()

    if len(diffs) == 0:
        return 30

    median_diff = diffs.median()

    if median_diff < 7:
        return 1
    elif median_diff < 14:
        return 7
    elif median_diff < 35:
        return 30
    else:
        return 365


def align_frequencies(series_dict: Dict[str, pd.Series],
                     target_freq: str = 'M') -> Dict[str, pd.Series]:
    """
    对齐不同频率的数据

    Parameters:
    -----------
    series_dict : Dict[str, pd.Series]
        时间序列字典
    target_freq : str
        目标频率，'D', 'W', 'M', 'Q', 'Y'

    Returns:
    --------
    Dict[str, pd.Series]
        对齐后的序列字典
    """
    aligned = {}

    freq_map = {
        'D': 'D',
        'W': 'W',
        'M': 'M',
        'Q': 'Q',
        'Y': 'Y'
    }

    for name, series in series_dict.items():
        if len(series) == 0:
            continue

        try:
            if target_freq == 'M':
                resampled = series.resample('M').last()
                resampled = resampled.ffill()
            elif target_freq == 'W':
                resampled = series.resample('W').last()
                resampled = resampled.ffill()
            elif target_freq == 'Q':
                resampled = series.resample('Q').last()
                resampled = resampled.ffill()
            elif target_freq == 'Y':
                resampled = series.resample('Y').last()
                resampled = resampled.ffill()
            else:
                resampled = series

            aligned[name] = resampled.dropna()

        except Exception as e:
            print(f"对齐{name}时出错: {e}")
            aligned[name] = series

    return aligned


if __name__ == '__main__':
    print("测试数据预处理模块...")

    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='M')

    raw_data = pd.Series(np.cumsum(np.random.randn(120)) + 50, index=dates)
    raw_data.iloc[20] = raw_data.iloc[20] * 3
    raw_data.iloc[50:55] = np.nan

    print("\n1. 测试预处理器:")
    preprocessor = IndicatorPreprocessor()

    print(f"  原始数据长度: {len(raw_data)}")
    print(f"  缺失值数量: {raw_data.isna().sum()}")

    cleaned = preprocessor.process_indicator(
        raw_data,
        remove_trend=True,
        handle_outliers=True,
        fill_missing=True,
        standardize_result=True
    )

    print(f"  处理后长度: {len(cleaned)}")
    print(f"  处理后缺失值: {cleaned.isna().sum()}")

    print("\n2. 测试平稳性检验:")
    is_stationary, p_value = preprocessor.check_stationarity(cleaned)
    print(f"  平稳性: {is_stationary}, p值: {p_value:.4f}")

    print("\n3. 测试同比转换:")
    yoy = preprocessor.transform_to_yoy(raw_data)
    print(f"  同比序列前5个值: {yoy.head().tolist()}")

    print("\n4. 测试指标选择器:")
    factor = pd.Series(np.random.randn(120), index=dates)
    indicators = {
        f'indicator_{i}': pd.Series(
            factor.values * (0.3 + np.random.rand()) + np.random.randn(120) * 0.3,
            index=dates
        ) for i in range(20)
    }

    selector = IndicatorSelector(min_explained_variance=0.15, min_indicators=10)
    selected = selector.select_indicators(indicators, factor)
    print(f"  选中指标数量: {len(selected)}")
    print(f"  选中指标: {selected[:5]}")

    print("\n5. 测试频率检测:")
    freq = detect_frequency(raw_data)
    print(f"  估计频率: {freq}天")
