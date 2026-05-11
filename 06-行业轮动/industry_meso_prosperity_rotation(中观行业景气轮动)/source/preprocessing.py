import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class Preprocessor:
    def __init__(self, rolling_window: int = 60):
        self.rolling_window = rolling_window

    def preprocess_total_indicator(self, data: pd.Series) -> pd.Series:
        data = data.copy()
        data = data.interpolate(method='linear', limit_direction='both')
        data = self._remove_outliers(data)
        yoy = data.pct_change(periods=12) * 100
        return yoy.fillna(0)

    def preprocess_price_indicator(self, data: pd.Series) -> pd.Series:
        data = data.copy()
        data = data.interpolate(method='linear', limit_direction='both')
        data = self._remove_outliers(data)
        yoy = data.pct_change(periods=12) * 100
        return yoy.fillna(0)

    def preprocess_yoy_ratio_indicator(self, data: pd.Series) -> pd.Series:
        data = data.copy()
        data = data.interpolate(method='linear', limit_direction='both')
        data = self._remove_outliers(data)
        return data

    def preprocess_diffusion_indicator(self, data: pd.Series) -> pd.Series:
        data = data.copy()
        data = data.interpolate(method='linear', limit_direction='both')
        data = self._remove_outliers(data)
        yoy = data.pct_change(periods=12) * 100
        return yoy.fillna(0)

    def preprocess_ratio_indicator(self, data: pd.Series) -> pd.Series:
        data = data.copy()
        data = data.interpolate(method='linear', limit_direction='both')
        data = self._remove_outliers(data)
        diff = data.diff()
        return diff.fillna(0)

    def _remove_outliers(self, data: pd.Series, k: float = 1.5) -> pd.Series:
        if len(data) < 4:
            return data

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        result = data.copy()
        result[(result < lower) | (result > upper)] = np.nan
        return result

    def seasonal_adjustment(self, data: pd.Series) -> pd.Series:
        if len(data) < 24:
            return data

        n_periods = 12
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(data, lamb=129600)
            seasonally_adjusted = data - trend
            return seasonally_adjusted.fillna(data)
        except:
            return data

    def hp_filter(self, data: pd.Series, lamb: int = 129600) -> Tuple[pd.Series, pd.Series]:
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(data, lamb=lamb)
            return cycle, trend
        except:
            return data, pd.Series(0, index=data.index)

    def simple_seasonal_adjustment(self, data: pd.Series) -> pd.Series:
        if len(data) < 12:
            return data

        result = data.copy()
        for month in range(1, 13):
            month_mask = data.index.month == month
            if month_mask.sum() > 1:
                monthly_mean = data[month_mask].mean()
                overall_mean = data.mean()
                if overall_mean != 0:
                    result[month_mask] = data[month_mask] * (overall_mean / monthly_mean)
        return result

    def align_to_reference(
        self,
        indicator: pd.Series,
        reference: pd.Series,
        lag: int
    ) -> pd.Series:
        if lag > 0:
            indicator_aligned = indicator.shift(lag)
        elif lag < 0:
            indicator_aligned = indicator.shift(lag)
        else:
            indicator_aligned = indicator.copy()

        common_index = indicator_aligned.index.intersection(reference.index)
        return indicator_aligned.loc[common_index]

    def rolling_window_preprocess(
        self,
        data: pd.Series,
        window: int
    ) -> pd.Series:
        if len(data) < window:
            return data

        result = data.copy()
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            window_mean = window_data.mean()
            window_std = window_data.std()
            if window_std != 0:
                zscore = (data.iloc[i] - window_mean) / window_std
                result.iloc[i] = zscore
        return result

    def get_latest_values(self, data: pd.Series, n: int = 4) -> List[float]:
        return data.dropna().tail(n).tolist()

    def check_stability(self, data: pd.Series) -> bool:
        recent = data.tail(4)
        return not recent.isna().any()

    def preprocess_financial_reference(
        self,
        data: pd.Series,
        method: str = "interpolate"
    ) -> pd.Series:
        data = data.copy()

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        data[(data < lower) | (data > upper)] = np.nan

        if method == "interpolate":
            data = data.interpolate(method='linear', limit_direction='both')
        elif method == "forward":
            data = data.ffill()

        return data.fillna(method='bfill')

    def convert_to_monthly_yoy(
        self,
        data: pd.Series
    ) -> pd.Series:
        yoy = data.pct_change(periods=12) * 100
        return yoy.fillna(0)

    def convert_to_quarterly_yoy(
        self,
        data: pd.Series
    ) -> pd.Series:
        yoy = data.pct_change(periods=4) * 100
        return yoy.fillna(0)

    def fill_missing_tail(
        self,
        data: pd.Series,
        max_fill: int = 3
    ) -> pd.Series:
        result = data.copy()
        tail_na_count = result.iloc[::-1].isna().cumsum().iloc[::-1]

        for i in range(len(result)):
            if pd.isna(result.iloc[i]):
                if tail_na_count.iloc[i] <= max_fill:
                    if i > 0 and not pd.isna(result.iloc[i-1]):
                        result.iloc[i] = result.iloc[i-1]

        return result


class IndicatorPreprocessor:
    def __init__(self, rolling_window: int = 60, min_valid_length: int = 36):
        self.rolling_window = rolling_window
        self.min_valid_length = min_valid_length
        self.preprocessor = Preprocessor(rolling_window)

    def preprocess_indicator(
        self,
        data: pd.Series,
        category: str,
        apply_seasonal_adjustment: bool = False,
        apply_hp_filter: bool = False
    ) -> pd.Series:
        if category == "total":
            processed = self.preprocessor.preprocess_total_indicator(data)
        elif category == "price":
            processed = self.preprocessor.preprocess_price_indicator(data)
        elif category == "yoy_ratio":
            processed = self.preprocessor.preprocess_yoy_ratio_indicator(data)
        elif category == "diffusion":
            processed = self.preprocessor.preprocess_diffusion_indicator(data)
        else:
            processed = data.copy()

        if apply_seasonal_adjustment:
            processed = self.preprocessor.simple_seasonal_adjustment(processed)

        if apply_hp_filter:
            try:
                from statsmodels.tsa.filters.hp_filter import hpfilter
                cycle, trend = hpfilter(processed, lamb=129600)
                processed = trend
            except:
                pass

        return processed

    def validate_indicator(
        self,
        data: pd.Series
    ) -> Tuple[bool, str]:
        valid_length = data.dropna().shape[0]

        if valid_length < self.min_valid_length:
            return False, f"Valid length {valid_length} < {self.min_valid_length}"

        recent_na_count = data.tail(4).isna().sum()
        if recent_na_count >= 4:
            return False, "Last 4 values are all NaN"

        if data.std() == 0:
            return False, "Standard deviation is zero"

        return True, "Valid"

    def preprocess_batch(
        self,
        indicator_dict: Dict[str, pd.Series],
        category_dict: Dict[str, str]
    ) -> Dict[str, pd.Series]:
        processed = {}

        for name, data in indicator_dict.items():
            category = category_dict.get(name, "total")
            try:
                processed[name] = self.preprocess_indicator(data, category)
            except Exception as e:
                print(f"Error preprocessing {name}: {e}")
                continue

        return processed


def interpolate_to_monthly(
    quarterly_data: pd.Series,
    method: str = "linear"
) -> pd.Series:
    if quarterly_data.index.freq is not None:
        monthly_index = pd.date_range(
            start=quarterly_data.index.min(),
            end=quarterly_data.index.max(),
            freq='M'
        )
    else:
        monthly_index = pd.date_range(
            start=quarterly_data.index.min(),
            end=quarterly_data.index.max(),
            freq='M'
        )

    result = pd.Series(index=monthly_index, dtype=float)

    quarterly_dates = quarterly_data.index
    for i, q_date in enumerate(quarterly_dates[:-1]):
        next_q_date = quarterly_dates[i + 1]
        value = quarterly_data.loc[q_date]

        start_month = q_date.month if hasattr(q_date, 'month') else (i * 3)
        end_month = next_q_date.month if hasattr(next_q_date, 'month') else ((i + 1) * 3)

        for month_offset in range(3):
            month = start_month + month_offset
            if month <= 12:
                target_date = q_date.replace(month=month) if hasattr(q_date, 'replace') else q_date
                if target_date in result.index:
                    result.loc[target_date] = value

    return result.fillna(method='ffill').fillna(method='bfill')


def calculate_rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int = 36
) -> pd.Series:
    return x.rolling(window).corr(y)


def main():
    preprocessor = Preprocessor(rolling_window=60)

    test_data = pd.Series(
        np.random.randn(100) * 10 + 100,
        index=pd.date_range('2016-01-01', periods=100, freq='M')
    )

    processed = preprocessor.preprocess_total_indicator(test_data)
    print(f"Original data shape: {test_data.shape}")
    print(f"Processed data shape: {processed.shape}")
    print(f"Processed data sample:\n{processed.tail()}")

    validator = IndicatorPreprocessor(rolling_window=60, min_valid_length=36)
    is_valid, msg = validator.validate_indicator(processed)
    print(f"\nValidation result: {is_valid}, {msg}")


if __name__ == "__main__":
    main()
