import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class IndicatorScore:
    name: str
    r2: float
    lead_lag: int
    dtw_dist: float
    total_score: float
    direction: int


class SimpleNowcasting:
    def __init__(self, rolling_window: int = 60):
        self.rolling_window = rolling_window

    def fit_predict(
        self,
        proxy_indicators: pd.DataFrame,
        financial_reference: pd.Series,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> pd.Series:
        n_indicators = proxy_indicators.shape[1]
        weights = np.ones(n_indicators) / n_indicators

        for iteration in range(max_iter):
            weights_old = weights.copy()

            combined_signal = proxy_indicators.values @ weights

            residuals = financial_reference.values - combined_signal

            for j in range(n_indicators):
                corr = np.corrcoef(proxy_indicators.values[:, j], residuals)[0, 1]
                if np.isnan(corr):
                    corr = 0

                gradient = -corr * np.std(proxy_indicators.values[:, j])

                weights[j] = weights[j] - 0.1 * gradient

            weights = np.maximum(weights, 0)
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights = weights / weights_sum

            weight_change = np.abs(weights - weights_old).sum()
            if weight_change < tol:
                break

        combined_signal = proxy_indicators.values @ weights
        result = pd.Series(
            combined_signal,
            index=proxy_indicators.index
        )
        return result

    def simple_average(
        self,
        proxy_indicators: pd.DataFrame
    ) -> pd.Series:
        return proxy_indicators.mean(axis=1)

    def weighted_average(
        self,
        proxy_indicators: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        if len(weights) != proxy_indicators.shape[1]:
            weights = np.ones(proxy_indicators.shape[1]) / proxy_indicators.shape[1]
        result = proxy_indicators.values @ weights
        return pd.Series(result, index=proxy_indicators.index)


class IndicatorEvaluator:
    def __init__(self, max_lag: int = 4):
        self.max_lag = max_lag

    def calculate_time_diff_r2(
        self,
        indicator: pd.Series,
        reference: pd.Series
    ) -> Tuple[float, int]:
        best_r2 = 0
        best_lag = 0

        for lag in range(-self.max_lag, self.max_lag + 1):
            indicator_aligned, reference_aligned = self._align_series(
                indicator, reference, lag
            )

            valid_mask = ~(indicator_aligned.isna() | reference_aligned.isna())
            if valid_mask.sum() < 10:
                continue

            x = indicator_aligned[valid_mask].values
            y = reference_aligned[valid_mask].values

            if np.std(x) == 0 or np.std(y) == 0:
                continue

            corr_matrix = np.corrcoef(x, y)
            corr = corr_matrix[0, 1]

            if np.isnan(corr):
                continue

            r2 = corr ** 2

            if r2 > best_r2:
                best_r2 = r2
                best_lag = lag

        return best_r2, best_lag

    def calculate_dtw_distance(
        self,
        indicator: pd.Series,
        reference: pd.Series
    ) -> float:
        indicator_clean = indicator.dropna()
        reference_clean = reference.dropna()

        min_len = min(len(indicator_clean), len(reference_clean))
        if min_len < 10:
            return np.inf

        x = indicator_clean.values[-min_len:]
        y = reference_clean.values[-min_len:]

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

    def calculate_avg_monthly_dtw(
        self,
        indicator: pd.Series,
        reference: pd.Series
    ) -> float:
        dtw = self.calculate_dtw_distance(indicator, reference)
        return dtw / len(indicator.dropna())

    def evaluate_indicator(
        self,
        indicator: pd.Series,
        reference: pd.Series,
        direction: int = 1
    ) -> IndicatorScore:
        r2, lead_lag = self.calculate_time_diff_r2(indicator, reference)

        avg_dtw = self.calculate_avg_monthly_dtw(indicator, reference)

        direction_penalty = 0
        if direction != 0:
            recent_indicator = indicator.tail(6).diff().mean()
            recent_reference = reference.tail(6).diff().mean()
            if recent_indicator * recent_reference * direction < 0:
                direction_penalty = 0.1

        total_score = r2 - 0.2 * (avg_dtw / 100) - direction_penalty

        return IndicatorScore(
            name=indicator.name if hasattr(indicator, 'name') else 'unknown',
            r2=r2,
            lead_lag=lead_lag,
            dtw_dist=avg_dtw,
            total_score=total_score,
            direction=direction
        )

    def _align_series(
        self,
        indicator: pd.Series,
        reference: pd.Series,
        lag: int
    ) -> Tuple[pd.Series, pd.Series]:
        if lag > 0:
            indicator_aligned = indicator.iloc[:-lag].values
            reference_aligned = reference.iloc[lag:].values
            index = indicator.index[:-lag]
        elif lag < 0:
            indicator_aligned = indicator.iloc[-lag:].values
            reference_aligned = reference.iloc[:lag].values
            index = reference.index[:lag]
        else:
            indicator_aligned = indicator.values
            reference_aligned = reference.values
            index = indicator.index

        return (
            pd.Series(indicator_aligned, index=index),
            pd.Series(reference_aligned, index=index)
        )


class IndicatorSelector:
    def __init__(
        self,
        min_valid_length: int = 36,
        top_k: int = 10
    ):
        self.min_valid_length = min_valid_length
        self.top_k = top_k
        self.evaluator = IndicatorEvaluator()

    def select_indicators(
        self,
        indicators: Dict[str, pd.Series],
        reference: pd.Series,
        direction_dict: Dict[str, int] = None
    ) -> List[str]:
        scores = []

        for name, indicator in indicators.items():
            if len(indicator.dropna()) < self.min_valid_length:
                continue

            direction = direction_dict.get(name, 1) if direction_dict else 1

            try:
                score = self.evaluator.evaluate_indicator(indicator, reference, direction)
                score.name = name
                scores.append(score)
            except Exception as e:
                continue

        scores.sort(key=lambda x: x.total_score, reverse=True)

        selected = []
        selected_names = set()
        for score in scores:
            if len(selected) >= self.top_k:
                break
            if score.name not in selected_names:
                selected.append(score.name)
                selected_names.add(score.name)

        return selected


class NowcastingModel:
    def __init__(
        self,
        rolling_window: int = 60,
        min_valid_length: int = 36,
        top_k_indicators: int = 10
    ):
        self.rolling_window = rolling_window
        self.min_valid_length = min_valid_length
        self.top_k_indicators = top_k_indicators
        self.nowcasting = SimpleNowcasting(rolling_window)
        self.selector = IndicatorSelector(min_valid_length, top_k_indicators)
        self.evaluator = IndicatorEvaluator()

    def fit(
        self,
        indicators: Dict[str, pd.Series],
        reference: pd.Series,
        direction_dict: Dict[str, int] = None
    ) -> Tuple[pd.Series, List[str], List[IndicatorScore]]:
        if len(indicators) == 0:
            return pd.Series(), [], []

        selected_names = self.selector.select_indicators(
            indicators, reference, direction_dict
        )

        proxy_df = pd.DataFrame({name: indicators[name] for name in selected_names})

        prosperity_index = self.nowcasting.fit_predict(proxy_df, reference)

        scores = []
        for name in selected_names:
            try:
                score = self.evaluator.evaluate_indicator(
                    indicators[name], reference,
                    direction_dict.get(name, 1) if direction_dict else 1
                )
                scores.append(score)
            except:
                pass

        return prosperity_index, selected_names, scores

    def predict(
        self,
        indicators: Dict[str, pd.Series],
        selected_names: List[str]
    ) -> pd.Series:
        proxy_df = pd.DataFrame({name: indicators[name] for name in selected_names if name in indicators})
        if proxy_df.empty:
            return pd.Series()
        return self.nowcasting.simple_average(proxy_df)


class IndustryNowcasting:
    def __init__(
        self,
        rolling_window: int = 60,
        min_valid_length: int = 36,
        top_k: int = 10
    ):
        self.rolling_window = rolling_window
        self.top_k = top_k
        self.models = {}

    def fit_industry(
        self,
        industry_name: str,
        indicators: Dict[str, pd.Series],
        reference: pd.Series,
        direction_dict: Dict[str, int] = None
    ) -> Tuple[pd.Series, List[str], List[IndicatorScore]]:
        model = NowcastingModel(
            rolling_window=self.rolling_window,
            min_valid_length=min_valid_length,
            top_k_indicators=self.top_k
        )
        景气指数, selected, scores = model.fit(indicators, reference, direction_dict)
        self.models[industry_name] = model
        return 景气指数, selected, scores

    def get_prosperity_index(
        self,
        industry_name: str,
        indicators: Dict[str, pd.Series],
        selected_names: List[str]
    ) -> pd.Series:
        if industry_name not in self.models:
            return pd.Series()
        return self.models[industry_name].predict(indicators, selected_names)


def calculate_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: Optional[int] = None
) -> float:
    if window is None:
        valid = ~(series1.isna() | series2.isna())
        if valid.sum() < 10:
            return 0
        return series1[valid].corr(series2[valid])
    else:
        return series1.rolling(window).corr(series2).mean()


def main():
    print("Testing Simple-Nowcasting Model...")

    np.random.seed(42)
    dates = pd.date_range('2016-01-01', '2022-06-30', freq='M')
    n = len(dates)

    reference = pd.Series(
        np.random.randn(n).cumsum() * 2 + 10,
        index=dates,
        name='ROE_TTM_yoy'
    )

    indicators = {
        f'indicator_{i}': pd.Series(
            reference.values + np.random.randn(n) * 0.5,
            index=dates,
            name=f'indicator_{i}'
        )
        for i in range(10)
    }

    model = NowcastingModel(rolling_window=60, top_k_indicators=6)
    prosperity_index, selected, scores = model.fit(indicators, reference)

    print(f"Selected indicators: {selected}")
    print(f"Prosperity index shape: {prosperity_index.shape}")
    print(f"Correlation with reference: {calculate_correlation(prosperity_index, reference):.4f}")

    for score in scores[:3]:
        print(f"  {score.name}: R2={score.r2:.3f}, Lag={score.lead_lag}, Score={score.total_score:.3f}")


if __name__ == "__main__":
    main()
