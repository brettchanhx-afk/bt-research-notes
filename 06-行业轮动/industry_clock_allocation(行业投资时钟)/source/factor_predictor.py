"""
因子预测模块
包括：相位判断法、因子动量法、复合策略
"""
import pandas as pd
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class FactorPredictor:
    def __init__(self, cycle_period=42):
        self.cycle_period = cycle_period
        self.phase_views = {}
        self.momentum_views = {}
        self.combined_views = {}

    def fit_sine_wave(self, series, period=None):
        """
        拟合正弦波获取周期参数
        """
        if len(series) < period // 2 if period else 24:
            return None

        t = np.arange(len(series))
        y = series.values

        if period is None:
            period = self.cycle_period

        try:
            def sine_func(x, A, phase, offset):
                return A * np.sin(2 * np.pi * x / period + phase) + offset

            p0 = [y.max() - y.min(), 0, y.mean()]

            popt, _ = curve_fit(sine_func, t, y, p0=p0, maxfev=5000)

            return {
                'amplitude': popt[0],
                'phase': popt[1],
                'offset': popt[2],
                'fitted': sine_func(t, *popt)
            }
        except:
            return None

    def calculate_phase(self, fitted_series, current_idx):
        """
        计算当前相位（相对于周期的位置）
        """
        if fitted_series is None or current_idx >= len(fitted_series):
            return 0

        phase = (2 * np.pi * current_idx / self.cycle_period) % (2 * np.pi)
        return phase

    def phase_judgment(self, factor_series, lookback=12):
        """
        相位判断法
        基于42个月周期判断因子当前所处位置
        """
        if len(factor_series) < self.cycle_period:
            return pd.Series(dtype=float)

        views = pd.Series(index=factor_series.index, dtype=float)

        for i in range(self.cycle_period, len(factor_series)):
            historical_data = factor_series.iloc[i - self.cycle_period:i]

            fit_result = self.fit_sine_wave(historical_data, self.cycle_period)

            if fit_result is not None:
                current_phase = (2 * np.pi * (len(historical_data) - 1)) / self.cycle_period + fit_result['phase']
                current_phase = current_phase % (2 * np.pi)

                factor_diff = factor_series.iloc[i] - factor_series.iloc[i-1] if i > 0 else 0

                top_range = (np.pi / 6, 5 * np.pi / 6)
                bottom_range = (7 * np.pi / 6, 11 * np.pi / 6)

                if top_range[0] <= current_phase <= top_range[1]:
                    if factor_diff > 0:
                        views.iloc[i] = 1
                    else:
                        views.iloc[i] = 0
                elif bottom_range[0] <= current_phase <= bottom_range[1]:
                    if factor_diff < 0:
                        views.iloc[i] = -1
                    else:
                        views.iloc[i] = 0
                elif current_phase < np.pi:
                    views.iloc[i] = 1
                else:
                    views.iloc[i] = -1
            else:
                if i > 0:
                    factor_diff = factor_series.iloc[i] - factor_series.iloc[i-1]
                    views.iloc[i] = 1 if factor_diff > 0 else (-1 if factor_diff < 0 else 0)
                else:
                    views.iloc[i] = 0

        views.iloc[:self.cycle_period] = 0

        self.phase_views = views
        return views

    def factor_momentum(self, factor_series, window=3, consecutive=2):
        """
        因子动量法
        比较当期因子值与过去N期均值
        """
        if len(factor_series) < window + 1:
            return pd.Series(dtype=float)

        rolling_mean = factor_series.rolling(window=window).mean()

        momentum = factor_series - rolling_mean.shift(1)

        momentum_diff = momentum.diff()

        views = pd.Series(index=factor_series.index, dtype=float)

        positive_count = 0
        negative_count = 0

        for i in range(len(momentum_diff)):
            if np.isnan(momentum_diff.iloc[i]):
                views.iloc[i] = 0
                continue

            if momentum_diff.iloc[i] > 0:
                positive_count += 1
                negative_count = 0
            elif momentum_diff.iloc[i] < 0:
                negative_count += 1
                positive_count = 0
            else:
                pass

            if positive_count >= consecutive:
                views.iloc[i] = 1
            elif negative_count >= consecutive:
                views.iloc[i] = -1
            else:
                views.iloc[i] = 0

        self.momentum_views = views
        return views

    def combined_prediction(self, phase_views=None, momentum_views=None):
        """
        复合策略：结合相位判断和因子动量
        """
        if phase_views is None:
            phase_views = self.phase_views
        if momentum_views is None:
            momentum_views = self.momentum_views

        common_idx = phase_views.index.intersection(momentum_views.index)

        combined = pd.Series(index=common_idx, dtype=float)

        for idx in common_idx:
            phase_val = phase_views.loc[idx]
            momentum_val = momentum_views.loc[idx]

            combined_val = phase_val + momentum_val

            if combined_val >= 1:
                combined.loc[idx] = 1
            elif combined_val <= -1:
                combined.loc[idx] = -1
            else:
                combined.loc[idx] = 0

        self.combined_views = combined
        return combined

    def predict_factor_direction(self, factor_series, method='combined',
                                 window=3, consecutive=2):
        """
        预测因子方向
        """
        if method == 'phase':
            return self.phase_judgment(factor_series)
        elif method == 'momentum':
            return self.factor_momentum(factor_series, window, consecutive)
        elif method == 'combined':
            phase = self.phase_judgment(factor_series)
            momentum = self.factor_momentum(factor_series, window, consecutive)
            return self.combined_prediction(phase, momentum)
        else:
            return pd.Series(0, index=factor_series.index)

    def rolling_predict(self, factor_series, method='combined'):
        """
        滚动预测（用于回测）
        """
        if len(factor_series) < self.cycle_period:
            return pd.Series(0, index=factor_series.index)

        predictions = pd.Series(index=factor_series.index, dtype=float)

        for i in range(self.cycle_period, len(factor_series)):
            historical = factor_series.iloc[:i]
            pred = self.predict_factor_direction(historical, method)
            predictions.iloc[i] = pred.iloc[-1] if len(pred) > 0 else 0

        predictions.iloc[:self.cycle_period] = 0

        return predictions

    def detect_cycle_peaks_troughs(self, factor_series, period=None):
        """
        检测周期拐点
        """
        if period is None:
            period = self.cycle_period

        if len(factor_series) < period:
            return [], []

        fit_result = self.fit_sine_wave(factor_series, period)

        if fit_result is None:
            return [], []

        fitted = fit_result['fitted']

        peaks = []
        troughs = []

        for i in range(1, len(fitted) - 1):
            if fitted[i] > fitted[i-1] and fitted[i] > fitted[i+1]:
                peaks.append(i)
            elif fitted[i] < fitted[i-1] and fitted[i] < fitted[i+1]:
                troughs.append(i)

        return peaks, troughs

    def get_current_regime(self, factor_series, date=None):
        """
        获取当前因子状态
        """
        if len(factor_series) == 0:
            return 'unknown'

        if date is not None and date in factor_series.index:
            idx = factor_series.index.get_loc(date)
        else:
            idx = len(factor_series) - 1

        if idx < self.cycle_period:
            return 'unknown'

        historical = factor_series.iloc[:idx+1]
        view = self.predict_factor_direction(historical).iloc[-1]

        if view == 1:
            return 'up'
        elif view == -1:
            return 'down'
        else:
            return 'neutral'

    def get_all_predictions(self, factor_dict):
        """
        获取所有因子的预测
        """
        predictions = {}

        for factor_name, factor_series in factor_dict.items():
            if len(factor_series) > 0:
                predictions[factor_name] = {
                    'phase': self.phase_judgment(factor_series),
                    'momentum': self.factor_momentum(factor_series),
                    'combined': self.combined_prediction(
                        self.phase_judgment(factor_series),
                        self.factor_momentum(factor_series)
                    )
                }

        return predictions
