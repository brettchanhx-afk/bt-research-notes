"""
领先指标筛选模块
包括：时差相关系数、K-L信息量、拐点匹配率、DTW距离
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
try:
    from dtaidistance import dtw
    DTAI_AVAILABLE = True
except ImportError:
    DTAI_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')


class IndicatorSelector:
    def __init__(self):
        self.results = {}

    def calculate_cross_correlation(self, series1, series2, max_lag=12):
        """
        计算时差相关系数
        series1: 候选指标
        series2: 基准指标
        返回: 最佳滞后期数、最大相关系数
        """
        if len(series1) < max_lag + 12 or len(series2) < max_lag + 12:
            return 0, 0

        s1 = series1.dropna()
        s2 = series2.dropna()

        common_dates = s1.index.intersection(s2.index)
        if len(common_dates) < 24:
            return 0, 0

        s1 = s1.loc[common_dates]
        s2 = s2.loc[common_dates]

        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                corr = s1.iloc[lag:].corr(s2.iloc[:-lag])
            elif lag < 0:
                corr = s1.iloc[:lag].corr(s2.iloc[-lag:])
            else:
                corr = s1.corr(s2)
            correlations.append((lag, corr))

        best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)

        return best_lag, best_corr

    def calculate_kl_divergence(self, series1, series2, n_bins=20):
        """
        计算K-L信息量
        衡量候选指标对基准指标的预测能力
        """
        s1 = series1.dropna()
        s2 = series2.dropna()

        common_dates = s1.index.intersection(s2.index)
        if len(common_dates) < 24:
            return float('inf')

        s1 = s1.loc[common_dates].values
        s2 = s2.loc[common_dates].values

        min_val = min(s1.min(), s2.min())
        max_val = max(s1.max(), s2.max())

        bins = np.linspace(min_val, max_val, n_bins + 1)

        p_dist, _ = np.histogram(s1, bins=bins, density=True)
        q_dist, _ = np.histogram(s2, bins=bins, density=True)

        p_dist = p_dist + 1e-10
        q_dist = q_dist + 1e-10

        p_dist = p_dist / p_dist.sum()
        q_dist = q_dist / q_dist.sum()

        kl_div = entropy(p_dist, q_dist)

        return kl_div

    def find_turning_points_bry_boschan(self, series, mcd=3):
        """
        Bry-Boschan算法找拐点
        """
        if len(series) < 2 * mcd + 6:
            return [], []

        series = series.dropna()
        if len(series) < 24:
            return [], []

        values = series.values
        n = len(values)

        peaks = []
        troughs = []

        for i in range(mcd, n - mcd):
            is_peak = True
            is_trough = True

            for j in range(1, mcd + 1):
                if values[i] <= values[i - j] or values[i] <= values[i + j]:
                    is_peak = False
                if values[i] >= values[i - j] or values[i] >= values[i + j]:
                    is_trough = False

            if is_peak:
                peaks.append(i)
            if is_trough:
                troughs.append(i)

        return peaks, troughs

    def calculate_turning_point_match_rate(self, series1, series2, window=12):
        """
        计算拐点匹配率
        series1: 候选指标（可能领先）
        series2: 基准指标
        """
        peaks1, troughs1 = self.find_turning_points_bry_boschan(series1)
        peaks2, troughs2 = self.find_turning_points_bry_boschan(series2)

        if len(peaks2) == 0 and len(troughs2) == 0:
            return 0, [], 0, 0

        total_turning_points = len(peaks2) + len(troughs2)
        if total_turning_points == 0:
            return 0, [], 0, 0

        matched = 0
        avg_lead_time = 0
        lead_times = []

        all_points2 = [(idx, 'peak') for idx in peaks2] + [(idx, 'trough') for idx in troughs2]
        all_points2 = sorted(all_points2, key=lambda x: x[0])

        used_points1 = set()

        for idx2, type2 in all_points2:
            best_match = None
            best_distance = float('inf')

            peak_list = [(p, 'peak') for p in peaks1]
            trough_list = [(t, 'trough') for t in troughs1]
            combined_list = peak_list + trough_list

            for idx1, type1 in combined_list:
                if idx1 in used_points1:
                    continue
                if type1 != type2:
                    continue

                distance = idx2 - idx1

                if distance < 0:
                    continue

                if distance <= window and distance < best_distance:
                    best_match = idx1
                    best_distance = distance

            if best_match is not None:
                matched += 1
                lead_times.append(best_distance)
                used_points1.add(best_match)

        if matched > 0:
            avg_lead_time = np.mean(lead_times)
        else:
            avg_lead_time = 0

        match_rate = matched / total_turning_points if total_turning_points > 0 else 0

        return match_rate, lead_times, avg_lead_time, matched

    def calculate_dtw_distance(self, series1, series2):
        """
        计算动态时间规整距离
        衡量曲线形态相似性
        """
        s1 = series1.dropna()
        s2 = series2.dropna()

        common_dates = s1.index.intersection(s2.index)
        if len(common_dates) < 12:
            return float('inf')

        s1_normalized = (s1.loc[common_dates] - s1.loc[common_dates].mean()) / s1.loc[common_dates].std()
        s2_normalized = (s2.loc[common_dates] - s2.loc[common_dates].mean()) / s2.loc[common_dates].std()

        try:
            if DTAI_AVAILABLE:
                dtw_distance = dtw.distance(s1_normalized.values.astype(np.float64),
                                            s2_normalized.values.astype(np.float64))
            else:
                dtw_distance = self._simple_dtw(s1_normalized.values, s2_normalized.values)
        except:
            dtw_distance = self._simple_dtw(s1_normalized.values, s2_normalized.values)

        return dtw_distance

    def _simple_dtw(self, s1, s2):
        """
        简化的DTW实现
        """
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )

        return dtw_matrix[n, m]

    def select_leading_indicators(self, candidate_series_dict, benchmark_series,
                                 max_lag=12, min_correlation=0.5,
                                 max_kl_divergence=2.0,
                                 min_match_rate=0.7,
                                 max_dtw_distance=1.0):
        """
        综合筛选领先指标
        """
        results = []

        for name, candidate in candidate_series_dict.items():
            if len(candidate) < 24 or len(benchmark_series) < 24:
                continue

            common_dates = candidate.index.intersection(benchmark_series.index)
            if len(common_dates) < 24:
                continue

            cand = candidate.loc[common_dates]
            bench = benchmark_series.loc[common_dates]

            lag, corr = self.calculate_cross_correlation(cand, bench, max_lag)

            if abs(corr) < min_correlation:
                continue

            if lag <= 0:
                continue

            kl_div = self.calculate_kl_divergence(cand, bench)

            match_rate, lead_times, avg_lead, matched = self.calculate_turning_point_match_rate(
                cand, bench, window=12
            )

            dtw_dist = self.calculate_dtw_distance(cand, bench)

            is_leading = (
                lag > 0 and
                abs(corr) >= min_correlation and
                match_rate >= min_match_rate and
                dtw_dist <= max_dtw_distance
            )

            results.append({
                'indicator': name,
                'lag': lag,
                'correlation': corr,
                'kl_divergence': kl_div,
                'match_rate': match_rate,
                'avg_lead_time': avg_lead,
                'dtw_distance': dtw_dist,
                'is_leading': is_leading,
                'lead_times': lead_times
            })

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            results_df = results_df.sort_values(
                ['match_rate', 'correlation', 'dtw_distance'],
                ascending=[False, False, True]
            )

        return results_df

    def evaluate_indicator_quality(self, series, name='indicator'):
        """
        评估单个指标的质量
        """
        quality = {
            'name': name,
            'data_length': len(series),
            'missing_ratio': series.isna().sum() / len(series) if len(series) > 0 else 1,
            'mean': series.mean() if len(series) > 0 else 0,
            'std': series.std() if len(series) > 0 else 0,
            'min': series.min() if len(series) > 0 else 0,
            'max': series.max() if len(series) > 0 else 0
        }

        if len(series) > 12:
            quality['autocorr_12'] = series.autocorr(lag=12)

        return quality

    def filter_high_correlation_pairs(self, series_dict, threshold=0.9):
        """
        筛选高相关性指标对，避免重复计算
        """
        names = list(series_dict.keys())
        n = len(names)
        high_corr_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                s1 = series_dict[names[i]].dropna()
                s2 = series_dict[names[j]].dropna()

                common = s1.index.intersection(s2.index)
                if len(common) < 12:
                    continue

                corr = s1.loc[common].corr(s2.loc[common])

                if abs(corr) >= threshold:
                    high_corr_pairs.append((names[i], names[j], corr))

        return high_corr_pairs

    def get_leading_indicator_summary(self, results_df):
        """
        获取领先指标筛选结果汇总
        """
        if results_df.empty:
            return {}

        leading_df = results_df[results_df['is_leading'] == True]

        summary = {
            'total_candidates': len(results_df),
            'leading_indicators': len(leading_df),
            'avg_lead_time': leading_df['avg_lead_time'].mean() if len(leading_df) > 0 else 0,
            'avg_correlation': leading_df['correlation'].mean() if len(leading_df) > 0 else 0,
            'avg_match_rate': leading_df['match_rate'].mean() if len(leading_df) > 0 else 0,
            'indicators': leading_df.to_dict('records') if len(leading_df) > 0 else []
        }

        return summary
