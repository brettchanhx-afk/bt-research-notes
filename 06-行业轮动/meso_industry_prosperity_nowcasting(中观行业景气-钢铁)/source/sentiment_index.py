"""
行业景气度指数构建模块
基于Nowcasting模型构建行业景气度指数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

from .nowcasting_model import NowcastingModel, IndicatorSelector
from .dfm_model import DynamicFactorModel
from .utils import (
    standardize_series, check_stationarity, calculate_direction_accuracy
)


class SteelIndustrySentimentIndex:
    """
    钢铁行业景气度指数构建器

    基于研报中的钢铁行业案例：
    - 150余个指标的指标库
    - 筛选出31个景气度代理指标
    """

    def __init__(self, n_factors: int = 3, n_indicators: int = 31,
                 interpretability_threshold: float = 0.20,
                 stationarity_pvalue: float = 0.10):
        """
        初始化钢铁行业景气度指数构建器

        Parameters:
        -----------
        n_factors : int
            隐含因子数量
        n_indicators : int
            指标数量
        interpretability_threshold : float
            解释度阈值
        stationarity_pvalue : float
            平稳性p值阈值
        """
        self.n_factors = n_factors
        self.n_indicators = n_indicators
        self.interpretability_threshold = interpretability_threshold
        self.stationarity_pvalue = stationarity_pvalue

        self.nowcasting_model: Optional[NowcastingModel] = None
        self.sentiment_index: Optional[pd.Series] = None
        self.selected_indicators: Optional[pd.DataFrame] = None
        self.indicator_stats: Optional[pd.DataFrame] = None

    def build_sentiment_index(self, indicator_data: pd.DataFrame,
                               benchmark: Optional[pd.Series] = None,
                               use_selector: bool = True) -> pd.Series:
        """
        构建行业景气度指数

        Parameters:
        -----------
        indicator_data : pd.DataFrame
            指标数据 (T x n_indicators)
        benchmark : pd.Series, optional
            基准序列（如ROE_TTM）
        use_selector : bool
            是否使用指标筛选器

        Returns:
        --------
        pd.Series
            景气度指数
        """
        print("开始构建行业景气度指数...")

        indicator_for_selection = indicator_data.copy()
        if benchmark is not None and len(benchmark) > 0:
            common_idx = indicator_data.index.intersection(benchmark.index)
            if len(common_idx) > 20:
                print(f"对齐到共同区间: {common_idx.min()} 到 {common_idx.max()}")
                indicator_for_selection = indicator_data.loc[common_idx].copy()

        if use_selector:
            selector = IndicatorSelector(
                interpretability_threshold=self.interpretability_threshold,
                stationarity_pvalue=self.stationarity_pvalue,
                min_indicators=min(15, len(indicator_for_selection.columns))
            )

            self.selected_indicators = selector.select_indicators(
                indicator_for_selection,
                benchmark
            )
        else:
            self.selected_indicators = indicator_data.copy()

        if len(self.selected_indicators.columns) < 3:
            print("警告：筛选后指标数量不足，使用原始指标")
            self.selected_indicators = indicator_data.copy()

        print(f"最终使用{len(self.selected_indicators.columns)}个指标构建景气度指数")

        selected_cols = self.selected_indicators.columns.tolist()
        full_data_for_model = indicator_data[selected_cols].copy()

        print(f"使用全量数据训练DFM模型: {full_data_for_model.shape}")

        self.nowcasting_model = NowcastingModel(
            n_factors=self.n_factors,
            smooth_window=3
        )

        self.nowcasting_model.fit(full_data_for_model)

        self.sentiment_index = self.nowcasting_model.sentiment_index.copy()

        self.indicator_stats = self._calculate_indicator_stats()

        print("行业景气度指数构建完成")
        return self.sentiment_index

    def _calculate_indicator_stats(self) -> pd.DataFrame:
        """
        计算各指标的统计信息

        Returns:
        --------
        pd.DataFrame
            指标统计信息
        """
        if self.selected_indicators is None:
            return pd.DataFrame()

        stats_list = []

        for col in self.selected_indicators.columns:
            series = self.selected_indicators[col].dropna()

            if len(series) < 5:
                continue

            is_stationary, p_value = check_stationarity(series, self.stationarity_pvalue)

            stats_list.append({
                'indicator': col,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'is_stationary': is_stationary,
                'stationarity_pvalue': p_value,
                'n_obs': len(series)
            })

        return pd.DataFrame(stats_list)

    def evaluate_direction_accuracy(self, benchmark: pd.Series) -> Dict[str, float]:
        """
        评估方向预测准确率

        Parameters:
        -----------
        benchmark : pd.Series
            基准序列（如ROE_TTM环比变化）

        Returns:
        --------
        Dict[str, float]
            准确率评估结果
        """
        if self.sentiment_index is None:
            raise ValueError("景气度指数尚未构建")

        common_idx = self.sentiment_index.index.intersection(benchmark.index)

        if len(common_idx) < 10:
            return {'current_accuracy': 0.0, 'predicted_accuracy': 0.0}

        sentiment_aligned = self.sentiment_index.loc[common_idx]
        benchmark_aligned = benchmark.loc[common_idx]

        sentiment_diff = sentiment_aligned.diff()
        benchmark_diff = benchmark_aligned.diff()

        sentiment_direction = np.sign(sentiment_diff)
        benchmark_direction = np.sign(benchmark_diff)

        current_accuracy = calculate_direction_accuracy(
            sentiment_direction.shift(1).dropna(),
            benchmark_direction.dropna()
        )

        predicted_accuracy = current_accuracy

        return {
            'current_accuracy': current_accuracy,
            'predicted_accuracy': predicted_accuracy,
            'n_samples': len(common_idx)
        }

    def get_indicator_weights(self) -> pd.Series:
        """
        获取指标权重

        Returns:
        --------
        pd.Series
            指标权重
        """
        if self.nowcasting_model is None:
            raise ValueError("模型尚未拟合")

        return self.nowcasting_model.get_indicator_weights()

    def rolling_build(self, full_data: pd.DataFrame,
                       window: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
        """
        滚动构建景气度指数

        Parameters:
        -----------
        full_data : pd.DataFrame
            完整指标数据
        window : int
            滚动窗口大小

        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            (滚动景气度指数, 滚动统计信息)
        """
        rolling_results = []
        rolling_stats = []

        n_samples = len(full_data)

        for i in range(window, n_samples + 1, 3):
            window_data = full_data.iloc[:i]

            try:
                sentiment = self.build_sentiment_index(window_data, use_selector=False)

                rolling_results.append({
                    'date': full_data.index[i-1] if i <= len(full_data) else full_data.index[-1],
                    'sentiment': sentiment.iloc[-1]
                })

                accuracy = self.evaluate_direction_accuracy(
                    full_data.iloc[:i].iloc[:, 0]
                ) if len(full_data.columns) > 0 else {}

                rolling_stats.append({
                    'date': full_data.index[i-1] if i <= len(full_data) else full_data.index[-1],
                    **accuracy
                })

            except Exception as e:
                print(f"滚动窗口 {i} 失败: {e}")
                continue

        rolling_sentiment = pd.DataFrame(rolling_results).set_index('date')['sentiment']
        rolling_stats_df = pd.DataFrame(rolling_stats).set_index('date')

        return rolling_sentiment, rolling_stats_df


class SentimentIndexComparison:
    """
    景气度指数对比分析
    """

    def __init__(self):
        self.indices: Dict[str, pd.Series] = {}
        self.comparison_results: Optional[pd.DataFrame] = None

    def add_index(self, name: str, index: pd.Series):
        """
        添加景气度指数

        Parameters:
        -----------
        name : str
            指数名称
        index : pd.Series
            景气度指数
        """
        self.indices[name] = index

    def compare_with_benchmark(self, benchmark: pd.Series) -> pd.DataFrame:
        """
        与基准对比

        Parameters:
        -----------
        benchmark : pd.Series
            基准序列

        Returns:
        --------
        pd.DataFrame
            对比结果
        """
        comparison_data = []

        for name, index in self.indices.items():
            common_idx = index.index.intersection(benchmark.index)

            if len(common_idx) < 5:
                continue

            index_aligned = index.loc[common_idx]
            benchmark_aligned = benchmark.loc[common_idx]

            index_diff = np.sign(index_aligned.diff())
            benchmark_diff = np.sign(benchmark_aligned.diff())

            accuracy = calculate_direction_accuracy(
                index_diff.shift(1).dropna(),
                benchmark_diff.dropna()
            )

            correlation = index_aligned.corr(benchmark_aligned)

            comparison_data.append({
                'index_name': name,
                'direction_accuracy': accuracy,
                'correlation': correlation,
                'n_samples': len(common_idx)
            })

        self.comparison_results = pd.DataFrame(comparison_data)
        return self.comparison_results

    def get_leading_lags(self, benchmark: pd.Series,
                          max_lag: int = 6) -> pd.DataFrame:
        """
        计算领先滞后关系

        Parameters:
        -----------
        benchmark : pd.Series
            基准序列
        max_lag : int
            最大滞后期数

        Returns:
        --------
        pd.DataFrame
            领先滞后分析结果
        """
        lag_results = []

        for name, index in self.indices.items():
            common_idx = index.index.intersection(benchmark.index)

            if len(common_idx) < max_lag * 2:
                continue

            index_aligned = index.loc[common_idx]
            benchmark_aligned = benchmark.loc[common_idx]

            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    shifted_index = index_aligned.iloc[lag:]
                    aligned_benchmark = benchmark_aligned.iloc[:-lag]
                elif lag < 0:
                    shifted_index = index_aligned.iloc[:lag]
                    aligned_benchmark = benchmark_aligned.iloc[-lag:]
                else:
                    shifted_index = index_aligned
                    aligned_benchmark = benchmark_aligned

                common = shifted_index.index.intersection(aligned_benchmark.index)

                if len(common) < 5:
                    continue

                corr = shifted_index.loc[common].corr(aligned_benchmark.loc[common])

                lag_results.append({
                    'index_name': name,
                    'lag': lag,
                    'correlation': corr
                })

        return pd.DataFrame(lag_results)
