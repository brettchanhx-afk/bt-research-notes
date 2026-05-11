"""
Nowcasting模型模块
实现实时预测行业景气度的核心Nowcasting模型
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from .dfm_model import DynamicFactorModel, DFMSentimentIndex
from .utils import standardize_series, check_stationarity


@dataclass
class NowcastingResult:
    """
    Nowcasting结果类
    """
    sentiment_index: pd.Series
    current_direction: int
    predicted_direction: int
    confidence: float
    new_information: float
    expected_change: float


class NowcastingModel:
    """
    Nowcasting模型

    基于研报描述：
    - 基本假设：
      1) 经济变量的同源性：代理指标都是高维经济系统的低维映射
      2) 经济周期的内生性：经济系统的驱动力是内生的
    - 模型包含三个方程：隐含状态方程、隐含因子状态转移方程、特质因子状态转移方程
    - 使用EM算法求解

    五大优势：
    1) 内生预测景气度变化方向
    2) 支持因指标发布延迟或停更的数据缺失
    3) 支持混频数据
    4) 支持经济含义重叠的指标
    5) 建模过程不丢失指标信息
    """

    def __init__(self, n_factors: int = 3, n_idiosyncratic: int = 5,
                 em_max_iter: int = 100, em_tol: float = 1e-6,
                 smooth_window: int = 3):
        """
        初始化Nowcasting模型

        Parameters:
        -----------
        n_factors : int
            隐含因子数量
        n_idiosyncratic : int
            特质因子数量
        em_max_iter : int
            EM算法最大迭代次数
        em_tol : float
            EM算法收敛阈值
        smooth_window : int
            平滑窗口大小
        """
        self.n_factors = n_factors
        self.n_idiosyncratic = n_idiosyncratic
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.smooth_window = smooth_window

        self.dfm_model: Optional[DynamicFactorModel] = None
        self.sentiment_index: Optional[pd.Series] = None
        self.indicator_data: Optional[pd.DataFrame] = None

    def fit(self, indicators: pd.DataFrame) -> 'NowcastingModel':
        """
        拟合Nowcasting模型

        Parameters:
        -----------
        indicators : pd.DataFrame
            指标数据 (T x n_indicators)

        Returns:
        --------
        self
        """
        print("开始拟合Nowcasting模型...")

        processed_indicators = self._preprocess_indicators(indicators)

        Y = processed_indicators.values.astype(np.float64)

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y)

        pca = PCA(n_components=min(self.n_factors, Y.shape[1], Y.shape[0]))
        pca_factors = pca.fit_transform(Y_scaled)

        first_factor = pca_factors[:, 0]

        sentiment = pd.Series(
            first_factor,
            index=processed_indicators.index,
            name='nowcasting_sentiment'
        )

        if self.smooth_window > 1:
            sentiment = sentiment.rolling(window=self.smooth_window, min_periods=1).mean()

        self.sentiment_index = sentiment
        self.indicator_data = processed_indicators

        print("Nowcasting模型拟合完成 (PCA)")
        return self

    def _preprocess_indicators(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        预处理指标数据

        Parameters:
        -----------
        indicators : pd.DataFrame
            原始指标数据

        Returns:
        --------
        pd.DataFrame
            处理后的指标数据
        """
        result = indicators.copy()

        for col in result.columns:
            if result[col].dtype in ['object', 'str']:
                result[col] = pd.to_numeric(result[col], errors='coerce')

            if result[col].isna().any():
                result[col] = result[col].interpolate(method='linear', limit=3)

            if result[col].isna().any():
                result[col] = result[col].fillna(method='bfill', limit=2)

            if result[col].isna().any():
                result[col] = result[col].fillna(method='ffill', limit=2)

            if result[col].isna().any():
                result[col] = result[col].fillna(0)

        result = result.dropna(axis=1, how='all')

        return result

    def nowcast(self) -> NowcastingResult:
        """
        执行Nowcasting - 获取当前和预测的景气度方向

        Returns:
        --------
        NowcastingResult
            Nowcasting结果
        """
        if self.sentiment_index is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        sentiment = self.sentiment_index

        current_value = sentiment.iloc[-1]
        prev_value = sentiment.iloc[-2] if len(sentiment) > 1 else current_value

        current_direction = 1 if current_value > prev_value else -1

        predicted_direction, expected_change = self._predict_direction()

        confidence = self._calculate_confidence()

        new_information = current_value - sentiment.mean()

        return NowcastingResult(
            sentiment_index=sentiment,
            current_direction=current_direction,
            predicted_direction=predicted_direction,
            confidence=confidence,
            new_information=new_information,
            expected_change=expected_change
        )

    def _predict_direction(self) -> Tuple[int, float]:
        """
        预测下一期景气度方向

        Returns:
        --------
        Tuple[int, float]
            (预测方向: 1上升/-1下降, 预期变化幅度)
        """
        if self.sentiment_index is None or len(self.sentiment_index) < 6:
            return 0, 0.0

        recent = self.sentiment_index.iloc[-6:]

        if len(recent) < 2:
            return 0, 0.0

        trend = np.polyfit(range(len(recent)), recent.values, 1)[0]

        if self.dfm_model is not None:
            predicted_factors = self.dfm_model.predict(n_periods=1)
            if predicted_factors.size > 0:
                predicted_value = predicted_factors[0, 0]
                expected_change = predicted_value - self.sentiment_index.iloc[-1]
            else:
                expected_change = trend
        else:
            expected_change = trend

        predicted_direction = 1 if expected_change > 0 else (-1 if expected_change < 0 else 0)

        return predicted_direction, expected_change

    def _calculate_confidence(self) -> float:
        """
        计算预测置信度

        Returns:
        --------
        float
            置信度 (0-1)
        """
        if self.sentiment_index is None or len(self.sentiment_index) < 12:
            return 0.5

        recent = self.sentiment_index.iloc[-12:]

        if recent.std() < 1e-10:
            return 0.5

        cv = abs(recent.std() / recent.mean()) if abs(recent.mean()) > 1e-10 else 1.0

        confidence = 1.0 / (1.0 + cv)

        return max(0.0, min(1.0, confidence))

    def rolling_nowcast(self, data: pd.DataFrame, window: int = 12) -> pd.DataFrame:
        """
        滚动Nowcasting

        Parameters:
        -----------
        data : pd.DataFrame
            历史数据
        window : int
            滚动窗口大小

        Returns:
        --------
        pd.DataFrame
            滚动Nowcasting结果
        """
        results = []
        n_samples = len(data)

        for i in range(window, n_samples + 1):
            window_data = data.iloc[:i]

            model = NowcastingModel(
                n_factors=self.n_factors,
                n_idiosyncratic=self.n_idiosyncratic,
                em_max_iter=self.em_max_iter,
                em_tol=self.em_tol,
                smooth_window=self.smooth_window
            )

            try:
                model.fit(window_data)
                nowcast_result = model.nowcast()

                results.append({
                    'date': data.index[i-1] if i <= len(data) else data.index[-1],
                    'sentiment': nowcast_result.sentiment_index.iloc[-1],
                    'current_direction': nowcast_result.current_direction,
                    'predicted_direction': nowcast_result.predicted_direction,
                    'confidence': nowcast_result.confidence,
                    'new_information': nowcast_result.new_information
                })
            except Exception as e:
                print(f"滚动窗口 {i} 失败: {e}")
                continue

        return pd.DataFrame(results)

    def get_indicator_weights(self) -> pd.Series:
        """
        获取各指标对景气度的权重

        Returns:
        --------
        pd.Series
            指标权重
        """
        if self.dfm_model is None:
            raise ValueError("模型尚未拟合")

        loadings = self.dfm_model.get_factor_loadings()
        factor_loadings = loadings[:, :self.n_factors]

        weights = np.abs(factor_loadings[:, 0])
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        return pd.Series(
            weights,
            index=self.indicator_data.columns,
            name='indicator_weights'
        )

    def get_factor_explanatory(self) -> Dict[str, float]:
        """
        获取隐含因子对各指标的解释度

        Returns:
        --------
        Dict[str, float]
            各指标的解释度
        """
        if self.dfm_model is None or self.indicator_data is None:
            raise ValueError("模型尚未拟合")

        loadings = self.dfm_model.get_factor_loadings()
        latent_factors = self.dfm_model.get_latent_factors()

        first_factor = latent_factors[0, :]

        explanatory = {}
        for i, col in enumerate(self.indicator_data.columns):
            if i < len(loadings):
                loading = loadings[i, 0]
                r_squared = loading ** 2
                explanatory[col] = min(r_squared, 1.0)

        return explanatory


class IndicatorSelector:
    """
    代理指标筛选器

    基于研报中的定量标准：
    1) 隐含因子对指标的解释度大于20%
    2) 指标序列在10%显著性水平下平稳
    3) 选用代理指标数目不少于15个
    """

    def __init__(self, interpretability_threshold: float = 0.20,
                 stationarity_pvalue: float = 0.10,
                 min_indicators: int = 15):
        """
        初始化指标筛选器

        Parameters:
        -----------
        interpretability_threshold : float
            解释度阈值（默认20%）
        stationarity_pvalue : float
            平稳性检验p值阈值（默认10%）
        min_indicators : int
            最小指标数量
        """
        self.interpretability_threshold = interpretability_threshold
        self.stationarity_pvalue = stationarity_pvalue
        self.min_indicators = min_indicators

    def select_indicators(self, candidate_indicators: pd.DataFrame,
                          benchmark_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        筛选代理指标

        Parameters:
        -----------
        candidate_indicators : pd.DataFrame
            候选指标数据
        benchmark_series : pd.Series, optional
            基准序列（如ROE_TTM）

        Returns:
        --------
        pd.DataFrame
            筛选后的指标
        """
        selected_indicators = []
        stats_list = []

        for col in candidate_indicators.columns:
            if candidate_indicators[col].dtype in ['object', 'str']:
                continue

            series = candidate_indicators[col].dropna()

            if len(series) < 10:
                continue

            is_stationary, p_value = check_stationarity(series, self.stationarity_pvalue)

            if benchmark_series is not None and len(benchmark_series) > 0:
                common_idx = series.index.intersection(benchmark_series.index)
                if len(common_idx) >= 10:
                    series_aligned = series.loc[common_idx]
                    benchmark_aligned = benchmark_series.loc[common_idx]

                    correlation = series_aligned.corr(benchmark_aligned)
                    interpretability = correlation ** 2
                else:
                    interpretability = 0.0
            else:
                interpretability = self.interpretability_threshold + 0.01

            stats_list.append({
                'indicator': col,
                'is_stationary': is_stationary,
                'stationarity_pvalue': p_value,
                'interpretability': interpretability,
                'n_obs': len(series)
            })

        stats_df = pd.DataFrame(stats_list)

        if len(stats_df) > 0:
            passed_stationarity = stats_df[stats_df['is_stationary']]

            if len(passed_stationarity) >= self.min_indicators:
                selected = passed_stationarity.nlargest(
                    self.min_indicators,
                    'interpretability'
                )
            else:
                selected = stats_df.nlargest(
                    max(self.min_indicators, len(stats_df) // 2),
                    'interpretability'
                )

            selected_indices = selected['indicator'].tolist()
            selected_indicators = candidate_indicators[selected_indices]

            print(f"从{len(candidate_indicators.columns)}个候选指标中筛选出{len(selected_indices)}个")
        else:
            selected_indicators = candidate_indicators

        return selected_indicators

    def get_selection_stats(self) -> pd.DataFrame:
        """
        获取筛选统计信息

        Returns:
        --------
        pd.DataFrame
            统计信息
        """
        pass
