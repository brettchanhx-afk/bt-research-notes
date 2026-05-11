"""
Nowcasting行业景气度预测项目
source模块：核心算法和数据处理模块
"""

from .config import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, NOTEBOOK_DIR,
    TUSHARE_TOKEN, TUSHARE_URL, TARGET_INDUSTRY, TARGET_INDUSTRY_CODE,
    START_DATE, END_DATE, LATENT_FACTOR_NUM, MAX_INDICATORS, MIN_INDICATORS,
    FACTOR_INTERPRETABILITY_THRESHOLD, STATIONARITY_P_VALUE_THRESHOLD,
    EM_MAX_ITERATIONS, EM_CONVERGENCE_THRESHOLD, ROLLING_WINDOW_SIZE
)

from .utils import (
    check_stationarity, standardize_series, calculate_indicator_explanatory,
    detect_data_gaps, fill_missing_with_interpolation,
    calculate_rolling_correlation, calculate_direction_accuracy, get_date_range
)

from .data_fetcher import (
    MultiSourceDataFetcher, SteelIndustryDataFetcher, DataCache, get_all_steel_indicators,
    LocalDataLoader
)

from .dfm_model import DynamicFactorModel, DFMSentimentIndex, DFMParameters

from .nowcasting_model import (
    NowcastingModel, NowcastingResult, IndicatorSelector
)

from .sentiment_index import (
    SteelIndustrySentimentIndex, SentimentIndexComparison
)

from .backtest import (
    IndustryTimingBacktest, GodViewBacktest, TimingComparison, BacktestResult
)

__version__ = '1.0.0'

__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'OUTPUT_DIR', 'NOTEBOOK_DIR',
    'TUSHARE_TOKEN', 'TUSHARE_URL', 'TARGET_INDUSTRY', 'TARGET_INDUSTRY_CODE',
    'START_DATE', 'END_DATE', 'LATENT_FACTOR_NUM', 'MAX_INDICATORS', 'MIN_INDICATORS',
    'FACTOR_INTERPRETABILITY_THRESHOLD', 'STATIONARITY_P_VALUE_THRESHOLD',
    'EM_MAX_ITERATIONS', 'EM_CONVERGENCE_THRESHOLD', 'ROLLING_WINDOW_SIZE',
    'check_stationarity', 'standardize_series', 'calculate_indicator_explanatory',
    'detect_data_gaps', 'fill_missing_with_interpolation',
    'calculate_rolling_correlation', 'calculate_direction_accuracy', 'get_date_range',
    'MultiSourceDataFetcher', 'SteelIndustryDataFetcher', 'DataCache', 'get_all_steel_indicators',
    'LocalDataLoader',
    'DynamicFactorModel', 'DFMSentimentIndex', 'DFMParameters',
    'NowcastingModel', 'NowcastingResult', 'IndicatorSelector',
    'SteelIndustrySentimentIndex', 'SentimentIndexComparison',
    'IndustryTimingBacktest', 'GodViewBacktest', 'TimingComparison', 'BacktestResult'
]
