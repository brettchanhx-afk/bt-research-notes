"""
行业景气度分析包
复现华泰证券-中观景气度之上游资源中游材料 (2021-10-14)

模块结构:
- data_fetcher: 数据获取模块
- nowcasting_model: Nowcasting模型核心
- industry_indicators: 行业指标库配置
- preprocessing: 数据预处理
- evaluation: 评价指标
- industry_analyzer: 行业景气度分析器
- visualization: 可视化
"""

from .data_fetcher import (
    init_tushare,
    get_industry_roe_ttm,
    get_macro_ppi,
    get_market_indicator,
    get_commodity_price,
    get_futures_data,
    get_index_daily,
    get_trade_calendar,
    IndustryDataLoader
)

from .nowcasting_model import (
    NowcastingModel,
    SentimentIndexBuilder,
    calculate_roe_reproduction,
    calculate_direction_accuracy
)

from .industry_indicators import (
    INDUSTRY_INDICATORS,
    INDICATOR_LOADINGS,
    IndustryIndicatorLibrary,
    get_all_industries,
    get_industry_summary
)

from .preprocessing import (
    IndicatorPreprocessor,
    IndicatorSelector,
    detect_frequency,
    align_frequencies
)

from .evaluation import (
    calculate_r2_score,
    calculate_correlation,
    calculate_roe_reproduction as eval_roe_reproduction,
    calculate_direction_signal,
    calculate_latest_direction_accuracy,
    calculate_prediction_direction_accuracy,
    evaluate_sentiment_index,
    compare_with_benchmark,
    SentimentIndexEvaluator
)

from .industry_analyzer import (
    IndustrySentimentAnalyzer,
    MultiIndustrySentimentAnalyzer
)

from .visualization import (
    SentimentIndexVisualizer,
    plot_industry_chain
)

__version__ = '1.0.0'
__author__ = 'Mesotrade'
__description__ = '行业景气度分析工具包'

__all__ = [
    'init_tushare',
    'get_industry_roe_ttm',
    'get_macro_ppi',
    'get_market_indicator',
    'get_commodity_price',
    'get_futures_data',
    'get_index_daily',
    'get_trade_calendar',
    'IndustryDataLoader',
    'NowcastingModel',
    'SentimentIndexBuilder',
    'calculate_roe_reproduction',
    'calculate_direction_accuracy',
    'INDUSTRY_INDICATORS',
    'INDICATOR_LOADINGS',
    'IndustryIndicatorLibrary',
    'get_all_industries',
    'get_industry_summary',
    'IndicatorPreprocessor',
    'IndicatorSelector',
    'detect_frequency',
    'align_frequencies',
    'calculate_r2_score',
    'calculate_correlation',
    'calculate_direction_signal',
    'calculate_latest_direction_accuracy',
    'calculate_prediction_direction_accuracy',
    'evaluate_sentiment_index',
    'compare_with_benchmark',
    'SentimentIndexEvaluator',
    'IndustrySentimentAnalyzer',
    'MultiIndustrySentimentAnalyzer',
    'SentimentIndexVisualizer',
    'plot_industry_chain'
]
