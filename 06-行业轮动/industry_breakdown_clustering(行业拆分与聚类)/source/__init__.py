"""
华泰证券行业拆分与聚类复现项目

该项目复现了华泰证券金工研究报告《确立研究对象：行业拆分与聚类》中的方法论
"""

from .data_fetcher import (
    get_industry_stocks,
    get_industry_index_daily,
    get_stock_daily,
    get_all_industry_members,
    get_industry_price_data,
    calculate_returns_from_prices,
    CITIC_INDUSTRIES,
    CITIC_INDUSTRY_CODES
)

from .industry_split import (
    IndustryReturnDivergence,
    IndustryFundamentalDivergence,
    IndustrySplitter
)

from .industry_cluster import (
    MonteCarloKMeans,
    MaximumSpanningTree,
    IndustryClustering
)

from .evaluation import (
    ReturnHomogeneityEvaluator,
    FundamentalHomogeneityEvaluator,
    ComprehensiveEvaluator
)

from .visualization import (
    DivergenceVisualizer,
    ClusterVisualizer,
    TimeSeriesVisualizer
)

from . import config

__version__ = '1.0.0'
__author__ = 'Quantitative Research Team'

__all__ = [
    'get_industry_stocks',
    'get_industry_index_daily',
    'get_stock_daily',
    'get_all_industry_members',
    'get_industry_price_data',
    'calculate_returns_from_prices',
    'IndustryReturnDivergence',
    'IndustryFundamentalDivergence',
    'IndustrySplitter',
    'MonteCarloKMeans',
    'MaximumSpanningTree',
    'IndustryClustering',
    'ReturnHomogeneityEvaluator',
    'FundamentalHomogeneityEvaluator',
    'ComprehensiveEvaluator',
    'DivergenceVisualizer',
    'ClusterVisualizer',
    'TimeSeriesVisualizer',
    'config'
]
