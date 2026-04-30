# -*- coding: utf-8 -*-
"""
ETF聚类优选系统
基于民生证券研报《ETF的聚类优选与热点趋势策略构建》

主要模块：
- data_loader: ETF数据获取
- clustering: K-means++聚类分析
- index_evaluator: 指数多维评价
- etf_evaluator: ETF产品筛选
- plot: 可视化
"""

from .data_loader import (
    get_etf_basic_info,
    get_etf_historical_data,
    get_index_constituents,
    get_benchmark_data,
    load_all_etf_data,
    save_data,
    load_data,
    generate_mock_etf_data,
)

from .clustering import (
    ETFIndexClustering,
    cluster_indices_by_constituents,
    create_similarity_features,
    evaluate_clustering_quality,
)

from .index_evaluator import (
    IndexEvaluator,
    evaluate_and_select_indices,
)

from .etf_evaluator import (
    ETFEvaluator,
    evaluate_and_select_etfs,
    generate_mock_etf_metrics,
)

from .plot import (
    ETFPlotter,
    plot_etf_clustering,
)

__all__ = [
    'get_etf_basic_info',
    'get_etf_historical_data',
    'get_index_constituents',
    'get_benchmark_data',
    'load_all_etf_data',
    'save_data',
    'load_data',
    'generate_mock_etf_data',
    'ETFIndexClustering',
    'cluster_indices_by_constituents',
    'create_similarity_features',
    'evaluate_clustering_quality',
    'IndexEvaluator',
    'evaluate_and_select_indices',
    'ETFEvaluator',
    'evaluate_and_select_etfs',
    'generate_mock_etf_metrics',
    'ETFPlotter',
    'plot_etf_clustering',
]
