# -*- coding: utf-8 -*-
"""
fund_stock_timing - 基金选股择时能力定量评价模型
T-M / H-M / C-L Model Implementation
华泰金工研究 | 2020-08-21
"""

from .factor import TMModel, HMModel, CLModel, StockTimingEvaluator
from .data_loader import load_all_data, load_fund_nav, load_benchmark
from .backtest import RollingTimingBacktest, PerformanceAttribution
from .utils import summary_stats, get_fund_name

__all__ = [
    'TMModel', 'HMModel', 'CLModel', 'StockTimingEvaluator',
    'load_all_data', 'load_fund_nav', 'load_benchmark',
    'RollingTimingBacktest', 'PerformanceAttribution',
    'summary_stats', 'get_fund_name',
]
