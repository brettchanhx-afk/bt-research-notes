# -*- coding: utf-8 -*-
"""
source 包：DEPI 基金分析核心模块
"""
from .data_loader import (
    get_fund_list,
    get_fund_nav_history,
    get_fund_fee_rate,
    get_benchmark_history,
    get_fund_info,
)
from .factor import (
    calc_returns,
    calc_excess_return,
    calc_volatility,
    calc_timing_ability,
    build_factor_table,
)
from .backtest import DEPIEngine, backtest_depi
from .plot import setup_chinese_font, plot_depi_timeseries, plot_depi_distribution
from .utils import standardize, normalize, parse_fund_code, save_df

__all__ = [
    'get_fund_list', 'get_fund_nav_history', 'get_fund_fee_rate',
    'get_benchmark_history', 'get_fund_info',
    'calc_returns', 'calc_excess_return', 'calc_volatility',
    'calc_timing_ability', 'build_factor_table',
    'DEPIEngine', 'backtest_depi',
    'setup_chinese_font', 'plot_depi_timeseries', 'plot_depi_distribution',
    'standardize', 'normalize', 'parse_fund_code', 'save_df',
]
