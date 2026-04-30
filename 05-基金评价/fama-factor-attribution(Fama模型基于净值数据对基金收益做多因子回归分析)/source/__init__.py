"""
Fama-French多因子归因分析模块

基于净值数据对基金收益做多因子回归分析
"""

from .data_loader import FundDataLoader, FactorDataLoader
from .factor import FamaFrenchFactorBuilder
from .backtest import FamaFrenchAttribution
from .plot import FamaFrenchPlotter
from .utils import calculate_metrics, format_results

__all__ = [
    'FundDataLoader',
    'FactorDataLoader', 
    'FamaFrenchFactorBuilder',
    'FamaFrenchAttribution',
    'FamaFrenchPlotter',
    'calculate_metrics',
    'format_results',
]

__version__ = '1.0.0'
