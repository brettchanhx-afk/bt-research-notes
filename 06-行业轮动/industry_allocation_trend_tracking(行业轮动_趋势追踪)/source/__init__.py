"""
趋势追踪行业配置策略
模块包初始化文件
"""

from .data_loader import DataLoader, load_asset_data, INDEX_CODES, INDUSTRY_CODES
from .indicators import TrendIndicatorCalculator, MA, EMA, MACD, ROC, DPO, TSI
from .backtest import BacktestEngine, BacktestResult, StrategyEvaluator
from .monte_carlo import MonteCarloSimulator, VirtualSequenceGenerator
from .cscv import CSCVTest, OverfittingDetector, StrategyRobustnessAnalyzer
from .strategy_builder import (
    TrendFollowingStrategyBuilder,
    AssetAllocationStrategyBuilder,
    IndustryRotationStrategyBuilder,
    StrategyConfig,
    build_asset_allocation_strategies,
    build_industry_rotation_strategies
)

__version__ = "1.0.0"
__author__ = "量化研究团队"

__all__ = [
    "DataLoader",
    "load_asset_data",
    "INDEX_CODES",
    "INDUSTRY_CODES",
    "TrendIndicatorCalculator",
    "MA",
    "EMA",
    "MACD",
    "ROC",
    "DPO",
    "TSI",
    "BacktestEngine",
    "BacktestResult",
    "StrategyEvaluator",
    "MonteCarloSimulator",
    "VirtualSequenceGenerator",
    "CSCVTest",
    "OverfittingDetector",
    "StrategyRobustnessAnalyzer",
    "TrendFollowingStrategyBuilder",
    "AssetAllocationStrategyBuilder",
    "IndustryRotationStrategyBuilder",
    "StrategyConfig",
    "build_asset_allocation_strategies",
    "build_industry_rotation_strategies",
]