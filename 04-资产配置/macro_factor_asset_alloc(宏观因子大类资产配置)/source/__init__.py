"""
宏观因子资产配置框架
基于国泰金工研报《基于宏观因子的大类资产配置框架》复现

模块列表:
- config: 配置参数
- data_fetcher: 数据获取
- macro_factors: 宏观因子构建
- factor_exposure: 因子暴露度计算
- portfolio_optimizer: 资产配置优化
- risk_analysis: 风险分析
- backtest: 回测引擎
- main: 主函数入口
"""

from .config import *
from .macro_factors import MacroFactorBuilder
from .factor_exposure import FactorExposureCalculator, FactorExposureWithPrior
from .portfolio_optimizer import PortfolioOptimizer, MacroFactorAllocator
from .risk_analysis import RiskAnalyzer, risk_attribution_analysis
from .backtest import BacktestEngine, BacktestResultAnalyzer

__version__ = "1.0.0"
__author__ = "量化工程团队"
__description__ = "基于宏观因子的大类资产配置框架"

__all__ = [
    'MacroFactorBuilder',
    'FactorExposureCalculator',
    'FactorExposureWithPrior',
    'PortfolioOptimizer',
    'MacroFactorAllocator',
    'RiskAnalyzer',
    'risk_attribution_analysis',
    'BacktestEngine',
    'BacktestResultAnalyzer',
]