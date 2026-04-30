# -*- coding: utf-8 -*-
"""
债券基金风格分析模块 - 基于净值回归方法

复现华泰证券研报《基于净值数据对债券基金久期和信用配置风格进行估计的方法》(2020-08-21)

核心功能：
- 久期风格估计：通过净值回归估计基金久期
- 信用风格估计：通过净值回归估计基金信用评分
"""

__version__ = "1.0.0"
__author__ = "QClaw Agent"

from .data_loader import BondDataLoader
from .factor import BondStyleEstimator
from .backtest import StyleBacktest
from .plot import BondStylePlotter
from .utils import *

__all__ = [
    "BondDataLoader",
    "BondStyleEstimator", 
    "StyleBacktest",
    "BondStylePlotter",
]
