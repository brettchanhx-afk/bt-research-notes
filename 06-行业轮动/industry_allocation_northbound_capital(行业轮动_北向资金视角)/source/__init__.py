"""
北向资金量化策略项目

该项目复现华泰证券金工深度研究报告《析精剖微：机构拆解看北向资金》
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

from . import config
from . import utils
from . import data
from . import factors
from . import strategies
from . import backtest
from . import visualization

__all__ = [
    "config",
    "utils",
    "data",
    "factors",
    "strategies",
    "backtest",
    "visualization",
]
