# -*- coding: utf-8 -*-
"""
Barra模型因子归因 - 源码包
"""

from .factor import (
    BarraFactorAttribution,
    FactorExposureCalculator,
    FactorReturnCalculator,
    FundExposureCalculator,
    FactorAttributionResult,
    CrossSectionResult
)

from .data_loader import BarraDataLoader

from .backtest import (
    BarraRollingBacktest,
    BacktestConfig,
    AttributionStabilityTest,
    PerformanceSummary
)

from .plot import BarraVisualizer
