"""
Decay-Weighted PCA Risk Parity Model Package
============================================

A quantitative asset allocation package implementing:
- Principal Components Risk Parity (PCRP) model
- Decay weighting for volatility and correlation estimation
- Trend following for expected return estimation

Author: Quantitative Research Team
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

from .config import *
from .utils import *

__all__ = [
    'config',
    'utils',
]