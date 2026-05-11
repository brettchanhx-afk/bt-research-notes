"""
Standalone test script for the project modules.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("项目模块导入测试")
print("=" * 80)

from source.config import (
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DECAY_WEIGHTING_PARAMS,
    TF_PARAMS,
    RISK_FREE_RATE,
    DATA_DIR,
    OUTPUT_DIR
)

from source.utils import (
    calculate_returns,
    calculate_volatility,
    exponential_decay_weights,
    calculate_max_drawdown,
    calculate_sharpe_ratio
)

print("\n1. 配置模块测试通过")
print(f"   起始日期: {DEFAULT_START_DATE}")
print(f"   衰减加权参数: {DECAY_WEIGHTING_PARAMS}")
print(f"   趋势跟踪参数: {TF_PARAMS}")

print("\n2. 工具模块测试通过")

np.random.seed(42)
dates = pd.date_range('2017-01-01', periods=100, freq='D')
assets = ['Asset_A', 'Asset_B', 'Asset_C']

returns_data = np.random.randn(100, 3) * 0.02
returns = pd.DataFrame(returns_data, index=dates, columns=assets)

prices_data = 100 + np.cumsum(returns_data, axis=0)
prices = pd.DataFrame(prices_data, index=dates, columns=assets)

print("\n3. 测试exponential_decay_weights...")
weights = exponential_decay_weights(60, 30)
print(f"   权重和: {weights.sum():.6f}")

print("\n4. 测试calculate_sharpe_ratio...")
sharpe = calculate_sharpe_ratio(returns.mean())
print(f"   夏普比率: {sharpe:.4f}")

print("\n5. 测试calculate_max_drawdown...")
cum_ret = (1 + returns).cumprod() - 1
max_dd = calculate_max_drawdown(cum_ret.mean(axis=1))
print(f"   最大回撤: {max_dd:.4f}")

print("\n" + "=" * 80)
print("基础模块测试完成!")
print("=" * 80)

print("\n注意: 完整模块导入需要先安装项目为包")
print("运行: pip install -e .")
print("或使用Jupyter Notebook: notebooks/pcrp_backtest.ipynb")