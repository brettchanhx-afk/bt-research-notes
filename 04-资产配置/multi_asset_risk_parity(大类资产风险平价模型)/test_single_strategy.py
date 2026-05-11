"""
简化测试脚本 - 测试单个策略回测
"""

import sys
sys.path.insert(0, 'source')

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("测试数据获取...")
from data_fetcher import fetch_all_assets
data = fetch_all_assets(start_date='20090101', end_date='20100101')
print(f"数据形状: {data.shape}")
print(f"时间范围: {data.index.min().date()} 至 {data.index.max().date()}")

print("\n测试风险平价策略...")
from backtest import BacktestEngine
engine = BacktestEngine(data)

try:
    result = engine.run_risk_parity_strategy()
    print(f"策略名称: {result.strategy_name}")
    print(f"权重数据: {len(result.weights)} 条")
    print(f"收益率数据: {len(result.portfolio_returns)} 条")
    print("✓ 风险平价策略测试成功")
except Exception as e:
    print(f"✗ 风险平价策略测试失败: {e}")

print("\n测试完成!")