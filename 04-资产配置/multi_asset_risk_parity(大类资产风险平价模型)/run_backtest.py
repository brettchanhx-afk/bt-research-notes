"""
完整回测脚本 - 运行所有策略并输出结果
"""

import sys
sys.path.insert(0, 'source')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

# 导入模块
from data_fetcher import fetch_all_assets
from backtest import BackTestEngine
from performance import PerformanceAnalyzer

print("=" * 60)
print("桥水全天候策略和风险平价模型完整回测")
print("=" * 60)

# 1. 获取数据
print("\n【步骤1】获取所有资产数据...")
data = fetch_all_assets(start_date='20080101', end_date='20260430')
print(f"数据获取完成，共 {len(data)} 条记录")
print(f"时间范围: {data.index.min().date()} 至 {data.index.max().date()}")

# 2. 初始化回测引擎
print("\n【步骤2】初始化回测引擎...")
engine = BackTestEngine(data)

# 3. 运行所有策略回测
print("\n【步骤3】运行策略回测...")
strategies = ['risk_parity', 'volatility_parity', 'fixed_weight', 'equal_weight', 'sharp_budget', 'leveraged_risk_parity', 'factor_risk_parity']
results = {}

for strategy in strategies:
    print(f"  运行 {strategy}...")
    result = engine.run_strategy(strategy)
    results[strategy] = result

# 4. 性能分析
print("\n【步骤4】性能分析...")
analyzer = PerformanceAnalyzer()
report = analyzer.generate_report(results)

# 打印性能报告
print("\n" + "=" * 60)
print("策略性能对比报告")
print("=" * 60)
print(report.to_string())

# 5. 保存结果
print("\n【步骤5】保存结果...")
report.to_csv('output/策略性能报告.csv', encoding='utf-8-sig')
print("  性能报告已保存至 output/策略性能报告.csv")

# 保存各策略净值曲线
for strategy, result in results.items():
    if 'portfolio_value' in result:
        result['portfolio_value'].to_csv(f'output/{strategy}_净值.csv', encoding='utf-8-sig')
        print(f"  {strategy} 净值曲线已保存")

print("\n" + "=" * 60)
print("回测完成！")
print("=" * 60)