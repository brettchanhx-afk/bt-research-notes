"""
完整回测脚本 - 运行所有策略并输出结果
"""

import sys
sys.path.insert(0, 'source')

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

print("=" * 60)
print("桥水全天候策略和风险平价模型完整回测")
print("=" * 60)

# 1. 获取数据
print("\n【步骤1】获取所有资产数据...")
from data_fetcher import fetch_all_assets
data = fetch_all_assets(start_date='20080101', end_date='20260430')
print(f"数据获取完成，共 {len(data)} 条记录")
print(f"时间范围: {data.index.min().date()} 至 {data.index.max().date()}")

# 2. 初始化回测引擎
print("\n【步骤2】初始化回测引擎...")
from backtest import BacktestEngine
engine = BacktestEngine(data)

# 3. 运行所有策略回测
print("\n【步骤3】运行策略回测...")
try:
    results = engine.run_all_strategies()
    print("  所有策略回测完成")
except Exception as e:
    print(f"  回测失败: {e}")
    results = {}

# 4. 性能分析
print("\n【步骤4】性能分析...")
from performance import generate_performance_summary

# 过滤掉失败的策略
valid_results = {k: v for k, v in results.items() if v is not None}
report = generate_performance_summary(valid_results)

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
for strategy, result in valid_results.items():
    if 'portfolio_value' in result:
        result['portfolio_value'].to_csv(f'output/{strategy}_净值.csv', encoding='utf-8-sig')
        print(f"  {strategy} 净值曲线已保存")

# 保存原始数据
data.to_csv('output/asset_prices.csv', encoding='utf-8-sig')
print("  原始资产价格数据已保存")

print("\n" + "=" * 60)
print("回测完成！")
print("=" * 60)