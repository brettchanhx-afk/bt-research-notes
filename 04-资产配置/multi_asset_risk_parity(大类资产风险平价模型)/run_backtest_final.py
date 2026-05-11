"""
完整回测脚本 - 使用真实补充数据
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
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
print("数据获取完成，共 {} 条记录".format(len(data)))
print("时间范围: {} 至 {}".format(data.index.min().date(), data.index.max().date()))

# 2. 初始化回测引擎
print("\n【步骤2】初始化回测引擎...")
from backtest import BacktestEngine
engine = BacktestEngine(data)

# 3. 运行所有策略回测
print("\n【步骤3】运行策略回测...")

# 风险平价策略
print("1. 风险平价策略...")
try:
    rp_result = engine.run_risk_parity_strategy()
    print("   完成")
except Exception as e:
    print("   失败:", e)

# 波动率倒数策略
print("2. 波动率倒数策略...")
try:
    vi_result = engine.run_volatility_inverse_strategy()
    print("   完成")
except Exception as e:
    print("   失败:", e)

# 固定资产比例策略
print("3. 固定资产比例策略...")
try:
    fr_result = engine.run_fixed_ratio_strategy()
    print("   完成")
except Exception as e:
    print("   失败:", e)

# 等权重策略
print("4. 等权重策略...")
try:
    ew_result = engine.run_equal_weight_strategy()
    print("   完成")
except Exception as e:
    print("   失败:", e)

# 夏普预算策略
print("5. 夏普预算策略...")
try:
    sb_result = engine.run_sharpe_budget_strategy()
    print("   完成")
except Exception as e:
    print("   失败:", e)

# 加杠杆风险平价策略
print("6. 加杠杆风险平价策略...")
try:
    lr_result = engine.run_leveraged_risk_parity_strategy(target_volatility=0.03)
    print("   完成")
except Exception as e:
    print("   失败:", e)

# 因子风险平价策略
print("7. 因子风险平价策略...")
try:
    frp_result = engine.run_factor_risk_parity_strategy()
    print("   完成")
except Exception as e:
    print("   失败:", e)

# 4. 性能分析
print("\n【步骤4】性能分析...")
from performance import generate_performance_summary, print_performance_report

results = engine.get_all_results()
if results:
    report = generate_performance_summary(results)
    print("\n策略性能对比报告")
    print("-" * 60)
    print(report.to_string())
    
    # 保存结果
    report.to_csv('output/策略性能报告.csv', encoding='utf-8-sig')
    print("\n性能报告已保存至 output/策略性能报告.csv")
    
    # 保存各策略净值
    for name, result in results.items():
        if hasattr(result, 'portfolio_returns'):
            nav = (1 + result.portfolio_returns).cumprod()
            nav.to_csv('output/{}_净值.csv'.format(name), encoding='utf-8-sig')
            print("{} 净值曲线已保存".format(name))
else:
    print("没有成功的回测结果")

# 保存原始数据
data.to_csv('output/asset_prices.csv', encoding='utf-8-sig')
print("原始资产价格数据已保存")

print("\n" + "=" * 60)
print("回测完成！")
print("=" * 60)