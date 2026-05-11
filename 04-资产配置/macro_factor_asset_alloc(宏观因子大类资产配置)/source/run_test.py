"""
运行测试脚本 - 逐步执行宏观因子资产配置框架
"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print('=' * 70)
print('Macro Factor Asset Allocation Framework - Test Run')
print('=' * 70)

# 1. 配置参数
from config import ALL_ASSETS, MACRO_FACTORS, BACKTEST_CONFIG, OUTPUT_DIR
print(f'\n[Step 1] Configuration loaded')
print(f'  Assets: {len(ALL_ASSETS)}')
print(f'  Factors: {MACRO_FACTORS}')
print(f'  Backtest period: {BACKTEST_CONFIG["start_date"]} to {BACKTEST_CONFIG["end_date"]}')

# 2. 生成模拟数据
print(f'\n[Step 2] Generating simulated data...')
np.random.seed(42)
dates = pd.date_range('2015-01-01', '2023-05-31', freq='M')
n = len(dates)
asset_returns = pd.DataFrame({
    '沪深300': np.random.randn(n) * 0.025 + 0.005,
    '中证500': np.random.randn(n) * 0.03 + 0.003,
    '中债国债': np.random.randn(n) * 0.008 + 0.002,
    '中债企业债': np.random.randn(n) * 0.01 + 0.002,
    '中证转债': np.random.randn(n) * 0.02 + 0.003,
    '南华工业品': np.random.randn(n) * 0.03 + 0.004,
    '南华农产品': np.random.randn(n) * 0.025 + 0.002,
    '布伦特原油': np.random.randn(n) * 0.04 + 0.001,
    '沪金': np.random.randn(n) * 0.025 + 0.003,
    '美元兑人民币': np.random.randn(n) * 0.015 + 0.001,
    '恒生指数': np.random.randn(n) * 0.03 + 0.002,
}, index=dates)
print(f'  Asset returns shape: {asset_returns.shape}')
print(f'  Date range: {asset_returns.index[0]} to {asset_returns.index[-1]}')

# 3. 构建宏观因子
from macro_factors import MacroFactorBuilder
print(f'\n[Step 3] Building macro factors...')
factor_builder = MacroFactorBuilder(n_factors=6)
factor_returns = factor_builder.construct_all_factors(asset_returns)
print(f'  Factor returns shape: {factor_returns.shape}')
print(f'  Factors: {factor_returns.columns.tolist()}')

# 4. 计算因子暴露
from factor_exposure import FactorExposureWithPrior
print(f'\n[Step 4] Computing factor exposures...')
exposure_calc = FactorExposureWithPrior(alpha=0.01)
exposure_matrix = exposure_calc.fit(asset_returns, factor_returns)
print(f'  Exposure matrix shape: {exposure_matrix.shape}')

# 5. 运行回测
from backtest import BacktestEngine, BacktestResultAnalyzer
print(f'\n[Step 5] Running backtest...')
backtest_engine = BacktestEngine(
    start_date='2015-01-01',
    end_date='2023-05-31',
    rebalance_freq='monthly',
    factor_deviation=0.05
)

all_results = {}
for factor in MACRO_FACTORS:
    print(f'  Testing {factor} factor deviation...')
    result = backtest_engine.run_factor_deviation_backtest(
        asset_returns, factor_returns, target_factor=factor
    )
    all_results[factor] = result

# 6. 生成汇总报告
print(f'\n[Step 6] Generating summary report...')
analyzer = BacktestResultAnalyzer(all_results)
summary = analyzer.generate_summary_report(factor_returns)

print('\n' + '=' * 70)
print('Backtest Results Summary')
print('=' * 70)
print(summary.to_string(index=False))

# 7. 保存结果
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_file = os.path.join(OUTPUT_DIR, 'factor_strategy_summary.csv')
summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f'\n[Results saved to: {summary_file}]')

print('\n' + '=' * 70)
print('Framework execution completed successfully!')
print('=' * 70)