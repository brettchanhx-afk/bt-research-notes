"""
主运行脚本 - 投资时钟行业配置策略
运行完整流程：数据获取 -> 因子合成 -> 策略回测 -> 结果输出
"""
import sys
sys.path.append('..')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from source.data_fetcher import DataFetcher
from source.factor_synthesis import FactorSynthesis
from source.factor_predictor import FactorPredictor
from source.asset_mapping import AssetMapping
from source.asset_strategy import AssetStrategy
from source.industry_strategy import IndustryStrategy
from source.backtest import BacktestEngine

OUTPUT_DIR = '../output'
DATA_DIR = '../data'
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

print('='*70)
print('投资时钟行业配置策略 - 完整运行')
print('='*70)

START_DATE = '20110101'
END_DATE = '20210630'

print('\n[1/8] 初始化数据获取器...')
fetcher = DataFetcher()
print(f'  tushare: {"可用" if fetcher.tushare_available else "不可用"}')
print(f'  akshare: {"可用" if fetcher.akshare_available else "不可用"}')
print(f'  baostock: {"可用" if fetcher.baostock_available else "不可用"}')

print('\n[2/8] 获取宏观因子数据...')
factor_dict = fetcher.get_all_macro_factors(start_date=START_DATE, end_date=END_DATE)
for name, series in factor_dict.items():
    print(f'  {name}: {len(series)} 条记录')
factor_df = pd.DataFrame(factor_dict)
factor_df.to_csv(os.path.join(DATA_DIR, '宏观因子_输出.csv'), index_label='date')
print(f'  -> 已保存至 data/宏观因子_输出.csv')

print('\n[3/8] 获取大类资产收益率...')
asset_returns = fetcher.get_asset_returns(start_date=START_DATE, end_date=END_DATE)
print(f'  资产数量: {len(asset_returns.columns)}')
print(f'  资产列表: {list(asset_returns.columns)}')
asset_returns.to_csv(os.path.join(DATA_DIR, '大类资产收益率_输出.csv'))
print(f'  -> 已保存至 data/大类资产收益率_输出.csv')

print('\n[4/8] 获取行业收益率...')
industry_returns = fetcher.get_industry_returns(start_date=START_DATE, end_date=END_DATE)
print(f'  行业数量: {len(industry_returns.columns)}')
industry_returns.to_csv(os.path.join(DATA_DIR, '行业收益率_输出.csv'))
print(f'  -> 已保存至 data/行业收益率_输出.csv')

print('\n[5/8] 初始化策略模块...')
factor_predictor = FactorPredictor(cycle_period=42)
asset_mapper = AssetMapping()
asset_strategy = AssetStrategy(target_volatility=0.05, max_leverage=2.0, risk_free_rate=0.04)
industry_strategy = IndustryStrategy(top_n=5, momentum_window=20)
backtester = BacktestEngine(initial_capital=1000000, transaction_cost=0.002, risk_free_rate=0.04)

print('\n[6/8] 构建投资时钟和资产映射...')
factor_predictions = factor_predictor.get_all_predictions(factor_dict)
asset_mapping = asset_mapper.build_macro_asset_mapping(factor_dict, asset_returns)

predictions_df = pd.DataFrame()
for factor_name, preds in factor_predictions.items():
    if 'combined' in preds:
        temp_df = pd.DataFrame({'date': preds['combined'].index, f'{factor_name}_view': preds['combined'].values})
        if predictions_df.empty:
            predictions_df = temp_df
        else:
            predictions_df = predictions_df.merge(temp_df, on='date')
if not predictions_df.empty:
    predictions_df.to_csv(os.path.join(DATA_DIR, '因子预测观点_输出.csv'), index=False)
    print(f'  -> 已保存至 data/因子预测观点_输出.csv')

clock_result = asset_mapper.build_growth_inflation_clock(
    factor_dict['growth'],
    factor_dict['inflation'],
    asset_returns
)
if 'clock' in clock_result:
    clock_df = pd.DataFrame({'date': pd.Series(clock_result['clock']).index, 'clock_state': clock_result['clock'].values})
    clock_df.to_csv(os.path.join(DATA_DIR, '投资时钟状态_输出.csv'), index=False)
    print(f'  -> 已保存至 data/投资时钟状态_输出.csv')

print('\n[7/8] 运行大类资产配置回测...')
asset_results = backtester.run_asset_backtest(
    asset_strategy,
    asset_returns,
    factor_dict,
    asset_mapping,
    start_date='2011-01-01',
    end_date='2021-06-30'
)
print(f'  累计收益: {asset_results.get("cumulative_return", 0):.2%}')
print(f'  年化收益: {asset_results.get("annual_return", 0):.2%}')
print(f'  夏普比率: {asset_results.get("sharpe_ratio", 0):.2f}')
print(f'  最大回撤: {asset_results.get("max_drawdown", 0):.2%}')

asset_results_df = pd.DataFrame({
    '指标': ['累计收益', '年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '月度胜率'],
    '大类资产策略': [
        f'{asset_results.get("cumulative_return", 0):.2%}',
        f'{asset_results.get("annual_return", 0):.2%}',
        f'{asset_results.get("annual_volatility", 0):.2%}',
        f'{asset_results.get("sharpe_ratio", 0):.2f}',
        f'{asset_results.get("max_drawdown", 0):.2%}',
        f'{asset_results.get("calmar_ratio", 0):.2f}',
        f'{asset_results.get("win_rate", 0):.2%}'
    ]
})

print('\n[8/8] 运行行业轮动回测...')
industry_mapping = asset_mapper.build_industry_mapping(factor_dict, industry_returns)
benchmark = asset_returns['沪深300'] if '沪深300' in asset_returns.columns else None

industry_results = backtester.run_industry_backtest(
    industry_strategy,
    industry_returns,
    factor_dict,
    industry_mapping,
    benchmark_returns=benchmark,
    start_date='2011-01-01',
    end_date='2021-06-30'
)
print(f'  累计收益: {industry_results.get("cumulative_return", 0):.2%}')
print(f'  年化收益: {industry_results.get("annual_return", 0):.2%}')
print(f'  夏普比率: {industry_results.get("sharpe_ratio", 0):.2f}')
print(f'  最大回撤: {industry_results.get("max_drawdown", 0):.2%}')

industry_results_df = pd.DataFrame({
    '指标': ['累计收益', '年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '月度胜率'],
    '行业轮动策略': [
        f'{industry_results.get("cumulative_return", 0):.2%}',
        f'{industry_results.get("annual_return", 0):.2%}',
        f'{industry_results.get("annual_volatility", 0):.2%}',
        f'{industry_results.get("sharpe_ratio", 0):.2f}',
        f'{industry_results.get("max_drawdown", 0):.2%}',
        f'{industry_results.get("calmar_ratio", 0):.2f}',
        f'{industry_results.get("win_rate", 0):.2%}'
    ]
})

comparison_df = asset_results_df.merge(industry_results_df, on='指标')
comparison_df.to_csv(os.path.join(RESULTS_DIR, '策略绩效对比.csv'), index=False)
print(f'\n  -> 已保存至 output/results/策略绩效对比.csv')

print('\n' + '='*70)
print('策略绩效对比')
print('='*70)
print(comparison_df.to_string(index=False))

print('\n[作图] 保存结果图表...')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

if 'portfolio_values' in asset_results and len(asset_results['portfolio_values']) > 0:
    values = asset_results['portfolio_values']
    axes[0, 0].plot(values.index, values.values, 'b-', linewidth=2)
    axes[0, 0].set_title('大类资产策略 - 组合价值', fontsize=14)
    axes[0, 0].set_xlabel('日期')
    axes[0, 0].set_ylabel('组合价值')
    axes[0, 0].grid(True, alpha=0.3)

if 'portfolio_values' in industry_results and len(industry_results['portfolio_values']) > 0:
    values = industry_results['portfolio_values']
    axes[0, 1].plot(values.index, values.values, 'r-', linewidth=2)
    axes[0, 1].set_title('行业轮动策略 - 组合价值', fontsize=14)
    axes[0, 1].set_xlabel('日期')
    axes[0, 1].set_ylabel('组合价值')
    axes[0, 1].grid(True, alpha=0.3)

for factor_name, series in factor_dict.items():
    if len(series) > 0:
        axes[1, 0].plot(series.index, series.values, label=factor_name, linewidth=1.5)
axes[1, 0].set_title('宏观因子走势', fontsize=14)
axes[1, 0].set_xlabel('日期')
axes[1, 0].set_ylabel('因子值')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

if 'returns_series' in asset_results and 'returns_series' in industry_results:
    asset_cum = (1 + asset_results['returns_series']).cumprod()
    industry_cum = (1 + industry_results['returns_series']).cumprod()
    axes[1, 1].plot(asset_cum.index, asset_cum.values, label='大类资产', linewidth=2)
    axes[1, 1].plot(industry_cum.index, industry_cum.values, label='行业轮动', linewidth=2)
    axes[1, 1].set_title('累计收益对比', fontsize=14)
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('累计收益')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'investment_clock_results.png'), dpi=150, bbox_inches='tight')
print(f'  -> 已保存至 output/results/investment_clock_results.png')

if 'returns_series' in asset_results:
    returns_df = pd.DataFrame({
        'date': asset_results['returns_series'].index,
        'asset_strategy_return': asset_results['returns_series'].values
    })
    if 'returns_series' in industry_results:
        returns_df['industry_strategy_return'] = industry_results['returns_series'].values
    returns_df.to_csv(os.path.join(RESULTS_DIR, '月度收益率.csv'), index=False)
    print(f'  -> 已保存至 output/results/月度收益率.csv')

print('\n' + '='*70)
print('运行完成！')
print('='*70)
print(f'数据文件: data/')
print(f'结果文件: output/results/')
