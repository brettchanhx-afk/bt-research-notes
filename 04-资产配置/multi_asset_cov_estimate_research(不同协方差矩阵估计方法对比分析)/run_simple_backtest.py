import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine
from source.evaluation import PortfolioEvaluator

print("=== 使用已有数据运行简化版回测 ===")
print("可用资产: 沪深300, 中证1000")
print()

data_fetcher = DataFetcher()
cov_estimator = CovarianceEstimator()
portfolio_builder = PortfolioBuilder()
backtest_engine = BacktestEngine(initial_capital=1000000)
evaluator = PortfolioEvaluator()

asset_config = {
    '沪深300': '000300.SH',
    '中证1000': '000852.SH',
}

start_date = '20170101'
end_date = '20231231'

available_data = {}
for asset_name, ts_code in asset_config.items():
    df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
    if len(df) > 0:
        available_data[asset_name] = df['returns']

if available_data:
    returns_df = pd.DataFrame(available_data).dropna()
    print(f"合并数据: {returns_df.shape[0]} 天, {returns_df.shape[1]} 个资产")
    print(f"时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    print()

    methods = ['sample_cov', 'risk_metrics', 'ledoit_wolf_single_factor']
    
    results = {}
    for method in methods:
        print(f"运行回测: {method}")
        result = backtest_engine.run_rolling_backtest(
            returns=returns_df,
            cov_estimator=cov_estimator,
            portfolio_builder=portfolio_builder,
            method=method,
            lookback_period=252,
            rebalance_freq='monthly',
            allow_short=False,
            portfolio_type='min_variance'
        )
        results[method] = result

    comparison_df = backtest_engine.compare_methods()
    print("\n=== 回测结果 ===")
    print(comparison_df.round(4))
    comparison_df.to_csv('output/backtest_results.csv', encoding='utf-8')
    print("\n结果已保存到: output/backtest_results.csv")

    plt.figure(figsize=(12, 6))
    for method, result in results.items():
        if result is not None and len(result) > 0:
            plt.plot(result['portfolio_value'].values, label=method)
    
    plt.title('不同协方差估计方法 - 最低波动组合价值曲线')
    plt.xlabel('时间')
    plt.ylabel('组合价值')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/min_variance_backtest.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: output/min_variance_backtest.png")

else:
    print("没有可用数据")

print()
print("=== 缺失数据说明 ===")
print("完整复现研报需要以下数据:")
print("1. 恒生指数")
print("2. 标普500")
print("3. 中债-国债总财富指数")
print("4. 中债-企业债总财富指数")
print("5. 南华商品指数")
print()
print("请将CSV文件放入 data/ 目录，包含 trade_date 和 close 列")