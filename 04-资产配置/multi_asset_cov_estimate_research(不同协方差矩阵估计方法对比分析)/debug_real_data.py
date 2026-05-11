import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine

print("=== 调试真实数据回测 ===")

# 初始化模块
data_fetcher = DataFetcher()
cov_estimator = CovarianceEstimator()
portfolio_builder = PortfolioBuilder()
backtest_engine = BacktestEngine(initial_capital=1000000)

# 获取真实数据
asset_config = {
    '沪深300': '000300.SH',
    '中证1000': '000852.SH',
}

start_date = '20170101'
end_date = '20231231'

print("\n=== 数据获取 ===")
available_data = {}
for asset_name, ts_code in asset_config.items():
    print(f"获取 {asset_name} ({ts_code})...")
    df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
    print(f"  结果: {len(df)} 条数据")
    if len(df) > 0:
        available_data[asset_name] = df['returns']
        print(f"  日期范围: {df.index[0].date()} 到 {df.index[-1].date()}")

if available_data:
    returns_df = pd.DataFrame(available_data).dropna()
    print(f"\n数据准备完成: {returns_df.shape[0]} 天, {returns_df.shape[1]} 个资产")
    print(f"时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    
    # 检查数据类型
    print(f"\n数据类型: {type(returns_df.index)}")
    print(f"索引类型: {type(returns_df.index[0])}")
    
    # 获取调仓日期
    lookback_period = 252
    monthly_idx = returns_df.resample('M').indices
    dates = sorted(list(monthly_idx.keys()))
    start_idx = int(lookback_period / 22)
    if start_idx < 12:
        start_idx = 12
    if start_idx >= len(dates):
        start_idx = len(dates) - 1
    rebalance_dates = dates[start_idx:]
    
    print(f"\n调仓日期信息:")
    print(f"  总月度日期数: {len(dates)}")
    print(f"  起始索引: {start_idx}")
    print(f"  调仓日期数: {len(rebalance_dates)}")
    if len(rebalance_dates) > 0:
        print(f"  第一个调仓日: {rebalance_dates[0]}")
        print(f"  最后一个调仓日: {rebalance_dates[-1]}")
    
    # 运行回测
    print("\n=== 运行回测 ===")
    result = backtest_engine.run_rolling_backtest(
        returns=returns_df,
        cov_estimator=cov_estimator,
        portfolio_builder=portfolio_builder,
        method='sample_cov',
        lookback_period=lookback_period,
        rebalance_freq='monthly',
        allow_short=False,
        portfolio_type='min_variance'
    )
    
    print(f"\n回测结果:")
    if result is not None:
        print(f"  结果行数: {len(result)}")
        print(f"  结果列: {result.columns.tolist()}")
        print(f"  前5行:\n{result.head()}")
        print(f"\n  初始价值: {1000000}")
        print(f"  最终价值: {result['portfolio_value'].iloc[-1]:.2f}")
    else:
        print("  结果为None")
    
    # 检查 self.results
    print(f"\nself.results内容: {backtest_engine.results}")
    
    # 测试 compare_methods
    comparison = backtest_engine.compare_methods()
    print(f"\n对比结果:\n{comparison}")
    
else:
    print("\n没有可用数据")
