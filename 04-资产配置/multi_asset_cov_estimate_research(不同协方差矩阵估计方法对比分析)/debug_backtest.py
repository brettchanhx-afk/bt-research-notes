import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine

print("=== 调试回测流程 ===")

# 初始化模块
data_fetcher = DataFetcher()
cov_estimator = CovarianceEstimator()
portfolio_builder = PortfolioBuilder()
backtest_engine = BacktestEngine(initial_capital=1000000)

# 获取数据
asset_config = {
    '沪深300': '000300.SH',
    '中证1000': '000852.SH',
}

start_date = '20170101'
end_date = '20231231'

available_data = {}
for asset_name, ts_code in asset_config.items():
    df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
    print(f"{asset_name}: {len(df)} 条数据")
    if len(df) > 0:
        available_data[asset_name] = df['returns']
        print(f"  日期范围: {df.index[0].date()} 到 {df.index[-1].date()}")

if available_data:
    returns_df = pd.DataFrame(available_data).dropna()
    print(f"\n数据准备完成: {returns_df.shape[0]} 天, {returns_df.shape[1]} 个资产")
    print(f"时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    
    # 测试单个方法
    method = 'sample_cov'
    print(f"\n=== 测试 {method} ===")
    
    # 检查lookback
    lookback_period = 252
    print(f"lookback_period: {lookback_period}")
    print(f"数据量是否足够: {len(returns_df) >= lookback_period + 30} ({len(returns_df)} >= {lookback_period + 30})")
    
    # 获取调仓日期
    monthly_idx = returns_df.resample('M').indices
    dates = sorted(list(monthly_idx.keys()))
    print(f"月度调仓日期数量: {len(dates)}")
    if len(dates) > 0:
        print(f"  第一个日期: {dates[0]}")
        print(f"  最后一个日期: {dates[-1]}")
    
    # 运行回测
    result = backtest_engine.run_rolling_backtest(
        returns=returns_df,
        cov_estimator=cov_estimator,
        portfolio_builder=portfolio_builder,
        method=method,
        lookback_period=lookback_period,
        rebalance_freq='monthly',
        allow_short=False,
        portfolio_type='min_variance'
    )
    
    print(f"\n回测结果: {result}")
    if result is not None:
        print(f"结果行数: {len(result)}")
        print(result.head())
    
    # 检查self.results
    print(f"\nself.results: {backtest_engine.results}")
    
else:
    print("没有可用数据")
