import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('.')

import numpy as np
import pandas as pd
from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine

print("=== 简单回测测试 ===")

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

start_date = '20200101'
end_date = '20210101'

print("\n1. 获取数据")
available_data = {}
for asset_name, ts_code in asset_config.items():
    df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
    print(f"  {asset_name}: {len(df)} 条")
    if len(df) > 0:
        available_data[asset_name] = df['returns']

if available_data:
    returns_df = pd.DataFrame(available_data).dropna()
    print(f"\n2. 数据准备完成: {returns_df.shape[0]} 天, {returns_df.shape[1]} 个资产")
    print(f"   时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    
    # 获取调仓日期
    lookback_period = 120
    print(f"\n3. 计算调仓日期 (lookback={lookback_period})")
    
    monthly_idx = returns_df.resample('M').indices
    dates = sorted(list(monthly_idx.keys()))
    print(f"   月度日期数: {len(dates)}")
    if len(dates) > 0:
        print(f"   日期列表: {[d.date() for d in dates]}")
    
    start_idx = int(lookback_period / 22)
    if start_idx < 6:
        start_idx = 6
    if start_idx >= len(dates):
        start_idx = len(dates) - 1
    
    rebalance_dates = dates[start_idx:]
    print(f"   起始索引: {start_idx}")
    print(f"   调仓日期数: {len(rebalance_dates)}")
    
    # 运行单方法回测
    print("\n4. 运行单方法回测")
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
    
    print(f"\n5. 回测结果")
    if result is not None:
        print(f"   结果行数: {len(result)}")
        print(f"   结果列: {result.columns.tolist()}")
        print(f"   前3行:")
        print(result.head(3))
        print(f"\n   初始价值: 1000000")
        print(f"   最终价值: {result['portfolio_value'].iloc[-1]:.2f}")
        
        # 测试 compare_methods
        print("\n6. 测试 compare_methods")
        backtest_engine.results = {'sample_cov': result}
        comparison = backtest_engine.compare_methods()
        print(f"   对比结果:\n{comparison}")
        
        # 保存结果
        comparison.to_csv('output/backtest_results.csv', encoding='utf-8-sig')
        print("\n7. 结果已保存到 output/backtest_results.csv")
        
    else:
        print("   结果为None")
    
else:
    print("没有可用数据")

print("\n=== 测试完成 ===")
