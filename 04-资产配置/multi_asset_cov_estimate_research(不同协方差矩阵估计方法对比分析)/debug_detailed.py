import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine

print("=== 详细调试回测流程 ===")

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

print("\n=== 数据获取 ===")
available_data = {}
for asset_name, ts_code in asset_config.items():
    print(f"获取 {asset_name} ({ts_code})...")
    df = data_fetcher.get_index_daily(ts_code, start_date, end_date)
    print(f"  结果: {len(df)} 条数据")
    if len(df) > 0:
        available_data[asset_name] = df['returns']
        print(f"  日期范围: {df.index[0].date()} 到 {df.index[-1].date()}")
        print(f"  收益率样本: {df['returns'].dropna().head(3).values}")
    else:
        print(f"  获取失败!")

if available_data:
    returns_df = pd.DataFrame(available_data).dropna()
    print(f"\n数据准备完成: {returns_df.shape[0]} 天, {returns_df.shape[1]} 个资产")
    print(f"时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    print(f"前5行数据:\n{returns_df.head()}")
    
    # 检查lookback
    lookback_period = 252
    print(f"\n=== 参数检查 ===")
    print(f"lookback_period: {lookback_period}")
    print(f"数据量是否足够: {len(returns_df) >= lookback_period + 30} ({len(returns_df)} >= {lookback_period + 30})")
    
    # 获取调仓日期
    print("\n=== 调仓日期 ===")
    monthly_idx = returns_df.resample('M').indices
    dates = sorted(list(monthly_idx.keys()))
    print(f"月度调仓日期数量: {len(dates)}")
    if len(dates) > 0:
        print(f"  前5个日期: {dates[:5]}")
        print(f"  后5个日期: {dates[-5:]}")
    
    # 计算start_idx
    start_idx = int(lookback_period / 22)
    print(f"\nstart_idx计算: {int(lookback_period / 22)} (252/22)")
    print(f"实际使用的start_idx: {max(start_idx, 12)}")
    
    # 运行回测
    print("\n=== 运行回测 ===")
    method = 'sample_cov'
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
    
    print(f"\n回测结果:")
    if result is not None:
        print(f"  结果行数: {len(result)}")
        print(f"  结果列: {result.columns.tolist()}")
        print(f"  前5行:\n{result.head()}")
    else:
        print(f"  结果为None")
    
    # 检查self.results
    print(f"\nself.results内容: {backtest_engine.results}")
    
    # 手动测试协方差估计
    print("\n=== 测试协方差估计 ===")
    test_data = returns_df.head(252)
    cov_matrix = cov_estimator.get_covariance(test_data, method='sample_cov')
    print(f"协方差矩阵:\n{cov_matrix}")
    
    # 测试组合构建
    print("\n=== 测试组合构建 ===")
    weights = portfolio_builder.minimum_variance_portfolio(cov_matrix, allow_short=False)
    print(f"权重: {weights}")
    print(f"权重和: {weights.sum()}")
    
else:
    print("\n没有可用数据")

# 保存调试数据
if available_data:
    returns_df.to_csv('output/debug_returns.csv')
    print("\n调试数据已保存到 output/debug_returns.csv")
