import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine

print("=== 模块测试 ===")

# 测试数据获取
print("\n1. 测试数据获取")
data_fetcher = DataFetcher()
df = data_fetcher.get_index_daily('000300.SH', '20200101', '20200131')
print(f"沪深300数据: {len(df)} 条")
if len(df) > 0:
    print(f"  日期范围: {df.index[0].date()} 到 {df.index[-1].date()}")
    print(f"  收益率均值: {df['returns'].mean():.6f}")

# 测试协方差估计
print("\n2. 测试协方差估计")
cov_estimator = CovarianceEstimator()
test_returns = pd.DataFrame({
    'A': np.random.randn(100) * 0.01,
    'B': np.random.randn(100) * 0.01
})
cov_matrix = cov_estimator.get_covariance(test_returns, method='sample_cov')
print(f"协方差矩阵形状: {cov_matrix.shape}")
print(f"协方差矩阵:\n{cov_matrix}")

# 测试组合构建
print("\n3. 测试组合构建")
portfolio_builder = PortfolioBuilder()
weights = portfolio_builder.minimum_variance_portfolio(cov_matrix, allow_short=False)
print(f"最小方差组合权重: {weights}")
print(f"权重和: {weights.sum():.6f}")

# 测试回测引擎
print("\n4. 测试回测引擎")
backtest_engine = BacktestEngine(initial_capital=1000000)

# 创建模拟数据
dates = pd.date_range('2020-01-01', '2021-12-31', freq='B')
sim_returns = pd.DataFrame({
    'A': np.random.randn(len(dates)) * 0.005,
    'B': np.random.randn(len(dates)) * 0.005
}, index=dates)

result = backtest_engine.run_rolling_backtest(
    returns=sim_returns,
    cov_estimator=cov_estimator,
    portfolio_builder=portfolio_builder,
    method='sample_cov',
    lookback_period=60,
    rebalance_freq='monthly',
    allow_short=False,
    portfolio_type='min_variance'
)

print(f"回测结果: {result}")
if result is not None:
    print(f"结果行数: {len(result)}")
    print(f"初始价值: {1000000}")
    print(f"最终价值: {result['portfolio_value'].iloc[-1]:.2f}")
    
    # 检查 self.results
    print(f"\nself.results内容: {backtest_engine.results}")
    
    # 测试 compare_methods
    comparison = backtest_engine.compare_methods()
    print(f"\n对比结果:\n{comparison}")
