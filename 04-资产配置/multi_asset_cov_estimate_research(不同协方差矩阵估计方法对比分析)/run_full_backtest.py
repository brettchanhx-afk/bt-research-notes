import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine
from source.evaluation import PortfolioEvaluator

print("=== 运行完整回测 ===")

# 初始化模块
data_fetcher = DataFetcher()
cov_estimator = CovarianceEstimator()
portfolio_builder = PortfolioBuilder()
backtest_engine = BacktestEngine(initial_capital=1000000)
evaluator = PortfolioEvaluator()

# 加载多资产数据
print("加载多资产数据...")
returns_df = data_fetcher.load_multi_asset_data('data/多资产行情序列.csv')

if returns_df is not None and len(returns_df) > 0:
    print(f"数据准备完成: {returns_df.shape[0]} 天, {returns_df.shape[1]} 个资产")
    print(f"资产列表: {returns_df.columns.tolist()}")
    print(f"时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    print()

    # 协方差估计方法列表
    methods = [
        'sample_cov',
        'ledoit_wolf_constant_variance',
        'ledoit_wolf_single_factor',
        'ledoit_wolf_constant_correlation',
        'risk_metrics',
    ]

    print("=== 运行最低波动组合回测（允许做空） ===")

    # 使用 run_multi_method_backtest 方法，允许做空以获得有意义的权重
    results = backtest_engine.run_multi_method_backtest(
        returns=returns_df,
        cov_estimator=cov_estimator,
        portfolio_builder=portfolio_builder,
        methods=methods,
        lookback_period=252,
        rebalance_freq='monthly',
        allow_short=True,  # 允许做空以获得有意义的最小方差组合
        portfolio_type='min_variance'
    )

    # 手动设置 self.results 以便 compare_methods 可以使用
    backtest_engine.results = results

    valid_methods = sum(1 for v in results.values() if v is not None)
    print(f"\n回测完成，有效方法数量: {valid_methods}")

    # 生成对比表格
    comparison_df = backtest_engine.compare_methods()
    print("\n=== 回测结果对比 ===")
    print(comparison_df.round(4))

    if not comparison_df.empty:
        comparison_df.to_csv('output/backtest_results.csv', encoding='utf-8-sig')
        print("\n结果已保存到: output/backtest_results.csv")

        # 绘制组合价值曲线
        plt.figure(figsize=(14, 7))
        for method, result in results.items():
            if result is not None and len(result) > 0:
                plt.plot(result['portfolio_value'].values, label=method, linewidth=1.5)

        plt.title('Different Covariance Estimation Methods - Minimum Variance Portfolio Value (Allow Short)', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Portfolio Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig('output/min_variance_backtest.png', dpi=150, bbox_inches='tight')
        print("组合价值曲线图已保存到: output/min_variance_backtest.png")

        # 绘制绩效指标对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        comparison_df['annualized_return'].plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('Annualized Return', fontsize=12)
        ax1.set_ylabel('Return')
        ax1.tick_params(axis='x', rotation=45)

        ax2 = axes[0, 1]
        comparison_df['annualized_volatility'].plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title('Annualized Volatility', fontsize=12)
        ax2.set_ylabel('Volatility')
        ax2.tick_params(axis='x', rotation=45)

        ax3 = axes[1, 0]
        comparison_df['sharpe_ratio'].plot(kind='bar', ax=ax3, color='green')
        ax3.set_title('Sharpe Ratio', fontsize=12)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.tick_params(axis='x', rotation=45)

        ax4 = axes[1, 1]
        comparison_df['max_drawdown'].plot(kind='bar', ax=ax4, color='red')
        ax4.set_title('Maximum Drawdown', fontsize=12)
        ax4.set_ylabel('Drawdown')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('output/performance_metrics.png', dpi=150, bbox_inches='tight')
        print("绩效指标对比图已保存到: output/performance_metrics.png")

        # 计算协方差矩阵RMSE
        print("\n=== 协方差矩阵RMSE分析 ===")
        true_cov = returns_df.cov().values
        rmse_results = {}
        for method in methods:
            est_cov = cov_estimator.get_covariance(returns_df, method=method)
            rmse = cov_estimator.compute_rmse(est_cov, true_cov)
            rmse_results[method] = rmse
            print(f"  {method}: RMSE = {rmse:.6f}")

        rmse_df = pd.DataFrame({'RMSE': rmse_results}).T
        rmse_df.to_csv('output/covariance_rmse.csv', encoding='utf-8-sig')
        print("\nRMSE结果已保存到: output/covariance_rmse.csv")

        # 绘制协方差矩阵热力图
        n_assets = min(5, returns_df.shape[1])
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        methods_to_plot = ['sample_cov', 'ledoit_wolf_single_factor', 'risk_metrics']

        for i, method in enumerate(methods_to_plot):
            cov_matrix = cov_estimator.get_covariance(returns_df, method=method)
            im = axes[i].imshow(cov_matrix[:n_assets, :n_assets], cmap='RdBu_r', aspect='auto', vmin=-0.001, vmax=0.001)
            axes[i].set_title(method, fontsize=12)
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.savefig('output/covariance_matrices.png', dpi=150, bbox_inches='tight')
        print("协方差矩阵热力图已保存到: output/covariance_matrices.png")

        # 保存调仓日期和权重
        weights_data = []
        for method, result in results.items():
            if result is not None and len(result) > 0:
                for idx, row in result.iterrows():
                    weights_data.append({
                        'method': method,
                        'date': idx,
                        'weights': row['weights']
                    })
        weights_df = pd.DataFrame(weights_data)
        weights_df.to_csv('output/rebalance_weights.csv', encoding='utf-8-sig', index=False)
        print("调仓权重已保存到: output/rebalance_weights.csv")
    else:
        print("\n警告：没有有效的回测结果")

else:
    print("没有可用数据，请检查 data/多资产行情序列.csv 文件")

print("\n=== 回测完成 ===")
print("\n输出文件清单:")
print("  - output/backtest_results.csv")
print("  - output/min_variance_backtest.png")
print("  - output/performance_metrics.png")
print("  - output/covariance_rmse.csv")
print("  - output/covariance_matrices.png")
print("  - output/rebalance_weights.csv")