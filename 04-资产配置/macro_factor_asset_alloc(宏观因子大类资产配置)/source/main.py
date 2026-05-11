"""
宏观因子资产配置框架 - 主函数模块
整合所有模块，实现完整的因子配置回测流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config import (
    ALL_ASSETS, MACRO_FACTORS, BACKTEST_CONFIG, OPTIMIZER_CONFIG, OUTPUT_DIR
)
from data_fetcher import get_all_assets_data, resample_to_monthly
from macro_factors import MacroFactorBuilder
from factor_exposure import FactorExposureWithPrior
from portfolio_optimizer import PortfolioOptimizer, MacroFactorAllocator
from risk_analysis import RiskAnalyzer, risk_attribution_analysis
from backtest import BacktestEngine, BacktestResultAnalyzer


def load_or_fetch_data(start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    加载或获取资产数据

    Parameters:
        start_date: 开始日期
        end_date: 结束日期
        force_refresh: 是否强制刷新数据

    Returns:
        资产月度收益率DataFrame
    """
    import os

    cache_file = os.path.join(OUTPUT_DIR, 'monthly_asset_returns.pkl')

    if not force_refresh and os.path.exists(cache_file):
        print("从缓存加载资产数据...")
        return pd.read_pickle(cache_file)

    print("正在从数据源获取资产数据...")
    asset_data = get_all_assets_data(start_date, end_date)

    if not asset_data:
        print("警告: 未能获取真实数据，使用模拟数据")
        return generate_simulated_data(start_date, end_date)

    monthly_returns = {}
    for asset, df in asset_data.items():
        if not df.empty:
            monthly = resample_to_monthly(df)
            monthly_returns[asset] = monthly['return']

    if not monthly_returns:
        print("警告: 月度数据为空，使用模拟数据")
        return generate_simulated_data(start_date, end_date)

    result = pd.DataFrame(monthly_returns)
    result.to_pickle(cache_file)
    print(f"数据已缓存至: {cache_file}")

    return result


def generate_simulated_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    生成模拟数据（当真实数据获取失败时）

    Parameters:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        模拟资产收益率DataFrame
    """
    print("生成模拟数据用于测试...")

    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n = len(dates)

    np.random.seed(42)

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

    return asset_returns


def run_macro_factor_allocation(start_date: str = None, end_date: str = None,
                                factor_deviation: float = 0.05,
                                use_simulated_data: bool = False) -> dict:
    """
    运行宏观因子资产配置主流程

    Parameters:
        start_date: 回测开始日期
        end_date: 回测结束日期
        factor_deviation: 因子偏离值
        use_simulated_data: 是否使用模拟数据

    Returns:
        回测结果字典
    """
    if start_date is None:
        start_date = BACKTEST_CONFIG['start_date']
    if end_date is None:
        end_date = BACKTEST_CONFIG['end_date']

    print("=" * 60)
    print("宏观因子资产配置框架")
    print("=" * 60)
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"因子偏离值: {factor_deviation}")
    print(f"使用模拟数据: {use_simulated_data}")
    print("=" * 60)

    if use_simulated_data:
        asset_returns = generate_simulated_data(start_date, end_date)
    else:
        asset_returns = load_or_fetch_data(start_date, end_date)

    print(f"\n资产收益率数据形状: {asset_returns.shape}")
    print(f"资产列表: {asset_returns.columns.tolist()}")

    print("\n构建宏观因子...")
    factor_builder = MacroFactorBuilder(n_factors=6)
    factor_returns = factor_builder.construct_all_factors(asset_returns)
    print(f"因子收益率数据形状: {factor_returns.shape}")
    print(f"因子列表: {factor_returns.columns.tolist()}")

    print("\n计算因子暴露...")
    exposure_calculator = FactorExposureWithPrior(alpha=0.01)
    exposure_matrix = exposure_calculator.fit(asset_returns, factor_returns)
    print(f"因子暴露矩阵形状: {exposure_matrix.shape}")

    print("\n运行回测...")
    backtest_engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=BACKTEST_CONFIG['rebalance_freq'],
        factor_deviation=factor_deviation
    )

    all_results = backtest_engine.run_all_factor_backtests(asset_returns, factor_returns)

    print("\n生成回测报告...")
    analyzer = BacktestResultAnalyzer(all_results)
    summary = analyzer.generate_summary_report(factor_returns)
    print("\n=== 因子偏离策略绩效汇总 ===")
    print(summary.to_string(index=False))

    print("\n保存结果...")
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_file = os.path.join(OUTPUT_DIR, 'factor_strategy_summary.csv')
    summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"汇总报告已保存至: {summary_file}")

    results = {
        'asset_returns': asset_returns,
        'factor_returns': factor_returns,
        'exposure_matrix': exposure_matrix,
        'backtest_results': all_results,
        'summary': summary
    }

    return results


def analyze_risk_decomposition(weights: np.ndarray, exposure_matrix: pd.DataFrame,
                               asset_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                               asset_names: list, factor_names: list) -> pd.DataFrame:
    """
    分析组合的宏观风险分解

    Parameters:
        weights: 资产权重
        exposure_matrix: 因子暴露矩阵
        asset_returns: 资产收益率
        factor_returns: 因子收益率
        asset_names: 资产名称
        factor_names: 因子名称

    Returns:
        风险分解DataFrame
    """
    risk_analyzer = RiskAnalyzer()

    risk_analyzer.compute_factor_covariance(factor_returns)
    risk_analyzer.compute_heterogeneous_variance(asset_returns)

    portfolio_risk, asset_risk = risk_attribution_analysis(
        weights, exposure_matrix.values, factor_returns, asset_returns,
        asset_names, factor_names
    )

    print("\n=== 组合宏观风险分解 ===")
    print(portfolio_risk.to_string(index=False))

    print("\n=== 各资产风险分解 (前5) ===")
    print(asset_risk.head().to_string(index=False))

    return portfolio_risk, asset_risk


def plot_factor_strategy_results(all_results: dict, factor_returns: pd.DataFrame,
                                save_dir: str = None):
    """
    绘制因子策略结果图

    Parameters:
        all_results: 所有因子回测结果
        factor_returns: 因子收益率
        save_dir: 保存目录
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

    for factor_name, result in all_results.items():
        print(f"\n绘制 {factor_name} 因子策略结果...")

        portfolio_values = result['portfolio_values']
        benchmark_values = result['benchmark_values']
        relative_values = portfolio_values / benchmark_values

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        axes[0].plot(relative_values.index, relative_values.values,
                   label=f'做多{factor_name}因子', linewidth=1.5, color='blue')
        axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{factor_name}因子 - 相对净值走势', fontsize=14)
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('相对净值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        factor_cumsum = (1 + factor_returns[factor_name]).cumprod()
        factor_cumsum = factor_cumsum / factor_cumsum.iloc[0]

        common_idx = relative_values.index.intersection(factor_cumsum.index)
        axes[1].plot(common_idx, relative_values.loc[common_idx].values,
                    label='相对净值', linewidth=1.5, color='blue')
        axes[1].plot(common_idx, factor_cumsum.loc[common_idx].values,
                    label=f'{factor_name}因子', linewidth=1.5, color='orange', alpha=0.7)
        axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title(f'相对净值 vs {factor_name}因子走势对比', fontsize=14)
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('标准化净值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'{factor_name}_factor_result.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")

        plt.show()


if __name__ == "__main__":
    print("运行宏观因子资产配置框架...")

    results = run_macro_factor_allocation(
        start_date='2015-01-01',
        end_date='2023-05-31',
        factor_deviation=0.05,
        use_simulated_data=True
    )

    print("\n" + "=" * 60)
    print("宏观因子资产配置框架运行完成")
    print("=" * 60)