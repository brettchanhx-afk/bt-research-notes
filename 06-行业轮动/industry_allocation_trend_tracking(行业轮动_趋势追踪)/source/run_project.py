"""
趋势追踪行业配置策略 - 主运行脚本
运行整个项目流程：数据获取 -> 指标计算 -> 策略回测 -> 结果输出
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from source import (
    DataLoader, TrendIndicatorCalculator, BacktestEngine,
    MonteCarloSimulator, CSCVTest, StrategyEvaluator
)

DATA_DIR = os.path.join(parent_dir, 'data')
OUTPUT_DIR = os.path.join(parent_dir, 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = '20100101'
END_DATE = '20200731'

def fetch_and_save_data():
    print("=" * 60)
    print("第一步：获取市场数据")
    print("=" * 60)

    loader = DataLoader()

    print("\n获取股票指数数据...")
    index_data = loader.get_multiple_indices(START_DATE, END_DATE)
    print(f"成功获取 {len(index_data)} 个指数数据")

    print("\n获取行业指数数据...")
    industry_data = loader.get_multiple_industries(START_DATE, END_DATE)
    print(f"成功获取 {len(industry_data)} 个行业数据")

    print("\n保存数据到 data 文件夹...")
    for name, df in index_data.items():
        safe_name = name.replace('/', '_').replace('\\', '_')
        filepath = os.path.join(DATA_DIR, f'index_{safe_name}.csv')
        df.to_csv(filepath, index=False)
        print(f"  保存: {filepath}")

    for name, df in industry_data.items():
        safe_name = name.replace('/', '_').replace('\\', '_')
        filepath = os.path.join(DATA_DIR, f'industry_{safe_name}.csv')
        df.to_csv(filepath, index=False)
        print(f"  保存: {filepath}")

    print("\n计算资产统计信息...")
    index_stats = loader.calculate_assets_statistics(index_data)
    index_stats.to_csv(os.path.join(DATA_DIR, 'index_statistics.csv'), index=False)
    print(f"  指数统计信息已保存")

    industry_stats = loader.calculate_assets_statistics(industry_data)
    industry_stats.to_csv(os.path.join(DATA_DIR, 'industry_statistics.csv'), index=False)
    print(f"  行业统计信息已保存")

    print("\n计算相关系数矩阵...")
    index_corr = loader.calculate_correlation_matrix(index_data)
    index_corr.to_csv(os.path.join(DATA_DIR, 'index_correlation_matrix.csv'))
    print(f"  指数相关系数矩阵已保存")

    industry_corr = loader.calculate_correlation_matrix(industry_data)
    industry_corr.to_csv(os.path.join(DATA_DIR, 'industry_correlation_matrix.csv'))
    print(f"  行业相关系数矩阵已保存")

    return index_data, industry_data, loader

def prepare_price_data(index_data, industry_data):
    print("\n" + "=" * 60)
    print("第二步：准备价格数据")
    print("=" * 60)

    index_prices = pd.DataFrame({
        name: df.set_index('trade_date')['close']
        for name, df in index_data.items()
        if 'close' in df.columns or 'Close' in df.columns
    })
    index_prices = index_prices.sort_index().dropna()

    industry_prices = pd.DataFrame({
        name: df.set_index('trade_date')['close']
        for name, df in industry_data.items()
        if 'close' in df.columns or 'Close' in df.columns
    })
    industry_prices = industry_prices.sort_index().dropna()

    print(f"指数价格数据形状: {index_prices.shape}")
    print(f"行业价格数据形状: {industry_prices.shape}")

    return index_prices, industry_prices

def calculate_indicators(prices_df, sample_size=10):
    print("\n" + "=" * 60)
    print("第三步：计算趋势追踪指标")
    print("=" * 60)

    calculator = TrendIndicatorCalculator()
    print(f"共实现 {len(calculator.indicator_funcs)} 个趋势追踪指标")

    all_signals = calculator.generate_all_signals(prices_df)
    print(f"共生成 {len(all_signals.columns)} 个指标信号")

    signal_sample = all_signals.iloc[:, :sample_size]
    signal_sample.to_csv(os.path.join(DATA_DIR, 'sample_signals.csv'))
    print(f"样本信号已保存到 data/sample_signals.csv")

    return calculator, all_signals

def run_backtests(prices_df, signals_df, strategy_name="Index"):
    print("\n" + "=" * 60)
    print(f"第四步：{strategy_name}策略回测")
    print("=" * 60)

    engine = BacktestEngine(rebalance_freq='monthly', commission_rate=0.001)
    evaluator = StrategyEvaluator()

    sample_cols = signals_df.columns[:20]
    sample_signals = signals_df[sample_cols].copy()
    sample_signals_aligned = sample_signals.reindex(prices_df.index, method='ffill')

    print(f"评估 {len(sample_cols)} 个指标...")
    results_df = evaluator.evaluate_multiple_indicators(
        prices_df, sample_signals_aligned, strategy_type='time_series'
    )

    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'{strategy_name.lower()}_backtest_results.csv'))
    print(f"回测结果已保存到 output/{strategy_name.lower()}_backtest_results.csv")

    print(f"\n{strategy_name}策略表现 Top 5:")
    print(results_df.head())

    best_indicator = results_df.index[0]
    best_result = results_df.iloc[0]

    print(f"\n表现最佳指标: {best_indicator}")
    print(f"  年化收益率: {best_result['annual_return']:.2%}")
    print(f"  夏普比率: {best_result['sharpe_ratio']:.4f}")
    print(f"  最大回撤: {best_result['max_drawdown']:.2%}")

    return results_df, best_indicator

def run_cscv_analysis(prices_df, signals_df, best_indicator):
    print("\n" + "=" * 60)
    print("第五步：CSCV过拟合检验")
    print("=" * 60)

    cscv = CSCVTest(n_splits=50)

    if best_indicator in signals_df.columns:
        signal = signals_df[best_indicator].reindex(prices_df.index, method='ffill')
        cscv_result = cscv.run_cscv_analysis(signal, prices_df.mean(axis=1))

        print(f"最佳指标 {best_indicator} CSCV检验结果:")
        print(f"  样本内夏普比率均值: {cscv_result['is_sharpe_mean']:.4f}")
        print(f"  样本外夏普比率均值: {cscv_result['oos_sharpe_mean']:.4f}")
        print(f"  过拟合概率: {cscv_result['overfitting_probability']:.4f}")

        robustness_results = []
        for col in signals_df.columns[:10]:
            try:
                signal = signals_df[col].reindex(prices_df.index, method='ffill')
                result = cscv.run_cscv_analysis(signal, prices_df.mean(axis=1))
                robustness_results.append({
                    'indicator': col,
                    'sharpe_ratio': result['selection_sharpe'],
                    'overfit_prob': result['overfitting_probability'],
                    'is_sharpe': result['is_sharpe_mean'],
                    'oos_sharpe': result['oos_sharpe_mean']
                })
            except:
                pass

        robustness_df = pd.DataFrame(robustness_results)
        robustness_df = robustness_df.sort_values('sharpe_ratio', ascending=False)
        robustness_df.to_csv(os.path.join(OUTPUT_DIR, 'cscv_robustness_results.csv'), index=False)
        print(f"\nCSCV鲁棒性检验结果已保存到 output/cscv_robustness_results.csv")

        return robustness_df

    return None

def run_monte_carlo_simulation():
    print("\n" + "=" * 60)
    print("第六步：蒙特卡洛模拟")
    print("=" * 60)

    simulator = MonteCarloSimulator(random_state=42)

    mu_range = [0.0001, 0.0005, 0.001]
    sigma_range = [0.005, 0.01, 0.02]
    rho_range = [0.01, 0.1, 0.2]

    print(f"单资产模拟参数:")
    print(f"  收益率范围: {mu_range}")
    print(f"  波动率范围: {sigma_range}")
    print(f"  自相关系数范围: {rho_range}")

    virtual_sequences = simulator.generate_single_asset_scenarios(
        n_days=100, n_scenarios=50,
        mu_range=mu_range, sigma_range=sigma_range, rho_range=rho_range
    )
    print(f"生成了 {len(virtual_sequences)} 个虚拟序列场景")

    mc_results = []
    for i, seq in enumerate(virtual_sequences[:27]):
        mc_results.append({
            'scenario_id': i,
            'mu': seq['mu'],
            'sigma': seq['sigma'],
            'rho': seq['rho'],
            'final_price': seq['prices'].iloc[-1],
            'price_return': (seq['prices'].iloc[-1] / seq['prices'].iloc[0] - 1)
        })

    mc_df = pd.DataFrame(mc_results)
    mc_df.to_csv(os.path.join(OUTPUT_DIR, 'monte_carlo_results.csv'), index=False)
    print(f"蒙特卡洛模拟结果已保存到 output/monte_carlo_results.csv")

    return mc_df

def generate_visualizations(index_prices, industry_prices, backtest_results, robustness_results):
    print("\n" + "=" * 60)
    print("第七步：生成可视化结果")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for col in index_prices.columns[:5]:
        normalized = index_prices[col] / index_prices[col].iloc[0] * 100
        ax1.plot(normalized.index, normalized, label=col, alpha=0.7)
    ax1.set_title('主要指数走势 (标准化)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('标准化价格 (初始=100)')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if len(backtest_results) > 0:
        top10 = backtest_results.head(10)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
        bars = ax2.barh(range(len(top10)), top10['sharpe_ratio'].values, color=colors)
        ax2.set_yticks(range(len(top10)))
        ax2.set_yticklabels(top10.index, fontsize=8)
        ax2.set_title('指标夏普比率 Top 10')
        ax2.set_xlabel('夏普比率')
        ax2.grid(True, alpha=0.3, axis='x')

    ax3 = axes[1, 0]
    if robustness_results is not None and len(robustness_results) > 0:
        ax3.scatter(robustness_results['sharpe_ratio'],
                   robustness_results['overfit_prob'],
                   alpha=0.6, s=100, c='steelblue')
        ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='阈值 0.5')
        ax3.set_title('夏普比率 vs 过拟合概率')
        ax3.set_xlabel('夏普比率')
        ax3.set_ylabel('过拟合概率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    industry_sample = industry_prices.iloc[:, :5]
    for col in industry_sample.columns:
        normalized = industry_sample[col] / industry_sample[col].iloc[0] * 100
        ax4.plot(normalized.index, normalized, label=col, alpha=0.7)
    ax4.set_title('行业指数走势 (标准化)')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_xlabel('日期')
    ax4.set_ylabel('标准化价格 (初始=100)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_results.png'), dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到 output/analysis_results.png")

    fig2, ax = plt.subplots(figsize=(12, 6))
    if len(backtest_results) > 0:
        metrics = backtest_results[['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']].head(15)
        metrics.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('策略回测指标对比')
        ax.set_xlabel('指标名称')
        ax.set_ylabel('数值')
        ax.legend(['年化收益率', '年化波动率', '夏普比率', '最大回撤'], loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_metrics.png'), dpi=150, bbox_inches='tight')
    print(f"回测指标图已保存到 output/backtest_metrics.png")

    plt.close('all')

def main():
    print("\n" + "=" * 60)
    print("趋势追踪行业配置策略 - 完整运行流程")
    print("=" * 60)

    try:
        index_data, industry_data, loader = fetch_and_save_data()

        index_prices, industry_prices = prepare_price_data(index_data, industry_data)

        calculator, index_signals = calculate_indicators(index_prices)

        index_backtest_results, best_index_indicator = run_backtests(
            index_prices, index_signals, "指数"
        )

        robustness_results = run_cscv_analysis(index_prices, index_signals, best_index_indicator)

        if len(industry_data) > 0:
            industry_prices, _ = prepare_price_data(industry_data, {})
            industry_signals = calculate_indicators(industry_prices)
            industry_backtest_results, best_industry_indicator = run_backtests(
                industry_prices, industry_signals, "行业"
            )

        mc_results = run_monte_carlo_simulation()

        generate_visualizations(index_prices, industry_prices, index_backtest_results, robustness_results)

        print("\n" + "=" * 60)
        print("运行完成！")
        print("=" * 60)
        print(f"\n数据文件保存在: {DATA_DIR}")
        print(f"输出文件保存在: {OUTPUT_DIR}")

        print("\n输出文件列表:")
        for f in os.listdir(OUTPUT_DIR):
            print(f"  - {f}")

    except Exception as e:
        print(f"\n运行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()