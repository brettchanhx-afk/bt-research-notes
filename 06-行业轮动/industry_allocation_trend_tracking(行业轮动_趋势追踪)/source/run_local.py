"""
趋势追踪行业配置策略 - 本地数据运行脚本
直接从 data 文件夹读取 CSV 数据，运行完整项目流程
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
DATA_DIR = os.path.join(parent_dir, 'data')
OUTPUT_DIR = os.path.join(parent_dir, 'output')

sys.path.insert(0, parent_dir)

from source import (
    TrendIndicatorCalculator, BacktestEngine,
    MonteCarloSimulator, CSCVTest, StrategyEvaluator
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_asset_class_data():
    print("=" * 60)
    print("加载大类资产数据")
    print("=" * 60)

    filepath = os.path.join(DATA_DIR, '大类资产数据2010_2026.csv')
    df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)

    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]

    df.columns = df.columns.get_level_values(0)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(axis=1, how='all')
    df = df.sort_index()

    print(f"大类资产数据形状: {df.shape}")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"资产列表: {list(df.columns)}")

    return df

def load_industry_data():
    print("\n" + "=" * 60)
    print("加载行业指数数据")
    print("=" * 60)

    filepath = os.path.join(DATA_DIR, '中信一级行业指数及收盘价2010_2026.csv')
    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.get_level_values(0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')
    df = df.sort_index()

    print(f"行业数据形状: {df.shape}")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"行业列表: {list(df.columns)}")

    return df

def load_commodity_data():
    print("\n" + "=" * 60)
    print("加载商品期货数据")
    print("=" * 60)

    filepath = os.path.join(DATA_DIR, '商品市场指数2010_2026.csv')
    df = pd.read_csv(filepath, header=0, index_col=0)

    while df.index[0] in ['指标名称', '频率', '单位']:
        df = df.iloc[1:]

    valid_dates = pd.to_datetime(df.index, errors='coerce')
    df = df[valid_dates.notna()]
    df.index = valid_dates[valid_dates.notna()]

    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    df = df.dropna(axis=1, how='all')
    df = df.sort_index()

    print(f"商品数据形状: {df.shape}")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"商品列表: {list(df.columns)}")

    return df

def calculate_statistics(prices_df, name="Asset"):
    print(f"\n计算 {name} 统计信息...")

    returns = prices_df.pct_change().dropna()

    stats = pd.DataFrame({
        '日均收益率': returns.mean(),
        '年化收益率': returns.mean() * 252,
        '年化波动率': returns.std() * np.sqrt(252),
        '夏普比率(无风险=0)': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
    })

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    stats['最大回撤'] = drawdown.min()

    return stats

def calculate_correlation_matrix(prices_df):
    returns = prices_df.pct_change().dropna()
    return returns.corr()

def calculate_indicators(prices_df):
    print("\n" + "=" * 60)
    print("计算趋势追踪指标")
    print("=" * 60)

    calculator = TrendIndicatorCalculator()
    print(f"共实现 {len(calculator.indicator_funcs)} 个趋势追踪指标")

    all_signals = calculator.generate_all_signals(prices_df)
    print(f"共生成 {len(all_signals.columns)} 个指标信号")

    return calculator, all_signals

def run_backtests(prices_df, signals_df, strategy_name="Strategy"):
    print("\n" + "=" * 60)
    print(f"{strategy_name} 策略回测")
    print("=" * 60)

    print(f"价格数据形状: {prices_df.shape}, 信号数据形状: {signals_df.shape}")

    evaluator = StrategyEvaluator()

    sample_cols = signals_df.columns[:min(10, len(signals_df.columns))]
    sample_signals = signals_df[sample_cols].copy()
    sample_signals_aligned = sample_signals.reindex(prices_df.index, method='ffill')

    print(f"评估 {len(sample_cols)} 个指标...")
    print(f"价格数据前5行:\n{prices_df.head()}")
    print(f"价格数据NaN数量: {prices_df.isna().sum().sum()}")
    print(f"各列NaN数量:\n{prices_df.isna().sum()}")

    prices_df = prices_df.fillna(method='ffill')
    prices_df = prices_df.clip(lower=0.001)
    print(f"填充后NaN数量: {prices_df.isna().sum().sum()}")

    try:
        results_df = evaluator.evaluate_multiple_indicators(
            prices_df, sample_signals_aligned, strategy_type='time_series'
        )
        if len(results_df) > 0:
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        else:
            print("警告: 没有有效的回测结果")
    except Exception as e:
        print(f"回测出错: {e}")
        import traceback
        traceback.print_exc()
        results_df = pd.DataFrame()

    return results_df

def run_cscv_analysis(prices_df, signals_df, best_indicator):
    print("\n" + "=" * 60)
    print("CSCV 过拟合检验")
    print("=" * 60)

    cscv = CSCVTest(n_splits=50)
    robustness_results = []

    for col in signals_df.columns[:10]:
        try:
            signal = signals_df[col].reindex(prices_df.index, method='ffill')
            benchmark = prices_df.mean(axis=1)
            result = cscv.run_cscv_analysis(signal, benchmark)

            robustness_results.append({
                'indicator': col,
                'sharpe_ratio': result.get('selection_sharpe', 0),
                'overfit_prob': result.get('overfitting_probability', 0.5),
                'is_sharpe': result.get('is_sharpe_mean', 0),
                'oos_sharpe': result.get('oos_sharpe_mean', 0)
            })
        except Exception as e:
            pass

    if robustness_results:
        robustness_df = pd.DataFrame(robustness_results)
        robustness_df = robustness_df.sort_values('sharpe_ratio', ascending=False)
        return robustness_df

    return pd.DataFrame()

def run_monte_carlo():
    print("\n" + "=" * 60)
    print("蒙特卡洛模拟")
    print("=" * 60)

    simulator = MonteCarloSimulator(random_state=42)

    virtual_sequences = simulator.generate_single_asset_scenarios(
        n_days=100, n_scenarios=50,
        mu_range=[0.0001, 0.0005, 0.001],
        sigma_range=[0.005, 0.01, 0.02],
        rho_range=[0.01, 0.1, 0.2]
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
    return mc_df

def generate_visualizations(asset_prices, industry_prices, backtest_results, robustness_results):
    print("\n" + "=" * 60)
    print("生成可视化结果")
    print("=" * 60)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    cols_to_plot = asset_prices.columns[:5]
    for col in cols_to_plot:
        if col in asset_prices.columns:
            normalized = asset_prices[col] / asset_prices[col].iloc[0] * 100
            ax1.plot(normalized.index, normalized.values, label=col, alpha=0.7)
    ax1.set_title('主要资产走势 (标准化)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price (Initial=100)')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if len(backtest_results) > 0:
        top_n = min(10, len(backtest_results))
        top_results = backtest_results.head(top_n)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
        bars = ax2.barh(range(top_n), top_results['sharpe_ratio'].values, color=colors)
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(top_results.index, fontsize=8)
        ax2.set_title('Indicator Sharpe Ratio Top 10')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3, axis='x')

    ax3 = axes[1, 0]
    if robustness_results is not None and len(robustness_results) > 0:
        ax3.scatter(robustness_results['sharpe_ratio'],
                   robustness_results['overfit_prob'],
                   alpha=0.6, s=100, c='steelblue')
        ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
        ax3.set_title('Sharpe Ratio vs Overfitting Probability')
        ax3.set_xlabel('Sharpe Ratio')
        ax3.set_ylabel('Overfitting Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    industry_sample_cols = industry_prices.columns[:5]
    for col in industry_sample_cols:
        if col in industry_prices.columns:
            normalized = industry_prices[col] / industry_prices[col].iloc[0] * 100
            ax4.plot(normalized.index, normalized.values, label=col, alpha=0.7)
    ax4.set_title('Industry Index Trend (Normalized)')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Normalized Price (Initial=100)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_results.png'), dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到 output/analysis_results.png")

    if len(backtest_results) > 0:
        fig2, ax = plt.subplots(figsize=(12, 6))
        metrics_cols = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
        available_cols = [c for c in metrics_cols if c in backtest_results.columns]
        if available_cols:
            metrics = backtest_results[available_cols].head(15)
            metrics.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Strategy Backtest Metrics')
            ax.set_xlabel('Indicator')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_metrics.png'), dpi=150, bbox_inches='tight')
        print(f"回测指标图已保存到 output/backtest_metrics.png")

    plt.close('all')

def main():
    print("\n" + "=" * 60)
    print("趋势追踪行业配置策略 - 本地数据运行")
    print("=" * 60)

    try:
        asset_prices = load_asset_class_data()
        industry_prices = load_industry_data()
        commodity_prices = load_commodity_data()

        asset_stats = calculate_statistics(asset_prices, "大类资产")
        asset_stats.to_csv(os.path.join(OUTPUT_DIR, 'asset_statistics.csv'))
        print("\n大类资产统计信息已保存")

        industry_stats = calculate_statistics(industry_prices, "行业指数")
        industry_stats.to_csv(os.path.join(OUTPUT_DIR, 'industry_statistics.csv'))
        print("行业指数统计信息已保存")

        commodity_stats = calculate_statistics(commodity_prices, "商品期货")
        commodity_stats.to_csv(os.path.join(OUTPUT_DIR, 'commodity_statistics.csv'))
        print("商品期货统计信息已保存")

        asset_corr = calculate_correlation_matrix(asset_prices)
        asset_corr.to_csv(os.path.join(OUTPUT_DIR, 'asset_correlation_matrix.csv'))

        industry_corr = calculate_correlation_matrix(industry_prices)
        industry_corr.to_csv(os.path.join(OUTPUT_DIR, 'industry_correlation_matrix.csv'))

        asset_calculator, asset_signals = calculate_indicators(asset_prices)
        asset_signals.to_csv(os.path.join(OUTPUT_DIR, 'asset_signals.csv'))

        industry_calculator, industry_signals = calculate_indicators(industry_prices)
        industry_signals.to_csv(os.path.join(OUTPUT_DIR, 'industry_signals.csv'))

        asset_backtest = run_backtests(asset_prices, asset_signals, "大类资产")
        if len(asset_backtest) > 0:
            asset_backtest.to_csv(os.path.join(OUTPUT_DIR, 'asset_backtest_results.csv'))
            best_asset_indicator = asset_backtest.index[0]
            print(f"\n大类资产最佳指标: {best_asset_indicator}")

        industry_backtest = run_backtests(industry_prices, industry_signals, "行业指数")
        if len(industry_backtest) > 0:
            industry_backtest.to_csv(os.path.join(OUTPUT_DIR, 'industry_backtest_results.csv'))
            best_industry_indicator = industry_backtest.index[0]
            print(f"行业指数最佳指标: {best_industry_indicator}")

        if len(asset_backtest) > 0:
            robustness_results = run_cscv_analysis(asset_prices, asset_signals, best_asset_indicator)
            if len(robustness_results) > 0:
                robustness_results.to_csv(os.path.join(OUTPUT_DIR, 'cscv_robustness_results.csv'), index=False)

        mc_results = run_monte_carlo()
        mc_results.to_csv(os.path.join(OUTPUT_DIR, 'monte_carlo_results.csv'), index=False)

        generate_visualizations(asset_prices, industry_prices, asset_backtest, robustness_results if 'robustness_results' in dir() else pd.DataFrame())

        print("\n" + "=" * 60)
        print("运行完成！")
        print("=" * 60)
        print(f"\n数据来源: {DATA_DIR}")
        print(f"输出文件保存在: {OUTPUT_DIR}")

        print("\n输出文件列表:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            print(f"  - {f}")

    except Exception as e:
        print(f"\n运行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()