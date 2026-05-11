import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source import (
    get_all_industries_data,
    load_cached_data,
    get_market_index_data,
    calculate_returns,
    CrowdednessIndicator,
    generate_crowdedness_signals,
    IndustryRotationStrategy,
    BacktestEngine,
    run_backtest,
    compare_strategies,
    save_backtest_results,
    plot_equity_curves,
    plot_drawdown_series,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_crowdedness_heatmap,
    plot_strategy_comparison,
    create_performance_summary
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("行业配置策略：拥挤度视角 - 项目运行")
    print("=" * 60)

    print("\n[Step 1] 获取申万行业指数数据...")
    industry_data = load_cached_data()

    if industry_data is None or len(industry_data) == 0:
        print("Fetching data from tushare (this may take a few minutes)...")
        industry_data = get_all_industries_data(
            start_date='2018-01-01',
            end_date='2023-12-31',
            save=True
        )
    else:
        print(f"已加载缓存数据，共 {len(industry_data)} 个行业")

    if industry_data is None or len(industry_data) == 0:
        print("错误：无法获取行业数据")
        return None

    print(f"\n成功获取 {len(industry_data)} 个行业的数据")

    print("\n[Step 2] 计算拥挤度指标...")
    print("生成拥挤度信号...")
    crowdedness_signals = generate_crowdedness_signals(industry_data)

    if crowdedness_signals is None or len(crowdedness_signals) == 0:
        print("错误：无法生成拥挤度信号")
        return None

    crowdedness_signals.to_pickle(os.path.join(DATA_DIR, 'crowdedness_signals.pkl'))
    print(f"拥挤度信号已保存，共 {len(crowdedness_signals)} 条")

    print("\n[Step 3] 运行行业轮动策略...")
    rotation = IndustryRotationStrategy(industry_data, crowdedness_signals)
    rotation.calculate_benchmark_returns()

    results = {}

    print("运行策略一：月度空头行业轮动...")
    try:
        strat1_returns, strat1_pos = rotation.strategy_one_monthly_short()
        print(f"策略一完成，收益序列长度: {len(strat1_returns)}")
        if len(strat1_returns) > 0:
            result1 = run_backtest(strat1_returns, strategy_name="策略一")
            results['strategy_1_monthly_short'] = result1
    except Exception as e:
        print(f"策略一运行失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n[Step 4] 生成可视化图表...")

    equity_curves = {k: v['equity_curve'] for k, v in results.items() if 'equity_curve' in v}
    drawdown_series = {k: v['drawdown'] for k, v in results.items() if 'drawdown' in v}
    returns_dict = {k: v['returns'] for k, v in results.items() if 'returns' in v}

    try:
        plot_equity_curves(equity_curves, save_path=os.path.join(OUTPUT_DIR, 'equity_curves.png'))
        print(" equity_curves.png")
    except Exception as e:
        print(f"生成equity_curves失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        plot_drawdown_series(drawdown_series, save_path=os.path.join(OUTPUT_DIR, 'drawdown_series.png'))
        print(" drawdown_series.png")
    except Exception as e:
        print(f"生成drawdown_series失败: {e}")

    try:
        plot_returns_distribution(returns_dict['strategy_1_monthly_short'], save_path=os.path.join(OUTPUT_DIR, 'returns_distribution.png'))
        print(" returns_distribution.png")
    except Exception as e:
        print(f"生成returns_distribution失败: {e}")

    try:
        plot_crowdedness_heatmap(crowdedness_signals, save_path=os.path.join(OUTPUT_DIR, 'crowdedness_heatmap.png'))
        print(" crowdedness_heatmap.png")
    except Exception as e:
        print(f"生成crowdedness_heatmap失败: {e}")

    try:
        plot_strategy_comparison(results, save_path=os.path.join(OUTPUT_DIR, 'strategy_comparison.png'))
        print(" strategy_comparison.png")
    except Exception as e:
        print(f"生成strategy_comparison失败: {e}")

    try:
        summary = create_performance_summary(results)
        summary.to_csv(os.path.join(OUTPUT_DIR, 'performance_summary.csv'))
        print(f" performance_summary.csv")
    except Exception as e:
        print(f"生成performance_summary失败: {e}")

    print("\n[Step 5] Save backtest results...")
    try:
        for name, result in results.items():
            save_backtest_results(result, OUTPUT_DIR, name)
        print("Backtest results saved")
    except Exception as e:
        print(f"Failed to save backtest results: {e}")

    print("\n" + "=" * 60)
    print("项目运行完成！")
    print("=" * 60)
    print(f"\n数据已保存至: {DATA_DIR}")
    print(f"结果已保存至: {OUTPUT_DIR}")

    return results

if __name__ == "__main__":
    results = main()