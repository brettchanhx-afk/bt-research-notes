# -*- coding: utf-8 -*-
"""
基金项目：基金业绩持续性量化评价
===================================
研报复现：华泰金工研报《定量评价基金的业绩持续性》

使用方法:
    python main.py                    # 使用默认基金池分析
    python main.py --fund 000628     # 分析指定基金
    python main.py --backtest         # 运行回测
"""

import argparse
import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.data_loader import (
    get_fund_nav,
    calculate_daily_returns,
    calculate_log_returns,
    get_benchmark_data,
    get_multiple_funds_data,
    merge_funds_returns
)
from source.factor import (
    cross_section_analysis,
    cross_section_single_fund,
    calculate_cpr,
    calculate_cpr_single_fund,
    hurst_analysis,
    comprehensive_persistence_analysis
)
from source.backtest import (
    persistence_based_backtest,
    calculate_performance_metrics,
    print_backtest_results,
    stratified_persistence_backtest
)
from source.plot import (
    plot_persistence_dashboard,
    plot_hurst_distribution,
    plot_backtest_results,
    plot_cpr_matrix
)
from source.utils import save_results, print_persistence_summary
from config import (
    BASE_DIR, OUTPUT_DIR, DATA_DIR,
    DEFAULT_FUND_POOL, RISK_FREE_RATE,
    BACKTEST_CONFIG
)


def analyze_single_fund(fund_code, start_date='2018-01-01', end_date='2023-12-31'):
    """
    分析单只基金的业绩持续性
    
    Parameters:
    -----------
    fund_code : str
        基金代码
    start_date : str
        开始日期
    end_date : str
        结束日期
        
    Returns:
    --------
    dict : 分析结果
    """
    print(f"\n{'='*60}")
    print(f"分析基金: {fund_code}")
    print(f"分析期间: {start_date} 至 {end_date}")
    print(f"{'='*60}")
    
    # 1. 获取数据
    print("\n[1/4] 获取基金净值数据...")
    nav_df = get_fund_nav(fund_code, start_date, end_date)
    
    if nav_df is None or len(nav_df) < 30:
        print(f"  [错误] 无法获取基金 {fund_code} 的数据或数据不足")
        return None
    
    # 2. 计算收益率
    print("\n[2/4] 计算收益率序列...")
    returns_df = calculate_daily_returns(nav_df)
    log_returns = calculate_log_returns(nav_df)
    
    fund_returns = returns_df.set_index('date')['daily_return']
    log_returns_series = log_returns.set_index('date')['log_return']
    
    # 获取基准数据
    print("\n[3/4] 获取基准数据...")
    benchmark_df = get_benchmark_data('000300', start_date, end_date)
    
    if benchmark_df is not None:
        benchmark_returns = benchmark_df.set_index('date')['nav'].pct_change().dropna()
        benchmark_returns.name = 'benchmark'
    else:
        benchmark_returns = None
        print("  [警告] 无法获取基准数据，将使用无风险利率")
    
    # 3. 综合分析
    print("\n[4/4] 执行业绩持续性分析...")
    
    # 三种方法分析
    results = {}
    
    # 方法1: 横截面分析法
    print("\n  --- 横截面分析法 ---")
    cs_result = cross_section_single_fund(
        fund_returns, 
        benchmark_returns,
        risk_free_rate=RISK_FREE_RATE
    )
    if cs_result:
        results['横截面分析法'] = cs_result
        print(f"    评价期超额收益: {cs_result.get('alpha1', 0):.4f}")
        print(f"    持有期超额收益: {cs_result.get('alpha2', 0):.4f}")
        print(f"    同号判断: {cs_result.get('same_sign', 'N/A')}")
        print(f"    持续性: {cs_result.get('persistence', 'N/A')}")
    
    # 方法2: CPR法
    print("\n  --- 交叉积比率法 (CPR) ---")
    cpr_result = calculate_cpr_single_fund(
        fund_returns, 
        n_periods=4,
        market_median_returns=None
    )
    if cpr_result:
        results['交叉积比率法'] = cpr_result
        print(f"    WW: {cpr_result.get('WW', 0)}, LL: {cpr_result.get('LL', 0)}")
        print(f"    WL: {cpr_result.get('WL', 0)}, LW: {cpr_result.get('LW', 0)}")
        print(f"    CPR: {cpr_result.get('CPR', 'N/A')}")
        print(f"    持续性: {cpr_result.get('persistence_verdict', 'N/A')}")
    
    # 方法3: Hurst指数法
    print("\n  --- Hurst指数法 ---")
    hurst_result = hurst_analysis(
        fund_returns,
        log_returns=log_returns_series,
        n_values=[4, 8, 16, 32],
        n_estimators=8
    )
    if hurst_result:
        results['Hurst指数法'] = hurst_result
        print(f"    Hurst指数 H: {hurst_result.get('H', 'N/A'):.4f}")
        print(f"    R方: {hurst_result.get('r_squared', 'N/A'):.4f}")
        print(f"    分类: {hurst_result.get('persistence_verdict', 'N/A')}")
    
    # 综合判断
    verdicts = [r.get('persistence_verdict', '') for r in results.values() 
                if isinstance(r, dict) and 'persistence_verdict' in r]
    
    persistence_count = sum(1 for v in verdicts if '持续' in v and '无' not in v)
    reversal_count = sum(1 for v in verdicts if '反转' in v)
    
    if persistence_count > reversal_count and persistence_count >= 2:
        overall = "业绩整体有持续性"
    elif reversal_count > persistence_count and reversal_count >= 2:
        overall = "业绩整体有反转倾向"
    elif persistence_count == reversal_count and persistence_count > 0:
        overall = "业绩持续性不明确"
    else:
        overall = "无法判断"
    
    results['综合判断'] = overall
    results['fund_code'] = fund_code
    
    # 4. 保存结果
    print("\n保存分析结果...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存JSON
    import json
    results_clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_clean[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, (np.integer, np.floating)):
                    results_clean[k][kk] = float(vv)
                elif isinstance(vv, (np.bool_, bool)):
                    results_clean[k][kk] = bool(vv)
                elif isinstance(vv, (list, tuple)):
                    results_clean[k][kk] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in vv]
                elif vv is not None:
                    results_clean[k][kk] = vv
        elif isinstance(v, (np.integer, np.floating)):
            results_clean[k] = float(v)
        elif isinstance(v, (np.bool_, bool)):
            results_clean[k] = bool(v)
        else:
            results_clean[k] = v
    
    json_path = os.path.join(OUTPUT_DIR, f"{fund_code}_persistence_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, ensure_ascii=False, indent=2)
    print(f"  [保存] JSON结果已保存至 {json_path}")
    
    # 5. 绘制图表
    print("\n生成可视化图表...")
    cumulative_returns = (1 + fund_returns).cumprod()
    cumulative_returns.name = fund_code
    
    fig = plot_persistence_dashboard(
        fund_code,
        results,
        cumulative_returns=cumulative_returns,
        save_path=os.path.join(OUTPUT_DIR, f"{fund_code}_persistence_dashboard.png")
    )
    
    # 打印摘要
    print_persistence_summary(fund_code, results)
    
    return results


def analyze_fund_pool(fund_codes, start_date='2018-01-01', end_date='2023-12-31'):
    """
    分析基金池的业绩持续性
    """
    print(f"\n{'='*60}")
    print(f"分析基金池: {len(fund_codes)} 只基金")
    print(f"分析期间: {start_date} 至 {end_date}")
    print(f"{'='*60}")
    
    # 获取所有基金数据
    print("\n[1/4] 获取基金池数据...")
    funds_data = get_multiple_funds_data(fund_codes, start_date, end_date, DATA_DIR)
    
    if len(funds_data) == 0:
        print("  [错误] 无法获取任何基金数据")
        return None
    
    print(f"  成功获取 {len(funds_data)} 只基金数据")
    
    # 合并收益率
    print("\n[2/4] 合并收益率数据...")
    merged_returns = merge_funds_returns(funds_data)
    
    if merged_returns is None:
        print("  [错误] 无法合并收益率数据")
        return None
    
    # 横截面分析
    print("\n[3/4] 横截面分析法...")
    return_cols = [col for col in merged_returns.columns if col.startswith('return_')]
    fund_returns_df = merged_returns[return_cols].dropna()
    
    cs_result = cross_section_analysis(
        fund_returns_df,
        benchmark_returns=None
    )
    
    if cs_result:
        print(f"  样本基金数: {cs_result.get('n_funds', 'N/A')}")
        print(f"  Beta系数: {cs_result.get('beta', 'N/A'):.4f}")
        print(f"  P值: {cs_result.get('p_value', 'N/A'):.4f}")
        print(f"  R方: {cs_result.get('r_squared', 'N/A'):.4f}")
        print(f"  持续性判断: {cs_result.get('persistence_verdict', 'N/A')}")
    
    # Hurst分析
    print("\n[4/4] Hurst指数分析...")
    hurst_results = {}
    
    for fund_code, nav_df in funds_data.items():
        returns_df = calculate_daily_returns(nav_df)
        log_returns = calculate_log_returns(nav_df)
        
        fund_ret = returns_df.set_index('date')['daily_return']
        log_ret = log_returns.set_index('date')['log_return']
        
        hurst = hurst_analysis(fund_ret, log_returns=log_ret)
        if hurst:
            hurst_results[fund_code] = hurst
    
    # 绘制Hurst分布
    if hurst_results:
        plot_hurst_distribution(
            hurst_results,
            title="基金池Hurst指数分布",
            save_path=os.path.join(OUTPUT_DIR, "fund_pool_hurst_distribution.png")
        )
        
        # 统计
        H_values = [h.get('H', 0.5) for h in hurst_results.values()]
        print(f"\n  Hurst指数统计:")
        print(f"    均值: {np.mean(H_values):.4f}")
        print(f"    中位数: {np.median(H_values):.4f}")
        print(f"    最小值: {np.min(H_values):.4f}")
        print(f"    最大值: {np.max(H_values):.4f}")
        print(f"    持续性(H>0.5): {sum(1 for h in H_values if h > 0.5)} 只")
        print(f"    反转性(H<0.5): {sum(1 for h in H_values if h < 0.5)} 只")
    
    return {
        'cross_section': cs_result,
        'hurst_results': hurst_results
    }


def run_backtest(fund_codes, start_date='2018-01-01', end_date='2023-12-31'):
    """
    运行基于业绩持续性的回测
    """
    print(f"\n{'='*60}")
    print("基于业绩持续性的基金筛选回测")
    print(f"{'='*60}")
    
    # 获取数据
    print("\n获取基金数据...")
    funds_data = get_multiple_funds_data(fund_codes, start_date, end_date, DATA_DIR)
    
    if len(funds_data) == 0:
        print("  [错误] 无法获取任何基金数据")
        return None
    
    # 获取基准
    benchmark_returns = get_benchmark_data('000300', start_date, end_date)
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.set_index('date')['nav'].pct_change().dropna()
    
    # 运行回测
    print("\n运行回测...")
    results = persistence_based_backtest(
        funds_data,
        benchmark_returns=benchmark_returns,
        method='hurst',
        rebalance_freq=BACKTEST_CONFIG['rebalance_freq'],
        top_n=BACKTEST_CONFIG['top_n'],
        persistence_threshold=BACKTEST_CONFIG['persistence_threshold']
    )
    
    # 打印结果
    print_backtest_results(results)
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存回测持仓记录
    holdings = results.get('holdings', [])
    if holdings:
        holdings_df = pd.DataFrame([
            {
                'date': h['date'],
                'funds': ','.join(h['funds']),
                'n_funds': len(h['funds'])
            }
            for h in holdings
        ])
        holdings_path = os.path.join(OUTPUT_DIR, "backtest_holdings.csv")
        holdings_df.to_csv(holdings_path, index=False, encoding='utf-8-sig')
        print(f"\n  [保存] 持仓记录已保存至 {holdings_path}")
    
    # 绘制回测结果
    fig = plot_backtest_results(
        results,
        save_path=os.path.join(OUTPUT_DIR, "backtest_results.png")
    )
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基金业绩持续性分析')
    parser.add_argument('--fund', type=str, help='单只基金代码')
    parser.add_argument('--pool', action='store_true', help='分析默认基金池')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--start', type=str, default='2018-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2023-12-31', help='结束日期')
    parser.add_argument('--codes', type=str, help='基金代码列表，逗号分隔')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("基金项目：基金业绩持续性量化评价")
    print("研报复现：华泰金工研报《定量评价基金的业绩持续性》")
    print("="*60)
    
    # 确定要分析的基金
    if args.fund:
        fund_codes = [args.fund]
        analyze_single_fund(args.fund, args.start, args.end)
    
    elif args.codes:
        fund_codes = [c.strip() for c in args.codes.split(',')]
        if args.backtest:
            run_backtest(fund_codes, args.start, args.end)
        else:
            analyze_fund_pool(fund_codes, args.start, args.end)
    
    elif args.pool:
        fund_codes = DEFAULT_FUND_POOL
        analyze_fund_pool(fund_codes, args.start, args.end)
    
    elif args.backtest:
        fund_codes = DEFAULT_FUND_POOL
        run_backtest(fund_codes, args.start, args.end)
    
    else:
        # 默认：分析单只基金示例
        print("\n未指定基金，默认分析: 000628 (大成高鑫股票A)")
        analyze_single_fund('000628', args.start, args.end)
    
    print("\n" + "="*60)
    print("分析完成!")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
