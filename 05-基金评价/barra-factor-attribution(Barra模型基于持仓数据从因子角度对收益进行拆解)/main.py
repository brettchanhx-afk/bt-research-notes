# -*- coding: utf-8 -*-
"""
main.py - Barra模型因子归因命令行入口

【用法】
    python main.py --fund 019888 --start 2022-01-01 --end 2024-12-31 --freq monthly

【版本】
v1.0  2026-04-28
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = os.path.join(project_root, 'source')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Barra模型因子归因分析')
    parser.add_argument('--fund', type=str, default='019888', help='基金代码')
    parser.add_argument('--start', type=str, default='2022-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--freq', type=str, default='monthly', choices=['monthly', 'quarterly'], help='频率')
    parser.add_argument('--window', type=int, default=24, help='滚动回归窗口（月）')
    parser.add_argument('--rf', type=float, default=0.03, help='无风险利率')
    return parser.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    import numpy as np

    from source.data_loader import BarraDataLoader
    from source.factor import BarraFactorAttribution
    from source.backtest import BarraRollingBacktest, BacktestConfig, PerformanceSummary
    from source.plot import BarraVisualizer

    # 初始化
    loader = BarraDataLoader()
    viz = BarraVisualizer(output_dir=os.path.join(project_root, 'output'))

    print(f"\n{'='*70}")
    print(f"Barra模型因子归因分析")
    print(f"基金: {args.fund}  期间: {args.start}~{args.end}  频率: {args.freq}")
    print(f"{'='*70}\n")

    # ---- 数据获取 ----
    print("[1/5] 获取基金收益率...")
    fund_returns = loader.get_fund_returns(args.fund, args.start, args.end, args.freq)
    if len(fund_returns) == 0:
        print("  未获取到真实数据，使用模拟收益率")
        np.random.seed(42)
        dates = pd.date_range(args.start, args.end, freq='M')
        fund_returns = pd.Series(
            np.random.randn(len(dates)) * 0.15 / np.sqrt(12) + 0.08 / 12,
            index=dates
        )

    print("[2/5] 获取因子收益率矩阵...")
    factor_returns = loader.get_factor_returns(args.start, args.end, args.freq)

    # ---- 全样本归因 ----
    print("[3/5] 全样本Barra归因...")
    attribution = BarraFactorAttribution()
    result = attribution.run(
        fund_returns, factor_returns,
        fund_code=args.fund,
        period=f'{args.start}~{args.end}'
    )

    # ---- 滚动回归 ----
    print("[4/5] 滚动回归回测...")
    config = BacktestConfig(
        rolling_window=args.window, min_periods=12,
        start_date=args.start, end_date=args.end
    )
    backtest = BarraRollingBacktest(config)
    bt = backtest.run(fund_returns, factor_returns, fund_code=args.fund)

    # ---- 绩效指标 ----
    print("[5/5] 计算绩效指标...")
    metrics = PerformanceSummary.calculate(fund_returns, factor_returns, result)
    PerformanceSummary.print_summary(metrics, fund_code=args.fund)

    # ---- 可视化 ----
    print("\n生成可视化图表...")
    viz.plot_factor_exposure(result, title=f'Barra因子暴露 | {args.fund}')
    viz.plot_factor_contribution_waterfall(result, title=f'Barra因子贡献分解 | {args.fund}')

    if bt:
        viz.plot_rolling_exposure(
            bt['rolling_exposure'], bt['rolling_significance'],
            title=f'滚动因子暴露 | {args.fund}'
        )
        viz.plot_alpha_time_series(
            bt['rolling_alpha'], title=f'滚动Alpha | {args.fund}'
        )
        viz.plot_attribution_dashboard(
            result, bt['rolling_exposure'], bt['rolling_alpha'],
            fund_code=args.fund
        )

    # ---- 导出结果 ----
    out_dir = os.path.join(project_root, 'output')
    os.makedirs(out_dir, exist_ok=True)

    exp_df = pd.DataFrame({
        'factor': result.factor_names,
        'exposure_b': result.b,
        't_stat': result.t_stats,
        'p_value': result.p_values,
        'sig': ['***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
                for p in result.p_values],
        'contrib_pct': [result.factor_contributions.get(n, 0) * 100 for n in result.factor_names]
    })
    exp_df.to_csv(os.path.join(out_dir, f'{args.fund}_barra_factor_exposure.csv'),
                  index=False, encoding='utf-8-sig')

    if bt:
        bt['rolling_exposure'].to_csv(
            os.path.join(out_dir, f'{args.fund}_rolling_exposure.csv'), encoding='utf-8-sig')
        bt['rolling_alpha'].to_csv(
            os.path.join(out_dir, f'{args.fund}_rolling_alpha.csv'), encoding='utf-8-sig')

    factor_returns.to_csv(os.path.join(out_dir, 'factor_returns.csv'), encoding='utf-8-sig')

    print(f"\n结果已导出到: {out_dir}")
    for f in os.listdir(out_dir):
        print(f"  {f}")

    print(f"\n{'='*70}")
    print("Barra因子归因分析完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
