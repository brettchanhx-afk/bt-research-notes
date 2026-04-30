# -*- coding: utf-8 -*-
"""
主程序 - 基金选股择时能力定量评价模型
T-M 模型 / H-M 模型 / C-L 模型

华泰金工研究 | 2020-08-21
使用方法:
    python main.py --fund 021181 --start 2021-01-01 --end 2026-04-28
    python main.py --fund 021181 --model TM  # 仅运行 T-M 模型
    python main.py --fund 021181 --rolling  # 启用滚动回测
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 添加 source 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source'))

from config import (
    START_DATE, END_DATE, BENCHMARK, RISK_FREE_RATE,
    OUTPUT_DIR, DATA_DIR, PLT_STYLE, CHINESE_FONT
)
from source.data_loader import load_all_data
from source.factor import StockTimingEvaluator, TMModel, HMModel, CLModel
from source.backtest import RollingTimingBacktest, PerformanceAttribution
from source.plot import (
    plot_timing_dashboard,
    plot_rolling_timing,
    ensure_output_dir,
)
from source.utils import get_fund_name, summary_stats

# ==================== 全局字体设置 ====================
def setup_font():
    """设置 matplotlib 中文字体（必须在 plt.style.use 之后）。"""
    matplotlib.rcParams['font.sans-serif'] = CHINESE_FONT
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 120


# ==================== 主分析函数 ====================
def analyze_fund(fund_code: str,
                start_date: str = START_DATE,
                end_date: str = END_DATE,
                benchmark: str = BENCHMARK,
                risk_free_rate: float = RISK_FREE_RATE,
                output_dir: str = OUTPUT_DIR,
                run_rolling: bool = True,
                window: int = 252,
                step: int = 21) -> dict:
    """
    对单只基金进行选股择时能力分析。

    参数:
        fund_code: 基金代码（如 '021181'）
        start_date: 开始日期
        end_date: 结束日期
        benchmark: 基准指数代码
        risk_free_rate: 年化无风险利率
        output_dir: 输出目录
        run_rolling: 是否运行滚动回测
        window: 滚动窗口长度（交易日）
        step: 滚动步长（交易日）
    返回:
        包含所有分析结果的字典
    """
    print(f"\n{'=' * 60}")
    print(f"  基金选股择时能力分析 | {fund_code} {get_fund_name(fund_code)}")
    print(f"{'=' * 60}")
    print(f"  分析区间: {start_date} 至 {end_date}")
    print(f"  基准指数: {benchmark}")
    print(f"  无风险利率: {risk_free_rate * 100:.2f}% 年化")
    print(f"{'=' * 60}\n")

    ensure_output_dir(output_dir)
    ensure_output_dir(DATA_DIR)

    # ==================== 1. 数据加载 ====================
    print("[Step 1/5] 加载数据...")
    try:
        fund_returns, bench_returns = load_all_data(
            fund_code, start_date, end_date, benchmark
        )
    except ValueError as e:
        print(f"数据加载失败: {e}")
        raise

    print(f"  有效数据: {len(fund_returns)} 个交易日")

    # ==================== 2. 基础统计 ====================
    print("\n[Step 2/5] 基础统计...")
    fund_stats = summary_stats(fund_returns['基金收益率'])
    bench_stats = summary_stats(bench_returns['基准收益率'])

    print(f"  基金年化收益: {fund_stats['ann_return']*100:.2f}%")
    print(f"  基金年化波动: {fund_stats['ann_vol']*100:.2f}%")
    print(f"  基金夏普比率: {fund_stats['sharpe']:.3f}")
    print(f"  基金最大回撤: {fund_stats['max_drawdown']*100:.2f}%")

    # ==================== 3. 三模型回归 ====================
    print("\n[Step 3/5] T-M / H-M / C-L 模型回归...")

    evaluator = StockTimingEvaluator(
        fund_returns['基金收益率'],
        bench_returns['基准收益率'],
        risk_free_rate
    )
    model_results = evaluator.evaluate()

    # 打印汇总表
    summary_df = evaluator.get_summary()
    print("\n  模型结果汇总:")
    print(summary_df.to_string(index=False))

    # ==================== 4. 滚动回测 ====================
    rolling_data = {}
    if run_rolling:
        print(f"\n[Step 4/5] 滚动窗口回测 (window={window}, step={step})...")

        bt = RollingTimingBacktest(
            fund_returns['基金收益率'],
            bench_returns['基准收益率'],
            window=window,
            risk_free_rate=risk_free_rate,
        )

        for model_name, model_cls in [('TM', 'TM'), ('HM', 'HM'), ('CL', 'CL')]:
            print(f"  运行 {model_name} 滚动回测...")
            rolling = bt.run(model=model_name, step=step)
            if not rolling.empty:
                rolling_data[model_name] = rolling
                print(f"    {model_name} 滚动窗口数: {len(rolling)}")

                # 绘制滚动图
                fig_path = os.path.join(output_dir, f'{fund_code}_rolling_{model_name}.png')
                plot_rolling_timing(rolling, model_name, fig_path,
                                    title=f'{fund_code} Rolling {model_name} Timing Ability')

    # ==================== 5. 输出结果 ====================
    print("\n[Step 5/5] 保存结果...")

    # 保存 JSON 结果
    output_data = {
        'fund_code': fund_code,
        'fund_name': get_fund_name(fund_code),
        'analysis_period': {
            'start': start_date,
            'end': end_date,
        },
        'benchmark': benchmark,
        'risk_free_rate': risk_free_rate,
        'data_points': int(len(fund_returns)),
        'fund_stats': {k: float(v) for k, v in fund_stats.items()},
        'bench_stats': {k: float(v) for k, v in bench_stats.items()},
        'model_results': {},
    }

    # 添加各模型结果（转换为 Python 原生类型）
    for model_name, res in model_results.items():
        clean_res = {}
        for k, v in res.items():
            if isinstance(v, (np.bool_, np.floating, np.integer)):
                clean_res[k] = float(v) if isinstance(v, (np.floating, np.integer)) else bool(v)
            elif isinstance(v, np.ndarray):
                clean_res[k] = v.tolist()
            else:
                clean_res[k] = v
        output_data['model_results'][model_name] = clean_res

    # JSON 序列化
    json_path = os.path.join(output_dir, f'{fund_code}_timing_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON 结果: {json_path}")

    # 仪表盘图
    # 将 DataFrame 转换为 dict 以传递给绘图函数
    dashboard_data = {k: v for k, v in model_results.items()}
    dashboard_data['excess_fund'] = fund_returns['基金收益率']
    dashboard_data['excess_bench'] = bench_returns['基准收益率']

    dashboard_path = os.path.join(output_dir, f'{fund_code}_timing_dashboard.png')
    plot_timing_dashboard(
        fund_code,
        get_fund_name(fund_code),
        dashboard_data,
        dashboard_path
    )

    # 保存滚动结果 CSV
    for model_name, rolling in rolling_data.items():
        csv_path = os.path.join(output_dir, f'{fund_code}_rolling_{model_name}.csv')
        rolling.to_csv(csv_path)
        print(f"  滚动CSV ({model_name}): {csv_path}")

    # 保存文本报告
    report_path = os.path.join(output_dir, f'{fund_code}_timing_report.txt')
    write_text_report(output_data, report_path)
    print(f"  文本报告: {report_path}")

    print(f"\n{'=' * 60}")
    print(f"  分析完成！结果已保存至: {output_dir}")
    print(f"{'=' * 60}")

    return output_data


def write_text_report(data: dict, output_path: str) -> None:
    """将分析结果写入文本报告。"""
    fund_code = data['fund_code']
    fund_name = data['fund_name']
    results = data['model_results']

    lines = []
    lines.append("=" * 65)
    lines.append(f"  基金选股择时能力定量评价报告")
    lines.append(f"  {fund_name} ({fund_code})")
    lines.append("=" * 65)
    lines.append("")

    # 基本信息
    lines.append(f"【基本信息】")
    lines.append(f"  基金代码: {fund_code}")
    lines.append(f"  基金名称: {fund_name}")
    lines.append(f"  分析区间: {data['analysis_period']['start']} 至 {data['analysis_period']['end']}")
    lines.append(f"  基准指数: {data['benchmark']}")
    lines.append(f"  无风险利率: {data['risk_free_rate']*100:.2f}% (年化)")
    lines.append(f"  数据点数: {data['data_points']} 个交易日")
    lines.append("")

    # 基础统计
    fs = data['fund_stats']
    lines.append(f"【基金业绩统计】")
    lines.append(f"  年化收益率: {fs['ann_return']*100:.2f}%")
    lines.append(f"  年化波动率: {fs['ann_vol']*100:.2f}%")
    lines.append(f"  夏普比率:   {fs['sharpe']:.3f}")
    lines.append(f"  最大回撤:   {fs['max_drawdown']*100:.2f}%")
    lines.append("")

    # 三模型结果
    lines.append(f"【选股择时能力评估】")
    lines.append("")

    model_descriptions = {
        'TM': 'T-M 模型 (Treynor-Mazuy, 1966)',
        'HM': 'H-M 模型 (Henriksson-Merton, 1981)',
        'CL': 'C-L 模型 (Chang-Lewellen, 1984)',
    }

    for model_name, desc in model_descriptions.items():
        if model_name not in results:
            continue
        r = results[model_name]

        lines.append(f"  ---- {desc} ----")
        lines.append(f"  Alpha (选股能力):   {r.get('alpha', 0):.6f}  "
                    f"(p={r.get('alpha_pvalue', 1):.4f}) "
                    f"{'显著' if r.get('alpha_significant', False) else '不显著'}")
        lines.append(f"  Beta2 (择时能力):   {r.get('beta2', 0):.6f}  "
                    f"(p={r.get('beta2_pvalue', 1):.4f}) "
                    f"{'显著' if r.get('beta2_significant', False) else '不显著'}")

        if model_name == 'TM':
            lines.append(f"  择时能力判断: {'有' if r.get('timing_ability', False) else '无'} "
                        f"(beta2>0 且 p<0.05)")
        elif model_name == 'HM':
            lines.append(f"  牛市Beta: {r.get('bull_beta', 0):.4f}  "
                        f"熊市Beta: {r.get('bear_beta', 0):.4f}")
            lines.append(f"  择时能力判断: {'有' if r.get('timing_ability', False) else '无'}")
        elif model_name == 'CL':
            diff = r.get('timing_diff', 0)
            lines.append(f"  多头Beta: {r.get('beta2', 0):.4f}  "
                        f"空头Beta: {r.get('beta1', 0):.4f}  "
                        f"差值: {diff:.4f}")
            lines.append(f"  择时能力判断: {'有' if r.get('timing_ability', False) else '无'} "
                        f"(beta2-beta1>0)")

        lines.append(f"  R-squared:   {r.get('r_squared', 0):.4f}")
        lines.append(f"  Adj R2:      {r.get('adj_r_squared', 0):.4f}")
        lines.append(f"  观测数:      {r.get('nobs', 0)}")
        lines.append("")

    # 综合判断
    lines.append(f"【综合判断】")
    timing_count = sum(1 for r in results.values() if r.get('timing_ability', False))
    stock_count = sum(1 for r in results.values() if r.get('stock_ability', False))

    if timing_count >= 2:
        lines.append(f"  择时能力: 确认（{timing_count}/3 模型显著）")
    elif timing_count == 1:
        lines.append(f"  择时能力: 弱确认（{timing_count}/3 模型显著）")
    else:
        lines.append(f"  择时能力: 未确认（0/3 模型显著）")

    if stock_count >= 2:
        lines.append(f"  选股能力: 确认（{stock_count}/3 模型显著）")
    elif stock_count == 1:
        lines.append(f"  选股能力: 弱确认（{stock_count}/3 模型显著）")
    else:
        lines.append(f"  选股能力: 未确认（0/3 模型显著）")

    lines.append("")
    lines.append("=" * 65)
    lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("方法论: 华泰金工研究 | Treynor-Mazuy / Henriksson-Merton / Chang-Lewellen")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ==================== 命令行入口 ====================
def main():
    parser = argparse.ArgumentParser(
        description='基金选股择时能力定量评价 - T-M / H-M / C-L 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --fund 021181
  python main.py --fund 021181 --start 2021-01-01 --end 2026-04-28
  python main.py --fund 021181 --rolling --window 252 --step 21
  python main.py --fund 000628 --benchmark 000001.SH
        """
    )

    parser.add_argument('--fund',     type=str, default='021181',
                        help='基金代码 (default: 021181)')
    parser.add_argument('--start',    type=str, default=START_DATE,
                        help=f'开始日期 (default: {START_DATE})')
    parser.add_argument('--end',      type=str, default=END_DATE,
                        help=f'结束日期 (default: {END_DATE})')
    parser.add_argument('--benchmark', type=str, default=BENCHMARK,
                        help=f'基准指数代码 (default: {BENCHMARK})')
    parser.add_argument('--rf',       type=float, default=RISK_FREE_RATE,
                        help=f'年化无风险利率 (default: {RISK_FREE_RATE})')
    parser.add_argument('--output',   type=str, default=None,
                        help='输出目录 (default: ./output)')
    parser.add_argument('--rolling',  action='store_true',
                        help='启用滚动回测')
    parser.add_argument('--window',   type=int, default=252,
                        help='滚动窗口长度，交易日 (default: 252)')
    parser.add_argument('--step',     type=int, default=21,
                        help='滚动步长，交易日 (default: 21)')
    parser.add_argument('--no-rolling', dest='rolling', action='store_false',
                        help='禁用滚动回测')

    args = parser.parse_args()

    # 设置字体
    plt.style.use(PLT_STYLE)
    setup_font()

    # 输出目录
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output', args.fund
    )

    # 执行分析
    try:
        analyze_fund(
            fund_code=args.fund,
            start_date=args.start,
            end_date=args.end,
            benchmark=args.benchmark,
            risk_free_rate=args.rf,
            output_dir=output_dir,
            run_rolling=args.rolling,
            window=args.window,
            step=args.step,
        )
    except Exception as e:
        print(f"\n分析出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
