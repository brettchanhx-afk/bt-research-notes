#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py — Barra 基金风格分析入口脚本

用法:
    python main.py                          # 分析 config.py 中配置的基金
    python main.py --fund 000628            # 指定基金代码
    python main.py --force                  # 强制重新下载数据
    python main.py --list                   # 批量分析多只基金
"""

import argparse
import logging
import sys
from datetime import datetime

import config
from barra import DataLoader, FactorBuilder, BarraRegression, BarraPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def analyze_fund(fund_code: str, fund_name: str, force_reload: bool = False):
    """单只基金完整分析流程。"""
    logger.info(f"\n{'='*60}")
    logger.info(f"开始分析: {fund_name} ({fund_code})")
    logger.info(f"{'='*60}")

    # 1. 数据加载
    loader = DataLoader(config.DATA_DIR, config.START_DATE, config.END_DATE)
    fund_nav = loader.load_fund_nav(fund_code, force_reload=force_reload)
    index_df = loader.load_all_indexes(config.FACTOR_INDEXES, force_reload=force_reload)

    # 2. 因子构建
    factor_builder = FactorBuilder(index_df)
    factors = factor_builder.build()

    # 3. 回归分析
    reg = BarraRegression(fund_nav["return"], factors)
    reg.fit()
    exposures = reg.exposures()
    rolling   = reg.rolling_fit(window=config.ROLLING_WINDOW)

    # 4. 业绩指标
    perf = BarraRegression.performance_metrics(fund_nav["return"])

    # 5. 可视化
    plotter = BarraPlotter(style=config.PLOT_STYLE, dpi=config.DPI)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path  = f"{config.OUTPUT_DIR}/report_{fund_code}_{timestamp}.png"
    plotter.plot_report(
        fund_nav=(1 + fund_nav["return"]).cumprod(),
        exposures=exposures,
        rolling=rolling,
        output_path=fig_path
    )

    # 6. 输出摘要
    logger.info(f"\n{'─'*60}")
    logger.info("分析结果摘要")
    logger.info(f"{'─'*60}")
    logger.info(f"累计收益: {perf['cum_return']*100:+.2f}%")
    logger.info(f"年化收益: {perf['annual_return']*100:+.2f}%")
    logger.info(f"夏普比率: {perf['sharpe']:.2f}")
    logger.info(f"最大回撤: {perf['max_drawdown']*100:.2f}%")
    logger.info(f"回归 R²:  {reg.results.rsquared:.4f}")
    logger.info(f"\n风格暴露:")
    for f, v in exposures.items():
        logger.info(f"  {f:12s}: {v:+.4f}")
    logger.info(f"{'─'*60}")

    return {
        "fund_code": fund_code,
        "fund_name": fund_name,
        "performance": perf,
        "exposures": exposures.to_dict(),
        "r2": reg.results.rsquared,
        "report_path": fig_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Barra 基金风格分析")
    parser.add_argument("--fund", type=str, help="基金代码（如 000628）")
    parser.add_argument("--force", action="store_true", help="强制重新下载数据")
    parser.add_argument("--list", action="store_true", help="批量分析 config.FUND_LIST")
    args = parser.parse_args()

    results = []

    if args.list:
        for item in config.FUND_LIST:
            r = analyze_fund(item["code"], item["name"], force_reload=args.force)
            results.append(r)
    elif args.fund:
        name = next((f["name"] for f in config.FUND_LIST if f["code"] == args.fund), args.fund)
        r = analyze_fund(args.fund, name, force_reload=args.force)
        results.append(r)
    else:
        # 默认分析第一个
        item = config.FUND_LIST[0]
        r = analyze_fund(item["code"], item["name"], force_reload=args.force)
        results.append(r)

    logger.info(f"\n✓ 分析完成，共 {len(results)} 只基金")
    return results


if __name__ == "__main__":
    main()
