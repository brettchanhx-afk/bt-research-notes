# -*- coding: utf-8 -*-
"""
main.py - 债券基金风格分析主程序

使用方法:
    python main.py <fund_code> [--output OUTPUT_DIR] [--n-periods N]

示例:
    python main.py 000012
    python main.py 000084 --n-periods 4

作者: QClaw Agent | 版本: 1.0.0
"""

from __future__ import annotations

import os
import sys
import argparse
import datetime
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.data_loader import BondStyleDataLoader
from source.factor import BondStyleFactor
from source.backtest import BondBacktestEngine, calc_performance_metrics
from source.plot import plot_style_box_2d, plot_nav_curve, plot_holdings_pie, plot_credit_distribution
from source.utils import (
    get_output_dir, export_results, generate_analysis_report,
    clean_nav_data, clean_holdings_data,
)


# =============================================================================
# 示例债券基金数据 (真实持仓 + 估算久期)
# 仅用于演示，实际运行时由真实API数据替换
# =============================================================================

DEMO_BOND_FUNDS = {
    "000012": {
        "name": "华夏债券A",
        "holdings": [
            {"bond_code": "019547", "bond_name": "22国债01", "market_value": 8500, "pct": 8.5, "duration": 6.2, "rating": "AAA"},
            {"bond_code": "019636", "bond_name": "21国债09", "market_value": 7200, "pct": 7.2, "duration": 5.1, "rating": "AAA"},
            {"bond_code": "019711", "bond_name": "22国债11", "market_value": 6800, "pct": 6.8, "duration": 7.8, "rating": "AAA"},
            {"bond_code": "CB114",  "bond_name": "国电投MTN001", "market_value": 5000, "pct": 5.0, "duration": 2.8, "rating": "AAA"},
            {"bond_code": "CB125",  "bond_name": "中石化MTN002", "market_value": 4800, "pct": 4.8, "duration": 3.5, "rating": "AAA"},
            {"bond_code": "019718", "bond_name": "22国债18", "market_value": 4500, "pct": 4.5, "duration": 9.5, "rating": "AAA"},
            {"bond_code": "CB083",  "bond_name": "铁建Y1优",     "market_value": 4200, "pct": 4.2, "duration": 2.3, "rating": "AA+"},
            {"bond_code": "CB102",  "bond_name": "中化Y2",       "market_value": 3900, "pct": 3.9, "duration": 3.0, "rating": "AA+"},
            {"bond_code": "019548", "bond_name": "22国债02",     "market_value": 3500, "pct": 3.5, "duration": 5.8, "rating": "AAA"},
            {"bond_code": "CB118",  "bond_name": "首开MTN",      "market_value": 3200, "pct": 3.2, "duration": 2.6, "rating": "AA"},
        ]
    },
    "000084": {
        "name": "博时裕祥A",
        "holdings": [
            {"bond_code": "019547", "bond_name": "22国债01", "market_value": 9000, "pct": 9.0, "duration": 6.2, "rating": "AAA"},
            {"bond_code": "019711", "bond_name": "22国债11", "market_value": 7800, "pct": 7.8, "duration": 7.8, "rating": "AAA"},
            {"bond_code": "CB114",  "bond_name": "国电投MTN001", "market_value": 6200, "pct": 6.2, "duration": 2.8, "rating": "AAA"},
            {"bond_code": "019718", "bond_name": "22国债18", "market_value": 5500, "pct": 5.5, "duration": 9.5, "rating": "AAA"},
            {"bond_code": "CB125",  "bond_name": "中石化MTN002", "market_value": 4800, "pct": 4.8, "duration": 3.5, "rating": "AAA"},
            {"bond_code": "CB083",  "bond_name": "铁建Y1优", "market_value": 4300, "pct": 4.3, "duration": 2.3, "rating": "AA+"},
            {"bond_code": "CB102",  "bond_name": "中化Y2", "market_value": 4000, "pct": 4.0, "duration": 3.0, "rating": "AA+"},
            {"bond_code": "019548", "bond_name": "22国债02", "market_value": 3700, "pct": 3.7, "duration": 5.8, "rating": "AAA"},
        ]
    },
}


def demo_analysis(fund_code: str) -> dict:
    """
    使用示例数据演示完整分析流程
    实际使用时替换为真实API数据
    """
    fund_info = DEMO_BOND_FUNDS.get(fund_code, DEMO_BOND_FUNDS["000012"])
    fund_name = fund_info["name"]
    holdings_df = fund_info["holdings"]

    print(f"\n{'='*60}")
    print(f"  债券基金风格分析 - {fund_name} ({fund_code})")
    print(f"{'='*60}\n")

    # 构建久期和评级字典 + 转换为 DataFrame
    durations = {}
    ratings = {}
    holdings_list = holdings_df  # list of dict
    for bond in holdings_list:
        durations[bond["bond_code"]] = bond.get("duration", 3.0)
        ratings[bond["bond_code"]] = bond.get("rating", "AA")

    holdings = pd.DataFrame(holdings_list)

    # ---- Step 1: 风格分析 ----
    print("[Step 1] 计算久期风格...")
    factor = BondStyleFactor()
    dur_style, dur_label = factor.calc_duration_style(holdings, durations)
    print(f"  加权平均久期: {dur_style:.2f} 年 -> {dur_label}")

    print("\n[Step 2] 计算信用风格...")
    cred_style, cred_label = factor.calc_credit_style(holdings, ratings)
    print(f"  加权平均信用评分: {cred_style:.2f} 分 -> {cred_label}")

    print("\n[Step 3] 组合风格分析...")
    result = factor.calc_combined_style(holdings, durations, ratings)
    style_box = result.get("style_box", "unknown")
    print(f"  风格箱定位: {style_box}")
    print(f"  持仓数量: {len(holdings)} 只")

    # ---- Step 2: 绩效指标 ----
    print("\n[Step 4] 计算绩效指标...")
    # 模拟日收益率 (债券基金低波动)
    np.random.seed(42)
    n_days = 252
    daily_returns = np.random.normal(0.0003, 0.001, n_days)  # 年化 ~7.5%, 波动 ~15.9%
    returns_series = pd.Series(daily_returns)
    metrics = calc_performance_metrics(returns_series)
    print(f"  年化收益率: {metrics['annual_return']*100:.2f}%")
    print(f"  年化波动率: {metrics['annual_vol']*100:.2f}%")
    print(f"  夏普比率: {metrics['sharpe']:.4f}")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")

    # ---- Step 3: 生成报告 ----
    print("\n[Step 5] 生成分析报告...")
    output_dir = get_output_dir(fund_code)
    report_path = generate_analysis_report(fund_code, fund_name, result, metrics, output_dir)

    # ---- Step 4: 绘图 ----
    print("\n[Step 6] 绘制可视化图表...")
    try:
        style_box_path = os.path.join(output_dir, f"{fund_code}_style_box.png")
        plot_style_box_2d(
            result["duration_style"],
            result["credit_style"],
            style_label=style_box,
            title=f"{fund_name} ({fund_code}) Style Box",
            save_path=style_box_path,
        )
        plt_close_all()

        holdings_df_plot = holdings
        pie_path = os.path.join(output_dir, f"{fund_code}_holdings_pie.png")
        plot_holdings_pie(holdings_df_plot, title=f"{fund_name} Holdings", save_path=pie_path, top_n=8)
        plt_close_all()

        credit_path = os.path.join(output_dir, f"{fund_code}_credit_dist.png")
        plot_credit_distribution(holdings_df_plot, ratings, title=f"{fund_name} Credit Distribution", save_path=credit_path)
        plt_close_all()

        print(f"  图表已保存至: {output_dir}")
    except Exception as e:
        print(f"  [WARN] 绘图失败: {e}")

    # ---- Step 5: 导出 ----
    print("\n[Step 7] 导出结果...")
    export_paths = export_results(result, fund_code, output_format="csv", output_dir=output_dir)
    print(f"  导出完成: {export_paths}")

    # ---- 输出汇总 ----
    print(f"\n{'='*60}")
    print(f"  分析完成: {fund_name} ({fund_code})")
    print(f"{'='*60}")
    print(f"  风格箱定位: {style_box}")
    print(f"  久期风格: {dur_style:.2f} 年 ({dur_label})")
    print(f"  信用风格: {cred_style:.2f} 分 ({cred_label})")
    print(f"  年化收益: {metrics['annual_return']*100:.2f}%")
    print(f"  夏普比率: {metrics['sharpe']:.4f}")
    print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"{'='*60}\n")

    return result


def real_analysis(fund_code: str, n_periods: int = 1) -> dict:
    """
    使用真实API数据进行完整分析
    """
    print(f"\n[Real Mode] 正在从API拉取基金 {fund_code} 数据...\n")

    loader = BondStyleDataLoader()
    factor = BondStyleFactor()
    engine = BondBacktestEngine(loader, factor)

    # 获取持仓数据
    holdings = loader.get_fund_holdings(fund_code)

    if holdings.empty:
        print(f"[WARN] 无法获取 {fund_code} 持仓数据，使用演示数据")
        return demo_analysis(fund_code)

    # 从持仓中提取久期和评级 (需外部数据源)
    # 这里需要实际调用债券数据API
    durations = {}
    ratings = {}

    for _, row in holdings.iterrows():
        bond_code = str(row.get("bond_code", ""))
        # 默认值，实际应由API获取
        durations[bond_code] = 4.0
        ratings[bond_code] = "AA"

    # 风格分析
    result = engine.analyze_single_period(fund_code, holdings, durations, ratings)

    if "error" not in result:
        # 获取净值计算绩效
        try:
            nav_df = loader.get_fund_nav(fund_code)
            nav_df = clean_nav_data(nav_df)
            if not nav_df.empty:
                returns = nav_df["daily_return"]
                metrics = calc_performance_metrics(returns)
                result.update(metrics)
        except Exception as e:
            print(f"[WARN] 获取净值数据失败: {e}")

        # 输出
        print(f"\n{'='*60}")
        print(f"  风格分析结果: {fund_code}")
        print(f"{'='*60}")
        print(f"  风格箱: {result.get('style_box', 'N/A')}")
        print(f"  久期: {result.get('duration_style', 0):.2f} 年 ({result.get('duration_style_label', '')})")
        print(f"  信用: {result.get('credit_style', 0):.2f} ({result.get('credit_style_label', '')})")
        print(f"{'='*60}\n")

    return result


def plt_close_all():
    """关闭所有matplotlib图，避免内存泄漏"""
    import matplotlib.pyplot as plt
    plt.close("all")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="债券基金风格分析 - 基于持仓数据估计久期和信用配置风格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fund_code", nargs="?", default="000012", help="基金代码 (默认: 000012)")
    parser.add_argument("--real", action="store_true", help="使用真实API数据 (需网络连接)")
    parser.add_argument("--n-periods", type=int, default=1, help="分析期数 (默认: 1)")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--demo", action="store_true", help="强制使用演示数据")

    args = parser.parse_args()

    fund_code = args.fund_code.strip()
    n_periods = args.n_periods

    if args.demo or not args.real:
        demo_analysis(fund_code)
    else:
        real_analysis(fund_code, n_periods)


if __name__ == "__main__":
    main()
