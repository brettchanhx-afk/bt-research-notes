# -*- coding: utf-8 -*-
"""
main.py - 债券基金风格分析主程序

使用方法:
    python main.py <fund_code> [--start START_DATE] [--end END_DATE] [--output OUTPUT_DIR]

示例:
    python main.py 000012
    python main.py 000012 --start 20230101 --end 20231231

复现研报：
    华泰证券《基于净值数据对债券基金久期和信用配置风格进行估计的方法》(2020-08-21)
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.data_loader import BondDataLoader
from source.factor import BondStyleEstimator
from source.backtest import StyleBacktest
from source.plot import BondStylePlotter
from source.utils import (
    align_dates, generate_report, print_summary, 
    save_results_to_json, export_to_csv
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='债券基金久期和信用配置风格分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py 000012                    # 分析华夏债券A
  python main.py 000012 --start 20230101   # 指定开始日期
  python main.py 000012 --window 90        # 使用90天滚动窗口
        """
    )
    
    parser.add_argument('fund_code', type=str, nargs='?', default='000012',
                       help='基金代码，如 000012，默认 000012')
    parser.add_argument('--start', type=str, default=None, 
                       help='开始日期，格式 YYYYMMDD，默认一年前')
    parser.add_argument('--end', type=str, default=None,
                       help='结束日期，格式 YYYYMMDD，默认今天')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录，默认 output/')
    parser.add_argument('--window', type=int, default=60,
                       help='滚动窗口大小（交易日），默认60')
    parser.add_argument('--step', type=int, default=20,
                       help='滚动步长（交易日），默认20')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日期
    if args.end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(args.end, '%Y%m%d')
    
    if args.start is None:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = datetime.strptime(args.start, '%Y%m%d')
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    fund_code = args.fund_code
    fund_name = f"基金{fund_code}"
    
    print(f"\n{'='*60}")
    print(f"债券基金风格分析 - {fund_name} ({fund_code})")
    print(f"分析区间: {start_str} 至 {end_str}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    output_dir = os.path.join(args.output, fund_code)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: 加载数据
    print("[Step 1] 加载数据...")
    loader = BondDataLoader()
    
    try:
        fund_df = loader.get_fund_nav(fund_code, 
                                     start_date.strftime('%Y%m%d'),
                                     end_date.strftime('%Y%m%d'))
        print(f"  [OK] 基金数据: {len(fund_df)} 条")
    except Exception as e:
        print(f"  [Warning] 获取基金数据失败: {e}")
        print("  使用模拟数据进行演示...")
        fund_df = loader._generate_mock_index_data("fund", start_str, end_str)
        fund_df["nav"] = fund_df["close"]
        fund_df["acc_nav"] = fund_df["close"]
    
    # 加载指数数据
    index_data = loader.load_all_index_data(start_str, end_str)
    print(f"  [OK] 指数数据: {len(index_data)} 个")
    
    # Step 2: 数据对齐
    print("\n[Step 2] 数据对齐...")
    all_data = {"fund": fund_df}
    for code, df in index_data.items():
        all_data[code] = df
    
    aligned_data = align_dates(all_data, date_col="date")
    
    # 提取收益率序列
    fund_returns = aligned_data["fund"].set_index("date")["daily_return"]
    index_returns = {
        code: df.set_index("date")["daily_return"] 
        for code, df in aligned_data.items() if code != "fund"
    }
    
    # 构建指数收益率矩阵
    X = pd.DataFrame(index_returns).dropna()
    y = fund_returns.loc[X.index].dropna()
    X = X.loc[y.index]
    
    print(f"  [OK] 对齐后数据: {len(X)} 个交易日")
    
    # Step 3: 风格估计
    print("\n[Step 3] 风格估计...")
    index_info = loader.get_index_info()
    estimator = BondStyleEstimator(index_info)
    
    fit_result = estimator.fit(X, y, method="ols")
    print(f"  回归 R-squared: {fit_result['r2']:.4f}")
    
    # 估计久期和信用
    dur_result = estimator.estimate_duration()
    cred_result = estimator.estimate_credit()
    style_box = estimator.get_style_box()
    
    print(f"  估计久期: {dur_result['estimated_duration']:.2f} 年 ({dur_result['style_label']})")
    print(f"  估计信用: {cred_result['estimated_credit']:.2f} 分 ({cred_result['style_label']})")
    print(f"  风格定位: {style_box['style_box']}")
    
    # Step 4: 滚动回测
    print("\n[Step 4] 滚动回测...")
    backtest = StyleBacktest(estimator)
    rolling_results = backtest.run_rolling_backtest(X, y, 
                                                    window=args.window, 
                                                    step=args.step)
    print(f"  [OK] 回测窗口数: {len(rolling_results)}")
    
    # Step 5: 稳定性分析
    print("\n[Step 5] 稳定性分析...")
    stability = backtest.calculate_style_stability()
    print(f"  久期稳定性: {stability['duration']['mean']:.2f} +/- {stability['duration']['std']:.2f}")
    print(f"  信用稳定性: {stability['credit']['mean']:.2f} +/- {stability['credit']['std']:.2f}")
    
    # Step 6: 生成图表
    print("\n[Step 6] 生成图表...")
    plotter = BondStylePlotter()
    
    # 风格演变图
    fig1 = plotter.plot_style_evolution(rolling_results,
                                        save_path=os.path.join(output_dir, f"{fund_code}_style_evolution.png"))
    plt_close(fig1)
    
    # 风格箱定位图
    fig2 = plotter.plot_style_box(style_box['duration'], style_box['credit'], fund_name,
                                 save_path=os.path.join(output_dir, f"{fund_code}_style_box.png"))
    plt_close(fig2)
    
    # 因子暴露图
    fig3 = plotter.plot_factor_exposure(fit_result['coef'], index_info,
                                       save_path=os.path.join(output_dir, f"{fund_code}_factor_exposure.png"))
    plt_close(fig3)
    
    # Step 7: 生成报告
    print("\n[Step 7] 生成报告...")
    
    # 汇总结果
    style_result = {
        "duration": dur_result['estimated_duration'],
        "credit": cred_result['estimated_credit'],
        "duration_label": dur_result['style_label'],
        "credit_label": cred_result['style_label'],
        "style_box": style_box['style_box'],
        "r2": fit_result['r2'],
        "coef": fit_result['coef'].to_dict()
    }
    
    # 保存结果
    save_results_to_json(style_result, os.path.join(output_dir, f"{fund_code}_style_result.json"))
    export_to_csv(rolling_results, os.path.join(output_dir, f"{fund_code}_rolling_results.csv"))
    
    # 生成文本报告
    generate_report(fund_code, fund_name, style_result, rolling_results, stability, output_dir)
    
    # 打印摘要
    print_summary(style_result, stability)
    
    print(f"\n[OK] 分析完成！结果已保存到: {output_dir}")
    
    return style_result


def plt_close(fig):
    """关闭图表释放内存"""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
