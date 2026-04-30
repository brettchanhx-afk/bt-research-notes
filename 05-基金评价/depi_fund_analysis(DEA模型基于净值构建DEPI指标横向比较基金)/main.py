# -*- coding: utf-8 -*-
"""
DEPI 基金绩效归因分析 - 主程序
复现研报：《DEA模型基于净值构建DEPI指标，横向比较同类基金》（华泰金工，2020-08-21）

执行方式：
  python main.py

本程序完整流程：
  1. 数据获取（efinance基金净值 + akshare费率 + baostock沪深300）
  2. 因子构建（volatility, fee_rate, timing_alpha, timing_beta, 超额收益R）
  3. CCR模型截面DEPI分析
  4. 滚动回测（季度调仓）
  5. 可视化输出
"""
import os
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

import config
from source.data_loader import (
    get_fund_list, get_fund_nav_history,
    get_fund_fee_rate, get_benchmark_history,
)
from source.factor import build_factor_table
from source.backtest import DEPIEngine, backtest_depi
from source.plot import (
    setup_chinese_font,
    plot_depi_distribution,
    plot_depi_timeseries,
    plot_depi_bar_topn,
)
from source.utils import save_df


def main():
    print('=' * 60)
    print('DEPI 基金绩效归因分析 - 研报复现')
    print('研报：DEA模型基于净值构建DEPI指标，横向比较同类基金')
    print('来源：华泰金工，2020-08-21')
    print('=' * 60)

    # ---- Step 1: 数据获取 ----
    print('\n[Step 1/5] 获取数据...')
    fund_list = get_fund_list(n=config.SAMPLE_SIZE)
    if fund_list.empty:
        print('[ERROR] 基金列表为空，请检查网络连接。')
        return
    fund_codes = fund_list['基金代码'].tolist()

    # 基金净值
    nav_dict = get_fund_nav_history(
        fund_codes, config.BACKTEST_START, config.BACKTEST_END
    )
    if not nav_dict:
        print('[ERROR] 基金净值数据为空。')
        return

    # 费率
    fee_df = get_fund_fee_rate(fund_codes)

    # 基准指数
    benchmark_df = get_benchmark_history(
        config.BENCHMARK_CODE, config.BACKTEST_START, config.BACKTEST_END
    )
    if benchmark_df.empty:
        print('[ERROR] 基准指数数据为空。')
        return
    benchmark_returns = benchmark_df['日收益率']

    print(f'\n  基金数量：{len(nav_dict)}')
    print(f'  回测区间：{config.BACKTEST_START} ~ {config.BACKTEST_END}')
    print(f'  无风险利率：{config.RISK_FREE_RATE:.1%}')
    print(f'  调仓频率：{config.REBALANCE_FREQ}')

    # ---- Step 2: 因子构建 ----
    print('\n[Step 2/5] 构建因子表...')
    factor_df = build_factor_table(
        nav_dict, benchmark_returns, fee_df, config.RISK_FREE_RATE
    )
    if factor_df.empty:
        print('[ERROR] 因子表为空。')
        return

    save_df(factor_df, 'factor_table.csv', str(config.DATA_DIR))
    print(factor_df.describe())

    # ---- Step 3: DEPI截面分析 ----
    print('\n[Step 3/5] DEPI截面分析...')
    engine = DEPIEngine()
    depi_result = engine.fit_transform(
        factor_df,
        output_col='超额收益R',
        input_cols=config.INPUT_INDICATORS
    )
    print(f'\n  统计摘要：{engine.get_summary()}')
    print(f'\n  DEPI Top-10:')
    print(depi_result[['基金代码', 'DEPI', 'DEPI_Rank', '超额收益R',
                       'volatility', 'fee_rate']].head(10).to_string(index=False))

    save_df(depi_result, 'depi_cross_section.csv', str(config.DATA_DIR))

    # ---- Step 4: 滚动回测 ----
    print('\n[Step 4/5] 滚动回测...')
    depi_ts = backtest_depi(
        nav_dict, benchmark_returns, fee_df,
        start=config.BACKTEST_START,
        end=config.BACKTEST_END,
        freq=config.REBALANCE_FREQ,
        input_cols=config.INPUT_INDICATORS,
    )
    if not depi_ts.empty:
        save_df(depi_ts, 'depi_timeseries.csv', str(config.DATA_DIR))

    # ---- Step 5: 可视化 ----
    print('\n[Step 5/5] 生成图表...')
    setup_chinese_font()

    # 图1：DEPI分布
    plot_depi_distribution(
        depi_result,
        title='DEPI Cross-sectional Distribution',
        save_path=str(config.OUTPUT_DIR / 'depi_distribution.png')
    )

    # 图2：Top-N柱状图
    plot_depi_bar_topn(
        depi_result, top_n=15,
        title='DEPI Top-15 Funds',
        save_path=str(config.OUTPUT_DIR / 'depi_top15.png')
    )

    # 图3：时间序列（Top5）
    if not depi_ts.empty:
        plot_depi_timeseries(
            depi_ts, top_n=5,
            title='DEPI Top-5 Funds Time Series',
            save_path=str(config.OUTPUT_DIR / 'depi_timeseries.png')
        )

    print('\n' + '=' * 60)
    print('分析完成！')
    print(f'数据保存目录：{config.DATA_DIR}')
    print(f'图表保存目录：{config.OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
