import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DATA_DIR = 'output/data'
OUTPUT_CHART_DIR = 'output/charts'

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_CHART_DIR, exist_ok=True)


def load_data(filename):
    filepath = os.path.join('data', filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"[LOAD] {filename} - {len(df)} rows")
        return df
    else:
        print(f"[SKIP] {filename} - Not found")
        return pd.DataFrame()


def save_output(df, filename):
    if len(df) == 0:
        return
    filepath = os.path.join(OUTPUT_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"[OUTPUT] {filename}")


def plot_and_save(figure, filename):
    filepath = os.path.join(OUTPUT_CHART_DIR, filename)
    figure.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[CHART] {filename}")
    plt.close(figure)


def main():
    print("=" * 60)
    print("行业配置策略：资金流向视角 - 回测分析")
    print("=" * 60)

    print("\n[1/5] 加载数据...")
    benchmark = load_data('benchmark_returns.csv')
    margin_indicators = []
    for f in os.listdir('data'):
        if f.startswith('margin_indicator'):
            margin_indicators.append(load_data(f))

    north_holdings = load_data('northbound_holdings.csv')
    industry_list = load_data('industry_list_l1.csv')
    etf_list = load_data('etf_list.csv')

    print("\n[2/5] 数据预处理...")
    if len(benchmark) > 0:
        benchmark['trade_date'] = pd.to_datetime(benchmark['trade_date'])
        benchmark = benchmark.sort_values('trade_date')
        benchmark['return'] = benchmark['close'].pct_change()
        benchmark['period'] = benchmark['trade_date'].dt.to_period('W')
        save_output(benchmark, 'benchmark_processed.csv')

    print("\n[3/5] 两融资金指标分析...")
    if len(margin_indicators) > 0:
        margin_analysis = []
        for df in margin_indicators:
            if len(df) > 0 and 'indicator_name' in df.columns:
                margin_analysis.append(df)

        if len(margin_analysis) > 0:
            combined_margin = pd.concat(margin_analysis, ignore_index=True)
            save_output(combined_margin, 'margin_indicators_combined.csv')

            if 'buy_amount' in combined_margin.columns:
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))

                ax1 = axes[0]
                grouped = combined_margin.groupby('period')['buy_amount'].sum()
                ax1.plot(range(len(grouped)), grouped.values)
                ax1.set_title('融资买入额时间序列')
                ax1.set_xlabel('周期')
                ax1.set_ylabel('融资买入额')
                ax1.grid(True, alpha=0.3)

                ax2 = axes[1]
                if 'balance' in combined_margin.columns:
                    balance_grouped = combined_margin.groupby('period')['balance'].last()
                    ax2.plot(range(len(balance_grouped)), balance_grouped.values)
                    ax2.set_title('融资余额时间序列')
                    ax2.set_xlabel('周期')
                    ax2.set_ylabel('融资余额')
                    ax2.grid(True, alpha=0.3)

                plot_and_save(fig, 'margin_analysis.png')

    print("\n[4/5] 北向资金分析...")
    if len(north_holdings) > 0:
        north_holdings['trade_date'] = pd.to_datetime(north_holdings['trade_date'])
        north_holdings = north_holdings.sort_values('trade_date')
        save_output(north_holdings, 'north_holdings_processed.csv')

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax1 = axes[0]
        holdings_by_date = north_holdings.groupby('trade_date')['ratio'].mean()
        ax1.plot(holdings_by_date.index, holdings_by_date.values)
        ax1.set_title('北向资金持股占比时间序列')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('持股占比')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        holdings_count = north_holdings.groupby('trade_date').size()
        ax2.bar(holdings_count.index, holdings_count.values)
        ax2.set_title('北向资金持股数量时间序列')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('持股数量')
        ax2.grid(True, alpha=0.3)

        plot_and_save(fig, 'north_analysis.png')

    print("\n[5/5] 生成策略评估报告...")

    report = []
    report.append("=" * 60)
    report.append("行业配置策略：资金流向视角 - 回测报告")
    report.append("=" * 60)
    report.append("")
    report.append("数据概况:")
    report.append(f"  - 基准收益数据: {len(benchmark)} 条")
    report.append(f"  - 两融指标数据: {sum(len(df) for df in margin_indicators)} 条")
    report.append(f"  - 北向持股数据: {len(north_holdings)} 条")
    report.append(f"  - 行业数量: {len(industry_list)} 个")
    report.append(f"  - ETF数量: {len(etf_list)} 个")
    report.append("")
    report.append("核心发现:")
    report.append("  1. 北向资金: 持股明细可获取,需归因到中信行业")
    report.append("  2. 两融资金: 融资余额、融资买入额等指标可构建")
    report.append("  3. ETF资金: 全市场ETF列表可获取,资金流向需计算")
    report.append("  4. 产业资本: 定向增发、限售解禁等接口受限")
    report.append("")
    report.append("数据限制:")
    report.append("  - 产业资本详细事件数据(tushare接口限制)")
    report.append("  - 行业层面资金流向需进一步处理")
    report.append("  - 建议补充Wind/Choice数据源")
    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    report_path = os.path.join(OUTPUT_DATA_DIR, 'backtest_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[OUTPUT] backtest_report.txt")

    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)
    print(f"\n数据输出: {OUTPUT_DATA_DIR}/")
    print(f"图表输出: {OUTPUT_CHART_DIR}/")

    return {
        'benchmark': benchmark,
        'margin_indicators': margin_indicators,
        'north_holdings': north_holdings,
        'industry_list': industry_list
    }


if __name__ == "__main__":
    results = main()