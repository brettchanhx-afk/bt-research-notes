import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'data'
OUTPUT_DATA_DIR = 'output/data'
OUTPUT_CHART_DIR = 'output/charts'

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_CHART_DIR, exist_ok=True)


def save_output(df, filename):
    if df is None or (isinstance(df, list) and len(df) == 0):
        return
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)
    if len(df) == 0:
        return
    filepath = os.path.join(OUTPUT_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"[OUTPUT] {filename} - {len(df)} rows")


def plot_and_save(figure, filename):
    filepath = os.path.join(OUTPUT_CHART_DIR, filename)
    figure.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[CHART] {filename}")
    plt.close(figure)


def load_csv_safe(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8', header=1)
    except:
        try:
            df = pd.read_csv(filepath, encoding='gbk', header=1)
        except:
            df = pd.read_csv(filepath, encoding='gb2312', header=1)
    df.columns = ['date'] + list(df.columns[1:])
    return df


def process_north_flow_data():
    print("\n[1] 处理北向资金数据...")

    df_value = load_csv_safe(os.path.join(DATA_DIR, '陆股通持股市值(亿元)-中信行业.csv'))
    df_ratio = load_csv_safe(os.path.join(DATA_DIR, '陆股通配置比例-中信行业.csv'))
    df_float = load_csv_safe(os.path.join(DATA_DIR, '陆股通持股市值占流通市值比-中信行业.csv'))

    print(f"  - 持股市值数据: {df_value.shape}")
    print(f"  - 配置比例数据: {df_ratio.shape}")
    print(f"  - 流通市值比数据: {df_float.shape}")

    df_value = df_value.rename(columns={df_value.columns[0]: 'date'})
    df_ratio = df_ratio.rename(columns={df_ratio.columns[0]: 'date'})
    df_float = df_float.rename(columns={df_float.columns[0]: 'date'})

    df_value['date'] = pd.to_datetime(df_value['date'], errors='coerce')
    df_ratio['date'] = pd.to_datetime(df_ratio['date'], errors='coerce')
    df_float['date'] = pd.to_datetime(df_float['date'], errors='coerce')

    df_value = df_value.dropna(subset=['date'])
    df_ratio = df_ratio.dropna(subset=['date'])
    df_float = df_float.dropna(subset=['date'])

    industry_cols = [c for c in df_value.columns if c != 'date']
    print(f"  - 行业数量: {len(industry_cols)}")

    df_value_long = df_value.melt(id_vars=['date'], var_name='industry_code', value_name='hold_mv')
    print("  - melt value done", flush=True)
    df_ratio_long = df_ratio.melt(id_vars=['date'], var_name='industry_code', value_name='config_ratio')
    print("  - melt ratio done", flush=True)
    df_float_long = df_float.melt(id_vars=['date'], var_name='industry_code', value_name='float_ratio')
    print("  - melt float done", flush=True)

    df_north = df_value_long.merge(df_ratio_long, on=['date', 'industry_code'], how='left')
    print("  - merge 1 done", flush=True)
    df_north = df_north.merge(df_float_long, on=['date', 'industry_code'], how='left')
    print("  - merge 2 done", flush=True)

    for col in ['hold_mv', 'config_ratio', 'float_ratio']:
        df_north[col] = pd.to_numeric(df_north[col], errors='coerce')

    df_north = df_north.sort_values(['industry_code', 'date'])
    print("  - sort done", flush=True)
    df_north['hold_mv_change'] = df_north.groupby('industry_code')['hold_mv'].diff()
    print("  - hold_mv_change done", flush=True)
    df_north['hold_mv_yoy'] = df_north.groupby('industry_code')['hold_mv'].pct_change(periods=12)
    print("  - hold_mv_yoy done", flush=True)
    df_north['config_ratio_yoy'] = df_north.groupby('industry_code')['config_ratio'].pct_change(periods=12)
    print("  - config_ratio_yoy done", flush=True)
    df_north['float_ratio_yoy'] = df_north.groupby('industry_code')['float_ratio'].pct_change(periods=12)
    print("  - float_ratio_yoy done", flush=True)

    df_north['period'] = df_north['date'].dt.to_period('M')
    df_north_monthly = df_north.copy()

    save_output(df_north, 'north_flow_processed.csv')
    save_output(df_north_monthly, 'north_flow_monthly.csv')

    print(f"  - 北向资金处理完成: {len(df_north)} rows")

    return df_north, df_north_monthly, industry_cols


def process_margin_data():
    print("\n[2] 处理两融资金数据...")

    margin_files = [f for f in os.listdir(DATA_DIR) if f.startswith('margin_indicator')]
    margin_dfs = []
    for f in margin_files:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        margin_dfs.append(df)

    if len(margin_dfs) > 0:
        df_margin = pd.concat(margin_dfs, ignore_index=True)
        save_output(df_margin, 'margin_all_indicators.csv')
        print(f"  - 两融指标处理完成: {len(df_margin)} rows")
        return df_margin
    return pd.DataFrame()


def process_benchmark():
    print("\n[3] 处理基准收益数据...")

    df_bench = pd.read_csv(os.path.join(DATA_DIR, 'benchmark_returns.csv'))
    df_bench['trade_date'] = pd.to_datetime(df_bench['trade_date'])
    df_bench = df_bench.sort_values('trade_date')
    df_bench['return'] = df_bench['close'].pct_change()
    df_bench['period'] = df_bench['trade_date'].dt.to_period('M')

    save_output(df_bench, 'benchmark_processed.csv')
    print(f"  - 基准收益处理完成: {len(df_bench)} rows")

    return df_bench


def calculate_north_indicators(df_north):
    print("\n[4] 计算北向资金指标...")

    df_north = df_north.copy()

    indicator_list = []

    df_orig = df_north[['date', 'industry_code', 'hold_mv', 'config_ratio', 'float_ratio']].copy()
    df_orig['indicator_name'] = 'north_hold_mv_M_orig'
    indicator_list.append(df_orig)

    df_yoy = df_north[['date', 'industry_code', 'hold_mv_yoy', 'config_ratio_yoy', 'float_ratio_yoy']].copy()
    df_yoy = df_yoy.rename(columns={
        'hold_mv_yoy': 'hold_mv_value',
        'config_ratio_yoy': 'config_ratio_value',
        'float_ratio_yoy': 'float_ratio_value'
    })
    df_yoy['indicator_name'] = 'north_hold_mv_M_yoy'
    df_yoy = df_yoy.rename(columns={'hold_mv_value': 'hold_mv', 'config_ratio_value': 'config_ratio', 'float_ratio_value': 'float_ratio'})
    indicator_list.append(df_yoy)

    for ind in indicator_list:
        save_output(ind, f"indicator_{ind['indicator_name'].iloc[0]}.csv")

    print(f"  - 北向资金指标计算完成: {len(indicator_list)} 个指标")

    return indicator_list


def run_stratification_test(df_north, df_benchmark, n_groups=5):
    print("\n[5] 运行分层回测...")

    df_merged = df_north.merge(
        df_benchmark[['trade_date', 'return']].rename(columns={'trade_date': 'date'}),
        on='date',
        how='inner'
    )

    if len(df_merged) == 0:
        print("  [WARN] 合并后无数据")
        return pd.DataFrame()

    df_merged = df_merged.dropna(subset=['hold_mv'])

    df_merged['group'] = df_merged.groupby('date')['hold_mv'].transform(
        lambda x: pd.qcut(x, q=n_groups, labels=range(1, n_groups+1), duplicates='drop')
    )

    group_stats = []
    for g in range(1, n_groups + 1):
        g_df = df_merged[df_merged['group'] == g]
        if len(g_df) > 0:
            stats = {
                'group': g,
                'mean_return': g_df['return'].mean(),
                'std_return': g_df['return'].std(),
                'count': len(g_df),
                'avg_hold_mv': g_df['hold_mv'].mean()
            }
            group_stats.append(stats)

    df_stats = pd.DataFrame(group_stats)
    save_output(df_stats, 'stratification_results.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.bar(df_stats['group'], df_stats['mean_return'] * 100)
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Mean Return (%)')
    ax1.set_title('North Flow Strategy - Stratification Test')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(df_stats['group'], df_stats['avg_hold_mv'])
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Avg Hold MV')
    ax2.set_title('North Flow Strategy - Avg Holdings by Group')
    ax2.grid(True, alpha=0.3)

    plot_and_save(fig, 'stratification_test.png')

    print(f"  - 分层回测完成")

    return df_stats


def run_long_short_test(df_north, df_benchmark, n_groups=5):
    print("\n[6] 运行多空回测...")

    df_merged = df_north.merge(
        df_benchmark[['trade_date', 'return']].rename(columns={'trade_date': 'date'}),
        on='date',
        how='inner'
    )

    if len(df_merged) == 0:
        print("  [WARN] 合并后无数据")
        return {}

    df_merged = df_merged.dropna(subset=['hold_mv'])

    df_merged['group'] = df_merged.groupby('date')['hold_mv'].transform(
        lambda x: pd.qcut(x, q=n_groups, labels=range(1, n_groups+1), duplicates='drop')
    )

    long_df = df_merged[df_merged['group'] == n_groups]
    short_df = df_merged[df_merged['group'] == 1]

    long_return = long_df['return'].mean() if len(long_df) > 0 else 0
    short_return = short_df['return'].mean() if len(short_df) > 0 else 0
    long_short_return = long_return - short_return

    results = {
        'long_return': long_return,
        'short_return': short_return,
        'long_short_return': long_short_return,
        'long_count': len(long_df),
        'short_count': len(short_df)
    }

    long_cum = (1 + long_df['return']).cumprod() if len(long_df) > 0 else pd.Series([1])
    short_cum = (1 + short_df['return']).cumprod() if len(short_df) > 0 else pd.Series([1])
    benchmark_cum = (1 + df_merged['return']).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(long_cum.values, label=f'Long (Group {n_groups})', linewidth=2)
    ax.plot(short_cum.values, label=f'Short (Group 1)', linewidth=2)
    ax.plot(benchmark_cum.values, label='Benchmark', linewidth=2, alpha=0.5)
    ax.set_xlabel('Period')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('North Flow Strategy - Long Short Portfolio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_and_save(fig, 'long_short_portfolio.png')

    print(f"  - 多空回测完成: Long={long_return:.4f}, Short={short_return:.4f}, L-S={long_short_return:.4f}")

    return results


def run_threshold_test(df_north, df_benchmark):
    print("\n[7] 运行阈值回测...")

    df_merged = df_north.merge(
        df_benchmark[['trade_date', 'return']].rename(columns={'trade_date': 'date'}),
        on='date',
        how='inner'
    )

    if len(df_merged) == 0:
        return {}

    df_merged = df_merged.dropna(subset=['hold_mv'])

    thresholds = {
        'long_90': df_merged['hold_mv'].quantile(0.9),
        'long_70': df_merged['hold_mv'].quantile(0.7),
        'long_50': df_merged['hold_mv'].quantile(0.5),
        'short_10': df_merged['hold_mv'].quantile(0.1),
        'short_30': df_merged['hold_mv'].quantile(0.3),
    }

    results = {}
    for name, thresh in thresholds.items():
        if 'long' in name:
            mask = df_merged['hold_mv'] >= thresh
        else:
            mask = df_merged['hold_mv'] <= thresh

        subset = df_merged[mask]
        if len(subset) > 0:
            results[name] = {
                'threshold': thresh,
                'mean_return': subset['return'].mean(),
                'count': len(subset)
            }

    df_threshold = pd.DataFrame(results).T
    save_output(df_threshold, 'threshold_test_results.csv')

    print(f"  - 阈值回测完成")

    return results


def calculate_annual_metrics(df_north, df_benchmark):
    print("\n[8] 计算年度指标...")

    df_merged = df_north.merge(
        df_benchmark[['trade_date', 'return']].rename(columns={'trade_date': 'date'}),
        on='date',
        how='inner'
    )

    if len(df_merged) == 0:
        return {}

    df_merged['year'] = df_merged['date'].dt.year

    df_merged['group'] = df_merged.groupby('date')['hold_mv'].transform(
        lambda x: pd.qcut(x, q=5, labels=range(1, 6), duplicates='drop')
    )

    long_df = df_merged[df_merged['group'] == 5]

    annual_returns = df_merged.groupby('year')['return'].agg(['mean', 'std'])
    annual_returns.columns = ['benchmark_return', 'benchmark_std']

    if len(long_df) > 0:
        long_annual = long_df.groupby('year')['return'].mean()
        annual_returns['long_return'] = long_annual

    annual_returns['excess_return'] = annual_returns['long_return'] - annual_returns['benchmark_return']

    save_output(annual_returns, 'annual_metrics.csv')

    print(f"  - 年度指标计算完成")

    return annual_returns


def plot_north_flow_time_series(df_north):
    print("\n[9] 绘制北向资金时间序列...")

    industry_agg = df_north.groupby('date').agg({
        'hold_mv': 'sum',
        'config_ratio': 'mean',
        'float_ratio': 'mean'
    }).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax1 = axes[0]
    ax1.plot(industry_agg['date'], industry_agg['hold_mv'])
    ax1.set_title('North Flow - Total Hold Market Value', fontsize=12)
    ax1.set_ylabel('Hold MV (100M)')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(industry_agg['date'], industry_agg['config_ratio'])
    ax2.set_title('North Flow - Average Config Ratio', fontsize=12)
    ax2.set_ylabel('Config Ratio (%)')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(industry_agg['date'], industry_agg['float_ratio'])
    ax3.set_title('North Flow - Average Float Ratio', fontsize=12)
    ax3.set_ylabel('Float Ratio (%)')
    ax3.grid(True, alpha=0.3)

    plot_and_save(fig, 'north_flow_time_series.png')

    print(f"  - 时间序列图完成")


def generate_backtest_report(df_stats, long_short_results, threshold_results, annual_metrics):
    print("\n[10] 生成回测报告...")

    report = []
    report.append("=" * 70)
    report.append("行业配置策略：资金流向视角 - 回测报告")
    report.append("=" * 70)
    report.append("")

    report.append("一、数据概况")
    report.append("-" * 50)
    report.append(f"  北向资金数据: 2017-01 至 2021-11")
    report.append(f"  中信行业数量: 30 个一级行业")
    report.append(f"  基准指数: 中证500 (000905.SH)")
    report.append("")

    report.append("二、分层回测结果")
    report.append("-" * 50)
    if len(df_stats) > 0:
        for _, row in df_stats.iterrows():
            report.append(f"  Group {int(row['group'])}: Return={row['mean_return']*100:.2f}%, AvgMV={row['avg_hold_mv']:.2f}")
    report.append("")

    report.append("三、多空回测结果")
    report.append("-" * 50)
    if long_short_results:
        report.append(f"  多头组合收益: {long_short_results.get('long_return', 0)*100:.2f}%")
        report.append(f"  空头组合收益: {long_short_results.get('short_return', 0)*100:.2f}%")
        report.append(f"  多空组合收益: {long_short_results.get('long_short_return', 0)*100:.2f}%")
    report.append("")

    report.append("四、阈值回测结果")
    report.append("-" * 50)
    if threshold_results:
        for name, res in threshold_results.items():
            report.append(f"  {name}: Return={res['mean_return']*100:.2f}%, Count={res['count']}")
    report.append("")

    report.append("五、年度超额收益")
    report.append("-" * 50)
    if len(annual_metrics) > 0:
        for year, row in annual_metrics.iterrows():
            if 'excess_return' in row and not pd.isna(row['excess_return']):
                report.append(f"  {year}: Excess Return={row['excess_return']*100:.2f}%")
    report.append("")

    report.append("=" * 70)
    report.append("报告生成完毕")
    report.append("=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    report_path = os.path.join(OUTPUT_DATA_DIR, 'backtest_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[OUTPUT] backtest_report.txt")

    return report_text


def main():
    print("=" * 70)
    print("行业配置策略：资金流向视角 - 完整回测")
    print("=" * 70)

    df_north, df_north_monthly, industry_cols = process_north_flow_data()

    df_margin = process_margin_data()

    df_benchmark = process_benchmark()

    indicator_list = calculate_north_indicators(df_north)

    df_stats = run_stratification_test(df_north, df_benchmark)

    long_short_results = run_long_short_test(df_north, df_benchmark)

    threshold_results = run_threshold_test(df_north, df_benchmark)

    annual_metrics = calculate_annual_metrics(df_north, df_benchmark)

    plot_north_flow_time_series(df_north)

    generate_backtest_report(df_stats, long_short_results, threshold_results, annual_metrics)

    print("\n" + "=" * 70)
    print("回测完成!")
    print("=" * 70)
    print(f"\n数据输出目录: {OUTPUT_DATA_DIR}/")
    print(f"图表输出目录: {OUTPUT_CHART_DIR}/")

    return {
        'df_north': df_north,
        'df_benchmark': df_benchmark,
        'stratification_results': df_stats,
        'long_short_results': long_short_results,
        'threshold_results': threshold_results,
        'annual_metrics': annual_metrics
    }


if __name__ == "__main__":
    results = main()