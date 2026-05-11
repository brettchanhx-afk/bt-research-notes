import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
OUTPUT_DIR = "output"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data():
    print("=" * 70)
    print("加载数据...")
    print("=" * 70)

    prosperity_file = os.path.join(DATA_DIR, '行业中观景气度指数.csv')
    index_file = os.path.join(DATA_DIR, '中信一级行业指数及收盘价2010_2026.csv')

    prosperity_df = pd.read_excel(prosperity_file)
    prosperity_df = prosperity_df.rename(columns={'Unnamed: 0': 'date'})
    prosperity_df['date'] = pd.to_datetime(prosperity_df['date'])
    prosperity_df = prosperity_df.set_index('date').sort_index()

    raw_df = pd.read_csv(index_file, encoding='utf-8-sig', header=None)

    codes = raw_df.iloc[0].tolist()
    names = raw_df.iloc[1].tolist()

    code_to_name = {}
    for i, (code, name) in enumerate(zip(codes, names)):
        if i > 0 and pd.notna(name):
            code_to_name[code] = name

    index_df = raw_df.iloc[2:].copy()
    index_df.columns = names
    index_df = index_df.rename(columns={index_df.columns[0]: 'date'})
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.set_index('date').sort_index()

    for col in index_df.columns:
        index_df[col] = pd.to_numeric(index_df[col], errors='coerce')

    common_industries = [col for col in prosperity_df.columns if col in index_df.columns]

    prosperity_df = prosperity_df[common_industries]
    index_df = index_df[common_industries]

    print(f"景气指数数据: {prosperity_df.shape}, 时间区间: {prosperity_df.index[0]} ~ {prosperity_df.index[-1]}")
    print(f"行业指数数据: {index_df.shape}, 时间区间: {index_df.index[0]} ~ {index_df.index[-1]}")
    print(f"匹配行业数量: {len(common_industries)}")
    print(f"行业列表: {common_industries}")

    return prosperity_df, index_df


def calculate_industry_returns(index_df):
    print("\n计算行业月度收益率...")
    monthly_prices = index_df.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    monthly_returns = monthly_returns.iloc[1:]
    return monthly_returns


def prosperity_state_analysis(prosperity_df, backtest_start='2016-04', backtest_end='2022-06'):
    print("\n" + "=" * 70)
    print("应用一：景气状态判断")
    print("=" * 70)

    prosper_df = prosperity_df.loc[backtest_start:backtest_end].copy()

    results = {}
    for industry in prosper_df.columns:
        series = prosper_df[industry].dropna()
        if len(series) < 12:
            continue

        median = series.median()
        std = series.std()

        current = series.iloc[-1]
        prev_3m = series.iloc[-4:-1].mean() if len(series) >= 4 else series.iloc[-1]

        if current > median:
            state = "高景气"
        elif current < median - std:
            state = "低景气"
        else:
            state = "中等景气"

        trend = "改善" if current > prev_3m else "恶化"

        results[industry] = {
            '当前值': round(current, 4),
            '中位数': round(median, 4),
            '标准差': round(std, 4),
            '状态': state,
            '趋势': trend,
            '边际变化': round(current - prev_3m, 4)
        }

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('当前值', ascending=False)

    print("\n各行业景气状态（最新一期）:")
    print(results_df.head(15).to_string())

    ensure_dir(OUTPUT_DIR)
    output_df = results_df.copy()
    output_df.to_csv(os.path.join(OUTPUT_DIR, 'prosperity_state_analysis.csv'))
    print(f"\n结果已保存至: {os.path.join(OUTPUT_DIR, 'prosperity_state_analysis.csv')}")

    return results_df


def single_industry_timing(prosperity_df, monthly_returns, backtest_start='2016-04', backtest_end='2022-06'):
    print("\n" + "=" * 70)
    print("应用二：单行业择时")
    print("=" * 70)

    prosper_df = prosperity_df.loc[backtest_start:backtest_end].copy()

    results = {}
    for industry in prosper_df.columns:
        if industry not in monthly_returns.columns:
            continue

        pros_series = prosper_df[industry].dropna()
        ret_series = monthly_returns[industry].dropna()

        if len(pros_series) < 12 or len(ret_series) < 12:
            continue

        median = pros_series.median()
        std = pros_series.std()

        timing_signals = []
        position = 0

        for i in range(len(pros_series)):
            current_date = pros_series.index[i]
            current_pros = pros_series.iloc[i]

            if i >= 4:
                prev_4m = pros_series.iloc[i-4:i].mean()
            else:
                prev_4m = pros_series.iloc[:i].mean() if i > 0 else current_pros

            if current_pros > median + 0.5 * std:
                signal = 1
                position = 1
            elif current_pros < median - 0.5 * std:
                signal = -1
                position = -1
            else:
                signal = 0
                position = 0

            if current_date in ret_series.index:
                actual_ret = ret_series.loc[current_date]
                timing_signals.append({
                    'date': current_date,
                    'signal': signal,
                    'position': position,
                    'return': actual_ret,
                    'prosperity': current_pros
                })

        if timing_signals:
            signal_df = pd.DataFrame(timing_signals)
            long_only = signal_df[signal_df['signal'] == 1]['return'].mean() if len(signal_df[signal_df['signal'] == 1]) > 0 else 0
            short_only = signal_df[signal_df['signal'] == -1]['return'].mean() if len(signal_df[signal_df['signal'] == -1]) > 0 else 0
            neutral = signal_df['return'].mean()

            results[industry] = {
                '高景气信号收益': round(long_only, 6) if long_only else 0,
                '低景气信号收益': round(short_only, 6) if short_only else 0,
                '中性收益': round(neutral, 6),
                '标准差': round(std, 4),
                '中位数': round(median, 4)
            }

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('高景气信号收益', ascending=False)

    print("\n单行业择时效果（各行业高景气信号收益）:")
    print(results_df.head(15).to_string())

    ensure_dir(OUTPUT_DIR)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'single_industry_timing_results.csv'))
    print(f"\n结果已保存至: {os.path.join(OUTPUT_DIR, 'single_industry_timing_results.csv')}")

    return results_df


def run_rotation_backtest(prosperity_df, monthly_returns, backtest_start='2016-04', backtest_end='2022-06', top_n=4):
    print("\n" + "=" * 70)
    print("应用三：行业间轮动策略")
    print("=" * 70)

    prosper_df = prosperity_df.loc[backtest_start:backtest_end].copy()
    industries = [col for col in prosper_df.columns if col in monthly_returns.columns]

    print(f"有效行业数量: {len(industries)}")

    dates = prosper_df.index.sort_values()
    window_size = 12

    rotation_returns = []
    benchmark_returns = []
    selected_history = []

    print("\n开始回测...")
    for i in range(window_size, len(dates) - 1):
        current_date = dates[i]

        window_pros = prosper_df.loc[:current_date].iloc[-window_size:]

        all_scores = {}
        for industry in industries:
            pros = window_pros[industry].dropna()
            if len(pros) < 4:
                continue

            current_vals = []
            for ind in industries:
                ind_pros = prosper_df.loc[:current_date, ind].dropna()
                if len(ind_pros) >= 1:
                    current_vals.append((ind, ind_pros.iloc[-1]))

            current_vals_sorted = sorted(current_vals, key=lambda x: x[1], reverse=True)
            ranks_first = {x[0]: len(current_vals_sorted) - idx for idx, x in enumerate(current_vals_sorted)}

            mom_vals = []
            for ind in industries:
                ind_pros = prosper_df.loc[:current_date, ind].dropna()
                if len(ind_pros) >= 2:
                    mom_vals.append((ind, ind_pros.diff(1).iloc[-1]))

            mom_vals_sorted = sorted(mom_vals, key=lambda x: x[1], reverse=True)
            ranks_second = {x[0]: len(mom_vals_sorted) - idx for idx, x in enumerate(mom_vals_sorted)}

            score = sum([ranks_first.get(industry, 0) for _ in range(3)]) / 3 + sum([ranks_second.get(industry, 0) for _ in range(2)]) / 2
            all_scores[industry] = score

        all_scores_sorted = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        selected_industries = [x[0] for x in all_scores_sorted[:top_n]]

        next_date = dates[i + 1]

        period_mask = (monthly_returns.index > current_date) & (monthly_returns.index <= next_date)

        if period_mask.sum() == 0:
            continue

        strategy_ret = 0
        for ind in selected_industries:
            if ind in monthly_returns.columns:
                ind_ret = monthly_returns.loc[period_mask, ind].mean()
                strategy_ret += ind_ret / top_n

        benchmark_ret = monthly_returns.loc[period_mask].mean(axis=1).mean()

        rotation_returns.append({'date': next_date, 'return': strategy_ret})
        benchmark_returns.append({'date': next_date, 'return': benchmark_ret})
        selected_history.append({'date': next_date, 'selected': ','.join(selected_industries)})

        if i % 10 == 0:
            print(f"  进度: {i}/{len(dates) - window_size - 1}")

    print(f"  完成: {len(rotation_returns)} 次调仓")

    rotation_df = pd.DataFrame(rotation_returns).set_index('date')
    benchmark_df = pd.DataFrame(benchmark_returns).set_index('date')

    rotation_series = rotation_df['return'].sort_index()
    benchmark_series = benchmark_df['return'].sort_index()

    common_idx = rotation_series.index.intersection(benchmark_series.index)
    rotation_series = rotation_series.loc[common_idx]
    benchmark_series = benchmark_series.loc[common_idx]

    n_periods = len(rotation_series)
    annual_factor = 12

    rotation_cum = (1 + rotation_series).prod()
    benchmark_cum = (1 + benchmark_series).prod()

    rotation_annual = rotation_cum ** (annual_factor / n_periods) - 1
    benchmark_annual = benchmark_cum ** (annual_factor / n_periods) - 1

    rotation_vol = rotation_series.std() * np.sqrt(12)
    benchmark_vol = benchmark_series.std() * np.sqrt(12)

    rotation_sharpe = (rotation_annual - 0.03) / rotation_vol if rotation_vol > 0 else 0
    benchmark_sharpe = (benchmark_annual - 0.03) / benchmark_vol if benchmark_vol > 0 else 0

    rotation_max_dd = (rotation_series.cumsum() - rotation_series.cumsum().cummax()).min()
    benchmark_max_dd = (benchmark_series.cumsum() - benchmark_series.cumsum().cummax()).min()

    excess_return = rotation_annual - benchmark_annual

    print("\n" + "=" * 70)
    print("行业轮动策略回测结果")
    print("=" * 70)
    print(f"\n{'指标':<20} {'轮动策略':>15} {'等权基准':>15}")
    print("-" * 55)
    print(f"{'年化收益率':<20} {rotation_annual:>14.2%} {benchmark_annual:>14.2%}")
    print(f"{'年化波动率':<20} {rotation_vol:>14.2%} {benchmark_vol:>14.2%}")
    print(f"{'夏普比率':<20} {rotation_sharpe:>15.2f} {benchmark_sharpe:>15.2f}")
    print(f"{'最大回撤':<20} {rotation_max_dd:>14.2%} {benchmark_max_dd:>14.2%}")
    print(f"\n超额年化收益: {excess_return:.2%}")

    results_summary = pd.DataFrame({
        '指标': ['年化收益率', '年化波动率', '夏普比率', '最大回撤'],
        '轮动策略': [f"{rotation_annual:.2%}", f"{rotation_vol:.2%}", f"{rotation_sharpe:.2f}", f"{rotation_max_dd:.2%}"],
        '等权基准': [f"{benchmark_annual:.2%}", f"{benchmark_vol:.2%}", f"{benchmark_sharpe:.2f}", f"{benchmark_max_dd:.2%}"]
    })
    results_summary.to_csv(os.path.join(OUTPUT_DIR, 'rotation_strategy_results.csv'), index=False)
    rotation_series.to_csv(os.path.join(OUTPUT_DIR, 'rotation_strategy_returns.csv'))
    benchmark_series.to_csv(os.path.join(OUTPUT_DIR, 'rotation_benchmark_returns.csv'))

    selected_df = pd.DataFrame(selected_history)
    selected_df.to_csv(os.path.join(OUTPUT_DIR, 'rotation_selected_industries.csv'), index=False)

    print(f"\n结果已保存至: {OUTPUT_DIR}/")

    return rotation_series, benchmark_series, selected_history


def plot_results(rotation_series, benchmark_series):
    print("\n生成图表...")

    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        rotation_cum = (1 + rotation_series).cumprod()
        benchmark_cum = (1 + benchmark_series).cumprod()

        axes[0].plot(rotation_cum.index, rotation_cum.values,
                    label='Rotation Strategy', color='blue', linewidth=2)
        axes[0].plot(benchmark_cum.index, benchmark_cum.values,
                    label='Equal Weight Benchmark', color='gray', linewidth=2, alpha=0.7)
        axes[0].set_title('Cumulative Returns: Rotation Strategy vs Benchmark', fontsize=14)
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(True)

        excess = rotation_series - benchmark_series
        axes[1].bar(excess.index, excess.values * 100, alpha=0.7, color='green')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('Monthly Excess Returns', fontsize=14)
        axes[1].set_ylabel('Excess Return (%)')
        axes[1].grid(True)

        drawdown = rotation_cum / rotation_cum.cummax() - 1
        axes[2].fill_between(drawdown.index, drawdown.values * 100, 0,
                            alpha=0.3, color='red')
        axes[2].set_title('Strategy Drawdown', fontsize=14)
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'rotation_strategy_performance.png'), dpi=150)
        print(f"图表已保存至: {os.path.join(OUTPUT_DIR, 'rotation_strategy_performance.png')}")
        plt.close()

    except Exception as e:
        print(f"图表生成失败: {str(e)}")


def main():
    print("=" * 70)
    print("中观景气视角行业轮动策略 - 完整复现")
    print("基于研报：行业配置策略-中观景气视角(2)")
    print("=" * 70)

    ensure_dir(OUTPUT_DIR)

    prosperity_df, index_df = load_data()

    monthly_returns = calculate_industry_returns(index_df)

    state_results = prosperity_state_analysis(prosperity_df)

    timing_results = single_industry_timing(prosperity_df, monthly_returns)

    rotation_returns, benchmark_returns, selected_history = run_rotation_backtest(
        prosperity_df, monthly_returns,
        backtest_start='2016-04', backtest_end='2022-06', top_n=4
    )

    plot_results(rotation_returns, benchmark_returns)

    print("\n" + "=" * 70)
    print("运行完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()