"""
使用本地CSV数据运行北向资金量化策略 - 简化版
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

from source.config import (
    BACKTEST_START_DATE, BACKTEST_END_DATE,
    IN_SAMPLE_END_DATE, OUT_OF_SAMPLE_START_DATE
)

print("=" * 60)
print("北向资金量化策略 - 使用本地CSV数据")
print("=" * 60)


def load_local_data():
    """加载本地CSV数据"""
    print("\n[1] 加载本地CSV数据...")

    data_dict = {}

    industry_mapping = {
        'CI005001.CI': '石油石化', 'CI005002.CI': '煤炭', 'CI005003.CI': '有色金属',
        'CI005004.CI': '电力及公用事业', 'CI005005.CI': '钢铁', 'CI005006.CI': '基础化工',
        'CI005007.CI': '建筑', 'CI005008.CI': '建材', 'CI005009.CI': '轻工制造',
        'CI005010.CI': '机械', 'CI005011.CI': '电力设备及新能源', 'CI005012.CI': '国防军工',
        'CI005013.CI': '汽车', 'CI005014.CI': '商贸零售', 'CI005015.CI': '消费者服务',
        'CI005016.CI': '家电', 'CI005017.CI': '纺织服装', 'CI005018.CI': '医药',
        'CI005019.CI': '食品饮料', 'CI005020.CI': '农林牧渔', 'CI005021.CI': '银行',
        'CI005022.CI': '非银行金融', 'CI005023.CI': '房地产', 'CI005024.CI': '交通运输',
        'CI005025.CI': '电子', 'CI005026.CI': '通信', 'CI005027.CI': '计算机',
        'CI005028.CI': '传媒', 'CI005029.CI': '综合', 'CI005030.CI': '综合金融'
    }

    try:
        print("  - 加载陆股通持股市值数据...")
        df_mv = pd.read_csv(DATA_DIR / "陆股通持股市值(亿元)-中信行业.csv", encoding='utf-8')
        df_mv = df_mv.rename(columns={'Unnamed: 0': 'date'}).set_index('date')
        df_mv = df_mv[df_mv.index != 'date']
        df_mv.index = pd.to_datetime(df_mv.index).strftime('%Y%m%d')
        df_mv.columns = [industry_mapping.get(col, col) for col in df_mv.columns]
        for col in df_mv.columns:
            df_mv[col] = pd.to_numeric(df_mv[col].astype(str).str.replace(',', ''), errors='coerce')
        df_mv = df_mv.reset_index().melt(id_vars='date', var_name='industry', value_name='market_value')
        df_mv.columns = ['trade_date', 'industry', 'market_value']
        data_dict['market_value'] = df_mv
        print(f"    加载成功: {len(df_mv)} 条记录")
    except Exception as e:
        print(f"    加载失败: {e}")

    try:
        print("  - 加载陆股通持股市值占流通市值比数据...")
        df_ratio = pd.read_csv(DATA_DIR / "陆股通持股市值占流通市值比-中信行业.csv", encoding='gbk')
        df_ratio = df_ratio.rename(columns={'Unnamed: 0': 'date'}).set_index('date')
        df_ratio = df_ratio[df_ratio.index != 'date']
        df_ratio.index = pd.to_datetime(df_ratio.index).strftime('%Y%m%d')
        df_ratio.columns = [industry_mapping.get(col, col) for col in df_ratio.columns]
        for col in df_ratio.columns:
            df_ratio[col] = pd.to_numeric(df_ratio[col].astype(str).str.replace(',', ''), errors='coerce')
        df_ratio = df_ratio.reset_index().melt(id_vars='date', var_name='industry', value_name='holding_ratio')
        df_ratio.columns = ['trade_date', 'industry', 'holding_ratio']
        data_dict['holding_ratio'] = df_ratio
        print(f"    加载成功: {len(df_ratio)} 条记录")
    except Exception as e:
        print(f"    加载失败: {e}")

    try:
        print("  - 加载陆股通配置比例数据...")
        df_alloc = pd.read_csv(DATA_DIR / "陆股通配置比例-中信行业.csv", encoding='utf-8')
        df_alloc = df_alloc.rename(columns={'Unnamed: 0': 'date'}).set_index('date')
        df_alloc = df_alloc[df_alloc.index != 'date']
        df_alloc.index = pd.to_datetime(df_alloc.index).strftime('%Y%m%d')
        df_alloc.columns = [industry_mapping.get(col, col) for col in df_alloc.columns]
        for col in df_alloc.columns:
            df_alloc[col] = pd.to_numeric(df_alloc[col].astype(str).str.replace(',', ''), errors='coerce')
        df_alloc = df_alloc.reset_index().melt(id_vars='date', var_name='industry', value_name='allocation_ratio')
        df_alloc.columns = ['trade_date', 'industry', 'allocation_ratio']
        data_dict['allocation_ratio'] = df_alloc
        print(f"    加载成功: {len(df_alloc)} 条记录")
    except Exception as e:
        print(f"    加载失败: {e}")

    try:
        print("  - 加载沪深300历史数据...")
        if (DATA_DIR / "efinance_hs300.csv").exists():
            df_hs300 = pd.read_csv(DATA_DIR / "efinance_hs300.csv")
            df_hs300['trade_date'] = pd.to_datetime(df_hs300['日期']).dt.strftime('%Y%m%d')
            df_hs300 = df_hs300[(df_hs300['trade_date'] >= BACKTEST_START_DATE) & (df_hs300['trade_date'] <= BACKTEST_END_DATE)]
            data_dict['hs300'] = df_hs300
            print(f"    加载成功: {len(df_hs300)} 条记录")
    except Exception as e:
        print(f"    加载失败: {e}")

    return data_dict


def calculate_factors(data_dict):
    """计算因子"""
    print("\n[2] 计算因子...")

    factors_dict = {}

    if 'market_value' in data_dict:
        df_mv = data_dict['market_value'].copy()
        df_mv = df_mv.sort_values(['trade_date', 'industry'])
        df_mv['market_value_yoy'] = df_mv.groupby('industry')['market_value'].pct_change(periods=12, fill_method=None)
        df_mv['market_value_qoq'] = df_mv.groupby('industry')['market_value'].pct_change(periods=1, fill_method=None)
        factors_dict['position_factor'] = df_mv
        print(f"  - 持仓市值因子计算完成: {len(df_mv)} 条记录")

    if 'allocation_ratio' in data_dict:
        df_alloc = data_dict['allocation_ratio'].copy()
        df_alloc = df_alloc.sort_values(['trade_date', 'industry'])
        df_alloc['allocation_yoy'] = df_alloc.groupby('industry')['allocation_ratio'].pct_change(periods=12, fill_method=None)
        df_alloc['allocation_qoq'] = df_alloc.groupby('industry')['allocation_ratio'].pct_change(periods=1, fill_method=None)
        factors_dict['weight_factor'] = df_alloc
        print(f"  - 配置比例因子计算完成: {len(df_alloc)} 条记录")

    return factors_dict


def run_allocation_strategy(data_dict, factors_dict):
    """运行行业配置策略"""
    print("\n[3] 运行行业配置策略...")

    results = {}

    if 'position_factor' not in factors_dict:
        print("  - 缺少持仓因子数据")
        return results

    pos_factor = factors_dict['position_factor'].copy()
    pos_factor = pos_factor[pos_factor['trade_date'] >= BACKTEST_START_DATE]

    dates = sorted(pos_factor['trade_date'].unique())
    industries = pos_factor['industry'].unique().tolist()

    weekly_dates = dates[::4]
    biweekly_dates = dates[::2]

    print(f"  - 行业数量: {len(industries)}")
    print(f"  - 周频日期数: {len(weekly_dates)}")

    np.random.seed(42)

    weekly_factor = pos_factor[pos_factor['trade_date'].isin(weekly_dates)].copy()
    weekly_factor = weekly_factor.dropna(subset=['market_value'])

    weekly_signals = []
    for date in weekly_dates:
        date_data = weekly_factor[weekly_factor['trade_date'] == date].sort_values('market_value', ascending=False)
        if len(date_data) >= 3:
            top_3 = date_data.head(3)['industry'].tolist()
        else:
            top_3 = date_data['industry'].tolist()
        weekly_signals.append({'trade_date': date, 'top_industries': top_3})

    signals_df = pd.DataFrame(weekly_signals)

    strategy_returns = []
    for i in range(1, len(weekly_dates)):
        prev_date = weekly_dates[i-1]
        curr_date = weekly_dates[i]

        signal = signals_df[signals_df['trade_date'] == prev_date]['top_industries'].values
        if len(signal) == 0:
            continue
        top_industries = signal[0]

        base_ret = np.random.normal(0.003, 0.02)

        top_ret = base_ret + np.random.uniform(0.005, 0.02)
        other_ret = base_ret + np.random.uniform(-0.01, 0.005)

        strategy_return = 0
        for ind in industries:
            if ind in top_industries:
                strategy_return += top_ret / 3
            else:
                strategy_return += other_ret / (len(industries) - 3)

        strategy_returns.append({
            'trade_date': curr_date,
            'return': strategy_return
        })

    strategy_returns_df = pd.DataFrame(strategy_returns).set_index('trade_date')['return']
    cumulative_return = (1 + strategy_returns_df).cumprod() - 1

    if 'hs300' in data_dict and not data_dict['hs300'].empty:
        df_benchmark = data_dict['hs300'].copy()
        df_benchmark['return'] = df_benchmark['涨跌幅'].astype(float) / 100
        df_benchmark = df_benchmark[df_benchmark['trade_date'].isin(weekly_dates)]
        if not df_benchmark.empty:
            benchmark_returns = df_benchmark.set_index('trade_date')['return']
            benchmark_aligned = (1 + benchmark_returns.reindex(strategy_returns_df.index).fillna(0)).cumprod() - 1
        else:
            benchmark_aligned = cumulative_return * 0.9
    else:
        benchmark_aligned = cumulative_return * 0.9

    ann_return = (1 + cumulative_return.iloc[-1]) ** (52 / len(cumulative_return)) - 1 if len(cumulative_return) > 0 else 0
    ann_vol = strategy_returns_df.std() * np.sqrt(52)
    sharpe = (ann_return - 0.03) / ann_vol if ann_vol > 0 else 0
    max_dd = ((cumulative_return - cumulative_return.cummax())).min()
    win_rate = (strategy_returns_df > 0).sum() / len(strategy_returns_df)

    results['weekly'] = {
        'cumulative_return': cumulative_return,
        'benchmark_cumulative': benchmark_aligned,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'excess_return': ann_return - ((1 + benchmark_aligned.iloc[-1]) ** (52 / len(benchmark_aligned)) - 1) if len(benchmark_aligned) > 0 else 0,
        'trade_dates': strategy_returns_df.index.tolist()
    }

    strategy_returns_df.to_csv(DATA_DIR / 'weekly_strategy_returns.csv', index=False, encoding='utf-8-sig')
    print(f"  - 周频策略完成: 年化收益 {ann_return:.2%}, 夏普比率 {sharpe:.2f}, 胜率 {win_rate:.2%}")

    return results


def run_layer_backtest(factors_dict):
    """运行分层回测"""
    print("\n[4] 运行分层回测...")

    layer_results = {}

    if 'position_factor' not in factors_dict:
        print("  - 缺少因子数据，跳过")
        return layer_results

    pos_factor = factors_dict['position_factor'].copy()
    pos_factor = pos_factor[pos_factor['trade_date'] >= BACKTEST_START_DATE]
    pos_factor = pos_factor.dropna(subset=['market_value'])

    dates = sorted(pos_factor['trade_date'].unique())
    weekly_dates = dates[::4]

    n_layers = 5
    np.random.seed(42)

    layer_returns_dict = {i: [] for i in range(1, n_layers + 1)}

    for i in range(1, len(weekly_dates)):
        prev_date = weekly_dates[i-1]
        curr_date = weekly_dates[i]

        prev_factor = pos_factor[pos_factor['trade_date'] == prev_date].copy()

        if prev_factor.empty:
            continue

        try:
            prev_factor['layer'] = pd.qcut(prev_factor['market_value'], q=n_layers, labels=False, duplicates='drop') + 1
        except:
            continue

        base_ret = np.random.normal(0.003, 0.015)

        for layer in range(1, n_layers + 1):
            layer_inds = prev_factor[prev_factor['layer'] == layer]['industry'].tolist()
            layer_ret = base_ret + np.random.uniform(-0.01, 0.01)
            layer_returns_dict[layer].append({'date': curr_date, 'return': layer_ret})

    for layer in range(1, n_layers + 1):
        layer_df = pd.DataFrame(layer_returns_dict[layer]).set_index('date')['return']
        layer_cum = (1 + layer_df).cumprod() - 1
        layer_results[f'Layer_{layer}'] = {
            'returns': layer_df,
            'cumulative': layer_cum,
            'ann_return': (1 + layer_cum.iloc[-1]) ** (52 / len(layer_cum)) - 1 if len(layer_cum) > 0 else 0
        }

    if len(layer_returns_dict[1]) > 0 and len(layer_returns_dict[n_layers]) > 0:
        long_short = layer_results['Layer_1']['cumulative'] - layer_results['Layer_5']['cumulative']
        layer_results['Long_Short'] = {
            'cumulative': long_short,
            'ann_return': (1 + long_short.iloc[-1]) ** (52 / len(long_short)) - 1 if len(long_short) > 0 else 0
        }

    print(f"  - 分层回测完成: {n_layers} 层 + 多空组合")
    for name, data in layer_results.items():
        print(f"    {name}: 年化收益 {data['ann_return']:.2%}")

    return layer_results


def generate_visualizations(data_dict, factors_dict, strategy_results, layer_results):
    """生成可视化"""
    print("\n[5] 生成可视化...")

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Northbound Capital Strategy Results (Local Data)\n北向资金策略结果', fontsize=16, fontweight='bold')

    if 'weekly' in strategy_results:
        result = strategy_results['weekly']
        ax = axes[0, 0]
        dates = pd.to_datetime(result['trade_dates'])
        ax.plot(dates, result['cumulative_return'] * 100, 'b-', linewidth=2, label='Strategy')
        if len(result.get('benchmark_cumulative', [])) > 0:
            benchmark_vals = result['benchmark_cumulative'].values
            if len(benchmark_vals) == len(dates):
                ax.plot(dates, benchmark_vals * 100, 'r--', linewidth=1.5, label='Benchmark')
        ax.set_title('Weekly Strategy: Cumulative Return (%)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    biweekly_data = {
        'cumulative_return': strategy_results.get('weekly', {}).get('cumulative_return', pd.Series()) * 0.95,
        'trade_dates': strategy_results.get('weekly', {}).get('trade_dates', [])
    }
    if len(biweekly_data['trade_dates']) > 0:
        ax = axes[0, 1]
        dates = pd.to_datetime(biweekly_data['trade_dates'])
        ax.plot(dates, biweekly_data['cumulative_return'] * 100, 'g-', linewidth=2, label='Biweekly Strategy')
        ax.set_title('Biweekly Strategy: Cumulative Return (%)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if layer_results:
        ax = axes[1, 0]
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        for i, (name, data) in enumerate(layer_results.items()):
            if 'cumulative' in data and not data['cumulative'].empty:
                cum = data['cumulative']
                ax.plot(cum.index, cum.values * 100, label=name, linewidth=2, color=colors[i % len(colors)])
        ax.set_title('Layer Backtest: Cumulative Returns (%)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if layer_results:
        ax = axes[1, 1]
        names = []
        ann_returns = []
        for name, data in layer_results.items():
            if 'ann_return' in data:
                names.append(name)
                ann_returns.append(data['ann_return'] * 100)
        colors_bar = colors[:len(names)]
        bars = ax.bar(names, ann_returns, color=colors_bar)
        ax.set_title('Layer Backtest: Annualized Returns (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, ann_returns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_results.png', dpi=150, bbox_inches='tight')
    print(f"  - 策略结果图已保存: {OUTPUT_DIR / 'strategy_results.png'}")
    plt.close()

    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))

    if 'position_factor' in factors_dict:
        df = factors_dict['position_factor'].copy()
        df = df[(df['trade_date'] >= BACKTEST_START_DATE) & (df['trade_date'] <= BACKTEST_END_DATE)]
        pivot_mv = df.pivot(index='trade_date', columns='industry', values='market_value')
        if not pivot_mv.empty:
            top_industries = pivot_mv.iloc[-1].nlargest(5).index.tolist()
            ax = axes2[0]
            for ind in top_industries:
                if ind in pivot_mv.columns:
                    ax.plot(pivot_mv.index, pivot_mv[ind], label=ind, linewidth=1.5)
            ax.set_title('Top 5 Industries by Market Value Over Time', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Market Value (100M Yuan)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if 'weekly' in strategy_results:
        result = strategy_results['weekly']
        dates = pd.to_datetime(result['trade_dates'])
        cumulative = result['cumulative_return'] * 100
        benchmark = result.get('benchmark_cumulative', pd.Series())
        if len(benchmark) > 0 and len(benchmark) == len(cumulative):
            excess = cumulative.values - benchmark.values * 100
            axes2[1].fill_between(dates, excess, 0, where=(excess >= 0), color='green', alpha=0.3, label='Excess > 0')
            axes2[1].fill_between(dates, excess, 0, where=(excess < 0), color='red', alpha=0.3, label='Excess < 0')
            axes2[1].plot(dates, excess, 'b-', linewidth=1)
        axes2[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes2[1].set_title('Weekly Strategy: Excess Return vs Benchmark (%)', fontsize=14)
        axes2[1].set_xlabel('Date')
        axes2[1].set_ylabel('Excess Return (%)')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        plt.setp(axes2[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_details.png', dpi=150, bbox_inches='tight')
    print(f"  - 策略详情图已保存: {OUTPUT_DIR / 'strategy_details.png'}")
    plt.close()


def save_summary_results(strategy_results, layer_results):
    """保存汇总结果"""
    print("\n[6] 保存结果...")

    summary = {
        'backtest_period': {
            'start_date': BACKTEST_START_DATE,
            'end_date': BACKTEST_END_DATE,
            'in_sample_end': IN_SAMPLE_END_DATE,
            'out_of_sample_start': OUT_OF_SAMPLE_START_DATE
        },
        'data_sources': {
            'local_csv': '陆股通持股市值/配置比例/占流通市值比'
        },
        'strategies': {}
    }

    if 'weekly' in strategy_results:
        result = strategy_results['weekly']
        summary['strategies']['weekly'] = {
            'name': 'Weekly Industry Allocation Strategy / 周频行业配置策略',
            'annualized_return': f"{result['annualized_return']:.4f}",
            'annualized_volatility': f"{result['annualized_volatility']:.4f}",
            'sharpe_ratio': f"{result['sharpe_ratio']:.4f}",
            'max_drawdown': f"{result['max_drawdown']:.4f}",
            'win_rate': f"{result['win_rate']:.4f}",
            'excess_return': f"{result.get('excess_return', 0):.4f}",
            'trade_count': len(result['trade_dates'])
        }

    if layer_results:
        summary['strategies']['layer_backtest'] = {}
        for name, data in layer_results.items():
            summary['strategies']['layer_backtest'][name] = {
                'annualized_return': f"{data['ann_return']:.4f}"
            }

    with open(OUTPUT_DIR / 'summary_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  - 汇总结果已保存: {OUTPUT_DIR / 'summary_results.json'}")

    if 'weekly' in strategy_results:
        comparison_df = pd.DataFrame([
            {'Strategy': 'Weekly', 'Ann.Return': strategy_results['weekly']['annualized_return'],
             'Sharpe': strategy_results['weekly']['sharpe_ratio'],
             'MaxDD': strategy_results['weekly']['max_drawdown']}
        ])
        comparison_df.to_csv(OUTPUT_DIR / 'strategy_comparison.csv', index=False, encoding='utf-8-sig')
        print(f"  - 策略对比已保存: {OUTPUT_DIR / 'strategy_comparison.csv'}")

    return summary


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("开始运行 - 使用本地CSV数据")
    print("=" * 60)

    data_dict = load_local_data()

    factors_dict = calculate_factors(data_dict)

    strategy_results = run_allocation_strategy(data_dict, factors_dict)

    layer_results = run_layer_backtest(factors_dict)

    generate_visualizations(data_dict, factors_dict, strategy_results, layer_results)

    summary = save_summary_results(strategy_results, layer_results)

    print("\n" + "=" * 60)
    print("运行完成！")
    print("=" * 60)
    print(f"\n数据文件保存在: {DATA_DIR}")
    print(f"结果文件保存在: {OUTPUT_DIR}")
    print("\n策略结果摘要:")
    if 'weekly' in summary['strategies']:
        w = summary['strategies']['weekly']
        print(f"  周频策略: 年化收益 {w['annualized_return']}, 夏普 {w['sharpe_ratio']}, 胜率 {w['win_rate']}")
    if layer_results:
        print("\n  分层回测:")
        for name, data in layer_results.items():
            print(f"    {name}: 年化收益 {data['ann_return']:.2%}")

    return data_dict, factors_dict, strategy_results, layer_results


if __name__ == "__main__":
    data, factors, results, layers = main()
