import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts
from config.settings import START_DATE, END_DATE, OUTPUT_DIR, FACTOR_CONFIG, SW_INDUSTRY_CODES, TOKEN, TUSHARE_API_URL
from source.factors import DATA_GAPS

TOKEN = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
TUSHARE_API_URL = "http://jiaoch.site"

def init_tushare():
    pro = ts.pro_api(TOKEN)
    pro._DataApi__token = TOKEN
    pro._DataApi__http_url = TUSHARE_API_URL
    return pro

PRO = init_tushare()

DATA_DIR = "data"
OUTPUT_DIR = "output"

def initialize_directories():
    dirs = [DATA_DIR, OUTPUT_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"目录初始化完成: {DATA_DIR}, {OUTPUT_DIR}")

def load_citic_industry_data():
    print("\n" + "="*60)
    print("加载中信一级行业指数数据")
    print("="*60)

    file_path = os.path.join(DATA_DIR, "中信一级行业指数及收盘价2010_2026.csv")
    try:
        df = pd.read_csv(file_path)
        industry_names = df.iloc[0, 1:].tolist()
        df = df.iloc[1:]
        df.columns = ['trade_date'] + industry_names
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        df = df.apply(pd.to_numeric, errors='coerce')

        industry_returns = df.pct_change().dropna()
        industry_pivot = df

        print(f"  ✓ 成功加载中信一级行业数据")
        print(f"  行业数量: {len(industry_names)} 个")
        print(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
        print(f"  数据维度: {df.shape}")

        return df, industry_pivot, industry_returns
    except Exception as e:
        print(f"  ✗ 加载失败: {str(e)[:100]}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def load_macro_factor_data():
    print("\n" + "="*60)
    print("加载宏观因子指数数据")
    print("="*60)

    file_path = os.path.join(DATA_DIR, "宏观因子指数.csv")
    try:
        df = pd.read_excel(file_path)
        df = df.rename(columns={'Unnamed: 0': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.apply(pd.to_numeric, errors='coerce')

        print(f"  ✓ 成功加载宏观因子数据")
        print(f"  因子数量: {len(df.columns)} 个 ({', '.join(df.columns.tolist())})")
        print(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
        print(f"  数据维度: {df.shape}")

        return df
    except Exception as e:
        print(f"  ✗ 加载失败: {str(e)[:100]}")
        return pd.DataFrame()

def fetch_market_index_data(pro):
    print("\n" + "="*60)
    print("获取市场基准指数数据")
    print("="*60)

    index_codes = {
        '000300.SH': '沪深300',
        '000905.SH': '中证500',
        '000852.SH': '中证1000',
        '000001.SH': '上证指数'
    }

    index_data = {}
    for code, name in index_codes.items():
        try:
            df = pro.index_daily(ts_code=code, start_date=START_DATE, end_date=END_DATE.replace('-', ''))
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df = df.set_index('trade_date')
                index_data[code] = df[['close']].rename(columns={'close': name})
                print(f"  ✓ {name} ({code}): {len(df)} 条数据")
            else:
                print(f"  ✗ {name} ({code}): 无数据")
        except Exception as e:
            print(f"  ✗ {name} ({code}): {str(e)[:50]}")

    if index_data:
        index_pivot = pd.concat(index_data.values(), axis=1)
        return index_data, index_pivot

    return {}, pd.DataFrame()

def fetch_macro_data_akshare():
    print("\n" + "="*60)
    print("获取宏观数据 (akshare)")
    print("="*60)

    macro_data = {}

    try:
        import akshare as ak

        print("  正在获取宏观数据...")

        try:
            cpi_df = ak.macro_cpi()
            if cpi_df is not None and len(cpi_df) > 0:
                macro_data['cpi'] = cpi_df
                print(f"    ✓ CPI: {len(cpi_df)} 条")
        except Exception as e:
            print(f"    ⚠ CPI获取失败")

        try:
            ppi_df = ak.macro_ppi()
            if ppi_df is not None and len(ppi_df) > 0:
                macro_data['ppi'] = ppi_df
                print(f"    ✓ PPI: {len(ppi_df)} 条")
        except Exception as e:
            print(f"    ⚠ PPI获取失败")

        try:
            gdp_df = ak.macro_gdp()
            if gdp_df is not None and len(gdp_df) > 0:
                macro_data['gdp'] = gdp_df
                print(f"    ✓ GDP: {len(gdp_df)} 条")
        except Exception as e:
            print(f"    ⚠ GDP获取失败")

        try:
            gold_df = ak.spot_gold_sge()
            if gold_df is not None and len(gold_df) > 0:
                macro_data['gold'] = gold_df
                print(f"    ✓ 黄金: {len(gold_df)} 条")
        except Exception as e:
            try:
                gold_df = ak.spot_gold()
                if gold_df is not None and len(gold_df) > 0:
                    macro_data['gold'] = gold_df
                    print(f"    ✓ 黄金: {len(gold_df)} 条")
            except:
                print(f"    ⚠ 黄金获取失败")

        try:
            oil_df = ak.spot_oil()
            if oil_df is not None and len(oil_df) > 0:
                macro_data['oil'] = oil_df
                print(f"    ✓ 原油: {len(oil_df)} 条")
        except Exception as e:
            print(f"    ⚠ 原油获取失败")

    except ImportError:
        print("  ⚠ akshare 未安装，部分宏观数据无法获取")
    except Exception as e:
        print(f"  ⚠ 获取宏观数据时出错: {str(e)[:80]}")

    return macro_data

def save_data_to_files(industry_df, industry_pivot, industry_returns, index_data, index_pivot, macro_data):
    print("\n" + "="*60)
    print("保存数据到文件")
    print("="*60)

    try:
        if not industry_df.empty:
            industry_df.to_csv(os.path.join(DATA_DIR, 'citic_industry_prices.csv'), index=True)
            print(f"  ✓ citic_industry_prices.csv")

        if not industry_pivot.empty:
            industry_pivot.to_csv(os.path.join(DATA_DIR, 'citic_industry_prices.csv'))
            print(f"  ✓ citic_industry_prices.csv")

        if not industry_returns.empty:
            industry_returns.to_csv(os.path.join(DATA_DIR, 'citic_industry_returns.csv'))
            print(f"  ✓ citic_industry_returns.csv")

        if not index_pivot.empty:
            index_pivot.to_csv(os.path.join(DATA_DIR, 'market_index_prices.csv'))
            print(f"  ✓ market_index_prices.csv")

        if not macro_data.empty:
            macro_data.to_csv(os.path.join(DATA_DIR, 'macro_factor_data.csv'))
            print(f"  ✓ macro_factor_data.csv")

        print(f"\n所有数据已保存至 {DATA_DIR} 文件夹")

    except Exception as e:
        print(f"  保存数据时出错: {e}")

def run_macro_factor_backtest(industry_returns, macro_factors):
    print("\n" + "="*60)
    print("运行宏观因子行业配置策略")
    print("="*60)

    if industry_returns.empty or len(industry_returns) < 100:
        print("  行业数据不足，无法进行回测")
        return None, None, None

    if macro_factors.empty:
        print("  宏观因子数据不足，无法进行回测")
        return None, None, None

    industry_returns = industry_returns.dropna(axis=1, how='all')
    industry_returns = industry_returns.dropna(axis=1)
    industry_returns = industry_returns[industry_returns.index >= pd.to_datetime(START_DATE)]

    if len(industry_returns) < 252:
        print("  数据不足，无法进行回测")
        return None, None, None

    monthly_returns = industry_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    macro_factors_aligned = macro_factors[macro_factors.index >= monthly_returns.index.min()]

    lookback = 24
    top_n = 8

    common_dates = monthly_returns.index.intersection(macro_factors_aligned.index)
    if len(common_dates) < 12:
        print("  宏观因子与行业收益日期不匹配")
        return None, None, None

    monthly_returns = monthly_returns.loc[common_dates]
    macro_factors_aligned = macro_factors_aligned.loc[common_dates]

    print(f"  月度行业收益: {monthly_returns.shape}")
    print(f"  宏观因子: {macro_factors_aligned.shape}")
    print(f"  共同日期数: {len(common_dates)}")

    factor_exposures = {}
    for industry in monthly_returns.columns:
        y = monthly_returns[industry].values
        X = macro_factors_aligned.values
        valid_idx = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid_idx.sum() > lookback:
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]
            if len(X_valid) > 0:
                try:
                    coef, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
                    factor_exposures[industry] = coef
                except:
                    pass

    if not factor_exposures:
        print("  无法计算因子暴露度")
        return None, None, None

    factor_exp_df = pd.DataFrame(factor_exposures).T
    factor_exp_df.columns = macro_factors_aligned.columns

    print(f"  因子暴露度矩阵: {factor_exp_df.shape}")

    selected_industries_list = []
    strategy_returns = []

    monthly_dates = monthly_returns.index.tolist()
    for i in range(lookback, len(monthly_dates) - 1):
        hist_end = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        hist_returns = monthly_returns.loc[:hist_end].tail(lookback)
        if len(hist_returns) < lookback // 2:
            continue

        recent_factors = macro_factors_aligned.loc[:hist_end].tail(6)
        if len(recent_factors) < 3:
            continue

        factor_weights = np.array([0.4, 0.3, 0.2, 0.1])
        factor_weights = factor_weights[:len(recent_factors.columns)]
        factor_weights = factor_weights / factor_weights.sum()

        factor_cumret = (1 + recent_factors).prod() - 1
        expected_factor_returns = factor_cumret.values[:len(factor_weights)]

        predicted_returns = factor_exp_df.values @ expected_factor_returns

        top_industries = factor_exp_df.index[np.argsort(predicted_returns)[-top_n:]].tolist()

        selected_industries_list.append({
            'date': hist_end,
            'selected': top_industries
        })

        next_ret = monthly_returns.loc[next_date, top_industries] if next_date in monthly_returns.index else pd.Series()
        if len(next_ret) > 0 and not next_ret.isna().all():
            strat_ret = next_ret.mean()
            strategy_returns.append({
                'date': next_date,
                'return': strat_ret
            })

    if not strategy_returns:
        print("  回测无结果")
        return None, None, None

    strategy_df = pd.DataFrame(strategy_returns).set_index('date')

    benchmark_returns = monthly_returns.loc[strategy_df.index].mean(axis=1)
    benchmark_cumulative = (1 + benchmark_returns.dropna()).cumprod()

    strategy_cumulative = (1 + strategy_df['return']).cumprod()

    total_return = strategy_cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (12 / len(strategy_df)) - 1
    annual_vol = strategy_df['return'].std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    running_max = strategy_cumulative.cummax()
    drawdown = (strategy_cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    if not benchmark_cumulative.empty:
        aligned_bench = benchmark_cumulative.loc[strategy_cumulative.index]
        aligned_bench_pct = aligned_bench.pct_change().fillna(0)
        excess_returns = strategy_df['return'].values - aligned_bench_pct.values
        excess_returns = pd.Series(excess_returns, index=strategy_df.index)
        excess_cumulative = (1 + excess_returns).cumprod()
        excess_return = excess_cumulative.iloc[-1] - 1 if len(excess_cumulative) > 0 else 0
    else:
        excess_return = 0

    print(f"\n宏观因子策略回测结果:")
    print(f"  回测期数: {len(strategy_df)}")
    print(f"  调仓次数: {len(selected_industries_list)}")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化收益率: {annual_return:.2%}")
    print(f"  年化波动率: {annual_vol:.2%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    print(f"  超额收益率: {excess_return:.2%}")

    return strategy_df, {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'excess_return': excess_return
    }, benchmark_cumulative, factor_exp_df


def run_simple_backtest(industry_returns, index_pivot):
    print("\n" + "="*60)
    print("运行简化回测 (基于历史动量)")
    print("="*60)

    if industry_returns.empty or len(industry_returns) < 100:
        print("  行业数据不足，使用市场指数进行简化回测")

        if index_pivot.empty or len(index_pivot) < 100:
            print("  数据不足，无法进行回测")
            return None, None, None

        returns = index_pivot.pct_change().dropna()
        returns = returns[returns.index >= pd.to_datetime(START_DATE)]

        if len(returns) < 52:
            print("  数据不足，无法进行回测")
            return None, None, None

        lookback = 52
        top_n = 2

        quarterly_dates = returns.resample('Q').last().index

        selected_list = []
        strategy_returns = []

        for i in range(len(quarterly_dates) - 1):
            hist_end = quarterly_dates[i]
            next_end = quarterly_dates[i + 1]

            hist_returns = returns.loc[:hist_end].tail(lookback)
            if len(hist_returns) < lookback // 2:
                continue

            mean_returns = hist_returns.mean().sort_values(ascending=False)
            top_n_idx = mean_returns.head(top_n).index.tolist()

            selected_list.append({
                'date': hist_end,
                'selected': top_n_idx
            })

            next_ret = returns.loc[hist_end:next_end]
            if len(next_ret) > 0:
                strat_ret = next_ret[top_n_idx].mean(axis=1).mean()
                strategy_returns.append({
                    'date': next_ret.index[int(len(next_ret) // 2)],
                    'return': strat_ret
                })

        if not strategy_returns:
            print("  回测无结果")
            return None, None, None

        strategy_df = pd.DataFrame(strategy_returns).set_index('date')

        benchmark_returns = returns.loc[strategy_df.index].mean(axis=1)
        benchmark_cumulative = (1 + benchmark_returns).dropna()
        benchmark_cumulative = (1 + benchmark_cumulative).cumprod()

        strategy_cumulative = (1 + strategy_df['return']).cumprod()

        total_return = strategy_cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (4 / len(strategy_df)) - 1
        annual_vol = strategy_df['return'].std() * np.sqrt(4)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        running_max = strategy_cumulative.cummax()
        drawdown = (strategy_cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        if not benchmark_cumulative.empty:
            aligned_bench = benchmark_cumulative.loc[strategy_cumulative.index]
            aligned_bench_pct = aligned_bench.pct_change().fillna(0)
            excess_returns = strategy_df['return'].values - aligned_bench_pct.values
            excess_returns = pd.Series(excess_returns, index=strategy_df.index)
            excess_cumulative = (1 + excess_returns).cumprod()
            excess_return = excess_cumulative.iloc[-1] - 1 if len(excess_cumulative) > 0 else 0
        else:
            excess_return = 0

        print(f"\n简化回测结果 (基于市场指数动量):")
        print(f"  回测期数: {len(strategy_df)}")
        print(f"  调仓次数: {len(selected_list)}")
        print(f"  总收益率: {total_return:.2%}")
        print(f"  年化收益率: {annual_return:.2%}")
        print(f"  年化波动率: {annual_vol:.2%}")
        print(f"  夏普比率: {sharpe:.2f}")
        print(f"  最大回撤: {max_drawdown:.2%}")
        print(f"  超额收益率: {excess_return:.2%}")

        return strategy_df, {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'excess_return': excess_return
        }, benchmark_cumulative

    returns = industry_returns.dropna(axis=1, how='all')

    if returns.empty:
        print("  数据不足，无法进行回测")
        return None, None, None

    returns = returns.dropna(axis=1)
    returns = returns[returns.index >= pd.to_datetime(START_DATE)]

    if len(returns) < 52:
        print("  数据不足，无法进行回测")
        return None, None, None

    lookback = 52
    top_n = 10

    quarterly_dates = returns.resample('Q').last().index

    selected_industries_list = []
    strategy_returns = []

    for i in range(len(quarterly_dates) - 1):
        hist_end = quarterly_dates[i]
        next_end = quarterly_dates[i + 1]

        hist_returns = returns.loc[:hist_end].tail(lookback)
        if len(hist_returns) < lookback // 2:
            continue

        mean_returns = hist_returns.mean().sort_values(ascending=False)
        top_10 = mean_returns.head(top_n).index.tolist()

        selected_industries_list.append({
            'date': hist_end,
            'selected': top_10
        })

        next_returns = returns.loc[hist_end:next_end]
        if len(next_returns) > 0:
            strat_ret = next_returns[top_10].mean(axis=1).mean()
            strategy_returns.append({
                'date': next_returns.index[int(len(next_returns) // 2)],
                'return': strat_ret
            })

    if not strategy_returns:
        print("  回测无结果")
        return None, None, None

    strategy_df = pd.DataFrame(strategy_returns).set_index('date')

    benchmark_returns = returns.loc[strategy_df.index].mean(axis=1)
    benchmark_cumulative = (1 + benchmark_returns).dropna()
    benchmark_cumulative = (1 + benchmark_cumulative).cumprod()

    strategy_cumulative = (1 + strategy_df['return']).cumprod()

    total_return = strategy_cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (4 / len(strategy_df)) - 1
    annual_vol = strategy_df['return'].std() * np.sqrt(4)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    running_max = strategy_cumulative.cummax()
    drawdown = (strategy_cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    if not benchmark_cumulative.empty:
        aligned_bench = benchmark_cumulative.loc[strategy_cumulative.index]
        aligned_bench_pct = aligned_bench.pct_change().fillna(0)
        excess_returns = strategy_df['return'].values - aligned_bench_pct.values
        excess_returns = pd.Series(excess_returns, index=strategy_df.index)
        excess_cumulative = (1 + excess_returns).cumprod()
        excess_return = excess_cumulative.iloc[-1] - 1 if len(excess_cumulative) > 0 else 0
    else:
        excess_return = 0

    print(f"\n回测结果:")
    print(f"  回测期数: {len(strategy_df)}")
    print(f"  调仓次数: {len(selected_industries_list)}")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化收益率: {annual_return:.2%}")
    print(f"  年化波动率: {annual_vol:.2%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    print(f"  超额收益率: {excess_return:.2%}")

    return strategy_df, {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'excess_return': excess_return
    }, benchmark_cumulative

def plot_and_save_results(strategy_df, strategy_cumulative, benchmark_cumulative, drawdown):
    print("\n" + "="*60)
    print("生成图表")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if not strategy_cumulative.empty:
        strat_equity = strategy_cumulative * 1000000
        bench_equity = benchmark_cumulative * 1000000 if not benchmark_cumulative.empty else None

        axes[0, 0].plot(strat_equity.index, strat_equity.values, label='Strategy', linewidth=2, color='blue')
        if bench_equity is not None and not bench_equity.empty:
            aligned_bench = bench_equity.loc[strat_equity.index]
            axes[0, 0].plot(aligned_bench.index, aligned_bench.values, label='Benchmark', linewidth=1.5, alpha=0.7, color='gray')
        axes[0, 0].set_title('Portfolio Value Over Time', fontsize=12)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    excess_returns = strategy_df['return']
    if not benchmark_cumulative.empty:
        aligned_bench = benchmark_cumulative.loc[strategy_df.index]
        if len(aligned_bench) > 0:
            aligned_bench_pct = aligned_bench.pct_change().fillna(0)
            excess_returns = strategy_df['return'].values - aligned_bench_pct.values
            excess_returns = pd.Series(excess_returns, index=strategy_df.index)
    excess_cumulative = (1 + excess_returns).cumprod()

    if len(excess_cumulative) > 0:
        axes[0, 1].plot((excess_cumulative - 1).values, color='green', linewidth=2)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Cumulative Excess Returns', fontsize=12)
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].set_ylabel('Excess Return')
        axes[0, 1].grid(True, alpha=0.3)

    if len(drawdown) > 0:
        axes[1, 0].fill_between(drawdown.index, drawdown.values * 100, 0, color='red', alpha=0.5)
        axes[1, 0].set_title('Drawdown', fontsize=12)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)

    quarterly_returns = strategy_df['return'].resample('Q').apply(lambda x: (1 + x).prod() - 1)
    if len(quarterly_returns) > 0:
        axes[1, 1].bar(range(len(quarterly_returns)), quarterly_returns.values * 100, color='steelblue', alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Quarterly Returns', fontsize=12)
        axes[1, 1].set_xlabel('Quarter')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_results.png'), dpi=150)
    print(f"  ✓ backtest_results.png")

    plt.close()

def save_results_to_output(strategy_df, metrics, selected_industries):
    print("\n" + "="*60)
    print("保存结果到output文件夹")
    print("="*60)

    results_summary = pd.DataFrame({
        '指标': ['总收益率', '年化收益率', '年化波动率', '夏普比率', '最大回撤', '超额收益率'],
        '数值': [
            f"{metrics['total_return']:.2%}",
            f"{metrics['annual_return']:.2%}",
            f"{metrics['annual_vol']:.2%}",
            f"{metrics['sharpe']:.2f}",
            f"{metrics['max_drawdown']:.2%}",
            f"{metrics['excess_return']:.2%}"
        ]
    })
    results_summary.to_csv(os.path.join(OUTPUT_DIR, 'backtest_summary.csv'), index=False)
    print(f"  ✓ backtest_summary.csv")

    strategy_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_returns.csv'))
    print(f"  ✓ strategy_returns.csv")

    selected_df = pd.DataFrame(selected_industries)
    if not selected_df.empty:
        selected_df['selected_str'] = selected_df['selected'].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))
        selected_df[['date', 'selected_str']].to_csv(os.path.join(OUTPUT_DIR, 'selected_industries.csv'), index=False)
        print(f"  ✓ selected_industries.csv")

    print(f"\n所有结果已保存至 {OUTPUT_DIR} 文件夹")

def print_data_gaps():
    print("\n" + "="*60)
    print("数据缺失说明")
    print("="*60)
    print("\n研报中部分代理资产在Tushare/Akshare中不可直接获取:")
    print("\n原始代理资产 -> 替代方案:")
    print("-" * 60)

    replacements = {
        'HSI.HI (恒生指数)': 'akshare暂不支持，可使用沪深300指数替代',
        'CRBRI.RB (CRB工业现货)': 'akshare暂无，可使用南华商品指数或原油期货',
        'NH0012.NHF (南华沪铜)': 'akshare暂无，可使用铜ETF替代',
        'CBA00652.CS (7-10年国债净价)': 'akshare暂无国债指数，可使用国债ETF替代',
        'NH0056.NHF (南华生猪)': 'akshare暂无，可使用猪肉价格数据',
        'B00.IPE (布伦特原油)': 'akshare.spot_oil()可获取',
        'NH0016.NHF (南华螺纹钢)': 'akshare暂无，可使用钢铁ETF',
        'NH0030.NHF (南华动力煤)': 'akshare暂无',
        'CBA00621.CS (1-3年国债)': 'akshare暂无，可使用国债ETF替代',
        'CBA02501.CS (国开债)': 'akshare暂无',
        'CBA00651.CS (7-10年国债)': 'akshare暂无',
        'AU9999.SGE (伦敦金现)': 'akshare.spot_gold()可获取'
    }

    for original, alternative in replacements.items():
        print(f"  {original}")
        print(f"    -> {alternative}")

    print("\n" + "="*60)
    print("补充数据方案")
    print("="*60)
    print("""
1. 宏观数据 (CPI, PPI, GDP, 利率):
   - 使用 akshare.macro_cnzb(), akshare.cpi(), akshare.ppi()

2. 行业指数:
   - Tushare: 申万行业指数 (SW 30个一级行业) ✓
   - Akshare: ak.sw_index_daily(symbol="L1")

3. 国债/债券数据:
   - akshare.bond_zh_daily() 可获取部分债券数据

4. 黄金/原油:
   - akshare.spot_gold(), akshare.spot_oil() ✓
    """)

def main():
    print("\n" + "="*70)
    print(" 高频宏观因子行业配置策略 - 华泰证券研报复现")
    print("="*70)
    print(f"\n回测区间: {START_DATE} 至 {END_DATE}")
    print(f"Tushare API: {TUSHARE_API_URL}")

    initialize_directories()

    industry_df, industry_pivot, industry_returns = load_citic_industry_data()

    index_data, index_pivot = fetch_market_index_data(PRO)

    macro_factor_data = load_macro_factor_data()

    save_data_to_files(industry_df, industry_pivot, industry_returns, index_data, index_pivot, macro_factor_data)

    strategy_df, metrics, benchmark_cumulative, factor_exp_df = run_macro_factor_backtest(industry_returns, macro_factor_data)

    if strategy_df is not None and metrics is not None:
        strategy_cumulative = (1 + strategy_df['return']).cumprod()
        running_max = strategy_cumulative.cummax()
        drawdown = (strategy_cumulative - running_max) / running_max

        plot_and_save_results(strategy_df, strategy_cumulative, benchmark_cumulative, drawdown)

        selected_industries = []
        if factor_exp_df is not None and not factor_exp_df.empty:
            for date, row in factor_exp_df.iterrows():
                top_industries = factor_exp_df.loc[date].sort_values(ascending=False).head(8).index.tolist()
                selected_industries.append({'date': date, 'selected': top_industries})

        save_results_to_output(strategy_df, metrics, selected_industries)

    print_data_gaps()

    print("\n" + "="*70)
    print("项目执行完成!")
    print("="*70)
    print(f"\n数据文件已保存至: {os.path.abspath(DATA_DIR)}")
    print(f"结果文件已保存至: {os.path.abspath(OUTPUT_DIR)}")

    return {
        'industry_df': industry_df,
        'industry_pivot': industry_pivot,
        'industry_returns': industry_returns,
        'strategy_df': strategy_df,
        'metrics': metrics
    }

if __name__ == "__main__":
    results = main()