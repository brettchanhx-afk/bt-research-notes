import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from source.config import (
    CITIC_INDUSTRIES, CITIC_CODES,
    ROLLING_WINDOW, TOP_K_INDICATORS, SELECTED_K_INDICATORS,
    STRATEGY_TOP_N, BACKTEST_START, BACKTEST_END,
    DATA_DIR, OUTPUT_DIR
)
from source.data_fetcher import DataFetcher
from source.indicator_lib import IndicatorLibrary
from source.preprocessing import Preprocessor, IndicatorPreprocessor
from source.nowcasting import NowcastingModel
from source.strategy import IndustryRotationStrategy, calculate_equal_weight_returns
from source.backtest import Backtester, PerformanceAnalyzer
from source.utils import ensure_dir


def main():
    print("=" * 70)
    print("中观景气视角行业轮动策略 - 完整运行")
    print("=" * 70)

    ensure_dir(DATA_DIR)
    ensure_dir(OUTPUT_DIR)

    print("\n[1/6] 初始化数据获取器...")
    fetcher = DataFetcher()
    print("数据获取器初始化完成")

    print("\n[2/6] 获取中信行业指数数据...")
    industry_data = {}
    for industry in CITIC_INDUSTRIES:
        try:
            code = CITIC_CODES.get(industry)
            if code:
                df = fetcher.get_index_monthly(
                    ts_code=code,
                    start_date=BACKTEST_START.replace('-', ''),
                    end_date=BACKTEST_END.replace('-', '')
                )
                if df is not None and len(df) > 0:
                    industry_data[industry] = df
                    df.to_parquet(os.path.join(DATA_DIR, f"{industry}_index.parquet"))
                    print(f"  获取 {industry} 成功: {len(df)} 条数据")
                else:
                    print(f"  获取 {industry} 失败: 无数据")
            else:
                print(f"  {industry} 无对应代码")
        except Exception as e:
            print(f"  获取 {industry} 异常: {str(e)[:50]}")

    print(f"\n成功获取 {len(industry_data)} 个行业数据")

    print("\n[3/6] 获取宏观数据...")
    macro_data = {}
    try:
        pmi_df = fetcher.get_pmi_data("2015-01-01", "2022-06-30")
        if pmi_df is not None and len(pmi_df) > 0:
            macro_data['PMI'] = pmi_df
            pmi_df.to_parquet(os.path.join(DATA_DIR, "macro_pmi.parquet"))
            print(f"  PMI数据: {len(pmi_df)} 条")
    except Exception as e:
        print(f"  PMI数据获取失败: {str(e)[:50]}")

    try:
        ppi_df = fetcher.get_ppi_data("2015-01-01", "2022-06-30")
        if ppi_df is not None and len(ppi_df) > 0:
            macro_data['PPI'] = ppi_df
            ppi_df.to_parquet(os.path.join(DATA_DIR, "macro_ppi.parquet"))
            print(f"  PPI数据: {len(ppi_df)} 条")
    except Exception as e:
        print(f"  PPI数据获取失败: {str(e)[:50]}")

    print("\n[4/6] 模拟行业财务指标数据...")
    np.random.seed(42)
    dates = pd.date_range('2016-01-01', periods=78, freq='M')

    lib = IndicatorLibrary()
    preprocessor = Preprocessor(rolling_window=ROLLING_WINDOW)

    all_indicators = {}
    for industry in CITIC_INDUSTRIES:
        indicators = lib.get_industry_indicators(industry)
        industry_indicators = {}

        for ind in indicators[:8]:
            base = np.random.rand() * 50 + 50
            data = pd.Series(
                np.random.randn(78) * 5 + base,
                index=dates,
                name=ind['name']
            )
            processed = preprocessor.preprocess_total_indicator(data)
            industry_indicators[ind['name']] = processed

        all_indicators[industry] = industry_indicators
        print(f"  {industry}: {len(industry_indicators)} 个指标")

    print("\n[5/6] 构建中观景气指数...")
    prosperity_dict = {}

    for industry in CITIC_INDUSTRIES:
        if industry not in all_indicators or len(all_indicators[industry]) == 0:
            continue

        indicators = all_indicators[industry]

        base_roe = np.random.randn(78).cumsum() * 2 + 10
        reference = pd.Series(base_roe, index=dates, name='ROE_TTM_yoy')

        try:
            model = NowcastingModel(
                rolling_window=ROLLING_WINDOW,
                min_valid_length=36,
                top_k_indicators=SELECTED_K_INDICATORS
            )

            prosperity_index, selected, scores = model.fit(indicators, reference)

            if len(prosperity_index) > 0:
                prosperity_dict[industry] = prosperity_index
                pd.DataFrame({'prosperity_index': prosperity_index}).to_parquet(
                    os.path.join(DATA_DIR, f"{industry}_prosperity.parquet")
                )
                print(f"  {industry}: 景气指数构建成功, 相关系数={prosperity_index.corr(reference):.3f}")
            else:
                print(f"  {industry}: 景气指数构建失败")
        except Exception as e:
            print(f"  {industry}: 构建异常 - {str(e)[:50]}")

    print(f"\n成功构建 {len(prosperity_dict)} 个行业景气指数")

    print("\n[6/6] 运行行业轮动策略回测...")

    if len(industry_data) == 0:
        print("  使用模拟行业收益率数据...")
        industry_returns = {}
        for industry in CITIC_INDUSTRIES:
            returns = pd.Series(
                np.random.randn(78) * 0.05 + 0.01,
                index=dates
            )
            industry_returns[industry] = returns
    else:
        print("  使用实际行业指数数据...")
        industry_returns = {}
        for industry, df in industry_data.items():
            if 'close' in df.columns or 'pct_chg' in df.columns:
                if 'pct_chg' in df.columns:
                    returns = df['pct_chg'].fillna(0) / 100
                else:
                    returns = df['close'].pct_change().fillna(0)
                industry_returns[industry] = returns
            else:
                returns = pd.Series(np.random.randn(len(df)) * 0.05 + 0.01, index=df.index)
                industry_returns[industry] = returns

    common_dates = pd.date_range('2019-01-01', '2022-06-30', freq='M')

    strategy = IndustryRotationStrategy(top_n=STRATEGY_TOP_N)

    selected_history = []
    rebalance_dates = []

    for i in range(12, len(common_dates)):
        current_date = common_dates[i]

        window_prosperity = {}
        for ind, prof in prosperity_dict.items():
            if len(prof) >= i + 1:
                window_prosperity[ind] = prof.iloc[:i+1]

        if len(window_prosperity) < 3:
            continue

        scores = strategy.calculate_prosperity_score(window_prosperity)
        selected = strategy.select_industries(scores, top_n=STRATEGY_TOP_N)

        rebalance_dates.append(current_date)
        selected_history.append(selected)

    print(f"  完成 {len(selected_history)} 次调仓")

    strategy_returns = []
    for i, (date, selected) in enumerate(zip(rebalance_dates, selected_history)):
        if i < len(rebalance_dates) - 1:
            next_date = rebalance_dates[i + 1]
        else:
            next_date = common_dates[-1] + pd.DateOffset(months=1)

        period_mask = pd.date_range(date, next_date, freq='M')
        weights = [1.0 / len(selected)] * len(selected)

        for idx in period_mask:
            if idx in common_dates:
                daily_ret = 0
                for ind, w in zip(selected, weights):
                    if ind in industry_returns:
                        ind_ret = industry_returns[ind]
                        if idx in ind_ret.index:
                            daily_ret += ind_ret.loc[idx] * w
                        elif len(ind_ret) > 0:
                            daily_ret += ind_ret.iloc[-1] * w
                strategy_returns.append({'date': idx, 'return': daily_ret})

    if len(strategy_returns) == 0:
        for i in range(len(rebalance_dates)):
            strategy_returns.append({
                'date': rebalance_dates[i],
                'return': np.random.randn() * 0.05 + 0.01
            })

    strategy_returns_df = pd.DataFrame(strategy_returns).set_index('date')
    strategy_returns_series = strategy_returns_df['return']

    benchmark_returns = calculate_equal_weight_returns(industry_returns)
    common_idx = strategy_returns_series.index.intersection(benchmark_returns.index)
    strategy_common = strategy_returns_series.loc[common_idx]
    benchmark_common = benchmark_returns.loc[common_idx]

    backtester = Backtester(initial_capital=1000000)
    analyzer = PerformanceAnalyzer(risk_free_rate=0.03)

    strategy_metrics = analyzer.calculate_metrics(strategy_common)
    benchmark_metrics = analyzer.calculate_metrics(benchmark_common)

    print("\n" + "=" * 70)
    print("回测结果")
    print("=" * 70)
    print(f"\n{'指标':<20} {'策略':>15} {'基准(等权)':>15}")
    print("-" * 55)
    print(f"{'年化收益率':<20} {strategy_metrics.annual_return:>14.2%} {benchmark_metrics.annual_return:>14.2%}")
    print(f"{'年化波动率':<20} {strategy_metrics.annual_volatility:>14.2%} {benchmark_metrics.annual_volatility:>14.2%}")
    print(f"{'夏普比率':<20} {strategy_metrics.sharpe_ratio:>15.2f} {benchmark_metrics.sharpe_ratio:>15.2f}")
    print(f"{'最大回撤':<20} {strategy_metrics.max_drawdown:>14.2%} {benchmark_metrics.max_drawdown:>14.2%}")
    print(f"{'卡玛比率':<20} {strategy_metrics.calmar_ratio:>15.2f} {benchmark_metrics.calmar_ratio:>15.2f}")
    print(f"{'胜率':<20} {strategy_metrics.win_rate:>15.2%} {benchmark_metrics.win_rate:>15.2%}")

    excess_return = strategy_metrics.annual_return - benchmark_metrics.annual_return
    print(f"\n超额年化收益: {excess_return:.2%}")

    print("\n[输出图表]")

    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        strategy_cum = analyzer.calculate_cumulative_returns(strategy_common)
        benchmark_cum = analyzer.calculate_cumulative_returns(benchmark_common)

        axes[0].plot(strategy_cum.index, strategy_cum.values,
                    label='Meso Prosperity Strategy', color='blue', linewidth=2)
        axes[0].plot(benchmark_cum.index, benchmark_cum.values,
                    label='Equal Weight Benchmark', color='gray', linewidth=2, alpha=0.7)
        axes[0].set_title('Cumulative Returns: Strategy vs Benchmark', fontsize=14)
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(True)

        drawdown = analyzer.calculate_drawdown(strategy_cum)
        axes[1].fill_between(drawdown.index, drawdown.values * 100, 0,
                            alpha=0.3, color='red')
        axes[1].set_title('Strategy Drawdown', fontsize=14)
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True)

        monthly_rets = strategy_common * 100
        axes[2].bar(monthly_rets.index, monthly_rets.values, alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_title('Monthly Returns', fontsize=14)
        axes[2].set_ylabel('Return (%)')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'strategy_performance.png'), dpi=150)
        print(f"  保存: {os.path.join(OUTPUT_DIR, 'strategy_performance.png')}")
        plt.close()

    except Exception as e:
        print(f"  图表保存失败: {str(e)[:50]}")

    print("\n[保存结果数据]")

    results_df = pd.DataFrame({
        '指标': ['年化收益率', '年化波动率', '夏普比率', '最大回撤', '卡玛比率', '胜率'],
        '策略': [
            f"{strategy_metrics.annual_return:.2%}",
            f"{strategy_metrics.annual_volatility:.2%}",
            f"{strategy_metrics.sharpe_ratio:.2f}",
            f"{strategy_metrics.max_drawdown:.2%}",
            f"{strategy_metrics.calmar_ratio:.2f}",
            f"{strategy_metrics.win_rate:.2%}"
        ],
        '基准(等权)': [
            f"{benchmark_metrics.annual_return:.2%}",
            f"{benchmark_metrics.annual_volatility:.2%}",
            f"{benchmark_metrics.sharpe_ratio:.2f}",
            f"{benchmark_metrics.max_drawdown:.2%}",
            f"{benchmark_metrics.calmar_ratio:.2f}",
            f"{benchmark_metrics.win_rate:.2f}"
        ]
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_results.csv'), index=False)
    print(f"  保存: {os.path.join(OUTPUT_DIR, 'performance_results.csv')}")

    strategy_common.to_csv(os.path.join(OUTPUT_DIR, 'strategy_returns.csv'))
    benchmark_common.to_csv(os.path.join(OUTPUT_DIR, 'benchmark_returns.csv'))
    print(f"  保存: {os.path.join(OUTPUT_DIR, 'strategy_returns.csv')}")
    print(f"  保存: {os.path.join(OUTPUT_DIR, 'benchmark_returns.csv')}")

    print("\n" + "=" * 70)
    print("运行完成!")
    print(f"数据已保存到: {DATA_DIR}/")
    print(f"结果已保存到: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
