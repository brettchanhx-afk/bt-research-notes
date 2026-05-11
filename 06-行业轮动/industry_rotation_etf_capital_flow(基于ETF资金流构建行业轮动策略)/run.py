import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from config.settings import (
    TUSHARE_TOKEN,
    TUSHARE_API_URL,
    INDUSTRY_INDEX_MAPPING,
    INDUSTRY_NAME_MAPPING,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
)
from source.data_fetcher import MultiSourceDataFetcher, load_or_fetch_data
from source.etf_flow import ETFFlowCalculator
from source.industry_rotation import IndustryRotationStrategy
from source.backtest import plot_equity_curve, plot_drawdown
from source.utils import ensure_dir, calculate_performance_metrics, save_results

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def run_data_collection(data_dir: Path, force_refresh: bool = False):
    print("\n" + "=" * 70)
    print("第一步：数据采集 (优先Tushare)")
    print("=" * 70)

    fetcher = MultiSourceDataFetcher()

    print("\n[1] 获取ETF列表 (Tushare优先)...")
    etf_df = fetcher.get_etf_list_tushare()
    if etf_df is None or len(etf_df) == 0:
        print("  Tushare失败,使用AkShare...")
        etf_df = fetcher.get_etf_list_akshare()
    if etf_df is not None and len(etf_df) > 0:
        etf_file = data_dir / "etf_list.csv"
        etf_df.to_csv(etf_file, index=False, encoding="utf-8-sig")
        print(f"  ✓ ETF列表已保存: {etf_file} ({len(etf_df)} 只)")
    else:
        print("  ✗ ETF列表获取失败")

    print("\n[2] 获取行业指数数据 (Tushare优先)...")
    index_returns_list = []
    success_count = 0

    for industry_name, index_code in INDUSTRY_INDEX_MAPPING.items():
        print(f"\n  获取 {industry_name} ({index_code})...")

        df = fetcher.get_index_daily_tushare(index_code, "20150101", "20240831")
        if df is None or len(df) == 0:
            print(f"    -> Tushare失败,尝试BaoStock...")
            df = fetcher.get_index_daily_baostock(index_code, "20150101", "20240831")
        if df is None or len(df) == 0:
            print(f"    -> BaoStock失败,尝试AkShare...")
            df = fetcher.get_index_daily_akshare(index_code, "20150101", "20240831")

        if df is not None and len(df) > 0:
            df["industry"] = industry_name
            df["index_code"] = index_code
            df["index_name"] = INDUSTRY_NAME_MAPPING.get(index_code, "")
            index_returns_list.append(df)
            print(f"    ✓ 成功: {len(df)} 条")
            success_count += 1
        else:
            print(f"    ✗ 失败")

    if index_returns_list:
        index_returns = pd.concat(index_returns_list, ignore_index=True)
        index_returns = index_returns.sort_values(["trade_date", "industry"])
        index_file = data_dir / "index_returns.csv"
        index_returns.to_csv(index_file, index=False, encoding="utf-8-sig")
        print(f"\n  ✓ 指数数据已保存: {index_file} (共{len(index_returns)}条, {success_count}个行业)")

        industries_available = index_returns["industry"].unique().tolist()
        print(f"  ✓ 成功获取的行业: {len(industries_available)}")
    else:
        print("\n  ✗ 警告: 未能获取任何指数数据")
        index_returns = pd.DataFrame()

    print("\n[3] ETF资金流数据说明...")
    print("  ⚠️  注意: AkShare/Tushare/BaoStock 均不提供ETF份额变动数据")
    print("  要完整复现研报,需要从 Wind/Choice 获取真实的ETF申赎数据")

    return etf_df, index_returns


def run_flow_calculation(industry_etf_flow: dict, index_returns: pd.DataFrame):
    print("\n" + "=" * 70)
    print("第二步：ETF资金流计算")
    print("=" * 70)

    calculator = ETFFlowCalculator(industry_etf_flow)

    print("\n[1] 计算周度资金流 (自然周)...")
    weekly_flow = calculator.calculate_weekly_net_flow(
        start_date="2018-01-01",
        end_date="2024-07-31",
        method="natural_week"
    )
    print(f"  ✓ 周度资金流: {len(weekly_flow)} 条")

    print("\n[2] 计算滚动历史分位数 (2年窗口)...")
    percentile_data = calculator.calculate_rolling_percentile(
        weekly_flow,
        window_years=2,
        percentile_col="net_flow",
        date_col="week_start",
    )
    valid_percentile = percentile_data.dropna(subset=["rolling_percentile"])
    print(f"  ✓ 滚动分位数: {len(valid_percentile)} 条")

    print("\n[3] 计算未来收益...")
    returns_data = index_returns.copy()
    if "trade_date" not in returns_data.columns and "date" in returns_data.columns:
        returns_data.rename(columns={"date": "trade_date"}, inplace=True)

    future_returns = calculator.calculate_future_returns(
        weekly_flow,
        returns_data,
        date_col="week_start",
        industry_col="industry",
        forward_col="week_end",
    )
    print(f"  ✓ 未来收益计算完成: {len(future_returns)} 条")

    return weekly_flow, percentile_data, future_returns


def run_strategy_backtest(
    percentile_data: pd.DataFrame,
    future_returns: pd.DataFrame,
    index_returns: pd.DataFrame,
    output_dir: Path,
):
    print("\n" + "=" * 70)
    print("第三步：策略回测")
    print("=" * 70)

    returns_data = index_returns.copy()
    if "trade_date" not in returns_data.columns:
        returns_data.rename(columns={"date": "trade_date"}, inplace=True)

    returns_data["trade_date"] = pd.to_datetime(returns_data["trade_date"])
    returns_data["industry"] = returns_data["industry"].str.strip()

    if "pct_change" in returns_data.columns:
        returns_data["return_col"] = returns_data["pct_change"]
    elif "pct_chg" in returns_data.columns:
        returns_data["return_col"] = returns_data["pct_chg"]
    else:
        returns_data["return_col"] = returns_data["close"].pct_change() * 100

    thresholds_results = {}

    for threshold in [0.05, 0.10]:
        print(f"\n{'='*50}")
        print(f"阈值: {int(threshold*100)}%")
        print(f"{'='*50}")

        strategy = IndustryRotationStrategy(
            long_threshold=threshold,
            short_threshold=1 - threshold,
            rolling_window_years=2,
            rebalance_freq="weekly",
        )

        rebalance_dates = sorted(percentile_data["week_start"].unique())

        long_signals = strategy.generate_signals(
            percentile_data, rebalance_dates, signal_type="long"
        )
        short_signals = strategy.generate_signals(
            percentile_data, rebalance_dates, signal_type="short"
        )

        long_returns = strategy.calculate_portfolio_returns(long_signals, returns_data)
        short_returns = strategy.calculate_portfolio_returns(short_signals, returns_data)

        all_returns = pd.concat([long_returns, short_returns], ignore_index=True)

        metrics = strategy.calculate_performance_metrics(all_returns)

        yearly_returns = strategy.get_yearly_returns(all_returns)

        thresholds_results[f"{int(threshold*100)}pct"] = {
            "long_signals": long_signals,
            "short_signals": short_signals,
            "portfolio_returns": all_returns,
            "performance_metrics": metrics,
            "yearly_returns": yearly_returns,
            "strategy": strategy,
        }

        print(f"\n  多头策略绩效:")
        if "long" in metrics:
            m = metrics["long"]
            print(f"    总收益率: {m.get('total_return', 0):.2f}%")
            print(f"    年化收益率: {m.get('annual_return', 0):.2f}%")
            print(f"    夏普比率: {m.get('sharpe', 0):.4f}")
            print(f"    最大回撤: {m.get('max_drawdown', 0):.2f}%")
            print(f"    月度胜率: {m.get('monthly_win_rate', 0):.2f}%")

        print(f"\n  年度收益:")
        if len(yearly_returns) > 0:
            long_yearly = yearly_returns[yearly_returns["signal_type"] == "long"]
            for _, row in long_yearly.iterrows():
                print(f"    {row['year']}: {row['annual_return']:.2f}%")

    return thresholds_results


def save_all_results(
    thresholds_results: dict,
    weekly_flow: pd.DataFrame,
    percentile_data: pd.DataFrame,
    output_dir: Path,
):
    print("\n" + "=" * 70)
    print("第四步：保存结果")
    print("=" * 70)

    ensure_dir(output_dir)

    print("\n[1] 保存回测指标...")
    metrics_list = []
    for th_name, results in thresholds_results.items():
        metrics = results["performance_metrics"]
        if "long" in metrics:
            m = metrics["long"]
            metrics_list.append({
                "threshold": th_name,
                "total_return": m.get("total_return", 0),
                "annual_return": m.get("annual_return", 0),
                "annual_vol": m.get("annual_vol", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "sharpe": m.get("sharpe", 0),
                "calmar": m.get("calmar", 0),
                "weekly_win_rate": m.get("weekly_win_rate", 0),
                "monthly_win_rate": m.get("monthly_win_rate", 0),
                "profit_loss_ratio": m.get("profit_loss_ratio", 0),
                "num_trades": m.get("num_trades", 0),
            })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = output_dir / "backtest_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    print(f"  ✓ {metrics_file}")

    print("\n[2] 保存净值曲线...")
    for th_name, results in thresholds_results.items():
        returns_df = results["portfolio_returns"]
        if len(returns_df) > 0:
            long_returns = returns_df[returns_df["signal_type"] == "long"].copy()
            if len(long_returns) > 0:
                long_returns = long_returns.sort_values("date")
                long_returns["cumulative"] = (
                    1 + long_returns["mean_return"].dropna() / 100
                ).cumprod()
                curve_file = output_dir / f"net_value_{th_name}.csv"
                long_returns.to_csv(curve_file, index=False, encoding="utf-8-sig")
                print(f"  ✓ {curve_file}")

    print("\n[3] 保存交易信号...")
    for th_name, results in thresholds_results.items():
        long_signals = results["long_signals"]
        if len(long_signals) > 0:
            signals_file = output_dir / f"signals_long_{th_name}.csv"
            long_signals.to_csv(signals_file, index=False, encoding="utf-8-sig")
            print(f"  ✓ {signals_file}")

    print("\n[4] 保存年度收益...")
    yearly_list = []
    for th_name, results in thresholds_results.items():
        yearly = results["yearly_returns"]
        if len(yearly) > 0:
            yearly["threshold"] = th_name
            yearly_list.append(yearly)

    if yearly_list:
        yearly_df = pd.concat(yearly_list, ignore_index=True)
        yearly_file = output_dir / "yearly_returns.csv"
        yearly_df.to_csv(yearly_file, index=False, encoding="utf-8-sig")
        print(f"  ✓ {yearly_file}")

    print("\n[5] 生成图表...")
    try:
        for th_name, results in thresholds_results.items():
            returns_df = results["portfolio_returns"]
            if len(returns_df) > 0:
                long_returns = returns_df[returns_df["signal_type"] == "long"].copy()
                if len(long_returns) > 0:
                    long_returns = long_returns.sort_values("date")
                    long_returns["cumulative"] = (
                        1 + long_returns["mean_return"].dropna() / 100
                    ).cumprod()

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                    ax1.plot(
                        pd.to_datetime(long_returns["date"]),
                        long_returns["cumulative"],
                        label=f"策略 (阈值={th_name})",
                        linewidth=2,
                    )
                    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
                    ax1.set_title(f"ETF行业轮动策略净值曲线 (阈值={th_name})", fontsize=14)
                    ax1.set_xlabel("日期")
                    ax1.set_ylabel("净值")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    cumulative = long_returns["cumulative"]
                    running_max = cumulative.cummax()
                    drawdown = (cumulative - running_max) / running_max * 100
                    ax2.fill_between(
                        range(len(drawdown)), drawdown, 0, alpha=0.3, color="red"
                    )
                    ax2.plot(range(len(drawdown)), drawdown, color="red", linewidth=1)
                    ax2.set_title("回撤曲线", fontsize=14)
                    ax2.set_xlabel("交易日")
                    ax2.set_ylabel("回撤 (%)")
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    fig_file = output_dir / f"strategy_chart_{th_name}.png"
                    plt.savefig(fig_file, dpi=150, bbox_inches="tight")
                    print(f"  ✓ {fig_file}")
                    plt.close()
    except Exception as e:
        print(f"  ⚠️  图表生成失败: {e}")

    print("\n[6] 保存原始数据...")
    if len(percentile_data) > 0:
        percentile_file = output_dir / "percentile_data.csv"
        percentile_data.to_csv(percentile_file, index=False, encoding="utf-8-sig")
        print(f"  ✓ {percentile_file}")

    print("\n" + "=" * 70)
    print("所有结果已保存至 output 文件夹")
    print("=" * 70)


def main():
    print("=" * 70)
    print("ETF行业轮动策略 - 华泰证券研报复现")
    print("基于ETF资金流构建行业轮动策略")
    print("=" * 70)

    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"

    ensure_dir(data_dir)
    ensure_dir(output_dir)

    etf_df, index_returns = run_data_collection(data_dir, force_refresh=False)

    if index_returns is None or len(index_returns) == 0:
        print("\n错误: 无法获取指数数据,使用模拟数据继续...")
        dates = pd.date_range("2018-01-01", "2024-07-31", freq="D")
        all_sim_data = []
        for industry in INDUSTRY_INDEX_MAPPING.keys():
            np.random.seed(hash(industry) % (2**32))
            sim_df = pd.DataFrame({
                "trade_date": dates,
                "open": np.random.uniform(1000, 5000, len(dates)),
                "high": np.random.uniform(1000, 5000, len(dates)),
                "low": np.random.uniform(1000, 5000, len(dates)),
                "close": np.random.uniform(1000, 5000, len(dates)),
                "volume": np.random.randint(1000000, 100000000, len(dates)),
                "pct_change": np.random.randn(len(dates)) * 2,
                "industry": industry,
                "index_code": INDUSTRY_INDEX_MAPPING[industry],
                "index_name": INDUSTRY_NAME_MAPPING.get(INDUSTRY_INDEX_MAPPING[industry], ""),
            })
            all_sim_data.append(sim_df)
        index_returns = pd.concat(all_sim_data, ignore_index=True)

    industry_etf_flow = {}
    for industry in INDUSTRY_INDEX_MAPPING.keys():
        dates = pd.date_range("2018-01-01", "2024-07-31", freq="D")
        np.random.seed(hash(industry) % (2**32))
        industry_etf_flow[industry] = pd.DataFrame({
            "trade_date": dates,
            "net_flow": np.random.randn(len(dates)) * 1000000,
            "nav": np.random.uniform(0.9, 1.1, len(dates)),
            "vol": np.random.randint(1000, 100000, len(dates)),
        })
        industry_etf_flow[industry]["trade_date"] = pd.to_datetime(industry_etf_flow[industry]["trade_date"])

    weekly_flow, percentile_data, future_returns = run_flow_calculation(
        industry_etf_flow, index_returns
    )

    thresholds_results = run_strategy_backtest(
        percentile_data, future_returns, index_returns, output_dir
    )

    save_all_results(
        thresholds_results, weekly_flow, percentile_data, output_dir
    )

    print("\n" + "=" * 70)
    print("项目说明")
    print("=" * 70)
    print("""
1. 数据来源 (优先顺序):
   - ETF列表: Tushare > AkShare
   - 指数数据: Tushare > BaoStock > AkShare
   - ETF资金流: 使用模拟数据 (需从Wind/Choice获取真实数据)

2. 输出文件:
   - backtest_metrics.csv: 回测绩效指标
   - net_value_*.csv: 净值曲线数据
   - signals_long_*.csv: 交易信号
   - yearly_returns.csv: 年度收益
   - percentile_data.csv: 滚动分位数数据
   - strategy_chart_*.png: 策略图表

3. 重要提示:
   - ETF资金流数据为模拟数据,不能直接用于实盘
   - 要完整复现研报,需要补充真实的ETF申赎数据
    """)

    print("\n程序运行完成!")
    return thresholds_results


if __name__ == "__main__":
    results = main()
