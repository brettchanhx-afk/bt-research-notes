import sys
import os
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from source import (
    DataLoader, ETFRotationStrategy, BacktestEngine,
    PerformanceAnalyzer, ETF_POOL,
    BACKTEST_CONFIG, STRATEGY_CONFIG
)


def main():
    print("=" * 60)
    print("多ETF轮动 + 趋势震荡双策略回测")
    print("=" * 60)

    config = BACKTEST_CONFIG.copy()
    config['start_date'] = '2021-01-01'
    config['end_date'] = '2024-06-30'
    config['initial_cash'] = 1000000
    config['daily_injection'] = 400

    print(f"\n回测配置:")
    print(f"  开始日期: {config['start_date']}")
    print(f"  结束日期: {config['end_date']}")
    print(f"  初始资金: {config['initial_cash']:,} 元")
    print(f"  每日定投: {config['daily_injection']} 元")

    print("\n" + "=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)

    data_loader = DataLoader()

    print(f"\n正在加载ETF数据 (使用akshare)...")
    etf_data = data_loader.load_all_data(
        etf_codes=list(ETF_POOL.keys()),
        start_date=config['start_date'],
        end_date=config['end_date'],
        use_cache=False
    )

    print(f"\n成功加载 {len(etf_data)} 个ETF数据")

    if len(etf_data) == 0:
        print("错误: 没有成功加载任何ETF数据，请检查网络连接")
        return

    print("\n正在获取交易日历...")
    for attempt in range(3):
        trading_dates = data_loader.get_trading_dates(config['start_date'], config['end_date'])
        if trading_dates:
            break
        print(f"获取失败，{data_loader.retry_delay}秒后重试...")
        time.sleep(data_loader.retry_delay)

    print(f"交易日数量: {len(trading_dates)}")

    if len(trading_dates) == 0:
        print("警告: 无法获取交易日历，从ETF数据提取")
        for code, df in etf_data.items():
            if df is not None and not df.empty:
                trading_dates = df.index.tolist()
                print(f"从 {code} 提取 {len(trading_dates)} 个交易日")
                break

    if len(trading_dates) == 0:
        print("错误: 无法获取任何交易日数据")
        return

    print("\n" + "=" * 60)
    print("步骤2: 运行回测")
    print("=" * 60)

    strategy = ETFRotationStrategy(ETF_POOL, STRATEGY_CONFIG)

    backtest = BacktestEngine(
        etf_data_dict=etf_data,
        trading_dates=trading_dates,
        strategy=strategy,
        config=config
    )

    print("\n开始回测...")
    results = backtest.run()
    print("回测完成!")

    print("\n" + "=" * 60)
    print("步骤3: 分析结果")
    print("=" * 60)

    analyzer = PerformanceAnalyzer(results)

    if results['equity'].empty:
        print("错误: 回测没有产生任何结果")
        return

    metrics = analyzer.calculate_metrics()

    print("\n回测结果:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("-" * 40)

    print("\n" + "=" * 60)
    print("步骤4: 生成报告")
    print("=" * 60)

    report_path = analyzer.generate_report()
    print(f"\n报告已保存至: {report_path}")

    if not results['trades'].empty:
        print(f"\n交易记录: {len(results['trades'])} 笔")

    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
