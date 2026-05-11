import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from source import (
    DataLoader, ETFRotationStrategy, BacktestEngine,
    PerformanceAnalyzer, ETF_POOL,
    BACKTEST_CONFIG, STRATEGY_CONFIG
)


def generate_mock_etf_data(etf_codes, start_date, end_date, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    etf_data = {}

    for code in etf_codes:
        base_price = np.random.uniform(0.5, 5.0)
        trend = np.random.uniform(-0.0002, 0.0005)
        volatility = np.random.uniform(0.01, 0.03)

        returns = np.random.normal(trend, volatility, len(dates))
        price_series = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'open': price_series * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'high': price_series * (1 + np.random.uniform(0, 0.01, len(dates))),
            'low': price_series * (1 - np.random.uniform(0, 0.01, len(dates))),
            'close': price_series,
            'volume': np.random.uniform(1e6, 1e8, len(dates)),
            'money': price_series * np.random.uniform(1e6, 1e8, len(dates))
        }, index=dates)

        etf_data[code] = df

    return etf_data, dates.tolist()


def main():
    print("=" * 60)
    print("多ETF轮动 + 趋势震荡双策略回测 (模拟数据)")
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
    print(f"  注意: 使用模拟数据进行演示")

    print("\n" + "=" * 60)
    print("步骤1: 生成模拟ETF数据")
    print("=" * 60)

    etf_data, trading_dates = generate_mock_etf_data(
        etf_codes=list(ETF_POOL.keys()),
        start_date=config['start_date'],
        end_date=config['end_date']
    )

    print(f"\n成功生成 {len(etf_data)} 个ETF模拟数据")
    print(f"交易日数量: {len(trading_dates)}")

    for code in list(etf_data.keys())[:3]:
        print(f"  {code}: {len(etf_data[code])} 条数据")

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
    print("步骤4: 生成报告和图表")
    print("=" * 60)

    _, report_path = analyzer.generate_report()
    print(f"\n报告已保存至: {report_path}")

    if not results['trades'].empty:
        print(f"\n交易记录: {len(results['trades'])} 笔")
        print("\n交易记录预览:")
        print(results['trades'].head(10).to_string())

    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)
    print("\n提示: 如需使用真实数据，请在您本地环境中运行")
    print("      pip install -r requirements.txt")
    print("      jupyter notebook ipynb/backtest_demo.ipynb")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
