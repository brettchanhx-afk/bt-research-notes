import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("1. 导入模块...")
try:
    from source import ETF_POOL, STRATEGY_CONFIG
    print(f"   ETF_POOL: {list(ETF_POOL.keys())[:3]}...")
    print(f"   STRATEGY_CONFIG: {STRATEGY_CONFIG}")
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n2. 生成模拟数据...")
import numpy as np
import pandas as pd

def generate_mock_data():
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', '2024-06-30', freq='B')
    data = {}
    for code in ['510050', '510300', '518880']:
        price = 1.0
        prices = [price]
        for _ in range(len(dates)-1):
            price *= (1 + np.random.normal(0.0003, 0.015))
            prices.append(price)
        df = pd.DataFrame({
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * len(dates),
            'money': [p * 1000000 for p in prices]
        }, index=dates)
        data[code] = df
    return data, dates.tolist()

etf_data, trading_dates = generate_mock_data()
print(f"   生成 {len(etf_data)} 个ETF数据")
print(f"   交易日数量: {len(trading_dates)}")

print("\n3. 初始化策略...")
try:
    from source import ETFRotationStrategy
    strategy = ETFRotationStrategy(ETF_POOL, STRATEGY_CONFIG)
    print("   策略初始化成功")
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n4. 初始化回测引擎...")
try:
    from source import BacktestEngine, BACKTEST_CONFIG
    backtest = BacktestEngine(
        etf_data_dict=etf_data,
        trading_dates=trading_dates,
        strategy=strategy,
        config=BACKTEST_CONFIG
    )
    print("   回测引擎初始化成功")
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n5. 运行回测...")
try:
    results = backtest.run()
    print(f"   回测完成!")
    print(f"   equity shape: {results['equity'].shape}")
    print(f"   trades count: {len(results['trades'])}")
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n6. 分析结果...")
try:
    from source import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(results)
    metrics = analyzer.calculate_metrics()
    print(f"   指标计算成功:")
    for k, v in metrics.items():
        print(f"     {k}: {v}")
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n7. 生成报告...")
try:
    report_path = analyzer.generate_report()
    print(f"   报告已保存: {report_path}")
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n完成!")
