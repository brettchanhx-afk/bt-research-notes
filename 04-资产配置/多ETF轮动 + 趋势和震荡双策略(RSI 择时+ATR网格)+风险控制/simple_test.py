import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Starting test...")

try:
    print("Importing ETF_POOL...")
    from source import ETF_POOL
    print(f"Success: {list(ETF_POOL.keys())[:3]}")
except Exception as e:
    print(f"Failed to import ETF_POOL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Importing BacktestEngine...")
try:
    from source import BacktestEngine
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Importing ETFRotationStrategy...")
try:
    from source import ETFRotationStrategy
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Creating strategy...")
try:
    strategy = ETFRotationStrategy(ETF_POOL)
    print(f"Success: {strategy}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Generating mock data...")
try:
    import numpy as np
    import pandas as pd

    dates = pd.date_range('2021-01-01', '2024-06-30', freq='B')
    np.random.seed(42)
    data = {}
    for code in ['510050']:
        price = 1.0
        prices = [price]
        for _ in range(len(dates)-1):
            price *= (1 + np.random.normal(0.0003, 0.015))
            prices.append(price)
        df = pd.DataFrame({
            'close': prices,
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.98 for p in prices],
            'volume': [1000000] * len(dates),
            'money': [p * 1000000 for p in prices]
        }, index=dates)
        data[code] = df
    print(f"Generated {len(data)} datasets with {len(dates)} dates")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Creating backtest engine...")
try:
    from source import BACKTEST_CONFIG
    backtest = BacktestEngine(
        etf_data_dict=data,
        trading_dates=dates.tolist(),
        strategy=strategy,
        config=BACKTEST_CONFIG
    )
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Running backtest...")
try:
    results = backtest.run()
    print(f"Success: equity shape = {results['equity'].shape}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Creating analyzer...")
try:
    from source import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(results)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Calculating metrics...")
try:
    metrics = analyzer.calculate_metrics()
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Generating report...")
try:
    _, report_path = analyzer.generate_report()
    print(f"Report saved to: {report_path}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Test completed successfully!")
