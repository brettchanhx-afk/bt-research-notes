import os
import sys
from datetime import datetime
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source import (
    CSVBacktest,
    Visualizer,
    RESULTS_DIR,
    BACKTEST_CONFIG,
)

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("Macro Factor Asset Allocation - Backtest Engine")
    print("Using Real CSV Data")
    print("=" * 60)

    start_date = BACKTEST_CONFIG["start_date"]
    end_date = BACKTEST_CONFIG["end_date"]
    initial_capital = BACKTEST_CONFIG["initial_capital"]

    print(f"\nBacktest Period: {start_date} to {end_date}")
    print(f"Initial Capital: {initial_capital:,.2f}")

    backtest = CSVBacktest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=BACKTEST_CONFIG["commission_rate"],
        stamp_tax=BACKTEST_CONFIG["stamp_tax"],
        rebalance_freq="monthly",
    )

    print("\n" + "-" * 40)
    print("Running Strategy Backtest...")
    print("-" * 40)

    results = backtest.run_backtest(use_macro_view=False)

    if results:
        print("\n" + "-" * 40)
        print("Backtest Results (Risk Parity Strategy)")
        print("-" * 40)
        print(f"Total Return: {results.get('total_return', 0) * 100:.2f}%")
        print(f"Annualized Return: {results.get('annualized_return', 0) * 100:.2f}%")
        print(f"Annualized Volatility: {results.get('annualized_volatility', 0) * 100:.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Win Rate: {results.get('win_rate', 0) * 100:.2f}%")

    print("\n" + "-" * 40)
    print("Running Benchmark Backtest...")
    print("-" * 40)

    benchmark_values = backtest.run_benchmark_backtest()

    if len(benchmark_values) > 0:
        benchmark_return = (benchmark_values.iloc[-1] / benchmark_values.iloc[0]) - 1
        print(f"Benchmark (Equal Weight) Total Return: {benchmark_return * 100:.2f}%")

    visualizer = Visualizer(output_dir=str(RESULTS_DIR))

    if results and "portfolio_values" in results:
        portfolio_values = results["portfolio_values"]
        weights_history = results.get("weights_history")

        print("\n" + "-" * 40)
        print("Generating Visualization Charts...")
        print("-" * 40)

        visualizer.generate_backtest_report(
            backtest_results=results,
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            weights_history=weights_history,
            save_dir=str(RESULTS_DIR),
        )

    output_results_path = RESULTS_DIR / "backtest_results.csv"
    backtest.save_results(str(output_results_path))

    print("\n" + "=" * 60)
    print("Backtest Completed Successfully!")
    print("=" * 60)

    return results, benchmark_values


if __name__ == "__main__":
    main()
