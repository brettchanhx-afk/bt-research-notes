import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from source import (
    load_csv_data, Backtest,
    plot_cumulative_returns, plot_drawdown, plot_metrics_comparison,
    plot_weights_heatmap, plot_annual_returns, plot_rolling_sharpe,
    OUTPUT_DIR, MOMENTUM_PARAMS
)

print("=" * 70)
print("Momentum-Enhanced Risk Parity Strategy - Full Backtest")
print("=" * 70)

print("\n[Step 1] Loading CSV data...")
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'momentum_enhanced_10assets_price_2013.csv')
print(f"Data file: {csv_path}")

data_dict = load_csv_data(csv_path)
print(f"Loaded data for {len(data_dict)} assets")
for name, df in data_dict.items():
    valid_prices = df['close'].dropna()
    print(f"  - {name}: {len(valid_prices)} price records, "
          f"period: {valid_prices.index[0].strftime('%Y-%m-%d')} ~ {valid_prices.index[-1].strftime('%Y-%m-%d')}")

print("\n[Step 2] Building daily returns...")
daily_returns = pd.DataFrame({k: v['returns'] for k, v in data_dict.items()})
daily_returns = daily_returns.dropna(how='all')
valid_returns = daily_returns.dropna()
print(f"Valid daily returns: {valid_returns.shape}")
print(f"Data period: {valid_returns.index[0].strftime('%Y-%m-%d')} ~ {valid_returns.index[-1].strftime('%Y-%m-%d')}")

print("\n[Step 3] Initializing backtest framework...")
bt = Backtest(valid_returns, transaction_cost=0.0005)
print(f"Transaction cost: 0.05% (bilateral)")

print("\n[Step 4] Running Risk Parity strategy backtest...")
try:
    bt.backtest_risk_parity(lookback_days=126, name='RiskParity')
    print("  ✓ Risk Parity backtest completed")
except Exception as e:
    print(f"  ✗ Risk Parity backtest failed: {e}")

print("\n[Step 5] Running Momentum Risk Budget strategy backtest (different k values)...")
for k in MOMENTUM_PARAMS['k_values']:
    try:
        bt.backtest_momentum_risk_budget(k=k, name=f'Momentum_k{k}')
        print(f"  ✓ Momentum Risk Budget (k={k}) backtest completed")
    except Exception as e:
        print(f"  ✗ Momentum Risk Budget (k={k}) backtest failed: {e}")

print("\n[Step 6] Backtest results summary...")
print("=" * 70)
summary = bt.get_metrics_summary()
print(summary.to_string(index=False))
print("=" * 70)

print("\n[Step 7] Generating visualization charts...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    strategies = ['RiskParity'] + [f'Momentum_k{k}' for k in MOMENTUM_PARAMS['k_values']]
    strategies = [s for s in strategies if s in bt.portfolio_returns]

    plot_cumulative_returns(
        bt,
        strategies=strategies,
        title='Cumulative Returns: Risk Parity vs Momentum Risk Budget',
        save_path=os.path.join(OUTPUT_DIR, 'cumulative_returns.png'),
        show=False
    )
    print("  ✓ Cumulative returns chart saved")

    plot_drawdown(
        bt,
        strategies=strategies,
        title='Drawdown Analysis',
        save_path=os.path.join(OUTPUT_DIR, 'drawdown.png'),
        show=False
    )
    print("  ✓ Drawdown chart saved")

    plot_metrics_comparison(
        bt,
        strategies=strategies,
        title='Strategy Metrics Comparison',
        save_path=os.path.join(OUTPUT_DIR, 'metrics_comparison.png'),
        show=False
    )
    print("  ✓ Metrics comparison chart saved")

    plot_annual_returns(
        bt,
        strategies=strategies,
        title='Annual Returns Comparison',
        save_path=os.path.join(OUTPUT_DIR, 'annual_returns.png'),
        show=False
    )
    print("  ✓ Annual returns chart saved")

    plot_rolling_sharpe(
        bt,
        strategies=strategies,
        window=252,
        title='Rolling Sharpe Ratio (252-day)',
        save_path=os.path.join(OUTPUT_DIR, 'rolling_sharpe.png'),
        show=False
    )
    print("  ✓ Rolling Sharpe ratio chart saved")

    for name in bt.weights_history.keys():
        safe_name = name.replace(' ', '_').replace('=', '')
        plot_weights_heatmap(
            bt.weights_history[name],
            title=f'Portfolio Weights - {name}',
            save_path=os.path.join(OUTPUT_DIR, f'weights_{safe_name}.png'),
            show=False
        )
        print(f"  ✓ Weights heatmap {name} saved")

except Exception as e:
    print(f"  ✗ Visualization generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[Step 8] Saving backtest results to CSV...")
try:
    returns_df = bt.get_all_returns()
    returns_df.to_csv(os.path.join(OUTPUT_DIR, 'portfolio_returns.csv'))
    print("  ✓ Strategy returns data saved")

    metrics_data = []
    for name, metrics in bt.metrics.items():
        metrics_data.append({
            'Strategy': name,
            'Total Return': f"{metrics['total_return']:.4f}",
            'Annualized Return': f"{metrics['annualized_return']:.4f}",
            'Annualized Volatility': f"{metrics['annualized_volatility']:.4f}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.4f}",
            'Max Drawdown': f"{metrics['max_drawdown']:.4f}",
            'Calmar Ratio': f"{metrics['calmar_ratio']:.4f}",
            'Win Rate': f"{metrics['win_rate']:.4f}"
        })
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), index=False)
    print("  ✓ Performance metrics data saved")

except Exception as e:
    print(f"  ✗ Data saving failed: {e}")

print("\n" + "=" * 70)
print("Backtest completed!")
print(f"All results saved to: {OUTPUT_DIR}")
print("=" * 70)

print("\nAvailable output files:")
for f in os.listdir(OUTPUT_DIR):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  - {f} ({size/1024:.1f} KB)")
