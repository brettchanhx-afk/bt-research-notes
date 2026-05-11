# -*- coding: utf-8 -*-
"""
Main program to run backtest
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from source.data_loader import fetch_global_index_data
from source.models import get_model_weights
from source.backtest import run_backtest, calculate_metrics
from config import BACKTEST_CONFIG

def main():
    print("="*60)
    print("HPCRP Global Asset Allocation Backtest")
    print("="*60)
    
    # 1. Get data
    print("\n[1] Fetching global index data...")
    returns = fetch_global_index_data()
    print(f"  Data: {len(returns)} days, {len(returns.columns)} indices")
    
    # 2. Define weight function
    def weights_func(model_name, hist_data):
        if model_name == 'HPCRP':
            return get_model_weights(model_name, hist_data, half_life=BACKTEST_CONFIG['HALF_LIFE'])
        return get_model_weights(model_name, hist_data)
    
    # 3. Run backtest for each model
    models = ['EW', 'EV', 'MV', 'MD', 'RP', 'PCRP', 'HPCRP']
    results = {}
    
    for model in models:
        print(f"\n[2] Backtesting {model}...")
        result = run_backtest(
            returns,
            weights_func,
            model,
            rebalance_freq=BACKTEST_CONFIG['REBALANCE_FREQ'],
            window=BACKTEST_CONFIG['WINDOW'],
            start_date=BACKTEST_CONFIG['BACKTEST_START'],
            end_date=BACKTEST_CONFIG['BACKTEST_END']
        )
        result['metrics'] = calculate_metrics(result['returns'], result['nav'])
        results[model] = result
        
        m = result['metrics']
        print(f"  Annual Return: {m['annual_return']:.2%}")
        print(f"  Max Drawdown: {m['max_drawdown']:.2%}")
        print(f"  Sharpe: {m['sharpe_ratio']:.3f}")
        print(f"  Calmar: {m['calmar_ratio']:.3f}")
    
    # 4. Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    rows = []
    for model, result in results.items():
        m = result['metrics']
        rows.append({
            'Model': model,
            'AnnReturn': f"{m['annual_return']:.2%}",
            'AnnVol': f"{m['annual_vol']:.2%}",
            'Sharpe': f"{m['sharpe_ratio']:.3f}",
            'MaxDD': f"{m['max_drawdown']:.2%}",
            'Calmar': f"{m['calmar_ratio']:.3f}"
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nDone!")

if __name__ == '__main__':
    main()