# -*- coding: utf-8 -*-
"""
William Sharpe Style Analysis - Main Entry

Usage:
    python main.py <fund_code> [--start_date START] [--end_date END] [--mock]

Examples:
    python main.py 021181
    python main.py 021181 --start_date 20230101 --end_date 20241231
    python main.py 021181 --mock  # Use mock data
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

from source.data_loader import StyleDataLoader
from source.factor import SharpeStyleModel, RollingStyleAnalyzer
from source.backtest import StyleDriftDetector, StyleBacktest
from source.plot import StyleVisualizer
from source.utils import (generate_style_report, save_results_to_json,
                          calculate_performance_metrics, format_date)


def parse_args():
    parser = argparse.ArgumentParser(
        description='William Sharpe Style Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py 021181
  python main.py 021181 --mock
  python main.py 021181 --start_date 20230101 --end_date 20241231
        """
    )
    
    parser.add_argument('fund_code', type=str, help='Fund code, e.g. 021181')
    parser.add_argument('--fund_name', type=str, default='', help='Fund name')
    parser.add_argument('--start_date', type=str, 
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y%m%d'),
                        help='Start date (YYYYMMDD), default 1 year ago')
    parser.add_argument('--end_date', type=str, 
                        default=datetime.now().strftime('%Y%m%d'),
                        help='End date (YYYYMMDD), default today')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--style_indices', type=str, nargs='+',
                        default=['000300.SH', '000905.SH', '000918.SH', '000919.SH'],
                        help='Style indices list')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print(" " * 15 + "William Sharpe Style Analysis")
    print("=" * 70)
    print()
    print(f"Fund Code: {args.fund_code}")
    print(f"Period: {args.start_date} - {args.end_date}")
    print(f"Style Indices: {', '.join(args.style_indices)}")
    print()
    
    os.makedirs(args.output, exist_ok=True)
    fund_output_dir = os.path.join(args.output, args.fund_code)
    os.makedirs(fund_output_dir, exist_ok=True)
    
    loader = StyleDataLoader()
    
    if args.mock:
        print("[INFO] Using mock data...")
        fund_df, index_df = loader.create_mock_data(
            fund_code=args.fund_code,
            periods=252,
            style_indices=args.style_indices
        )
        fund_returns = fund_df.set_index('日期')['daily_return']
    else:
        try:
            print("[INFO] Fetching fund NAV data...")
            fund_df = loader.get_fund_nav(args.fund_code, args.start_date, args.end_date)
            fund_returns = fund_df.set_index('日期')['daily_return']
            print(f"[OK] Got {len(fund_df)} NAV records")
        except Exception as e:
            print(f"[ERROR] Failed to fetch fund data: {e}")
            print("[INFO] Switching to mock data...")
            fund_df, index_df = loader.create_mock_data(
                fund_code=args.fund_code,
                periods=252,
                style_indices=args.style_indices
            )
            fund_returns = fund_df.set_index('日期')['daily_return']
    
    try:
        print("[INFO] Fetching index data...")
        if args.mock or 'index_df' in locals():
            pass
        else:
            index_df = loader.get_index_returns(
                args.style_indices, args.start_date, args.end_date
            )
        print(f"[OK] Got {len(index_df)} trading days")
    except Exception as e:
        print(f"[ERROR] Failed to fetch index data: {e}")
        print("[INFO] Using mock data...")
        _, index_df = loader.create_mock_data(
            fund_code=args.fund_code,
            periods=len(fund_returns),
            style_indices=args.style_indices
        )
    
    common_dates = fund_returns.index.intersection(index_df.index)
    fund_returns = fund_returns.loc[common_dates]
    index_df = index_df.loc[common_dates]
    
    print(f"[OK] Aligned data: {len(common_dates)} trading days")
    print()
    
    # Step 1: Overall Style Analysis
    print("-" * 70)
    print("[Step 1] Overall Style Analysis")
    print("-" * 70)
    
    model = SharpeStyleModel(args.style_indices)
    style_result = model.fit(fund_returns, index_df)
    
    print("\nStyle Exposures:")
    for idx, exp in style_result['exposures'].sort_values(ascending=False).items():
        bar = "#" * int(exp * 50)
        print(f"  {idx:15s}: {exp:.4f} ({exp:.2%}) {bar}")
    
    print(f"\nModel Fit:")
    print(f"  R-squared = {style_result['r_squared']:.4f}")
    print(f"  Tracking Error = {style_result['tracking_error']:.4f} (annualized)")
    
    style_label = model.get_style_label()
    print(f"\nStyle Label: {style_label}")
    print()
    
    # Step 2: Style Drift Detection
    print("-" * 70)
    print("[Step 2] Style Drift Detection (SDS Metric)")
    print("-" * 70)
    
    detector = StyleDriftDetector()
    sub_period_df = detector.analyze_sub_periods(
        fund_returns, index_df, args.style_indices, n_periods=4
    )
    
    print("\nSub-period Analysis:")
    print(sub_period_df.to_string(index=False))
    
    sds_score = detector.compute_sds()
    print(f"\nSDS Style Drift Metric: {sds_score:.4f}")
    
    if sds_score < 0.1:
        print("  -> Style highly stable [OK]")
    elif sds_score < 0.2:
        print("  -> Style relatively stable")
    elif sds_score < 0.3:
        print("  -> Style has some fluctuations [!]")
    else:
        print("  -> High style drift risk [!!]")
    
    drift_result = detector.check_style_drift(sub_period_df)
    print(f"\n{drift_result['analysis']}")
    print()
    
    # Step 3: Visualization
    print("-" * 70)
    print("[Step 3] Generating Visualizations")
    print("-" * 70)
    
    visualizer = StyleVisualizer()
    
    fig1 = visualizer.plot_style_exposure(
        style_result['exposures'],
        title=f"{args.fund_code} Style Exposure Analysis",
        save_path=os.path.join(fund_output_dir, 'style_exposure.png')
    )
    print("[OK] Style exposure chart saved")
    
    fig2 = visualizer.plot_return_comparison(
        fund_returns,
        style_result['fitted_returns'],
        title=f"{args.fund_code} Actual vs Fitted Returns",
        save_path=os.path.join(fund_output_dir, 'return_comparison.png')
    )
    print("[OK] Return comparison chart saved")
    
    fig3 = visualizer.plot_sds_analysis(
        sub_period_df, sds_score,
        save_path=os.path.join(fund_output_dir, 'sds_analysis.png')
    )
    print("[OK] SDS analysis chart saved")
    
    print()
    
    # Step 4: Performance Metrics
    print("-" * 70)
    print("[Step 4] Performance Metrics")
    print("-" * 70)
    
    perf_metrics = calculate_performance_metrics(fund_returns)
    
    print(f"\nAnnualized Return: {perf_metrics['ann_return']:.2%}")
    print(f"Annualized Volatility: {perf_metrics['ann_volatility']:.2%}")
    print(f"Sharpe Ratio: {perf_metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {perf_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {perf_metrics['win_rate']:.2%}")
    print()
    
    # Step 5: Generate Report
    print("-" * 70)
    print("[Step 5] Generating Analysis Report")
    print("-" * 70)
    
    report_text = generate_style_report(
        args.fund_code, args.fund_name,
        style_result, drift_result, perf_metrics
    )
    
    report_path = os.path.join(fund_output_dir, 'style_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[OK] Text report saved: {report_path}")
    
    results = {
        'fund_code': args.fund_code,
        'fund_name': args.fund_name,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'period': {'start': args.start_date, 'end': args.end_date},
        'style_analysis': style_result,
        'drift_analysis': drift_result,
        'performance': perf_metrics
    }
    
    json_path = os.path.join(fund_output_dir, 'results.json')
    save_results_to_json(results, json_path)
    
    print()
    print("=" * 70)
    print("Analysis Complete!")
    print(f"Output Directory: {os.path.abspath(fund_output_dir)}")
    print("=" * 70)
    
    print("\n" + report_text)


if __name__ == '__main__':
    main()
