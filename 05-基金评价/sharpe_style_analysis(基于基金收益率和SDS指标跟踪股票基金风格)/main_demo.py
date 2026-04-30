# -*- coding: utf-8 -*-
"""
威廉·夏普风格分析 - 演示版本（使用真实数据）
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

from source.factor import SharpeStyleModel, RollingStyleAnalyzer, compute_sds
from source.backtest import StyleDriftDetector
from source.plot import StyleVisualizer
from source.utils import calculate_performance_metrics

print("=" * 70)
print(" " * 15 + "William Sharpe Style Analysis")
print("=" * 70)
print()

# 基金配置
FUND_CODE = '021181'
FUND_NAME = '中欧价值精选混合A'
START_DATE = '20240101'
END_DATE = '20250427'

print(f"Fund Code: {FUND_CODE}")
print(f"Fund Name: {FUND_NAME}")
print(f"Period: {START_DATE} - {END_DATE}")
print()

# 使用AKShare获取数据
print("-" * 70)
print("[Data] Fetching fund and index data...")
print("-" * 70)

try:
    import akshare as ak
    
    # 获取基金净值
    print("\n[1] Fetching fund NAV...")
    fund_df = ak.fund_open_fund_info_em(symbol=FUND_CODE, indicator="单位净值走势")
    fund_df['日期'] = pd.to_datetime(fund_df['净值日期'])
    fund_df = fund_df[(fund_df['日期'] >= START_DATE) & (fund_df['日期'] <= END_DATE)]
    fund_df['nav'] = fund_df['单位净值'].astype(float)
    fund_df = fund_df.sort_values('日期').reset_index(drop=True)
    fund_df['daily_return'] = fund_df['nav'].pct_change()
    fund_returns = fund_df.set_index('日期')['daily_return'].dropna()
    print(f"[OK] Fund data: {len(fund_returns)} trading days")
    
    # 获取指数数据
    print("\n[2] Fetching style indices...")
    
    # 定义要获取的指数
    indices = {
        '000300': 'CSI300',
        '000905': 'CSI500', 
        '000918': 'CSI300_Growth',
        '000919': 'CSI300_Value'
    }
    
    index_returns = {}
    for code, name in indices.items():
        try:
            idx_df = ak.index_zh_a_hist(symbol=code, period="daily",
                                        start_date=START_DATE, end_date=END_DATE)
            idx_df['日期'] = pd.to_datetime(idx_df['日期'])
            idx_df = idx_df.sort_values('日期')
            idx_df['return'] = idx_df['收盘'].pct_change()
            returns = idx_df.set_index('日期')['return'].dropna()
            index_returns[f"{code}.SH"] = returns
            print(f"  [OK] {code}.SH ({name}): {len(returns)} days")
        except Exception as e:
            print(f"  [WARN] {code}.SH failed: {e}")
    
    # 对齐日期
    common_dates = fund_returns.index
    for code, ret in index_returns.items():
        common_dates = common_dates.intersection(ret.index)
    
    fund_returns = fund_returns.loc[common_dates]
    index_df = pd.DataFrame({k: v.loc[common_dates] for k, v in index_returns.items()})
    
    print(f"\n[OK] Aligned data: {len(common_dates)} common trading days")
    print(f"  Date range: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
    
except Exception as e:
    print(f"[ERROR] Data fetching failed: {e}")
    sys.exit(1)

print()

# ==================== Step 1: Overall Style Analysis ====================
print("-" * 70)
print("[Step 1] Overall Style Analysis")
print("-" * 70)

style_indices = list(index_returns.keys())
model = SharpeStyleModel(style_indices)
style_result = model.fit(fund_returns, index_df)

print("\nStyle Exposures:")
exposures_sorted = style_result['exposures'].sort_values(ascending=False)
for idx, exp in exposures_sorted.items():
    bar = "#" * int(exp * 50)
    name = indices.get(idx.replace('.SH', ''), idx)
    print(f"  {name:15s} ({idx}): {exp:.4f} ({exp:.2%}) {bar}")

print(f"\nModel Fit Statistics:")
print(f"  R-squared: {style_result['r_squared']:.4f}")
print(f"  Tracking Error: {style_result['tracking_error']:.4f} (annualized)")
print(f"  Residual Std: {np.std(style_result['residuals']):.4f}")

style_label = model.get_style_label()
print(f"\nStyle Label: {style_label}")

# 判断主要风格
main_exposure = exposures_sorted.iloc[0]
main_index = exposures_sorted.index[0]
print(f"  Primary Style: {main_index} ({main_exposure:.2%})")

print()

# ==================== Step 2: Style Drift Detection ====================
print("-" * 70)
print("[Step 2] Style Drift Detection (SDS Metric)")
print("-" * 70)

detector = StyleDriftDetector()
sub_period_df = detector.analyze_sub_periods(
    fund_returns, index_df, style_indices, n_periods=4
)

if len(sub_period_df) > 0:
    print("\nSub-period Analysis:")
    display_cols = ['period', 'start_date', 'end_date', 'n_days', 'r_squared', 'style_label']
    print(sub_period_df[display_cols].to_string(index=False))
    
    # 显示各期暴露
    print("\nStyle Exposures by Period:")
    exp_cols = [c for c in sub_period_df.columns if c.startswith('exp_')]
    for _, row in sub_period_df.iterrows():
        print(f"\n  Period {int(row['period'])} ({row['start_date']} to {row['end_date']}):")
        for col in exp_cols:
            idx_name = col.replace('exp_', '')
            val = row[col]
            print(f"    {idx_name}: {val:.4f}")
    
    sds_score = detector.compute_sds()
    print(f"\n" + "=" * 50)
    print(f"SDS Style Drift Metric: {sds_score:.4f}")
    print("=" * 50)
    
    if sds_score < 0.1:
        print("  -> Style highly stable [OK]")
        drift_level = "Low"
    elif sds_score < 0.2:
        print("  -> Style relatively stable")
        drift_level = "Medium-Low"
    elif sds_score < 0.3:
        print("  -> Style has some fluctuations [!]")
        drift_level = "Medium"
    else:
        print("  -> High style drift risk [!!]")
        drift_level = "High"
    
    print(f"\n  Drift Level: {drift_level}")
    
    # 检查风格漂移
    drift_result = detector.check_style_drift(sub_period_df)
    print(f"\n  Consistency Score: {drift_result['consistency_score']:.4f}")
    print(f"  Style Drift Detected: {drift_result['has_drift']}")
    
    if drift_result['has_drift']:
        print("\n  [!] Style drift events:")
        for event in drift_result.get('drift_periods', []):
            print(f"    Period {event['from_period']}: {event['from_style']} -> {event['to_style']}")
    else:
        print("\n  [OK] Style remains stable across all periods")
else:
    print("[WARN] Insufficient data for sub-period analysis")
    sds_score = None

print()

# ==================== Step 3: Rolling Window Analysis ====================
print("-" * 70)
print("[Step 3] Rolling Window Style Analysis")
print("-" * 70)

rolling_analyzer = RollingStyleAnalyzer(window=63, step=21)  # 3-month window, 1-month step
rolling_results = rolling_analyzer.analyze(fund_returns, index_df, style_indices)

print(f"\nCompleted {len(rolling_results)} rolling window analyses")
print(f"\nRecent 5 windows:")
exp_cols = [c for c in rolling_results.columns if c.startswith('exp_')]
available_cols = ['end_date', 'r_squared'] + exp_cols
available_cols = [c for c in available_cols if c in rolling_results.columns]
recent = rolling_results.tail(5)[available_cols]
print(recent.to_string(index=False))

print()

# ==================== Step 4: Performance Metrics ====================
print("-" * 70)
print("[Step 4] Performance Metrics")
print("-" * 70)

perf_metrics = calculate_performance_metrics(fund_returns)

print(f"\nReturn Metrics:")
print(f"  Total Return: {perf_metrics['total_return']:.2%}")
print(f"  Annualized Return: {perf_metrics['ann_return']:.2%}")

print(f"\nRisk Metrics:")
print(f"  Annualized Volatility: {perf_metrics['ann_volatility']:.2%}")
print(f"  Max Drawdown: {perf_metrics['max_drawdown']:.2%}")

print(f"\nRisk-Adjusted Metrics:")
print(f"  Sharpe Ratio: {perf_metrics['sharpe_ratio']:.4f}")
print(f"  Calmar Ratio: {perf_metrics['calmar_ratio']:.4f}")

print(f"\nOther Metrics:")
print(f"  Win Rate: {perf_metrics['win_rate']:.2%}")
print(f"  Profit/Loss Ratio: {perf_metrics['profit_loss_ratio']:.4f}")

print()

# ==================== Step 5: Summary Report ====================
print("=" * 70)
print(" " * 20 + "ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
Fund Information:
  Code: {FUND_CODE}
  Name: {FUND_NAME}
  Analysis Period: {common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}
  Trading Days: {len(common_dates)}

Style Analysis Results:
  Primary Style: {main_index} ({main_exposure:.2%})
  Style Label: {style_label}
  R-squared: {style_result['r_squared']:.4f} ({'Strong' if style_result['r_squared'] > 0.7 else 'Moderate' if style_result['r_squared'] > 0.4 else 'Weak'} explanatory power)
  Tracking Error: {style_result['tracking_error']:.2%}

Style Drift Assessment:
  SDS Metric: {sds_score:.4f}
  Drift Level: {drift_level}
  Status: {'Stable' if not drift_result.get('has_drift', False) else 'Drift Detected'}

Performance Summary:
  Annualized Return: {perf_metrics['ann_return']:.2%}
  Sharpe Ratio: {perf_metrics['sharpe_ratio']:.4f}
  Max Drawdown: {perf_metrics['max_drawdown']:.2%}
""")

print("=" * 70)
print("Analysis Complete!")
print("=" * 70)

# Save results
print("\n[Saving Results...]")
os.makedirs(f'output/{FUND_CODE}', exist_ok=True)

# Save to CSV
results_summary = pd.DataFrame([{
    'fund_code': FUND_CODE,
    'fund_name': FUND_NAME,
    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
    'start_date': common_dates[0].strftime('%Y-%m-%d'),
    'end_date': common_dates[-1].strftime('%Y-%m-%d'),
    'trading_days': len(common_dates),
    'primary_style': main_index,
    'primary_exposure': main_exposure,
    'style_label': style_label,
    'r_squared': style_result['r_squared'],
    'tracking_error': style_result['tracking_error'],
    'sds_score': sds_score,
    'drift_level': drift_level if sds_score else 'N/A',
    'ann_return': perf_metrics['ann_return'],
    'ann_volatility': perf_metrics['ann_volatility'],
    'sharpe_ratio': perf_metrics['sharpe_ratio'],
    'max_drawdown': perf_metrics['max_drawdown'],
}])

results_summary.to_csv(f'output/{FUND_CODE}/analysis_summary.csv', index=False, encoding='utf-8-sig')
print(f"[OK] Summary saved: output/{FUND_CODE}/analysis_summary.csv")

# Save style exposures
exposures_df = pd.DataFrame([style_result['exposures']])
exposures_df.to_csv(f'output/{FUND_CODE}/style_exposures.csv', index=False, encoding='utf-8-sig')
print(f"[OK] Exposures saved: output/{FUND_CODE}/style_exposures.csv")

# Save sub-period results if available
if len(sub_period_df) > 0:
    sub_period_df.to_csv(f'output/{FUND_CODE}/sub_period_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] Sub-period results saved: output/{FUND_CODE}/sub_period_analysis.csv")

print("\n[Done]")
