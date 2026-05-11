"""
使用真实数据运行宏观因子资产配置框架
"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print('=' * 70)
print('Macro Factor Asset Allocation Framework - Real Data Run')
print('=' * 70)

from config import ALL_ASSETS, MACRO_FACTORS, BACKTEST_CONFIG, OUTPUT_DIR, TUSHARE_TOKEN, TUSHARE_URL
import tushare as ts

# 初始化tushare
token = TUSHARE_TOKEN
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = TUSHARE_URL

print(f'\n[Step 1] Tushare initialized')

# 获取资产数据
def get_index_data(ts_code, start_date, end_date):
    """获取指数数据"""
    try:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        df['return'] = df['pct_chg'] / 100
        return df[['close', 'return']]
    except Exception as e:
        print(f'  Error fetching {ts_code}: {e}')
        return None

def get_stock_data(ts_code, start_date, end_date):
    """获取股票数据"""
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        df['return'] = df['close'].pct_change()
        return df[['close', 'return']]
    except Exception as e:
        print(f'  Error fetching {ts_code}: {e}')
        return None

def resample_to_monthly(data):
    """转换为月度数据"""
    monthly = data.resample('M').last()
    first_price = data['close'].resample('M').first()
    monthly['return'] = (monthly['close'] / first_price) - 1
    monthly.loc[monthly['return'].isna(), 'return'] = 0
    return monthly[['close', 'return']].dropna()

# 资产代码映射
asset_codes = {
    '沪深300': ('index', '000300.SH'),
    '中证500': ('index', '000905.SH'),
    '中证转债': ('stock', '000832.SH'),
    '恒生指数': ('index', 'HSI.HI'),
}

start_date = '2015-01-01'
end_date = '2023-05-31'

print(f'\n[Step 2] Fetching real market data from Tushare...')
print(f'  Period: {start_date} to {end_date}')

all_data = {}
for asset_name, (asset_type, code) in asset_codes.items():
    print(f'  Fetching {asset_name} ({code})...', end=' ')
    if asset_type == 'index':
        df = get_index_data(code, start_date, end_date)
    else:
        df = get_stock_data(code, start_date, end_date)

    if df is not None and not df.empty:
        monthly = resample_to_monthly(df)
        all_data[asset_name] = monthly['return']
        print(f'OK - {len(monthly)} months')
    else:
        print(f'FAILED')

# 检查获取到的数据
print(f'\n[Step 3] Data summary:')
print(f'  Successfully fetched {len(all_data)} assets')

if len(all_data) < 3:
    print('  WARNING: Not enough real data, falling back to simulated data')
    use_simulated = True
else:
    use_simulated = False
    asset_returns = pd.DataFrame(all_data)
    print(f'  Data shape: {asset_returns.shape}')
    print(f'  Date range: {asset_returns.index[0]} to {asset_returns.index[-1]}')

if use_simulated:
    # 使用模拟数据
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2023-05-31', freq='M')
    n = len(dates)
    asset_returns = pd.DataFrame({
        '沪深300': np.random.randn(n) * 0.025 + 0.005,
        '中证500': np.random.randn(n) * 0.03 + 0.003,
        '中债国债': np.random.randn(n) * 0.008 + 0.002,
        '中债企业债': np.random.randn(n) * 0.01 + 0.002,
        '中证转债': np.random.randn(n) * 0.02 + 0.003,
        '南华工业品': np.random.randn(n) * 0.03 + 0.004,
        '南华农产品': np.random.randn(n) * 0.025 + 0.002,
        '布伦特原油': np.random.randn(n) * 0.04 + 0.001,
        '沪金': np.random.randn(n) * 0.025 + 0.003,
        '美元兑人民币': np.random.randn(n) * 0.015 + 0.001,
        '恒生指数': np.random.randn(n) * 0.03 + 0.002,
    }, index=dates)
    print('  Using SIMULATED data')

# 构建宏观因子
from macro_factors import MacroFactorBuilder
print(f'\n[Step 4] Building macro factors...')
factor_builder = MacroFactorBuilder(n_factors=6)
factor_returns = factor_builder.construct_all_factors(asset_returns)
print(f'  Factor returns shape: {factor_returns.shape}')
print(f'  Factors: {factor_returns.columns.tolist()}')

# 计算因子暴露
from factor_exposure import FactorExposureWithPrior
print(f'\n[Step 5] Computing factor exposures...')
exposure_calc = FactorExposureWithPrior(alpha=0.01)
exposure_matrix = exposure_calc.fit(asset_returns, factor_returns)
print(f'  Exposure matrix shape: {exposure_matrix.shape}')

# 运行回测
from backtest import BacktestEngine, BacktestResultAnalyzer
print(f'\n[Step 6] Running backtest...')
backtest_engine = BacktestEngine(
    start_date='2015-01-01',
    end_date='2023-05-31',
    rebalance_freq='monthly',
    factor_deviation=0.05
)

all_results = {}
for factor in MACRO_FACTORS:
    print(f'  Testing {factor} factor deviation...')
    result = backtest_engine.run_factor_deviation_backtest(
        asset_returns, factor_returns, target_factor=factor
    )
    all_results[factor] = result

# 生成汇总报告
print(f'\n[Step 7] Generating summary report...')
analyzer = BacktestResultAnalyzer(all_results)
summary = analyzer.generate_summary_report(factor_returns)

print('\n' + '=' * 70)
print('Backtest Results Summary')
print('=' * 70)
print(summary.to_string(index=False))

# 保存结果
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_file = os.path.join(OUTPUT_DIR, 'factor_strategy_summary_real.csv')
summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f'\n[Results saved to: {summary_file}]')

print('\n' + '=' * 70)
print('Framework execution completed successfully!')
print('=' * 70)