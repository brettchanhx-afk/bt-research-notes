"""
使用本地CSV数据运行宏观因子资产配置框架
"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os

print('=' * 70)
print('Macro Factor Asset Allocation Framework - Real Data Run')
print('=' * 70)

DATA_DIR = 'd:/Documents/trae_projects/macro_factor_asset_alloc/data'
OUTPUT_DIR = 'd:/Documents/trae_projects/macro_factor_asset_alloc/output'

def load_seven_assets_price():
    """加载7种资产价格数据"""
    print('\n[Step 1] Loading seven assets price data...')
    file_path = os.path.join(DATA_DIR, 'seven_assets_price_2013_PCA.csv')

    df = pd.read_csv(file_path, index_col=0, encoding='gbk', skiprows=[1])
    df.index = pd.to_datetime(df.index, format='%Y/%m/%d')

    df = df.replace('--', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(how='all')

    price_cols = {
        '000300.SH': '沪深300',
        '000905.SH': '中证500',
        'CBA00601.CB': '中债国债',
        'CBA02001.CB': '中债企业债',
        'NHCI.SL': '南华商品',
        'BRN0Y.ICE': '布伦特原油',
    }

    df = df.rename(columns=price_cols)

    asset_returns = df.pct_change()
    asset_returns = asset_returns.dropna(how='all')

    print(f'  Assets: {list(asset_returns.columns)}')
    print(f'  Date range: {asset_returns.index[0]} to {asset_returns.index[-1]}')
    print(f'  Shape: {asset_returns.shape}')

    return df, asset_returns

def load_original_macro_factors():
    """加载原始宏观因子数据"""
    print('\n[Step 2] Loading original macro factors...')
    file_path = os.path.join(DATA_DIR, 'original_macro_factor_2013.csv')

    df = pd.read_csv(file_path, index_col=0, encoding='utf-8')
    df.index = pd.to_datetime(df.index, format='%Y-%m', errors='coerce')

    df = df.replace('', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(how='all', subset=df.columns[:1])
    df = df[df.index.notna()]

    print(f'  Factors: {list(df.columns)}')
    print(f'  Date range: {df.index[0]} to {df.index[-1]}')
    print(f'  Shape: {df.shape}')

    return df

def load_high_frequency_macro_factors():
    """加载高频宏观因子数据"""
    print('\n[Step 3] Loading high frequency macro factors...')
    file_path = os.path.join(DATA_DIR, 'high_frequency_macro_factor_portfolio.csv')

    df = pd.read_csv(file_path, index_col=0, encoding='utf-8')

    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]

    df = df.replace('', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(how='all')

    print(f'  Assets: {list(df.columns)}')
    print(f'  Date range: {df.index[0]} to {df.index[-1]}')
    print(f'  Shape: {df.shape}')

    return df

def resample_to_monthly(daily_data):
    """日频数据转为月频"""
    monthly_prices = daily_data.resample('M').last()
    monthly_returns = daily_data.resample('M').apply(lambda x: (1 + x).prod() - 1)
    return monthly_prices, monthly_returns

def build_pca_factors(asset_returns, n_factors=6):
    """使用PCA从资产收益中提取因子"""
    from sklearn.decomposition import PCA

    returns_clean = asset_returns.fillna(0)

    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns_clean)

    factor_names = [f'PC{i+1}' for i in range(n_factors)]
    factor_df = pd.DataFrame(factor_returns, index=asset_returns.index, columns=factor_names)

    explained_var = pca.explained_variance_ratio_
    print(f'\n  PCA Explained Variance:')
    for i, var in enumerate(explained_var):
        print(f'    PC{i+1}: {var:.2%}')

    return factor_df, pca

def standardize_macro_factors(macro_df):
    """标准化宏观因子"""
    from sklearn.preprocessing import StandardScaler

    macro_clean = macro_df.fillna(method='ffill').fillna(0)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(macro_clean)

    return pd.DataFrame(standardized, index=macro_df.index, columns=macro_df.columns)

def run_backtest_with_real_data():
    """使用真实数据运行回测"""
    price_df, asset_returns = load_seven_assets_price()

    print('\n[Step 4] Resampling to monthly data...')
    monthly_prices, monthly_returns = resample_to_monthly(asset_returns)
    monthly_returns = monthly_returns.dropna(how='all')
    monthly_returns = monthly_returns.fillna(0)

    monthly_returns = monthly_returns.loc['2013-06':'2023-05']

    print(f'  Monthly returns shape: {monthly_returns.shape}')
    print(f'  Date range: {monthly_returns.index[0]} to {monthly_returns.index[-1]}')

    print('\n[Step 5] Building PCA factors from asset returns...')
    pca_factors, pca_model = build_pca_factors(monthly_returns, n_factors=6)

    print('\n[Step 6] Loading macro factors...')
    original_factors = load_original_macro_factors()
    original_factors = original_factors.loc['2013-06':'2023-05']

    macro_factors = standardize_macro_factors(original_factors)

    asset_names = ['沪深300', '中证500', '中债国债', '中债企业债', '南华商品', '布伦特原油']
    asset_returns_final = monthly_returns[asset_names].fillna(0)

    factor_returns_final = pca_factors.loc[asset_returns_final.index]

    print(f'\n  Asset returns shape: {asset_returns_final.shape}')
    print(f'  Factor returns shape: {factor_returns_final.shape}')
    print(f'  Common date range: {asset_returns_final.index[0]} to {asset_returns_final.index[-1]}')

    print('\n[Step 7] Computing factor exposures...')
    from factor_exposure import FactorExposureWithPrior
    exposure_calc = FactorExposureWithPrior(alpha=0.01)

    exposure_matrix = exposure_calc.fit(asset_returns_final, factor_returns_final)
    print(f'  Exposure matrix shape: {exposure_matrix.shape}')

    print('\n[Step 8] Running backtest...')
    from backtest import BacktestEngine, BacktestResultAnalyzer

    backtest_engine = BacktestEngine(
        start_date='2013-06-01',
        end_date='2023-05-31',
        rebalance_freq='monthly',
        factor_deviation=0.05
    )

    MACRO_FACTORS = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']

    all_results = {}
    for factor in MACRO_FACTORS:
        print(f'  Testing {factor} factor deviation...')

        result = backtest_engine.run_factor_deviation_backtest(
            asset_returns_final,
            factor_returns_final[[factor]],
            target_factor=factor
        )
        all_results[factor] = result

    print('\n[Step 9] Generating summary report...')
    analyzer = BacktestResultAnalyzer(all_results)
    summary = analyzer.generate_summary_report(factor_returns_final)

    print('\n' + '=' * 70)
    print('Backtest Results Summary (Real Data)')
    print('=' * 70)
    print(summary.to_string(index=False))

    summary_file = os.path.join(OUTPUT_DIR, 'factor_strategy_summary_real_data.csv')
    summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f'\n[Results saved to: {summary_file}]')

    return all_results, asset_returns_final, factor_returns_final, exposure_matrix

if __name__ == "__main__":
    all_results, asset_returns, factor_returns, exposure_matrix = run_backtest_with_real_data()

    print('\n' + '=' * 70)
    print('Framework execution completed successfully!')
    print('=' * 70)