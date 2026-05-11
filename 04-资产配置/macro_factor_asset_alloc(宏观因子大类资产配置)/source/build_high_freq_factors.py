"""
构建高频化宏观因子序列并绘制分析图
复现国君量化配置团队的高频化宏观因子体系
"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'd:/Documents/trae_projects/macro_factor_asset_alloc/output'
DATA_DIR = 'd:/Documents/trae_projects/macro_factor_asset_alloc/data'

def load_high_frequency_data():
    """加载高频化因子所需数据"""
    print('[Step 1] Loading high frequency data...')
    file_path = os.path.join(DATA_DIR, 'high_frequency_macro_factor_portfolio.csv')

    df = pd.read_csv(file_path, index_col=0, encoding='utf-8')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(how='all')
    df = df.sort_index()

    df.columns = ['恒生指数', 'CRB工业', '南华铜', '房地产', '生猪价格', '布伦特原油',
                  '螺纹钢', '中债国债', '中债信用债', '中债3-5年国债', '美元指数',
                  '申万大盘PE', '申万小盘PE']

    print(f'  Shape: {df.shape}')
    print(f'  Date range: {df.index[0]} to {df.index[-1]}')
    return df

def compute_asset_returns(prices):
    """计算资产收益率"""
    returns = prices.pct_change()
    returns = returns.dropna(how='all')
    returns = returns.fillna(0)
    return returns

def build_portfolio_based_factors(daily_returns):
    """
    使用资产组合法构建高频化宏观因子
    参考国君量化配置团队方法：
    - 增长因子: 股票类资产
    - 通胀因子: 商品类资产
    - 利率因子: 债券类资产
    - 信用因子: 信用债与国债利差
    - 汇率因子: 美元指数
    - 流动性因子: PE比率变化
    """
    print('[Step 2] Building portfolio-based high frequency factors...')

    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns = monthly_returns.dropna(how='all')
    monthly_returns = monthly_returns.fillna(0)

    growth_assets = ['恒生指数', '房地产']
    inflation_assets = ['CRB工业', '南华铜', '布伦特原油', '螺纹钢', '生猪价格']
    rate_assets = ['中债国债', '中债3-5年国债']
    credit_assets = ['中债信用债']
    fx_assets = ['美元指数']
    liquidity_assets = ['申万大盘PE', '申万小盘PE']

    def build_factor(factor_name, assets, monthly_returns):
        """构建单个因子"""
        available_assets = [a for a in assets if a in monthly_returns.columns]
        if not available_assets:
            return None

        factor_returns = monthly_returns[available_assets].mean(axis=1)
        return factor_returns

    factors = {}
    factors['增长(高频)'] = build_factor('增长', growth_assets, monthly_returns)
    factors['通胀(高频)'] = build_factor('通胀', inflation_assets, monthly_returns)
    factors['利率(高频)'] = build_factor('利率', rate_assets, monthly_returns)
    factors['信用(高频)'] = build_factor('信用', credit_assets, monthly_returns)
    factors['汇率(高频)'] = build_factor('汇率', fx_assets, monthly_returns)
    factors['流动性(高频)'] = build_factor('流动性', liquidity_assets, monthly_returns)

    high_freq_factors = pd.DataFrame(factors)
    high_freq_factors = high_freq_factors.dropna()

    print(f'  High frequency factors shape: {high_freq_factors.shape}')
    print(f'  Factors: {list(high_freq_factors.columns)}')

    return high_freq_factors, monthly_returns

def build_pca_factors(monthly_returns, n_factors=6):
    """使用PCA从资产收益中提取因子"""
    print('[Step 3] Building PCA factors...')

    asset_names = ['恒生指数', 'CRB工业', '南华铜', '房地产', '布伦特原油', '螺纹钢',
                   '中债国债', '中债信用债', '中债3-5年国债', '美元指数']

    available_assets = [a for a in asset_names if a in monthly_returns.columns]

    returns_subset = monthly_returns[available_assets].fillna(0)

    pca = PCA(n_components=min(n_factors, len(available_assets)))
    pca_returns = pca.fit_transform(returns_subset)

    pca_factor_names = [f'PC{i+1}' for i in range(pca_returns.shape[1])]
    pca_factors = pd.DataFrame(pca_returns, index=monthly_returns.index, columns=pca_factor_names)

    print(f'  PCA factors shape: {pca_factors.shape}')
    print(f'  Explained variance: {pca.explained_variance_ratio_}')

    return pca_factors, pca

def standardize_factors(factors):
    """标准化因子"""
    scaler = StandardScaler()
    factors_std = pd.DataFrame(
        scaler.fit_transform(factors),
        index=factors.index,
        columns=factors.columns
    )
    return factors_std

def plot_high_freq_factor_returns(high_freq_factors):
    """绘制高频化因子收益"""
    print('[Step 4] Plotting high frequency factor returns...')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(high_freq_factors.columns):
        cumulative = (1 + high_freq_factors[col]).cumprod()
        axes[i].plot(cumulative.index, cumulative.values, linewidth=2, color='steelblue')
        axes[i].axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        axes[i].set_title(col, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Cumulative Return')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('High Frequency Macro Factor Cumulative Returns', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_high_freq_factor_cumulative.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')

def plot_factor_comparison(high_freq_factors, pca_factors):
    """绘制高频化因子与PCA因子对比"""
    print('[Step 5] Plotting factor comparison...')

    high_freq_std = standardize_factors(high_freq_factors)
    pca_std = standardize_factors(pca_factors)

    common_dates = high_freq_std.index.intersection(pca_std.index)
    high_freq_aligned = high_freq_std.loc[common_dates]
    pca_aligned = pca_std.loc[common_dates]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    factor_mapping = {
        '增长(高频)': 'PC1',
        '通胀(高频)': 'PC2',
        '利率(高频)': 'PC3',
        '信用(高频)': 'PC4',
        '汇率(高频)': 'PC5',
        '流动性(高频)': 'PC6'
    }

    for i, (hf_name, pca_name) in enumerate(factor_mapping.items()):
        if hf_name in high_freq_aligned.columns and pca_name in pca_aligned.columns:
            hf_cum = (1 + high_freq_aligned[hf_name]).cumprod()
            pca_cum = (1 + pca_aligned[pca_name]).cumprod()

            axes[i].plot(hf_cum.index, hf_cum.values, linewidth=2,
                        label='Portfolio-based', color='blue')
            axes[i].plot(pca_cum.index, pca_cum.values, linewidth=2,
                        label='PCA-based', color='red', alpha=0.7)
            axes[i].axhline(y=1, color='gray', linestyle='--', alpha=0.7)
            axes[i].set_title(f'{hf_name} vs {pca_name}', fontsize=12, fontweight='bold')
            axes[i].legend(loc='upper left')
            axes[i].grid(True, alpha=0.3)

    plt.suptitle('High Frequency Factors vs PCA Factors Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_high_freq_vs_pca_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')

def plot_factor_correlation_matrix(high_freq_factors):
    """绘制高频化因子相关性矩阵"""
    print('[Step 6] Plotting factor correlation matrix...')

    corr = high_freq_factors.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f'{corr.values[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=12)

    ax.set_title('High Frequency Factor Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_high_freq_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')

def plot_factor_rolling_correlation(high_freq_factors, pca_factors):
    """绘制高频化因子与PCA因子的滚动相关性"""
    print('[Step 7] Plotting rolling correlation...')

    high_freq_std = standardize_factors(high_freq_factors)
    pca_std = standardize_factors(pca_factors)

    common_dates = high_freq_std.index.intersection(pca_std.index)
    high_freq_aligned = high_freq_std.loc[common_dates]
    pca_aligned = pca_std.loc[common_dates]

    factor_mapping = {
        '增长(高频)': 'PC1',
        '通胀(高频)': 'PC2',
        '利率(高频)': 'PC3',
        '信用(高频)': 'PC4',
        '汇率(高频)': 'PC5',
        '流动性(高频)': 'PC6'
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    window = 12
    for i, (hf_name, pca_name) in enumerate(factor_mapping.items()):
        if hf_name in high_freq_aligned.columns and pca_name in pca_aligned.columns:
            rolling_corr = high_freq_aligned[hf_name].rolling(window).corr(pca_aligned[pca_name])

            axes[i].plot(rolling_corr.index, rolling_corr.values, linewidth=2, color='green')
            axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            axes[i].set_ylim(-1, 1)
            axes[i].set_title(f'{hf_name} vs {pca_name}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Rolling Correlation (12M)')
            axes[i].grid(True, alpha=0.3)

    plt.suptitle('Rolling Correlation: Portfolio-based vs PCA-based Factors', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_high_freq_rolling_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')

def plot_factor_exposure_analysis(high_freq_factors):
    """绘制因子暴露度分析"""
    print('[Step 8] Plotting factor exposure analysis...')

    high_freq_std = standardize_factors(high_freq_factors)

    asset_names = ['恒生指数', 'CRB工业', '南华铜', '房地产', '布伦特原油', '螺纹钢',
                   '中债国债', '中债信用债', '中债3-5年国债', '美元指数']

    available_assets = [a for a in asset_names if a in high_freq_factors.columns]

    fig, ax = plt.subplots(figsize=(12, 8))

    exposure_data = []
    for asset in available_assets:
        if asset in high_freq_factors.columns:
            corr = high_freq_std.corrwith(high_freq_factors[asset])
            exposure_data.append(corr.values)

    exposure_matrix = pd.DataFrame(exposure_data, index=available_assets, columns=high_freq_factors.columns)
    exposure_matrix = exposure_matrix.astype(float)

    im = ax.imshow(exposure_matrix.values, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(high_freq_factors.columns)))
    ax.set_yticks(np.arange(len(available_assets)))
    ax.set_xticklabels(high_freq_factors.columns, rotation=45, ha='right')
    ax.set_yticklabels(available_assets)

    for i in range(len(available_assets)):
        for j in range(len(high_freq_factors.columns)):
            text = ax.text(j, i, f'{exposure_matrix.values[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)

    ax.set_title('Asset-Factor Exposure Matrix (Correlation-based)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_high_freq_exposure_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')

def plot_monthly_heatmap(high_freq_factors):
    """绘制高频化因子月度收益热力图"""
    print('[Step 9] Plotting monthly returns heatmap...')

    monthly = high_freq_factors * 100

    fig, ax = plt.subplots(figsize=(14, 8))

    data = monthly.values.T
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    ax.set_yticks(np.arange(len(monthly.columns)))
    ax.set_yticklabels(monthly.columns)

    dates = monthly.index
    tick_positions = np.linspace(0, len(dates)-1, min(24, len(dates)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime('%Y-%m') for i in tick_positions], rotation=45, ha='right')

    ax.set_title('High Frequency Factor Monthly Returns (%)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Return (%)')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_high_freq_monthly_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')

def save_factor_data(high_freq_factors, pca_factors):
    """保存因子数据到CSV"""
    print('[Step 10] Saving factor data...')

    combined = pd.concat([high_freq_factors, pca_factors], axis=1)
    combined = combined.dropna()

    save_path = os.path.join(OUTPUT_DIR, 'high_frequency_factors.csv')
    combined.to_csv(save_path, encoding='utf-8-sig')
    print(f'  Saved: {save_path}')

def main():
    print('=' * 70)
    print('Building High Frequency Macro Factors')
    print('=' * 70)

    high_freq_data = load_high_frequency_data()

    daily_returns = compute_asset_returns(high_freq_data)

    high_freq_factors, monthly_returns = build_portfolio_based_factors(daily_returns)

    pca_factors, pca_model = build_pca_factors(monthly_returns, n_factors=6)

    plot_high_freq_factor_returns(high_freq_factors)
    plot_factor_comparison(high_freq_factors, pca_factors)
    plot_factor_correlation_matrix(high_freq_factors)
    plot_factor_rolling_correlation(high_freq_factors, pca_factors)
    plot_factor_exposure_analysis(high_freq_factors)
    plot_monthly_heatmap(high_freq_factors)

    save_factor_data(high_freq_factors, pca_factors)

    print('\n' + '=' * 70)
    print('High frequency factor analysis completed!')
    print('=' * 70)

    files = [f for f in os.listdir(OUTPUT_DIR) if 'high_freq' in f]
    print('\nGenerated files:')
    for f in sorted(files):
        print(f'  - {f}')

if __name__ == "__main__":
    main()