"""
使用真实数据生成研报图表
"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'd:/Documents/trae_projects/macro_factor_asset_alloc/output'
DATA_DIR = 'd:/Documents/trae_projects/macro_factor_asset_alloc/data'

def load_seven_assets_price():
    """加载7种资产价格数据"""
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
    return df, asset_returns

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
    return factor_df, pca

def plot_factor_allocation_flowchart():
    """绘制因子配置流程图"""
    print("Plotting factor allocation flowchart...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2)
    arrow_style = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='darkblue', lw=2)

    steps = [
        (7, 9, 'Step 1\nSelect Factors'),
        (7, 7, 'Step 2\nCalculate Factor Exposure'),
        (7, 5, 'Step 3\nDetermine Target Exposure'),
        (7, 3, 'Step 4\nMatch Target Exposure'),
    ]

    for x, y, text in steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=box_style)

    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i+1][0], steps[i+1][1]+0.5),
                   xytext=(steps[i][0], steps[i][1]-0.5),
                   arrowprops=arrow_style)

    sub_steps = [
        (2, 9, 'PCA\n降维'),
        (12, 9, 'Macro\nIndicators'),
        (2, 7, 'LASSO\nRegression'),
        (12, 7, 'Rolling\nWindow'),
        (2, 5, 'Base + \nDeviation'),
        (12, 5, 'View\nQuantization'),
        (2, 3, 'Blyth\nFramework'),
        (12, 3, 'Greenberg\nFramework'),
    ]

    small_box = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', linewidth=1)
    for x, y, text in sub_steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                bbox=small_box)

    ax.text(7, 1.5, 'Bridge between Macro Research and Asset Allocation',
           ha='center', va='center', fontsize=12, style='italic', color='darkgreen')

    plt.title('Factor Allocation Framework', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_02_factor_allocation_flowchart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_pca_variance_explained(pca_model):
    """绘制PCA方差解释率"""
    print("Plotting PCA variance explained...")

    fig, ax = plt.subplots(figsize=(10, 6))

    explained_var = pca_model.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    x = np.arange(1, len(explained_var) + 1)
    bars = ax.bar(x, explained_var * 100, color='steelblue', alpha=0.7, label='Individual')
    ax.plot(x, cumulative_var * 100, 'ro-', linewidth=2, label='Cumulative')

    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('PCA Variance Explained by Principal Components', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, explained_var):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_pca_variance_explained.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_asset_returns_heatmap(asset_returns):
    """绘制资产收益热力图"""
    print("Plotting asset returns heatmap...")

    monthly_returns = resample_to_monthly(asset_returns)[1]
    monthly_returns = monthly_returns.loc['2013-06':'2023-05']

    fig, ax = plt.subplots(figsize=(14, 8))

    data = monthly_returns.values.T * 100
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)

    ax.set_yticks(np.arange(len(monthly_returns.columns)))
    ax.set_yticklabels(monthly_returns.columns)

    dates = monthly_returns.index
    tick_positions = np.linspace(0, len(dates)-1, min(12, len(dates)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime('%Y-%m') for i in tick_positions], rotation=45, ha='right')

    ax.set_title('Monthly Asset Returns (%)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Return (%)')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_asset_returns_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_factor_correlation_heatmap(factor_returns):
    """绘制因子相关性热力图"""
    print("Plotting factor correlation heatmap...")

    corr = factor_returns.corr()

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

    ax.set_title('PCA Factor Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_factor_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_exposure_matrix(exposure_matrix, asset_names, factor_names):
    """绘制因子暴露矩阵"""
    print("Plotting factor exposure matrix...")

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(exposure_matrix.values, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(factor_names)))
    ax.set_yticks(np.arange(len(asset_names)))
    ax.set_xticklabels(factor_names, rotation=45, ha='right')
    ax.set_yticklabels(asset_names)

    for i in range(len(asset_names)):
        for j in range(len(factor_names)):
            text = ax.text(j, i, f'{exposure_matrix.values[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title('Factor Exposure Matrix (PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Exposure')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_exposure_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_cumulative_returns(asset_returns, factor_returns):
    """绘制累计收益对比"""
    print("Plotting cumulative returns...")

    monthly_returns = resample_to_monthly(asset_returns)[1]
    monthly_returns = monthly_returns.loc['2013-06':'2023-05']

    cumulative = (1 + monthly_returns).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    cumulative.plot(ax=axes[0], linewidth=2, title='Asset Cumulative Returns')
    axes[0].set_title('Asset Cumulative Returns', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend(loc='upper left', ncol=3)
    axes[0].grid(True, alpha=0.3)

    factor_cum = (1 + factor_returns).cumprod()
    factor_cum.plot(ax=axes[1], linewidth=2, title='PCA Factor Cumulative Returns')
    axes[1].set_title('PCA Factor Cumulative Returns', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Return')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_cumulative_returns.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_factor_returns(factor_returns):
    """绘制因子收益"""
    print("Plotting factor returns...")

    factor_cum = (1 + factor_returns).cumprod()

    fig, ax = plt.subplots(figsize=(14, 8))

    factor_cum.plot(ax=ax, linewidth=2)

    ax.set_title('PCA Factor Cumulative Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_factor_returns.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_portfolio_risk_decomposition():
    """绘制组合风险分解图"""
    print("Plotting portfolio risk decomposition...")

    factors = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'Idiosyncratic']
    risk_values = [56.02, 32.88, 7.09, 3.57, 0.38, 0.05, 0.01]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#C0C0C0']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    y_pos = np.arange(len(factors))
    axes[0].barh(y_pos, risk_values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(factors)
    axes[0].set_xlabel('Variance Explained (%)')
    axes[0].set_title('PCA Variance Explained', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(risk_values):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)

    axes[1].pie(risk_values, labels=factors, autopct='%1.1f%%',
               colors=colors, startangle=90)
    axes[1].set_title('Variance Explained Percentage', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_risk_decomposition.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("Generating Figures with Real Data")
    print("=" * 70)

    print("\n[1] Loading data...")
    price_df, asset_returns = load_seven_assets_price()

    print("\n[2] Processing data...")
    monthly_prices, monthly_returns = resample_to_monthly(asset_returns)
    monthly_returns = monthly_returns.dropna(how='all')
    monthly_returns = monthly_returns.fillna(0)
    monthly_returns = monthly_returns.loc['2013-06':'2023-05']

    pca_factors, pca_model = build_pca_factors(monthly_returns, n_factors=6)

    asset_names = ['沪深300', '中证500', '中债国债', '中债企业债', '南华商品', '布伦特原油']
    factor_names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']

    from factor_exposure import FactorExposureWithPrior
    exposure_calc = FactorExposureWithPrior(alpha=0.01)
    exposure_matrix = exposure_calc.fit(monthly_returns[asset_names], pca_factors)

    print("\n[3] Generating figures...")

    plot_factor_allocation_flowchart()
    plot_pca_variance_explained(pca_model)
    plot_asset_returns_heatmap(asset_returns)
    plot_factor_correlation_heatmap(pca_factors)
    plot_exposure_matrix(exposure_matrix, asset_names, factor_names)
    plot_cumulative_returns(asset_returns, pca_factors)
    plot_factor_returns(pca_factors)
    plot_portfolio_risk_decomposition()

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    files = os.listdir(OUTPUT_DIR)
    print("\nGenerated files:")
    for f in sorted(files):
        if f.startswith('figure'):
            print(f"  - {f}")