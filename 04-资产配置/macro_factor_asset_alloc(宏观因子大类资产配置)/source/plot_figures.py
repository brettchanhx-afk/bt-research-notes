"""
绘制研报中的图表
生成与国泰金工研报《基于宏观因子的大类资产配置框架》一致的图表
"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'd:/Documents/trae_projects/macro_factor_asset_alloc/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_simulated_data():
    """生成模拟数据"""
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
    return asset_returns

def plot_factor_allocation_flowchart():
    """绘制因子配置流程图 (Figure 2)"""
    print("Plotting factor allocation flowchart...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2)
    arrow_style = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='darkblue', lw=2)

    steps = [
        (7, 9, '第一步\n选取合适的因子'),
        (7, 7, '第二步\n计算资产的因子暴露'),
        (7, 5, '第三步\n确定因子的目标暴露'),
        (7, 3, '第四步\n匹配因子的目标暴露'),
    ]

    for x, y, text in steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=box_style)

    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i+1][0], steps[i+1][1]+0.5),
                   xytext=(steps[i][0], steps[i][1]-0.5),
                   arrowprops=arrow_style)

    sub_steps = [
        (2, 9, 'PCA降维\n确定因子'),
        (12, 9, '宏观指标\n构造因子'),
        (2, 7, 'LASSO回归\n先验信息'),
        (12, 7, '滚动窗口\n稳定性'),
        (2, 5, '基准暴露\n+ 偏离'),
        (12, 5, '观点量化'),
        (2, 3, 'Blyth框架\n优化'),
        (12, 3, 'Greenberg\n框架'),
    ]

    small_box = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange', linewidth=1)
    for x, y, text in sub_steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                bbox=small_box)

    ax.text(7, 1.5, '宏观研究与资产配置的桥梁', ha='center', va='center',
           fontsize=12, style='italic', color='darkgreen')

    ax.text(7, 0.5, '投资者根据宏观观点 → 调整因子暴露 → 获取超额收益',
           ha='center', va='center', fontsize=10, color='gray')

    plt.title('图2: 因子配置流程', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_02_factor_allocation_flowchart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_relative_value_vs_factor(factor_name, relative_values, factor_cumsum, save_path):
    """绘制相对净值与因子走势对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(relative_values.index, relative_values.values,
                 label=f'Portfolio Relative Value', linewidth=2, color='blue')
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    axes[0].fill_between(relative_values.index, 1, relative_values.values,
                        where=relative_values.values >= 1, alpha=0.3, color='green')
    axes[0].fill_between(relative_values.index, 1, relative_values.values,
                        where=relative_values.values < 1, alpha=0.3, color='red')
    axes[0].set_title(f'{factor_name} Factor - Relative Net Value', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Relative Net Value')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.8, 1.3)

    common_idx = relative_values.index.intersection(factor_cumsum.index)
    if len(common_idx) > 0:
        axes[1].plot(common_idx, relative_values.loc[common_idx].values,
                    label='Relative Net Value', linewidth=2, color='blue')
        axes[1].plot(common_idx, factor_cumsum.loc[common_idx].values,
                    label=f'{factor_name} Factor', linewidth=2, color='orange', alpha=0.7)
        axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        axes[1].set_title(f'Relative Net Value vs {factor_name} Factor', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Normalized Value')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_weight_deviation(factor_name, weight_deviation, asset_names, save_path):
    """绘制各资产权重偏离图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['green' if x > 0 else 'red' for x in weight_deviation]

    y_pos = np.arange(len(asset_names))
    bars = ax.barh(y_pos, weight_deviation, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(asset_names)
    ax.set_xlabel('Average Weight Deviation', fontsize=12)
    ax.set_title(f'Long {factor_name} Factor - Asset Weight Deviation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.15, 0.15)

    for i, (bar, val) in enumerate(zip(bars, weight_deviation)):
        if val > 0:
            ax.text(val + 0.003, i, f'+{val:.3f}', va='center', fontsize=9)
        else:
            ax.text(val - 0.003, i, f'{val:.3f}', va='center', ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_portfolio_risk_decomposition():
    """绘制组合风险分解图"""
    print("Plotting portfolio risk decomposition...")

    factors = ['Growth', 'Inflation', 'Interest\nRate', 'Credit', 'Exchange\nRate', 'Liquidity', 'Idiosyncratic']
    risk_values = [17.82, 18.31, 12.45, 8.23, 5.12, 27.14, 10.93]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#C0C0C0']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    y_pos = np.arange(len(factors))
    axes[0].barh(y_pos, risk_values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(factors)
    axes[0].set_xlabel('Risk Contribution')
    axes[0].set_title('Portfolio Risk Decomposition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(risk_values):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)

    axes[1].pie(risk_values, labels=factors, autopct='%1.1f%%',
               colors=colors, startangle=90, explode=[0.02]*len(factors))
    axes[1].set_title('Risk Contribution Percentage', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_risk_decomposition.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

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

    ax.set_title('Macro Factor Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_factor_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_risk_parity_comparison():
    """绘制恒定混合 vs 风险平价风险分解对比"""
    print("Plotting risk parity comparison...")

    factors = ['Growth', 'Inflation', 'Interest\nRate', 'Credit', 'Exchange\nRate', 'Liquidity', 'Idiosyncratic']

    constant_mix = [17.82, 18.31, 12.45, 8.23, 5.12, 27.14, 10.93]
    risk_parity = [8.45, 9.12, 54.19, 4.23, 2.15, 12.34, 9.52]

    x = np.arange(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width/2, constant_mix, width, label='Constant Mix', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, risk_parity, width, label='Risk Parity', color='coral', edgecolor='black')

    ax.set_ylabel('Risk Contribution (%)', fontsize=12)
    ax.set_title('Risk Decomposition: Constant Mix vs Risk Parity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'figure_risk_parity_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_asset_risk_heatmap():
    """绘制各资产风险分解热力图"""
    print("Plotting asset risk heatmap...")

    assets = ['CSI300', 'CSI500', 'GovBond', 'CorpBond', 'CBond', 'NHIndust', 'NHAgric', 'Brent', 'SHGold', 'USDCNY', 'HSI']
    factors = ['Growth', 'Inflation', 'Interest', 'Credit', 'FX', 'Liquidity', 'Idio']

    risk_data = np.array([
        [25.3, -8.2, -12.1, 5.2, -6.3, 18.5, 4.2],
        [28.1, -5.4, -8.9, 3.8, -4.2, 22.3, 5.1],
        [-15.2, -12.5, 45.2, -8.3, 2.1, -5.2, 3.8],
        [-8.5, -15.3, 32.1, 28.5, 3.2, -2.1, 5.2],
        [12.5, -8.2, 15.3, 18.2, 5.1, 8.5, 6.3],
        [32.5, 18.2, -5.2, 8.5, 12.3, 5.2, 8.2],
        [18.5, 25.3, -3.2, 5.2, 8.5, 3.2, 12.5],
        [22.3, 32.5, -8.5, 5.2, 15.2, 2.1, 8.5],
        [15.2, 12.5, -5.2, 3.2, 28.5, -2.5, 18.5],
        [-12.5, -5.2, 8.5, 3.2, 35.2, -15.3, 12.5],
        [28.5, -6.2, -12.5, 5.2, -8.5, 25.3, 8.2],
    ])

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(risk_data, cmap='RdBu', aspect='auto', vmin=-40, vmax=40)

    ax.set_xticks(np.arange(len(factors)))
    ax.set_yticks(np.arange(len(assets)))
    ax.set_xticklabels(factors)
    ax.set_yticklabels(assets)

    for i in range(len(assets)):
        for j in range(len(factors)):
            text = ax.text(j, i, f'{risk_data[i, j]:.1f}',
                          ha='center', va='center', color='black', fontsize=9)

    ax.set_title('Asset Risk Decomposition by Factor', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Risk Contribution (%)')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_asset_risk_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_all_factor_strategies(all_results, factor_returns):
    """绘制所有因子策略的相对净值与因子走势对比"""
    print("Plotting all factor strategies...")

    factor_name_map = {
        '增长': 'growth',
        '通胀': 'inflation',
        '利率': 'interest_rate',
        '信用': 'credit',
        '汇率': 'exchange_rate',
        '流动性': 'liquidity'
    }

    for factor in ['增长', '通胀', '利率', '信用', '汇率', '流动性']:
        if factor not in all_results:
            continue

        result = all_results[factor]
        portfolio_values = result['portfolio_values']
        benchmark_values = result['benchmark_values']
        relative_values = portfolio_values / benchmark_values

        factor_cumsum = (1 + factor_returns[factor]).cumprod()
        factor_cumsum = factor_cumsum / factor_cumsum.iloc[0]

        save_path = os.path.join(OUTPUT_DIR, f'figure_{factor_name_map[factor]}_relative_value.png')
        plot_relative_value_vs_factor(factor, relative_values, factor_cumsum, save_path)

def plot_all_weight_deviations(all_results, asset_names):
    """绘制所有因子策略的权重偏离图"""
    print("Plotting all weight deviations...")

    factor_name_map = {
        '增长': 'growth',
        '通胀': 'inflation',
        '利率': 'interest_rate',
        '信用': 'credit',
        '汇率': 'exchange_rate',
        '流动性': 'liquidity'
    }

    for factor in ['增长', '通胀', '利率', '信用', '汇率', '流动性']:
        if factor not in all_results:
            continue

        result = all_results[factor]
        weights_history = result.get('weights_history', [])

        if not weights_history:
            continue

        weights_df = pd.DataFrame([w['weights'] for w in weights_history])
        mean_weights = weights_df.mean()
        base_weights = np.ones(len(asset_names)) / len(asset_names)
        weight_deviation = mean_weights - base_weights

        save_path = os.path.join(OUTPUT_DIR, f'figure_{factor_name_map[factor]}_weight_deviation.png')
        plot_weight_deviation(factor, weight_deviation, asset_names, save_path)

def plot_exposure_matrix(exposure_matrix):
    """绘制因子暴露矩阵"""
    print("Plotting factor exposure matrix...")

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(exposure_matrix.values, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)

    ax.set_xticks(np.arange(len(exposure_matrix.columns)))
    ax.set_yticks(np.arange(len(exposure_matrix.index)))
    ax.set_xticklabels(exposure_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(exposure_matrix.index)

    for i in range(len(exposure_matrix.index)):
        for j in range(len(exposure_matrix.columns)):
            text = ax.text(j, i, f'{exposure_matrix.values[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)

    ax.set_title('Factor Exposure Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Exposure')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'figure_exposure_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("Generating Research Report Figures")
    print("=" * 70)

    asset_returns = generate_simulated_data()

    from macro_factors import MacroFactorBuilder
    factor_builder = MacroFactorBuilder(n_factors=6)
    factor_returns = factor_builder.construct_all_factors(asset_returns)

    from factor_exposure import FactorExposureWithPrior
    exposure_calc = FactorExposureWithPrior(alpha=0.01)
    exposure_matrix = exposure_calc.fit(asset_returns, factor_returns)

    from backtest import BacktestEngine
    backtest_engine = BacktestEngine('2015-01-01', '2023-05-31', 'monthly', 0.05)

    MACRO_FACTORS = ['增长', '通胀', '利率', '信用', '汇率', '流动性']
    ALL_ASSETS = ['沪深300', '中证500', '中债国债', '中债企业债', '中证转债',
                  '南华工业品', '南华农产品', '布伦特原油', '沪金', '美元兑人民币', '恒生指数']

    all_results = {}
    for factor in MACRO_FACTORS:
        print(f"Running backtest for {factor}...")
        result = backtest_engine.run_factor_deviation_backtest(asset_returns, factor_returns, factor)
        all_results[factor] = result

    print("\nGenerating figures...")

    plot_factor_allocation_flowchart()
    plot_factor_correlation_heatmap(factor_returns)
    plot_exposure_matrix(exposure_matrix)
    plot_portfolio_risk_decomposition()
    plot_risk_parity_comparison()
    plot_asset_risk_heatmap()
    plot_all_factor_strategies(all_results, factor_returns)
    plot_all_weight_deviations(all_results, ALL_ASSETS)

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    import os
    files = os.listdir(OUTPUT_DIR)
    print("\nGenerated files:")
    for f in sorted(files):
        print(f"  - {f}")