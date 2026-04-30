# -*- coding: utf-8 -*-
"""
可视化绘图模块
===================================
用于绘制业绩持续性分析结果的各种图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    # 清除字体缓存
    cache_dir = matplotlib.get_cachedir()
    font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
    if os.path.exists(font_cache):
        try:
            os.remove(font_cache)
        except OSError:
            pass
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120

# 在任何style.use之前调用
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
setup_chinese_font()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, PLOT_CONFIG


# ============================================================
# 收益率曲线图
# ============================================================

def plot_cumulative_returns(fund_returns, benchmark_returns=None, 
                           title="基金累计收益曲线",
                           save_path=None):
    """
    绘制累计收益率曲线
    
    Parameters:
    -----------
    fund_returns : dict or DataFrame
        基金收益率数据
    benchmark_returns : Series, optional
        基准收益率
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 转换输入为DataFrame
    if isinstance(fund_returns, dict):
        returns_df = pd.DataFrame(fund_returns)
    else:
        returns_df = fund_returns.copy()
    
    # 计算累计收益
    cumulative = (1 + returns_df.fillna(0)).cumprod()
    
    # 绘制基金曲线
    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col], label=col, linewidth=1.5)
    
    # 绘制基准曲线
    if benchmark_returns is not None:
        bm_cumulative = (1 + benchmark_returns.fillna(0)).cumprod()
        ax.plot(bm_cumulative.index, bm_cumulative, 
               label='Benchmark', linewidth=2, linestyle='--', color='gray')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [保存] 图表已保存至 {save_path}")
    
    plt.show()
    return fig


def plot_returns_comparison(fund_returns, period1_label="Period 1", 
                           period2_label="Period 2",
                           title="Two Period Returns Comparison",
                           save_path=None):
    """
    绘制两个期间的收益对比散点图（用于横截面分析）
    
    Parameters:
    -----------
    fund_returns : DataFrame
        包含 period1 和 period2 列的数据框
    period1_label : str
        第一期间标签
    period2_label : str
        第二期间标签
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制散点
    ax.scatter(fund_returns.iloc[:, 0], fund_returns.iloc[:, 1], 
              alpha=0.6, s=50)
    
    # 添加参考线
    max_val = max(fund_returns.iloc[:, 0].max(), fund_returns.iloc[:, 1].max())
    min_val = min(fund_returns.iloc[:, 0].min(), fund_returns.iloc[:, 1].min())
    
    # 45度线
    ax.plot([min_val, max_val], [min_val, max_val], 
           'k--', linewidth=1, label='y=x')
    
    # 添加趋势线
    z = np.polyfit(fund_returns.iloc[:, 0], fund_returns.iloc[:, 1], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    ax.set_xlabel(period1_label)
    ax.set_ylabel(period2_label)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围相同
    ax.set_xlim(min_val - 0.1 * abs(max_val - min_val), 
                 max_val + 0.1 * abs(max_val - min_val))
    ax.set_ylim(min_val - 0.1 * abs(max_val - min_val), 
                 max_val + 0.1 * abs(max_val - min_val))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [保存] 图表已保存至 {save_path}")
    
    plt.show()
    return fig


# ============================================================
# Hurst指数分析图
# ============================================================

def plot_hurst_rs_analysis(log_rs_values, log_n_values, hurst_result,
                           title="Hurst指数回归分析",
                           save_path=None):
    """
    绘制R/S分析回归图
    
    Parameters:
    -----------
    log_rs_values : list
        log(R/S)值
    log_n_values : list
        log(n)值
    hurst_result : dict
        Hurst分析结果
    save_path : str, optional
        保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点
    ax.scatter(log_n_values, log_rs_values, s=100, c='blue', 
              alpha=0.7, label='Data Points')
    
    # 绘制拟合线
    H = hurst_result.get('H', 0.5)
    c = hurst_result.get('c', 1)
    
    x_fit = np.linspace(min(log_n_values), max(log_n_values), 100)
    y_fit = np.log(c) + H * x_fit
    
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
           label=f'Fit: H={H:.3f}')
    
    ax.set_xlabel('log(n)')
    ax.set_ylabel('log(R/S)')
    ax.set_title(f'{title}\nHurst Index H = {H:.4f}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加R方
    r2 = hurst_result.get('r_squared', 0)
    ax.text(0.05, 0.95, f'R^2 = {r2:.4f}', 
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [保存] 图表已保存至 {save_path}")
    
    plt.show()
    return fig


def plot_hurst_distribution(fund_results, title="Hurst指数分布",
                           save_path=None):
    """
    绘制多只基金的Hurst指数分布直方图
    
    Parameters:
    -----------
    fund_results : dict or DataFrame
        {fund_code: hurst_result}
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 提取H值
    if isinstance(fund_results, dict):
        H_values = [v.get('H', np.nan) for v in fund_results.values()]
        labels = list(fund_results.keys())
    else:
        H_values = fund_results.iloc[:, 0].tolist()
        labels = fund_results.index.tolist()
    
    H_values = [h for h in H_values if not np.isnan(h)]
    
    if len(H_values) == 0:
        print("  [警告] 无有效的Hurst指数数据")
        return None
    
    # 绘制直方图
    ax.hist(H_values, bins=20, alpha=0.7, color='steelblue', edgecolor='white')
    
    # 添加分界线
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
              label='H=0.5 (Random)')
    ax.axvline(x=np.median(H_values), color='green', linestyle='--', 
              linewidth=2, label=f'Median H={np.median(H_values):.3f}')
    
    ax.set_xlabel('Hurst Index (H)')
    ax.set_ylabel('Number of Funds')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加分类区域
    ax.axvspan(0, 0.4, alpha=0.1, color='red', label='Reversal')
    ax.axvspan(0.6, 1, alpha=0.1, color='green', label='Persistence')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [保存] 图表已保存至 {save_path}")
    
    plt.show()
    return fig


# ============================================================
# CPR分析图
# ============================================================

def plot_cpr_matrix(ww, ll, wl, lw, title="Cross-Product Ratio Analysis",
                   save_path=None):
    """
    绘制CPR分析的四格矩阵图
    
    Parameters:
    -----------
    ww, ll, wl, lw : int
        各组合的数量
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 计算CPR
    if wl > 0 and lw > 0:
        cpr = (ww * ll) / (wl * lw)
    else:
        cpr = np.nan
    
    # 创建矩阵数据
    matrix = np.array([[ww, wl], [lw, ll]])
    
    # 绘制热力图
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal')
    
    # 添加数值标签
    labels = [['WW', 'WL'], ['LW', 'LL']]
    colors = ['white', 'white', 'white', 'black']
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{matrix[i, j]}', 
                   ha='center', va='center', fontsize=20, 
                   color=text_color, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Winner (W)', 'Loser (L)'])
    ax.set_yticklabels(['Next: Winner', 'Next: Loser'])
    
    # 计算CPR并显示
    if not np.isnan(cpr):
        cpr_text = f'CPR = {cpr:.2f}'
        if cpr > 1:
            verdict = 'Persistence'
            color = 'green'
        elif cpr < 1:
            verdict = 'Reversal'
            color = 'red'
        else:
            verdict = 'Random'
            color = 'gray'
        
        ax.text(0.5, -0.15, cpr_text, ha='center', va='top', 
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, -0.25, verdict, ha='center', va='top', 
               fontsize=14, color=color, transform=ax.transAxes)
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [保存] 图表已保存至 {save_path}")
    
    plt.show()
    return fig


# ============================================================
# 综合分析仪表盘
# ============================================================

def plot_persistence_dashboard(fund_code, analysis_results,
                             cumulative_returns=None,
                             title=None,
                             save_path=None):
    """
    绘制业绩持续性分析综合仪表盘
    
    Parameters:
    -----------
    fund_code : str
        基金代码
    analysis_results : dict
        综合分析结果
    cumulative_returns : Series, optional
        累计收益率
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    if title is None:
        title = f'基金 {fund_code} 业绩持续性分析'
    
    fig = plt.figure(figsize=(16, 12))
    
    # 创建子图布局
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. 累计收益率曲线
    ax1 = fig.add_subplot(gs[0, :])
    if cumulative_returns is not None:
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                'b-', linewidth=2)
        ax1.fill_between(cumulative_returns.index, 0, 
                         cumulative_returns.values, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title(f'{fund_code} Cumulative Return')
        ax1.grid(True, alpha=0.3)
    
    # 2. 横截面分析
    ax2 = fig.add_subplot(gs[1, 0])
    if '横截面分析法' in analysis_results:
        cs = analysis_results['横截面分析法']
        if 'alpha1' in cs and 'alpha2' in cs:
            ax2.scatter([cs['alpha1']], [cs['alpha2']], s=200, c='blue', alpha=0.7)
            max_val = max(abs(cs['alpha1']), abs(cs['alpha2'])) * 1.2
            ax2.plot([-max_val, max_val], [-max_val, max_val], 
                    'k--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Period 1 Excess Return')
            ax2.set_ylabel('Period 2 Excess Return')
            verdict = cs.get('persistence', 'N/A')
            ax2.set_title(f'Cross-Section Analysis\n{verdict}')
            ax2.set_xlim(-max_val, max_val)
            ax2.set_ylim(-max_val, max_val)
            ax2.grid(True, alpha=0.3)
            ax2.text(0.05, 0.95, f'Same Sign: {cs.get("same_sign", "N/A")}', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top')
    
    # 3. CPR分析
    ax3 = fig.add_subplot(gs[1, 1])
    if '交叉积比率法' in analysis_results:
        cpr = analysis_results['交叉积比率法']
        if 'WW' in cpr:
            matrix = np.array([[cpr.get('WW', 0), cpr.get('WL', 0)],
                              [cpr.get('LW', 0), cpr.get('LL', 0)]])
            im = ax3.imshow(matrix, cmap='YlOrRd', aspect='equal')
            labels = [['WW', 'WL'], ['LW', 'LL']]
            for i in range(2):
                for j in range(2):
                    color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
                    ax3.text(j, i, f'{labels[i][j]}\n{matrix[i, j]}', 
                           ha='center', va='center', fontsize=14, 
                           color=color, fontweight='bold')
            ax3.set_xticks([0, 1])
            ax3.set_yticks([0, 1])
            ax3.set_xticklabels(['Winner', 'Loser'])
            ax3.set_yticklabels(['Winner', 'Loser'])
            cpr_val = cpr.get('CPR', np.nan)
            verdict = cpr.get('persistence_verdict', 'N/A')
            ax3.set_title(f'CPR Matrix\nCPR={cpr_val:.2f} ({verdict})')
            plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # 4. Hurst指数
    ax4 = fig.add_subplot(gs[2, 0])
    if 'Hurst指数法' in analysis_results:
        hurst = analysis_results['Hurst指数法']
        H = hurst.get('H', np.nan)
        
        # 绘制H值仪表
        if not np.isnan(H):
            bars = ax4.barh(['Hurst Index'], [H], color='steelblue', alpha=0.7)
            ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                       label='Random (H=0.5)')
            
            # 颜色编码
            if H > 0.5:
                bars[0].set_color('green')
                label_color = 'green'
                label_text = 'Persistence'
            else:
                bars[0].set_color('red')
                label_color = 'red'
                label_text = 'Reversal'
            
            ax4.set_xlim(0, 1)
            ax4.text(H, 0, f' H={H:.3f}', va='center', fontsize=12, fontweight='bold')
            ax4.text(0.05, 0.95, label_text, transform=ax4.transAxes, 
                    fontsize=14, fontweight='bold', color=label_color,
                    verticalalignment='top')
            
            verdict = hurst.get('persistence_verdict', 'N/A')
            ax4.set_title(f'Hurst Index Analysis\n{verdict}')
            ax4.legend(loc='lower right')
            ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. 综合判断
    ax5 = fig.add_subplot(gs[2, 1])
    verdicts = []
    if '横截面分析法' in analysis_results:
        cs = analysis_results['横截面分析法']
        verdicts.append(('Cross-Section', cs.get('persistence', 'N/A')))
    if '交叉积比率法' in analysis_results:
        cpr = analysis_results['交叉积比率法']
        verdicts.append(('CPR', cpr.get('persistence_verdict', 'N/A')))
    if 'Hurst指数法' in analysis_results:
        hurst = analysis_results['Hurst指数法']
        verdicts.append(('Hurst', hurst.get('persistence_verdict', 'N/A')))
    
    if verdicts:
        methods = [v[0] for v in verdicts]
        colors = []
        for _, v in verdicts:
            if '持续' in v and '无' not in v:
                colors.append('green')
            elif '反转' in v:
                colors.append('red')
            else:
                colors.append('gray')
        
        y_pos = np.arange(len(methods))
        ax5.barh(y_pos, [1]*len(methods), color=colors, alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(methods)
        ax5.set_xlim(0, 1.5)
        ax5.set_xticks([])
        
        for i, (method, v) in enumerate(verdicts):
            ax5.text(0.5, i, v, ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')
        
        ax5.set_title('Comprehensive Assessment')
    
    # 添加总标题
    overall = analysis_results.get('综合判断', 'N/A')
    fig.suptitle(f'{title}\nOverall: {overall}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [保存] 仪表盘已保存至 {save_path}")
    
    plt.show()
    return fig


# ============================================================
# 回测结果可视化
# ============================================================

def plot_backtest_results(backtest_results, save_path=None):
    """
    绘制回测结果
    
    Parameters:
    -----------
    backtest_results : dict
        回测结果
    save_path : str, optional
        保存路径
    """
    returns = backtest_results.get('returns', [])
    
    if len(returns) == 0:
        print("  [警告] 无回测收益数据")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 累计收益曲线
    ax1 = axes[0, 0]
    returns_df = pd.DataFrame(returns)
    cumulative = (1 + returns_df['return']).cumprod()
    ax1.plot(returns_df.index, cumulative, 'b-', linewidth=2)
    ax1.fill_between(returns_df.index, 1, cumulative, alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax1.set_title('Portfolio Cumulative Return')
    ax1.set_xlabel('Rebalance Period')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True, alpha=0.3)
    
    # 2. 收益分布
    ax2 = axes[0, 1]
    ax2.hist(returns_df['return'], bins=20, alpha=0.7, color='steelblue', 
            edgecolor='white')
    ax2.axvline(x=returns_df['return'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {returns_df["return"].mean():.3f}')
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_title('Return Distribution')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 绩效指标
    ax3 = axes[1, 0]
    perf = backtest_results.get('performance', {})
    if perf:
        metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
        values = []
        labels = []
        for m in metrics:
            if m in perf:
                values.append(abs(perf[m]))
                labels.append(m.replace('_', '\n'))
        
        colors = ['green' if 'return' in l.lower() else 'orange' for l in labels]
        bars = ax3.bar(labels, values, color=colors, alpha=0.7)
        ax3.set_title('Performance Metrics')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2%}', ha='center', va='bottom', fontsize=10)
    
    # 4. 持仓变化
    ax4 = axes[1, 1]
    holdings = backtest_results.get('holdings', [])
    if holdings:
        dates = [h['date'] for h in holdings]
        n_holdings = [len(h['funds']) for h in holdings]
        ax4.plot(range(len(dates)), n_holdings, 'o-', linewidth=2, markersize=8)
        ax4.set_title('Number of Holdings Over Time')
        ax4.set_xlabel('Rebalance Period')
        ax4.set_ylabel('Number of Holdings')
        ax4.set_ylim(0, max(n_holdings) * 1.2 if n_holdings else 10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  [保存] 回测结果图已保存至 {save_path}")
    
    plt.show()
    return fig
