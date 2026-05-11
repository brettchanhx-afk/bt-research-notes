"""
可视化模块
绘制回测结果图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# Windows 中文字体配置
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def setup_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120


def plot_nav_curve(results, save_path=None):
    """
    绘制净值曲线
    
    Args:
        results: 回测结果字典 {model: result}
        save_path: 保存路径
    """
    setup_chinese_font()
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (model, result) in enumerate(results.items()):
        nav = result['nav']
        color = colors[i % len(colors)]
        plt.plot(nav.index, nav.values, label=model, linewidth=1.5, color=color)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Net Asset Value', fontsize=12)
    plt.title('Global Asset Allocation Strategy - Net Asset Value Curve', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"净值曲线已保存至: {save_path}")
    
    plt.close()


def plot_weights_heatmap(weights_history, dates, asset_names, model_name, save_path=None):
    """
    绘制权重热力图
    
    Args:
        weights_history: 权重历史列表
        dates: 调仓日期
        asset_names: 资产名称列表
        model_name: 模型名称
        save_path: 保存路径
    """
    setup_chinese_font()
    
    # 转换为DataFrame
    weights_df = pd.DataFrame(weights_history, index=dates, columns=asset_names)
    
    plt.figure(figsize=(14, 6))
    
    # 绘制热力图
    im = plt.imshow(weights_df.T.values, aspect='auto', cmap='YlOrRd')
    
    # 设置轴标签
    plt.xlabel('Rebalance Date', fontsize=12)
    plt.ylabel('Asset', fontsize=12)
    plt.title(f'{model_name} - Weights Distribution', fontsize=14)
    
    # 设置x轴刻度
    n_ticks = min(10, len(dates))
    tick_positions = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
    tick_labels = [dates[i].strftime('%Y-%m') for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # 设置y轴刻度
    plt.yticks(range(len(asset_names)), asset_names)
    
    # 颜色条
    cbar = plt.colorbar(im, label='Weight')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"权重热力图已保存至: {save_path}")
    
    plt.close()


def plot_risk_contribution(weights_history, returns_history, asset_names, model_name, save_path=None):
    """
    绘制风险贡献分布图
    
    Args:
        weights_history: 权重历史列表
        returns_history: 收益率历史列表
        asset_names: 资产名称列表
        model_name: 模型名称
        save_path: 保存路径
    """
    setup_chinese_font()
    
    from source.models import risk_contribution
    
    # 计算每次调仓的风险贡献
    risk_contribs = []
    for weights, returns in zip(weights_history, returns_history):
        if returns is not None and len(returns) > 0:
            cov = returns.cov().values
            rc = risk_contribution(weights, cov)
            risk_contribs.append(rc)
    
    if not risk_contribs:
        print("无法计算风险贡献")
        return
    
    # 计算平均风险贡献
    avg_rc = np.mean(risk_contribs, axis=0)
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(asset_names))
    bars = plt.bar(x, avg_rc, color='steelblue', alpha=0.8)
    
    plt.xlabel('Asset', fontsize=12)
    plt.ylabel('Average Risk Contribution', fontsize=12)
    plt.title(f'{model_name} - Average Risk Contribution', fontsize=14)
    plt.xticks(x, asset_names, rotation=45)
    
    # 添加数值标签
    for bar, val in zip(bars, avg_rc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"风险贡献图已保存至: {save_path}")
    
    plt.close()


def plot_annual_returns(results, save_path=None):
    """
    绘制年度收益柱状图
    
    Args:
        results: 回测结果字典
        save_path: 保存路径
    """
    setup_chinese_font()
    
    # 计算各年度收益
    annual_returns = {}
    
    for model, result in results.items():
        returns = result['returns']
        returns_df = returns.to_frame('return')
        returns_df['year'] = returns_df.index.year
        annual = returns_df.groupby('year')['return'].apply(lambda x: (1+x).prod() - 1)
        annual_returns[model] = annual
    
    # 转换为DataFrame
    annual_df = pd.DataFrame(annual_returns)
    
    plt.figure(figsize=(14, 6))
    
    x = np.arange(len(annual_df))
    width = 0.12
    colors = plt.cm.Set2(np.linspace(0, 1, len(annual_df.columns)))
    
    for i, col in enumerate(annual_df.columns):
        offset = (i - len(annual_df.columns)/2 + 0.5) * width
        bars = plt.bar(x + offset, annual_df[col].values * 100, width, 
                      label=col, color=colors[i], alpha=0.8)
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Annual Return (%)', fontsize=12)
    plt.title('Annual Returns Comparison', fontsize=14)
    plt.xticks(x, annual_df.index)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加零线
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"年度收益图已保存至: {save_path}")
    
    plt.close()


def plot_correlation_matrix(returns, asset_names, save_path=None):
    """
    绘制相关系数矩阵
    
    Args:
        returns: 收益率数据
        asset_names: 资产名称
        save_path: 保存路径
    """
    setup_chinese_font()
    
    corr = returns.corr()
    
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    
    plt.colorbar(im, label='Correlation')
    
    # 设置刻度
    plt.xticks(range(len(asset_names)), asset_names, rotation=45)
    plt.yticks(range(len(asset_names)), asset_names)
    
    # 添加数值
    for i in range(len(asset_names)):
        for j in range(len(asset_names)):
            text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    plt.title('Global Index Correlation Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"相关系数矩阵已保存至: {save_path}")
    
    plt.close()


def plot_drawdown(nav, model_name, save_path=None):
    """
    绘制回撤曲线
    
    Args:
        nav: 净值序列
        model_name: 模型名称
        save_path: 保存路径
    """
    setup_chinese_font()
    
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    
    plt.figure(figsize=(12, 4))
    
    plt.fill_between(drawdown.index, drawdown.values * 100, 0, 
                    alpha=0.3, color='red')
    plt.plot(drawdown.index, drawdown.values * 100, 
            color='red', linewidth=1)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.title(f'{model_name} - Drawdown', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"回撤曲线已保存至: {save_path}")
    
    plt.close()


def plot_all_results(results, returns, output_dir='output'):
    """
    绘制所有结果图表
    
    Args:
        results: 回测结果字典
        returns: 原始收益率数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 净值曲线
    plot_nav_curve(results, os.path.join(output_dir, 'nav_curve.png'))
    
    # 2. 年度收益
    plot_annual_returns(results, os.path.join(output_dir, 'annual_returns.png'))
    
    # 3. 相关系数矩阵
    plot_correlation_matrix(returns, returns.columns.tolist(), 
                          os.path.join(output_dir, 'correlation_matrix.png'))
    
    print(f"\n所有图表已保存至: {output_dir}")


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    dates = pd.date_range('2010-01-01', periods=1000, freq='D')
    nav = pd.Series((1 + np.random.randn(1000) * 0.01).cumprod(), index=dates)
    
    plot_drawdown(nav, 'Test')
    print("测试完成")