"""
plot.py - 可视化绘图模块
提供Brinson归因结果的可视化展示
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（Windows系统）
def setup_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120
    
    # 清除字体缓存确保生效
    try:
        import os
        cache_dir = matplotlib.get_cachedir()
        font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
        if os.path.exists(font_cache):
            os.remove(font_cache)
    except:
        pass

# 初始化字体设置
setup_chinese_font()


class BrinsonVisualizer:
    """Brinson归因可视化器"""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        初始化可视化器
        
        Parameters:
            style: matplotlib样式
        """
        self.style = style
        plt.style.use(style)
        setup_chinese_font()  # 重新设置字体
    
    def plot_attribution_waterfall(
        self,
        attribution_result: Dict[str, float],
        title: str = "Brinson绩效归因",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制归因瀑布图
        
        Parameters:
            attribution_result: 归因结果字典，包含total, allocation, selection, interaction
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 准备数据
        categories = ['基准收益', '类别配置', '个券选择', '交互作用', '实际组合']
        
        # 计算累积值
        q1 = attribution_result.get('q1', 0)
        allocation = attribution_result.get('allocation', 0)
        selection = attribution_result.get('selection', 0)
        interaction = attribution_result.get('interaction', 0)
        
        values = [
            q1,
            q1 + allocation,
            q1 + allocation + selection,
            q1 + allocation + selection + interaction,
            attribution_result.get('q4', q1 + allocation + selection + interaction)
        ]
        
        # 绘制瀑布图
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, (cat, val) in enumerate(zip(categories, values)):
            if i == 0:
                # 基准收益
                ax.bar(i, val, color=colors[i], alpha=0.8, label=cat)
                ax.text(i, val/2, f'{val*100:.2f}%', ha='center', va='center', fontsize=10)
            elif i == len(categories) - 1:
                # 实际组合
                ax.bar(i, val, color=colors[i], alpha=0.8, label=cat)
                ax.text(i, val/2, f'{val*100:.2f}%', ha='center', va='center', fontsize=10)
            else:
                # 中间环节（增量）
                prev_val = values[i-1]
                increment = val - prev_val
                ax.bar(i, increment, bottom=prev_val, color=colors[i], alpha=0.8, label=cat)
                ax.text(i, prev_val + increment/2, f'{increment*100:.2f}%', 
                       ha='center', va='center', fontsize=10)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=15, ha='right')
        ax.set_ylabel('收益率', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left')
        
        # 添加网格
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        return fig
    
    def plot_sector_contribution(
        self,
        sector_contribution: pd.DataFrame,
        date: Optional[str] = None,
        top_n: int = 10,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制各行业归因贡献图
        
        Parameters:
            sector_contribution: 行业贡献DataFrame，包含sector, allocation, selection, interaction列
            date: 日期（用于标题）
            top_n: 显示前N个行业
            figsize: 图表尺寸
            save_path: 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 按总贡献排序
        df = sector_contribution.copy()
        df['total_abs'] = df['total'].abs()
        df = df.nlargest(top_n, 'total_abs')
        
        sectors = df['sector'].values
        
        # 子图1: 总贡献
        ax1 = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in df['total']]
        ax1.barh(sectors, df['total'] * 100, color=colors, alpha=0.7)
        ax1.set_xlabel('贡献 (%)', fontsize=10)
        ax1.set_title('总超额贡献', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 子图2: 类别配置贡献
        ax2 = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in df['allocation']]
        ax2.barh(sectors, df['allocation'] * 100, color=colors, alpha=0.7)
        ax2.set_xlabel('贡献 (%)', fontsize=10)
        ax2.set_title('类别配置贡献', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 子图3: 个券选择贡献
        ax3 = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in df['selection']]
        ax3.barh(sectors, df['selection'] * 100, color=colors, alpha=0.7)
        ax3.set_xlabel('贡献 (%)', fontsize=10)
        ax3.set_title('个券选择贡献', fontsize=12, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 子图4: 交互作用贡献
        ax4 = axes[1, 1]
        colors = ['green' if x > 0 else 'red' for x in df['interaction']]
        ax4.barh(sectors, df['interaction'] * 100, color=colors, alpha=0.7)
        ax4.set_xlabel('贡献 (%)', fontsize=10)
        ax4.set_title('交互作用贡献', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 总标题
        title = 'Brinson行业归因贡献'
        if date:
            title += f' ({date})'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        return fig
    
    def plot_time_series_attribution(
        self,
        attribution_df: pd.DataFrame,
        date_col: str = 'date',
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制归因时间序列图
        
        Parameters:
            attribution_df: 归因结果DataFrame，包含date, allocation, selection, interaction列
            date_col: 日期列名
            figsize: 图表尺寸
            save_path: 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        df = attribution_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        dates = df[date_col]
        
        # 子图1: 各类收益贡献时间序列
        ax1 = axes[0]
        ax1.plot(dates, df['allocation'] * 100, label='类别配置', linewidth=2, marker='o', markersize=4)
        ax1.plot(dates, df['selection'] * 100, label='个券选择', linewidth=2, marker='s', markersize=4)
        ax1.plot(dates, df['interaction'] * 100, label='交互作用', linewidth=2, marker='^', markersize=4)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('贡献 (%)', fontsize=11)
        ax1.set_title('Brinson归因时间序列', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 累计归因贡献
        ax2 = axes[1]
        df['cum_allocation'] = (1 + df['allocation']).cumprod() - 1
        df['cum_selection'] = (1 + df['selection']).cumprod() - 1
        df['cum_interaction'] = (1 + df['interaction']).cumprod() - 1
        df['cum_total'] = (1 + df['total']).cumprod() - 1
        
        ax2.plot(dates, df['cum_allocation'] * 100, label='类别配置累计', linewidth=2)
        ax2.plot(dates, df['cum_selection'] * 100, label='个券选择累计', linewidth=2)
        ax2.plot(dates, df['cum_interaction'] * 100, label='交互作用累计', linewidth=2)
        ax2.plot(dates, df['cum_total'] * 100, label='总超额累计', linewidth=2, color='black')
        ax2.set_ylabel('累计贡献 (%)', fontsize=11)
        ax2.set_title('累计归因贡献', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 四象限收益
        ax3 = axes[2]
        ax3.plot(dates, df['q1'] * 100, label='Q1 基准组合', linewidth=2, alpha=0.7)
        ax3.plot(dates, df['q2'] * 100, label='Q2 类别配置组合', linewidth=2, alpha=0.7)
        ax3.plot(dates, df['q3'] * 100, label='Q3 股票选择组合', linewidth=2, alpha=0.7)
        ax3.plot(dates, df['q4'] * 100, label='Q4 实际组合', linewidth=2, alpha=0.7)
        ax3.set_ylabel('收益率 (%)', fontsize=11)
        ax3.set_xlabel('日期', fontsize=11)
        ax3.set_title('Brinson四象限收益', fontsize=13, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        return fig
    
    def plot_attribution_summary(
        self,
        attribution_result: Dict[str, float],
        title: str = "归因结果摘要",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制归因结果摘要图（饼图/柱状图组合）
        
        Parameters:
            attribution_result: 归因结果字典
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 数据准备
        allocation = attribution_result.get('allocation', 0)
        selection = attribution_result.get('selection', 0)
        interaction = attribution_result.get('interaction', 0)
        total = attribution_result.get('total', 0)
        
        # 左图: 绝对贡献柱状图
        ax1 = axes[0]
        categories = ['类别配置', '个券选择', '交互作用']
        values = [allocation * 100, selection * 100, interaction * 100]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel('贡献 (%)', fontsize=11)
        ax1.set_title('各因素贡献分解', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10)
        
        # 右图: 相对占比饼图（只显示正值）
        ax2 = axes[1]
        positive_values = [max(0, v) for v in [allocation, selection, interaction]]
        if sum(positive_values) > 0:
            ax2.pie(positive_values, labels=categories, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax2.set_title('正贡献占比', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '无正贡献', ha='center', va='center', transform=ax2.transAxes)
        
        fig.suptitle(f"{title}\n总超额收益: {total*100:.2f}%", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        return fig
    
    def plot_heatmap(
        self,
        attribution_df: pd.DataFrame,
        value_col: str = 'total',
        date_col: str = 'date',
        sector_col: str = 'sector',
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制行业归因热力图
        
        Parameters:
            attribution_df: 归因结果DataFrame
            value_col: 数值列名
            date_col: 日期列名
            sector_col: 行业列名
            figsize: 图表尺寸
            save_path: 保存路径
        
        Returns:
            plt.Figure: 图表对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 透视表
        pivot_df = attribution_df.pivot(index=sector_col, columns=date_col, values=value_col)
        
        # 绘制热力图
        sns.heatmap(pivot_df * 100, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': '贡献 (%)'})
        
        ax.set_title(f'{value_col}归因热力图', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel('行业', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        return fig


def create_full_report(
    single_period_df: pd.DataFrame,
    multi_period_result: Dict[str, float],
    sector_contribution_df: pd.DataFrame,
    output_dir: str = './output'
):
    """
    创建完整的归因报告（包含所有图表）
    
    Parameters:
        single_period_df: 单期归因结果DataFrame
        multi_period_result: 多期累计归因结果字典
        sector_contribution_df: 行业贡献DataFrame
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = BrinsonVisualizer()
    
    # 1. 多期累计归因瀑布图
    print("生成多期累计归因瀑布图...")
    visualizer.plot_attribution_waterfall(
        multi_period_result,
        title="多期累计Brinson归因",
        save_path=os.path.join(output_dir, 'multi_period_waterfall.png')
    )
    
    # 2. 归因结果摘要
    print("生成归因结果摘要图...")
    visualizer.plot_attribution_summary(
        multi_period_result,
        save_path=os.path.join(output_dir, 'attribution_summary.png')
    )
    
    # 3. 时间序列归因图
    print("生成时间序列归因图...")
    visualizer.plot_time_series_attribution(
        single_period_df,
        save_path=os.path.join(output_dir, 'time_series_attribution.png')
    )
    
    # 4. 行业贡献图（最新一期）
    if 'date' in sector_contribution_df.columns:
        latest_date = sector_contribution_df['date'].max()
        latest_contrib = sector_contribution_df[sector_contribution_df['date'] == latest_date]
        print(f"生成行业贡献图 ({latest_date})...")
        visualizer.plot_sector_contribution(
            latest_contrib,
            date=latest_date,
            save_path=os.path.join(output_dir, 'sector_contribution.png')
        )
    
    print(f"\n所有图表已保存至: {output_dir}")


if __name__ == "__main__":
    # 测试可视化模块
    print("测试可视化模块...")
    
    visualizer = BrinsonVisualizer()
    
    # 测试归因结果
    test_result = {
        'total': 0.085,
        'allocation': 0.025,
        'selection': 0.045,
        'interaction': 0.015,
        'q1': 0.10,
        'q2': 0.125,
        'q3': 0.145,
        'q4': 0.185
    }
    
    # 测试瀑布图
    print("\n1. 测试瀑布图:")
    visualizer.plot_attribution_waterfall(test_result, title="测试归因瀑布图")
    plt.show()
    
    # 测试摘要图
    print("\n2. 测试摘要图:")
    visualizer.plot_attribution_summary(test_result, title="测试归因摘要")
    plt.show()
    
    print("\n可视化模块测试完成!")
