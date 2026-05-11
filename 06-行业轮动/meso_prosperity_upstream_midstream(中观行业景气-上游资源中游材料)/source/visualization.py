"""
可视化模块
绘制景气度指数及相关分析图表

参考研报: 华泰证券-中观景气度之上游资源中游材料 (2021-10-14)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SentimentIndexVisualizer:
    """景气度指数可视化器"""

    def __init__(self, industry_name: str):
        self.industry_name = industry_name

    def plot_sentiment_index_with_roe(self,
                                     sentiment_index: pd.Series,
                                     roe: pd.Series,
                                     title: Optional[str] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制景气度指数与ROE_TTM对比图

        Parameters:
        -----------
        sentiment_index : pd.Series
            景气度指数
        roe : pd.Series
            ROE_TTM
        title : str, optional
            图表标题
        save_path : str, optional
            保存路径

        Returns:
        --------
        plt.Figure
        """
        if title is None:
            title = f'{self.industry_name}行业景气度指数与ROE_TTM'

        fig, ax1 = plt.subplots(figsize=(14, 6))

        common_idx = sentiment_index.index.intersection(roe.index)
        if len(common_idx) == 0:
            print("警告: 没有公共日期点")
            return fig

        index_aligned = sentiment_index.loc[common_idx]
        roe_aligned = roe.loc[common_idx]

        index_norm = (index_aligned - index_aligned.min()) / (index_aligned.max() - index_aligned.min())
        roe_norm = (roe_aligned - roe_aligned.min()) / (roe_aligned.max() - roe_aligned.min())

        color1 = '#1f77b4'
        color2 = '#ff7f0e'

        ax1.plot(index_norm.index, index_norm.values, color=color1, linewidth=2,
                label='景气度指数(标准化)')
        ax1.set_xlabel('日期', fontsize=12)
        ax1.set_ylabel('景气度指数(标准化)', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(roe_norm.index, roe_norm.values, color=color2, linewidth=2,
                label='ROE_TTM(标准化)')
        ax2.set_ylabel('ROE_TTM(标准化)', color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_direction_signals(self,
                              index: pd.Series,
                              roe: pd.Series,
                              window: int = 3,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制方向信号对比图

        Parameters:
        -----------
        index : pd.Series
            景气度指数
        roe : pd.Series
            ROE_TTM
        window : int
            滚动窗口
        save_path : str, optional
            保存路径

        Returns:
        --------
        plt.Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        index_rolled = index.rolling(window).mean()
        roe_rolled = roe.rolling(window).mean()

        index_change = index_rolled.diff(12)
        roe_change = roe_rolled.diff(4)

        common_idx = index_change.index.intersection(roe_change.index)

        index_dir = (index_change.loc[common_idx] > 0).astype(int)
        roe_dir = (roe_change.loc[common_idx] > 0).astype(int)

        ax1 = axes[0]
        ax1.fill_between(index_dir.index, 0, index_dir.values, alpha=0.5,
                        color='#1f77b4', label='景气度指数方向')
        ax1.set_ylabel('方向信号', fontsize=12)
        ax1.set_title(f'{self.industry_name} - 最新一期方向信号', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)

        ax2 = axes[1]
        ax2.fill_between(roe_dir.index, 0, roe_dir.values, alpha=0.5,
                        color='#ff7f0e', label='ROE_TTM方向')
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('方向信号', fontsize=12)
        ax2.set_title(f'{self.industry_name} - ROE_TTM季度变化方向', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_loadings(self,
                     loadings: pd.DataFrame,
                     top_n: int = 10,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制因子载荷图

        Parameters:
        -----------
        loadings : pd.DataFrame
            因子载荷数据，包含'indicator'和'loading'列
        top_n : int
            显示前n个指标
        save_path : str, optional
            保存路径

        Returns:
        --------
        plt.Figure
        """
        if loadings.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '没有载荷数据', ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        loadings_sorted = loadings.sort_values('loading', key=abs, ascending=False)
        loadings_top = loadings_sorted.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#d62728' if x < 0 else '#2ca02c' for x in loadings_top['loading']]

        bars = ax.barh(range(len(loadings_top)), loadings_top['loading'].values,
                      color=colors, alpha=0.7)

        ax.set_yticks(range(len(loadings_top)))
        ax.set_yticklabels(loadings_top['indicator'].values)
        ax.set_xlabel('标准化载荷系数', fontsize=12)
        ax.set_title(f'{self.industry_name} - 因子载荷 (Top {top_n})', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        for i, (bar, val) in enumerate(zip(bars, loadings_top['loading'].values)):
            if val >= 0:
                ax.text(val + 0.02, i, f'+{val:.2f}', va='center', fontsize=9)
            else:
                ax.text(val - 0.02, i, f'{val:.2f}', va='center', ha='right', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_multi_industry_comparison(self,
                                      indices_dict: Dict[str, pd.Series],
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制多行业景气度指数对比图

        Parameters:
        -----------
        indices_dict : Dict[str, pd.Series]
            行业名称到景气度指数的字典
        save_path : str, optional
            保存路径

        Returns:
        --------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.Set2(np.linspace(0, 1, len(indices_dict)))

        for (industry, index), color in zip(indices_dict.items(), colors):
            if len(index) == 0:
                continue

            index_norm = (index - index.min()) / (index.max() - index.min())

            ax.plot(index_norm.index, index_norm.values, color=color,
                   linewidth=2, label=industry, alpha=0.8)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('标准化景气度指数', fontsize=12)
        ax.set_title('多行业景气度指数对比', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_evaluation_metrics(self,
                               metrics: Dict[str, Dict],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制评估指标对比图

        Parameters:
        -----------
        metrics : Dict[str, Dict]
            评估指标字典
        save_path : str, optional
            保存路径

        Returns:
        --------
        plt.Figure
        """
        industries = list(metrics.keys())

        global_roe = [metrics[i].get('global', {}).get('roe_reproduction', 0) for i in industries]
        realtime_roe = [metrics[i].get('realtime', {}).get('roe_reproduction', 0)
                        if 'realtime' in metrics[i] else 0 for i in industries]

        global_dir = [metrics[i].get('global', {}).get('latest_direction_accuracy', 0)
                     for i in industries]
        realtime_dir = [metrics[i].get('realtime', {}).get('latest_direction_accuracy', 0)
                       if 'realtime' in metrics[i] else 0 for i in industries]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(industries))
        width = 0.35

        ax1 = axes[0]
        bars1 = ax1.bar(x - width/2, global_roe, width, label='全局指数', color='#1f77b4', alpha=0.7)
        bars2 = ax1.bar(x + width/2, realtime_roe, width, label='实时指数', color='#ff7f0e', alpha=0.7)

        ax1.set_xlabel('行业', fontsize=12)
        ax1.set_ylabel('ROE复现度 (R²)', fontsize=12)
        ax1.set_title('各行业ROE复现度对比', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(industries, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.0)

        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, global_dir, width, label='全局指数', color='#1f77b4', alpha=0.7)
        bars4 = ax2.bar(x + width/2, realtime_dir, width, label='实时指数', color='#ff7f0e', alpha=0.7)

        ax2.set_xlabel('行业', fontsize=12)
        ax2.set_ylabel('方向预测准确率', fontsize=12)
        ax2.set_title('各行业方向预测准确率对比', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(industries, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def plot_industry_chain(industry_name: str,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制行业产业链结构图

    Parameters:
    -----------
    industry_name : str
        行业名称
    save_path : str, optional
        保存路径

    Returns:
    --------
    plt.Figure
    """
    chain_data = {
        '石油石化': {
            'upstream': ['原油', '天然气'],
            'midstream': ['炼油', '化工原料'],
            'downstream': ['成品油', '化工产品', '沥青', '石蜡'],
            'applications': ['交通运输', '基础化工', '制造业']
        },
        '煤炭': {
            'upstream': ['原煤'],
            'midstream': ['动力煤', '炼焦煤', '喷吹煤', '无烟煤'],
            'downstream': ['火力发电', '冶金', '煤化工'],
            'applications': ['电力', '钢铁', '化工', '建材']
        },
        '有色金属': {
            'upstream': ['金属矿物'],
            'midstream': ['基本金属', '贵金属', '小金属'],
            'downstream': ['铜', '铝', '锌', '铅', '锡', '镍', '金', '银'],
            'applications': ['建材', '电子', '电力设备', '新能源']
        },
        '钢铁': {
            'upstream': ['铁矿石', '焦炭'],
            'midstream': ['粗钢', '钢材'],
            'downstream': ['螺纹钢', '线材', '热卷', '中板', '冷轧'],
            'applications': ['地产', '建筑', '汽车', '机械']
        },
        '基础化工': {
            'upstream': ['石油石化产品', '煤炭', '原盐'],
            'midstream': ['塑料', '橡胶', '化学纤维', '化肥农药'],
            'downstream': ['合成材料', '新型材料'],
            'applications': ['家电', '汽车', '纺织', '农业', '电子']
        },
        '建材': {
            'upstream': ['煤炭', '熟料', '纯碱', '非金属矿物'],
            'midstream': ['水泥', '玻璃', '玻璃纤维', '陶瓷'],
            'downstream': ['建筑用结构件', '装饰件'],
            'applications': ['建筑', '房地产', '汽车', '轻工']
        }
    }

    if industry_name not in chain_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'不支持的行业: {industry_name}', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    data = chain_data[industry_name]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    colors = {
        'upstream': '#66b3ff',
        'midstream': '#99ff99',
        'downstream': '#ffcc99',
        'applications': '#ff9999'
    }

    y_positions = {
        'upstream': 0.85,
        'midstream': 0.55,
        'downstream': 0.25,
        'applications': 0.05
    }

    labels = {
        'upstream': '上游原材料',
        'midstream': '中游产品',
        'downstream': '下游产成品',
        'applications': '应用领域'
    }

    for stage, (stage_name, items) in enumerate(data.items()):
        y = y_positions[stage]

        ax.text(0.02, y, labels[stage_name], fontsize=12, fontweight='bold',
               transform=ax.transAxes, verticalalignment='center')

        box_width = 0.22 * len(items)
        for i, item in enumerate(items):
            x = 0.15 + i * 0.15
            if stage == 'applications':
                x = 0.15 + i * 0.12

            rect = plt.Rectangle((x - 0.05, y - 0.08), 0.1, 0.12,
                                 facecolor=colors[stage_name], edgecolor='black',
                                 linewidth=1, transform=ax.transAxes)
            ax.add_patch(rect)

            ax.text(x, y - 0.02, item, fontsize=9, ha='center', va='center',
                   transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{industry_name} - 产业链结构', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == '__main__':
    print("测试可视化模块...")

    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='M')

    sentiment_index = pd.Series(np.cumsum(np.random.randn(120)) * 0.1, index=dates)
    roe = pd.Series(np.cumsum(np.random.randn(120)) * 0.5 + 10, index=dates)

    visualizer = SentimentIndexVisualizer('石油石化')

    print("\n1. 测试景气度指数与ROE对比图:")
    fig1 = visualizer.plot_sentiment_index_with_roe(sentiment_index, roe)
    print(f"  图表已创建: {fig1}")

    print("\n2. 测试方向信号图:")
    fig2 = visualizer.plot_direction_signals(sentiment_index, roe)
    print(f"  图表已创建: {fig2}")

    print("\n3. 测试载荷图:")
    loadings = pd.DataFrame({
        'indicator': ['指标1', '指标2', '指标3', '指标4', '指标5'],
        'loading': [0.85, 0.72, -0.45, 0.65, 0.38]
    })
    fig3 = visualizer.plot_loadings(loadings)
    print(f"  图表已创建: {fig3}")

    print("\n4. 测试产业链图:")
    fig4 = plot_industry_chain('石油石化')
    print(f"  图表已创建: {fig4}")

    plt.close('all')
    print("\n所有测试完成!")
