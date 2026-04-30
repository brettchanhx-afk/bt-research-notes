# -*- coding: utf-8 -*-
"""
可视化模块：ETF聚类优选系统图表绘制
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150

try:
    setup_chinese_font()
except Exception:
    pass


class ETFPlotter:
    """
    ETF聚类优选系统可视化类
    """
    
    def __init__(self, output_dir: str = None, style: str = 'seaborn-v0_8-whitegrid'):
        self.output_dir = output_dir or 'output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.style.use(style)
        setup_chinese_font()
    
    def plot_clustering_results(
        self,
        cluster_result: pd.DataFrame,
        similarity_matrix: pd.DataFrame = None,
        save_path: str = None
    ):
        """
        绘制聚类结果
        
        Parameters
        ----------
        cluster_result : pd.DataFrame
            聚类结果
        similarity_matrix : pd.DataFrame, optional
            相似度矩阵
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 图1：聚类分布饼图
        ax1 = axes[0]
        cluster_counts = cluster_result['cluster'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        
        ax1.pie(
            cluster_counts.values,
            labels=[f'Cluster {i}\n({v} indices)' for i, v in enumerate(cluster_counts.values)],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('ETF Index Clustering Distribution', fontsize=14)
        
        # 图2：聚类分布柱状图
        ax2 = axes[1]
        bars = ax2.bar(
            cluster_counts.index,
            cluster_counts.values,
            color=colors,
            edgecolor='black',
            alpha=0.8
        )
        ax2.set_xlabel('Cluster ID', fontsize=12)
        ax2.set_ylabel('Number of Indices', fontsize=12)
        ax2.set_title('Indices per Cluster', fontsize=14)
        
        # 添加数值标签
        for bar, count in zip(bars, cluster_counts.values):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                str(count),
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_cluster_similarity(
        self,
        similarity_matrix: pd.DataFrame,
        cluster_labels: np.ndarray = None,
        save_path: str = None
    ):
        """
        绘制相似度热力图
        
        Parameters
        ----------
        similarity_matrix : pd.DataFrame
            相似度矩阵
        cluster_labels : np.ndarray, optional
            聚类标签
        save_path : str, optional
            保存路径
        """
        if similarity_matrix is None or len(similarity_matrix) == 0:
            print("相似度矩阵为空，跳过绘制")
            return None
        
        # 限制显示数量
        max_display = 30
        if len(similarity_matrix) > max_display:
            similarity_matrix = similarity_matrix.iloc[:max_display, :max_display]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            similarity_matrix.values,
            cmap='YlOrRd',
            center=similarity_matrix.values.mean(),
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            cbar_kws={'label': 'Similarity'}
        )
        
        ax.set_title('Index Similarity Heatmap (Top 30)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_evaluation_scores(
        self,
        evaluation: pd.DataFrame,
        score_col: str = 'comprehensive_score',
        top_n: int = 20,
        save_path: str = None
    ):
        """
        绘制评价得分排名
        
        Parameters
        ----------
        evaluation : pd.DataFrame
            评价结果
        score_col : str
            得分列名
        top_n : int
            显示前N个
        save_path : str, optional
            保存路径
        """
        if score_col not in evaluation.columns:
            print(f"列 {score_col} 不存在，使用默认列")
            return None
        
        # 排序并取前N
        top_etfs = evaluation.nlargest(top_n, score_col)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_etfs)))
        
        bars = ax.barh(
            range(len(top_etfs)),
            top_etfs[score_col].values,
            color=colors,
            edgecolor='black',
            alpha=0.8
        )
        
        # 设置标签
        labels = top_etfs['fund_name'].tolist() if 'fund_name' in top_etfs.columns else top_etfs['index_code'].tolist()
        ax.set_yticks(range(len(top_etfs)))
        ax.set_yticklabels(labels, fontsize=9)
        
        ax.set_xlabel(score_col, fontsize=12)
        ax.set_title(f'Top {top_n} ETF Evaluation Scores ({score_col})', fontsize=14)
        ax.invert_yaxis()
        
        # 添加数值标签
        for bar, score in zip(bars, top_etfs[score_col].values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{score:.3f}',
                ha='left',
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_backtest_results(
        self,
        backtest_result: pd.DataFrame,
        save_path: str = None
    ):
        """
        绘制回测结果
        
        Parameters
        ----------
        backtest_result : pd.DataFrame
            回测结果
        save_path : str, optional
            保存路径
        """
        if 'date' not in backtest_result.columns and 'period' not in backtest_result.columns:
            print("回测结果缺少日期列")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 确定日期列
        date_col = 'date' if 'date' in backtest_result.columns else 'period'
        
        # 图1：累计收益对比
        ax1 = axes[0, 0]
        if 'portfolio_value' in backtest_result.columns:
            ax1.plot(
                backtest_result[date_col],
                backtest_result['portfolio_value'],
                label='Selected ETF Portfolio',
                linewidth=2
            )
        if 'benchmark_value' in backtest_result.columns:
            ax1.plot(
                backtest_result[date_col],
                backtest_result['benchmark_value'],
                label='Benchmark (CSI 300)',
                linewidth=2,
                alpha=0.7
            )
        if 'all_etf_value' in backtest_result.columns:
            ax1.plot(
                backtest_result[date_col],
                backtest_result['all_etf_value'],
                label='All ETF Average',
                linewidth=2,
                alpha=0.5
            )
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_title('Cumulative Return Comparison', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：超额收益
        ax2 = axes[0, 1]
        if 'excess_return' in backtest_result.columns:
            ax2.bar(
                backtest_result[date_col],
                backtest_result['excess_return'] * 100,
                color='steelblue',
                alpha=0.7
            )
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Excess Return (%)', fontsize=12)
        ax2.set_title('Excess Return vs Benchmark', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 图3：收益分布箱线图
        ax3 = axes[1, 0]
        if 'period_return' in backtest_result.columns:
            returns = backtest_result['period_return'] * 100
            bp = ax3.boxplot([returns], labels=['Selected Portfolio'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
        
        if 'benchmark_return' in backtest_result.columns:
            bm_returns = backtest_result['benchmark_return'] * 100
            bp2 = ax3.boxplot([bm_returns], positions=[2], widths=0.6, patch_artist=True)
            bp2['boxes'][0].set_facecolor('lightyellow')
        
        ax3.set_ylabel('Period Return (%)', fontsize=12)
        ax3.set_title('Return Distribution', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 图4：按类型统计
        ax4 = axes[1, 1]
        if 'etf_type' in backtest_result.columns:
            type_returns = backtest_result.groupby('etf_type')['period_return'].mean() * 100
            colors = plt.cm.Set2(np.linspace(0, 1, len(type_returns)))
            bars = ax4.bar(type_returns.index, type_returns.values, color=colors, edgecolor='black')
            ax4.set_xlabel('ETF Type', fontsize=12)
            ax4.set_ylabel('Average Return (%)', fontsize=12)
            ax4.set_title('Return by ETF Type', fontsize=14)
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, ret in zip(bars, type_returns.values):
                ax4.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f'{ret:.2f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_cluster_comparison(
        self,
        evaluation: pd.DataFrame,
        cluster_id: int,
        save_path: str = None
    ):
        """
        绘制同类指数对比
        
        Parameters
        ----------
        evaluation : pd.DataFrame
            评价结果
        cluster_id : int
            聚类ID
        save_path : str, optional
            保存路径
        """
        cluster_data = evaluation[evaluation['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            print(f"聚类 {cluster_id} 无数据")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ROE对比
        ax1 = axes[0, 0]
        if 'roe_ttm' in cluster_data.columns:
            valid_data = cluster_data.dropna(subset=['roe_ttm'])
            if len(valid_data) > 0:
                ax1.barh(
                    range(len(valid_data)),
                    valid_data['roe_ttm'].values,
                    color='steelblue',
                    alpha=0.8
                )
                labels = valid_data['index_code'].tolist()
                ax1.set_yticks(range(len(valid_data)))
                ax1.set_yticklabels(labels, fontsize=8)
                ax1.set_xlabel('ROE TTM (%)', fontsize=12)
                ax1.set_title(f'Cluster {cluster_id}: ROE Comparison', fontsize=12)
        
        # 营收同比对比
        ax2 = axes[0, 1]
        if 'revenue_yoy' in cluster_data.columns:
            valid_data = cluster_data.dropna(subset=['revenue_yoy'])
            if len(valid_data) > 0:
                colors = ['green' if x > 0 else 'red' for x in valid_data['revenue_yoy']]
                ax2.barh(
                    range(len(valid_data)),
                    valid_data['revenue_yoy'].values,
                    color=colors,
                    alpha=0.8
                )
                ax2.set_yticks(range(len(valid_data)))
                ax2.set_yticklabels(valid_data['index_code'].tolist(), fontsize=8)
                ax2.set_xlabel('Revenue YoY (%)', fontsize=12)
                ax2.set_title(f'Cluster {cluster_id}: Revenue Growth', fontsize=12)
        
        # 夏普比率对比
        ax3 = axes[1, 0]
        if 'sharpe_all' in cluster_data.columns:
            valid_data = cluster_data.dropna(subset=['sharpe_all']).nlargest(10, 'sharpe_all')
            if len(valid_data) > 0:
                ax3.barh(
                    range(len(valid_data)),
                    valid_data['sharpe_all'].values,
                    color='orange',
                    alpha=0.8
                )
                ax3.set_yticks(range(len(valid_data)))
                ax3.set_yticklabels(valid_data['index_code'].tolist(), fontsize=8)
                ax3.set_xlabel('Sharpe Ratio', fontsize=12)
                ax3.set_title(f'Cluster {cluster_id}: Sharpe Ratio (Top 10)', fontsize=12)
        
        # 综合得分对比
        ax4 = axes[1, 1]
        score_col = 'comprehensive_score' if 'comprehensive_score' in cluster_data.columns else 'financial_score'
        if score_col in cluster_data.columns:
            valid_data = cluster_data.dropna(subset=[score_col]).nlargest(10, score_col)
            if len(valid_data) > 0:
                ax4.barh(
                    range(len(valid_data)),
                    valid_data[score_col].values,
                    color='purple',
                    alpha=0.8
                )
                ax4.set_yticks(range(len(valid_data)))
                ax4.set_yticklabels(valid_data['index_code'].tolist(), fontsize=8)
                ax4.set_xlabel(score_col, fontsize=12)
                ax4.set_title(f'Cluster {cluster_id}: {score_col} (Top 10)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_etf_comparison(
        self,
        etf_data: pd.DataFrame,
        save_path: str = None
    ):
        """
        绘制ETF综合对比
        
        Parameters
        ----------
        etf_data : pd.DataFrame
            ETF数据
        save_path : str, optional
            保存路径
        """
        if len(etf_data) == 0:
            print("ETF数据为空")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 费率分布
        ax1 = axes[0, 0]
        if 'mgmt_fee' in etf_data.columns:
            fee_data = (etf_data['mgmt_fee'] + etf_data.get('custody_fee', 0)) * 100
            ax1.hist(fee_data, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Fee Rate (%)', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title('Fee Rate Distribution', fontsize=14)
        
        # 2. 规模分布
        ax2 = axes[0, 1]
        if 'scale' in etf_data.columns:
            ax2.hist(etf_data['scale'], bins=20, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Scale (100M Yuan)', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Scale Distribution', fontsize=14)
        
        # 3. 流动性分布
        ax3 = axes[0, 2]
        if 'avg_daily_volume' in etf_data.columns:
            ax3.hist(etf_data['avg_daily_volume'], bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Avg Daily Volume (100M)', fontsize=12)
            ax3.set_ylabel('Count', fontsize=12)
            ax3.set_title('Liquidity Distribution', fontsize=14)
        
        # 4. 综合得分分布
        ax4 = axes[1, 0]
        if 'comprehensive_score' in etf_data.columns:
            ax4.hist(etf_data['comprehensive_score'], bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Comprehensive Score', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title('Comprehensive Score Distribution', fontsize=14)
        
        # 5. ETF类型分布
        ax5 = axes[1, 1]
        if 'etf_type' in etf_data.columns:
            type_counts = etf_data['etf_type'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
            ax5.pie(type_counts.values, labels=type_counts.index, colors=colors, autopct='%1.1f%%')
            ax5.set_title('ETF Type Distribution', fontsize=14)
        
        # 6. 费率vs规模
        ax6 = axes[1, 2]
        if 'mgmt_fee' in etf_data.columns and 'scale' in etf_data.columns:
            fee_rate = (etf_data['mgmt_fee'] + etf_data.get('custody_fee', 0)) * 100
            ax6.scatter(etf_data['scale'], fee_rate, alpha=0.5, c='steelblue')
            ax6.set_xlabel('Scale (100M Yuan)', fontsize=12)
            ax6.set_ylabel('Fee Rate (%)', fontsize=12)
            ax6.set_title('Fee vs Scale', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        plt.show()
        return fig


# ==================== 便捷函数 ====================
def plot_etf_clustering(evaluation: pd.DataFrame, output_dir: str = 'output', **kwargs):
    """
    便捷函数：绘制ETF聚类结果
    
    Parameters
    ----------
    evaluation : pd.DataFrame
        评价结果
    output_dir : str
        输出目录
    """
    plotter = ETFPlotter(output_dir)
    
    # 绘制评价得分
    if 'comprehensive_score' in evaluation.columns:
        plotter.plot_evaluation_scores(
            evaluation,
            score_col='comprehensive_score',
            save_path=os.path.join(output_dir, 'etf_scores_top20.png')
        )
    
    return plotter


# ==================== 测试函数 ====================
if __name__ == '__main__':
    print("测试可视化模块...")
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = ETFPlotter(output_dir)
    
    # 测试聚类结果绘制
    np.random.seed(42)
    cluster_result = pd.DataFrame({
        'index_code': [f'00000{i}.SH' for i in range(30)],
        'cluster': np.random.randint(0, 6, 30)
    })
    
    fig = plotter.plot_clustering_results(
        cluster_result,
        save_path=os.path.join(output_dir, 'test_clustering.png')
    )
    
    print("\n可视化测试完成！")
