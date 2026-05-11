"""
可视化模块 - 绘制行业拆分和聚类结果图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class DivergenceVisualizer:
    """
    分化度可视化类
    """

    def __init__(self, figsize=(14, 8)):
        """
        Parameters:
        -----------
        figsize : tuple
            图表大小
        """
        self.figsize = figsize

    def plot_return_divergence_ranking(self, divergence_df, save_path=None):
        """
        绘制收益分化度排名

        Parameters:
        -----------
        divergence_df : pd.DataFrame
            分化度数据
        save_path : str
            保存路径
        """
        plt.figure(figsize=self.figsize)

        industries = divergence_df['industry'].values
        x = np.arange(len(industries))
        width = 0.25

        plt.barh(x - width, divergence_df['ls_rank'].values, width,
                label='Long-Short Return', color='#FF6B6B')
        plt.barh(x, divergence_df['reg_rank'].values, width,
                label='Regression R2', color='#4ECDC4')
        plt.barh(x + width, divergence_df['corr_rank'].values, width,
                label='Correlation', color='#45B7D1')

        plt.yticks(x, industries, fontsize=8)
        plt.xlabel('Ranking (Higher = More Divergent)', fontsize=12)
        plt.ylabel('Industry', fontsize=12)
        plt.title('Industry Return Divergence Ranking', fontsize=14)
        plt.legend(loc='lower right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()

    def plot_fundamental_divergence(self, divergence_df, metric='avg_div', save_path=None):
        """
        绘制基本面分化度

        Parameters:
        -----------
        divergence_df : pd.DataFrame
            分化度数据
        metric : str
            指标列名
        save_path : str
            保存路径
        """
        plt.figure(figsize=self.figsize)

        sorted_df = divergence_df.sort_values(metric, ascending=True)

        plt.barh(sorted_df['industry'].values, sorted_df[metric].values,
                color='#96CEB4', edgecolor='#45B7D1')

        plt.xlabel('Fundamental Divergence (CV)', fontsize=12)
        plt.ylabel('Industry', fontsize=12)
        plt.title('Industry Fundamental Divergence', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()

    def plot_split_comparison(self, eval_result, save_path=None):
        """
        绘制拆分前后对比

        Parameters:
        -----------
        eval_result : dict
            评估结果
        save_path : str
            保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        if 'return_evaluation' in eval_result:
            ret_eval = eval_result['return_evaluation']

            if 'original' in ret_eval:
                orig = ret_eval['original']
                labels = ['Original\n(All)', orig['industry']]
                intra_values = [orig['intra_industry_corr']]
                inter_values = [orig['inter_industry_corr']]

                if 'split' in ret_eval:
                    for sub, metrics in ret_eval['split'].items():
                        labels.append(f"\n{sub}")
                        intra_values.append(metrics['intra_industry_corr'])
                        inter_values.append(metrics['inter_industry_corr'])

                x = np.arange(len(labels))
                width = 0.35

                axes[0].bar(x - width/2, intra_values, width,
                           label='Intra-Industry', color='#FF6B6B')
                axes[0].bar(x + width/2, inter_values, width,
                           label='Inter-Industry', color='#4ECDC4')

                axes[0].set_ylabel('Correlation', fontsize=12)
                axes[0].set_title('Return Homogeneity Comparison', fontsize=14)
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(labels, fontsize=9)
                axes[0].legend()
                axes[0].grid(axis='y', alpha=0.3)

        if 'fundamental_evaluation' in eval_result:
            fund_eval = eval_result['fundamental_evaluation']
            categories = ['Original\nVariance', 'Split\nVariance']
            values = [fund_eval['original_variance'], fund_eval['split_variance']]

            axes[1].bar(categories, values, color=['#45B7D1', '#96CEB4'],
                       edgecolor='#333333')
            axes[1].set_ylabel('Joint Variance', fontsize=12)
            axes[1].set_title('Fundamental Homogeneity Comparison', fontsize=14)
            axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()


class ClusterVisualizer:
    """
    聚类结果可视化类
    """

    def __init__(self, figsize=(16, 12)):
        """
        Parameters:
        -----------
        figsize : tuple
            图表大小
        """
        self.figsize = figsize

    def plot_similarity_matrix(self, similarity_matrix, save_path=None):
        """
        绘制相似度矩阵热力图

        Parameters:
        -----------
        similarity_matrix : pd.DataFrame
            相似度矩阵
        save_path : str
            保存路径
        """
        plt.figure(figsize=self.figsize)

        plt.imshow(similarity_matrix.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Similarity Probability')

        n_industries = len(similarity_matrix.columns)
        tick_positions = np.arange(0, n_industries, max(1, n_industries // 15))
        tick_labels = [similarity_matrix.columns[i] for i in tick_positions]

        plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=8)
        plt.yticks(tick_positions, tick_labels, fontsize=8)

        plt.title('Industry Similarity Matrix (Monte Carlo K-Means)', fontsize=14)
        plt.xlabel('Industry', fontsize=12)
        plt.ylabel('Industry', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()

    def plot_mst_network(self, mst_edges, cluster_labels, save_path=None):
        """
        绘制最大生成树网络

        Parameters:
        -----------
        mst_edges : list
            MST边列表
        cluster_labels : dict
            聚类标签
        save_path : str
            保存路径
        """
        if not HAS_NETWORKX:
            print("请安装networkx库以绘制网络图")
            return

        plt.figure(figsize=self.figsize)

        G = nx.Graph()
        all_nodes = set()
        for edge in mst_edges:
            G.add_edge(edge[0], edge[1], weight=edge[2])
            all_nodes.add(edge[0])
            all_nodes.add(edge[1])

        for node in all_nodes:
            if node not in G.nodes():
                G.add_node(node)

        style_colors = {
            '周期': '#FF6B6B',
            '消费': '#4ECDC4',
            '金融': '#45B7D1',
            '成长': '#96CEB4',
            '稳定': '#FFEAA7'
        }

        node_colors = []
        for node in G.nodes():
            if node in cluster_labels:
                style = cluster_labels[node].get('style', 'Unknown')
                node_colors.append(style_colors.get(style, 'gray'))
            else:
                node_colors.append('gray')

        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                             node_size=1200, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                             width=2, alpha=0.6)

        edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                   font_size=7, alpha=0.6)

        nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, markersize=10, label=style)
                  for style, color in style_colors.items()]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=9)

        plt.title('Maximum Spanning Tree Industry Network', fontsize=14)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()

    def plot_cluster_result(self, cluster_result, save_path=None):
        """
        绘制聚类结果

        Parameters:
        -----------
        cluster_result : dict
            聚类结果
        save_path : str
            保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        style_colors = {
            '周期': '#FF6B6B',
            '消费': '#4ECDC4',
            '金融': '#45B7D1',
            '成长': '#96CEB4',
            '稳定': '#FFEAA7'
        }

        y_positions = {}
        current_y = 0
        y_spacing = 1

        for style, themes in cluster_result.items():
            y_positions[style] = current_y
            for theme, industries in themes.items():
                for i, industry in enumerate(industries):
                    color = style_colors.get(style, 'gray')
                    ax.barh(current_y - i * 0.3, 1, height=0.25,
                          color=color, edgecolor='white', alpha=0.8)
                    ax.text(-0.1, current_y - i * 0.3, industry,
                          ha='right', va='center', fontsize=8)

            current_y -= len(industries) * 0.3 + 1

        ax.set_xlim(-3, 1.5)
        ax.set_ylim(current_y - 2, y_positions[list(y_positions.keys())[0]] + 1)

        legend_elements = [plt.Rectangle((0, 0), 1, 1,
                   facecolor=color, edgecolor='white', label=style)
                  for style, color in style_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.set_title('Industry Cluster Result', fontsize=14)
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()


class TimeSeriesVisualizer:
    """
    时间序列可视化类
    """

    def __init__(self, figsize=(14, 8)):
        """
        Parameters:
        -----------
        figsize : tuple
            图表大小
        """
        self.figsize = figsize

    def plot_subindustry_price(self, price_data_dict, subindustries, save_path=None):
        """
        绘制子行业价格走势

        Parameters:
        -----------
        price_data_dict : dict
            子行业价格数据
        subindustries : list
            子行业列表
        save_path : str
            保存路径
        """
        plt.figure(figsize=self.figsize)

        for subindustry in subindustries:
            if subindustry in price_data_dict:
                data = price_data_dict[subindustry]
                if 'close' in data.columns:
                    normalized = data['close'] / data['close'].iloc[0] * 100
                    plt.plot(normalized.index, normalized.values,
                           label=subindustry, linewidth=1.5)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (Base=100)', fontsize=12)
        plt.title('Sub-Industry Price Trend', fontsize=14)
        plt.legend(loc='best', fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()

    def plot_correlation_matrix(self, corr_matrix, industries, save_path=None):
        """
        绘制相关系数矩阵

        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            相关系数矩阵
        industries : list
            行业列表
        save_path : str
            保存路径
        """
        plt.figure(figsize=(12, 10))

        ordered_industries = [ind for ind in industries if ind in corr_matrix.columns]
        ordered_corr = corr_matrix.loc[ordered_industries, ordered_industries]

        plt.imshow(ordered_corr.values, cmap='RdYlGn', aspect='auto',
                  vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')

        n = len(ordered_industries)
        tick_positions = np.arange(0, n, max(1, n // 10))
        tick_labels = [ordered_industries[i] for i in tick_positions]

        plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=8)
        plt.yticks(tick_positions, tick_labels, fontsize=8)

        plt.title('Industry Correlation Matrix', fontsize=14)

        for i in range(n):
            for j in range(n):
                if abs(ordered_corr.iloc[i, j]) > 0.5:
                    plt.text(j, i, f'{ordered_corr.iloc[i, j]:.2f}',
                           ha='center', va='center', fontsize=6)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()


if __name__ == "__main__":
    print("测试可视化模块...")
    print(f"NetworkX可用: {HAS_NETWORKX}")
    visualizer = DivergenceVisualizer()
    print("可视化器初始化完成")
