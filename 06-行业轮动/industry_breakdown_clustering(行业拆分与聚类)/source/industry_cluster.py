"""
行业聚类模块 - 基于K-means和最大生成树算法构建行业关联网络
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class MonteCarloKMeans:
    """
    蒙特卡洛K-means聚类
    """

    def __init__(self, n_clusters=5, n_simulations=1000, random_state=42):
        """
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        n_simulations : int
            模拟次数
        random_state : int
            随机种子
        """
        self.n_clusters = n_clusters
        self.n_simulations = n_simulations
        self.random_state = random_state

    def single_kmeans(self, returns_df):
        """
        单次K-means聚类

        Parameters:
        -----------
        returns_df : pd.DataFrame
            行业收益率矩阵

        Returns:
        --------
        dict
            聚类结果
        """
        valid_data = returns_df.dropna(axis=1).T

        if valid_data.shape[0] < self.n_clusters:
            return None

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(valid_data)

        result = pd.DataFrame({
            'industry': valid_data.index,
            'cluster': labels
        })

        return result

    def calculate_similarity_matrix(self, returns_df):
        """
        计算行业相似度矩阵

        Parameters:
        -----------
        returns_df : pd.DataFrame
            行业收益率矩阵

        Returns:
        --------
        pd.DataFrame
            行业对被归为一类的概率矩阵
        """
        industries = returns_df.columns.tolist()
        n_industries = len(industries)
        similarity_matrix = np.zeros((n_industries, n_industries))

        for _ in range(self.n_simulations):
            cluster_result = self.single_kmeans(returns_df)
            if cluster_result is None:
                continue

            for i in range(n_industries):
                for j in range(i, n_industries):
                    ind_i = industries[i]
                    ind_j = industries[j]
                    if ind_i in cluster_result['industry'].values and ind_j in cluster_result['industry'].values:
                        cluster_i = cluster_result[cluster_result['industry'] == ind_i]['cluster'].values[0]
                        cluster_j = cluster_result[cluster_result['industry'] == ind_j]['cluster'].values[0]
                        if cluster_i == cluster_j:
                            similarity_matrix[i, j] += 1
                            similarity_matrix[j, i] += 1

        similarity_matrix /= self.n_simulations

        return pd.DataFrame(similarity_matrix, index=industries, columns=industries)


class MaximumSpanningTree:
    """
    最大生成树算法（基于Kruskal算法）
    """

    def __init__(self, similarity_matrix):
        """
        Parameters:
        -----------
        similarity_matrix : pd.DataFrame
            行业相似度矩阵
        """
        self.similarity_matrix = similarity_matrix
        self.n_nodes = len(similarity_matrix)
        self.nodes = list(similarity_matrix.columns)
        self.parent = {node: node for node in self.nodes}
        self.rank = {node: 0 for node in self.nodes}

    def find(self, x):
        """
        并查集查找

        Parameters:
        -----------
        x : str
            节点

        Returns:
        --------
        str
            根节点
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        并查集合并

        Parameters:
        -----------
        x : str
            节点1
        y : str
            节点2

        Returns:
        --------
        bool
            是否合并成功
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            return True
        return False

    def build_mst(self):
        """
        构建最大生成树

        Returns:
        --------
        list
            边列表[(节点1, 节点2, 权重), ...]
        """
        edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                weight = self.similarity_matrix.iloc[i, j]
                edges.append((self.nodes[i], self.nodes[j], weight))

        edges.sort(key=lambda x: x[2], reverse=True)

        mst_edges = []
        for edge in edges:
            node1, node2, weight = edge
            if self.union(node1, node2):
                mst_edges.append((node1, node2, weight))
                if len(mst_edges) == self.n_nodes - 1:
                    break

        return mst_edges

    def get_clusters(self):
        """
        获取聚类结果

        Returns:
        --------
        dict
            聚类标签
        """
        clusters = {}
        for node in self.nodes:
            root = self.find(node)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(node)

        return clusters


class IndustryClustering:
    """
    行业聚类主类
    """

    CLUSTER_RESULT = {
        '周期': {
            '上游资源': ['石油石化', '煤炭', '有色金属'],
            '中游材料': ['钢铁', '建材', '基础化工'],
            '中游制造': ['机械', '电力设备及新能源', '国防军工']
        },
        '消费': {
            '可选消费': ['汽车', '家电', '酒类'],
            '必须消费': ['食品', '饮料', '纺织服装', '医药', '农林牧渔',
                       '消费者服务', '商贸零售', '轻工制造']
        },
        '金融': {
            '大金融': ['银行', '证券', '保险', '多元金融', '综合金融', '房地产']
        },
        '成长': {
            'TMT': ['计算机', '电子', '传媒', '通信']
        },
        '稳定': {
            '公共产业': ['电力及公用事业', '交通运输', '建筑']
        }
    }

    def __init__(self, n_clusters=5, n_simulations=1000):
        """
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        n_simulations : int
            蒙特卡洛模拟次数
        """
        self.n_clusters = n_clusters
        self.n_simulations = n_simulations
        self.mc_kmeans = MonteCarloKMeans(n_clusters, n_simulations)
        self.similarity_matrix = None
        self.mst_edges = None
        self.clusters = None

    def fit(self, returns_df):
        """
        训练聚类模型

        Parameters:
        -----------
        returns_df : pd.DataFrame
            行业收益率矩阵
        """
        self.similarity_matrix = self.mc_kmeans.calculate_similarity_matrix(returns_df)

        mst = MaximumSpanningTree(self.similarity_matrix)
        self.mst_edges = mst.build_mst()
        self.clusters = mst.get_clusters()

    def get_max_weight_network(self):
        """
        获取最大权值边构建的网络

        Returns:
        --------
        dict
            每个行业与其最相似行业的连接
        """
        network = {}
        for industry in self.similarity_matrix.columns:
            row = self.similarity_matrix[industry].drop(industry)
            if len(row) > 0:
                max_similar = row.idxmax()
                max_weight = row.max()
                network[industry] = {'similar': max_similar, 'weight': max_weight}

        return network

    def get_cluster_labels(self):
        """
        获取聚类标签

        Returns:
        --------
        dict
            行业到聚类结果的映射
        """
        labels = {}
        for cluster_id, industries in self.clusters.items():
            for industry in industries:
                labels[industry] = cluster_id

        return labels

    def get_predefined_cluster_labels(self):
        """
        获取预定义的五大风格聚类标签

        Returns:
        --------
        dict
            行业到风格/板块的映射
        """
        labels = {}
        for style, themes in self.CLUSTER_RESULT.items():
            for theme, industries in themes.items():
                for industry in industries:
                    labels[industry] = {'style': style, 'theme': theme}

        return labels

    def visualize_network(self, save_path=None):
        """
        可视化行业关联网络

        Parameters:
        -----------
        save_path : str
            图片保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            G = nx.Graph()

            for industry in self.similarity_matrix.columns:
                G.add_node(industry)

            for edge in self.mst_edges:
                node1, node2, weight = edge
                G.add_edge(node1, node2, weight=weight)

            plt.figure(figsize=(20, 16))

            style_colors = {
                '周期': '#FF6B6B',
                '消费': '#4ECDC4',
                '金融': '#45B7D1',
                '成长': '#96CEB4',
                '稳定': '#FFEAA7'
            }

            labels = self.get_predefined_cluster_labels()
            node_colors = []
            for node in G.nodes():
                if node in labels:
                    style = labels[node]['style']
                    node_colors.append(style_colors.get(style, 'gray'))
                else:
                    node_colors.append('gray')

            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

            plt.title('Industry Association Network (Maximum Spanning Tree)', fontsize=16)
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"网络图已保存至: {save_path}")

            plt.show()

        except ImportError:
            print("请安装matplotlib和networkx库以进行可视化")


if __name__ == "__main__":
    print("测试行业聚类模块...")
    print(f"预定义聚类结果包含 {len(IndustryClustering.CLUSTER_RESULT)} 个风格")
    total_industries = sum(
        sum(len(industries) for industries in themes.values())
        for themes in IndustryClustering.CLUSTER_RESULT.values()
    )
    print(f"预定义聚类结果共 {total_industries} 个细分行业")
