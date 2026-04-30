# -*- coding: utf-8 -*-
"""
聚类分析模块：基于K-means++的ETF跟踪指数聚类
基于民生证券研报《ETF的聚类优选与热点趋势策略构建》
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')


class ETFIndexClustering:
    """
    ETF跟踪指数聚类分析器
    
    使用K-means++算法对ETF跟踪指数进行聚类，根据成分股相似度
    将指数划分为若干类别，便于后续同类比较和优选。
    
    Parameters
    ----------
    n_clusters : int, optional
        聚类数量，默认None自动计算为n/5
    init : str, default='k-means++'
        初始化方法
    n_init : int, default=10
        运行次数
    max_iter : int, default=300
        最大迭代次数
    random_state : int, default=42
        随机种子
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        init: str = 'k-means++',
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        self.feature_matrix_ = None
        self.silhouette_score_ = None
        
    def build_similarity_matrix(self, constituents_dict: dict) -> pd.DataFrame:
        """
        构建成分股相似度矩阵
        
        Parameters
        ----------
        constituents_dict : dict
            字典，key为指数代码，value为成分股权重DataFrame
        
        Returns
        -------
        pd.DataFrame
            相似度矩阵
        """
        indices = list(constituents_dict.keys())
        n = len(indices)
        
        # 构建成分股-权重矩阵
        all_stocks = set()
        for df in constituents_dict.values():
            if df is not None and len(df) > 0:
                all_stocks.update(df['con_code'].tolist())
        
        all_stocks = sorted(list(all_stocks))
        
        # 创建股票到索引的映射
        stock_to_idx = {stock: i for i, stock in enumerate(all_stocks)}
        
        # 构建特征矩阵
        feature_matrix = np.zeros((n, len(all_stocks)))
        
        for i, idx_code in enumerate(indices):
            df = constituents_dict.get(idx_code)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    stock_code = row['con_code']
                    weight = row['weight']
                    if stock_code in stock_to_idx:
                        feature_matrix[i, stock_to_idx[stock_code]] = weight
        
        self.feature_matrix_ = feature_matrix
        self.indices_ = indices
        
        return pd.DataFrame(feature_matrix, index=indices, columns=all_stocks)
    
    def compute_similarity(self, X: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度矩阵
        
        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        
        Returns
        -------
        np.ndarray
            相似度矩阵
        """
        # 归一化
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        similarity = X_norm @ X_norm.T
        return similarity
    
    def fit_predict(
        self,
        X: np.ndarray,
        indices: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        拟合并预测聚类标签
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            特征矩阵
        indices : list, optional
            指数代码列表
        
        Returns
        -------
        np.ndarray
            聚类标签
        """
        if isinstance(X, pd.DataFrame):
            self.feature_matrix_ = X.values
            self.indices_ = X.index.tolist() if X.index is not None else None
            X = X.values
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 自动计算聚类数量
        if self.n_clusters is None:
            self.n_clusters = max(2, len(X) // 5)
            print(f"自动设置聚类数量: {self.n_clusters}")
        
        # K-means++聚类
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        self.cluster_labels_ = self.model.fit_predict(X_scaled)
        self.cluster_centers_ = self.model.cluster_centers_
        
        # 计算轮廓系数
        if len(X) >= 2 and len(set(self.cluster_labels_)) >= 2:
            try:
                self.silhouette_score_ = silhouette_score(X_scaled, self.cluster_labels_)
                print(f"轮廓系数: {self.silhouette_score_:.4f}")
            except Exception as e:
                print(f"计算轮廓系数失败: {e}")
        
        return self.cluster_labels_
    
    def get_cluster_info(self) -> pd.DataFrame:
        """
        获取聚类信息
        
        Returns
        -------
        pd.DataFrame
            包含指数代码和聚类标签的DataFrame
        """
        if self.cluster_labels_ is None:
            raise ValueError("模型尚未训练，请先调用fit_predict方法")
        
        result = pd.DataFrame({
            'index_code': self.indices_,
            'cluster': self.cluster_labels_
        })
        
        return result
    
    def get_cluster_summary(self) -> dict:
        """
        获取聚类摘要信息
        
        Returns
        -------
        dict
            聚类统计信息
        """
        if self.cluster_labels_ is None:
            raise ValueError("模型尚未训练")
        
        cluster_info = self.get_cluster_info()
        summary = {
            'n_clusters': self.n_clusters,
            'silhouette_score': self.silhouette_score_,
            'cluster_distribution': cluster_info['cluster'].value_counts().to_dict(),
            'indices_per_cluster': {}
        }
        
        for cluster_id in range(self.n_clusters):
            indices = cluster_info[cluster_info['cluster'] == cluster_id]['index_code'].tolist()
            summary['indices_per_cluster'][cluster_id] = {
                'count': len(indices),
                'indices': indices
            }
        
        return summary
    
    def kmeans_plusplus_init(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        K-means++初始化质心选择
        
        Parameters
        ----------
        X : np.ndarray
            数据点
        n_clusters : int
            聚类数量
        
        Returns
        -------
        np.ndarray
            初始质心
        """
        n_samples = X.shape[0]
        centroids = np.zeros((n_clusters, X.shape[1]))
        
        # 随机选择第一个质心
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]
        
        # 选择剩余质心
        for k in range(1, n_clusters):
            # 计算每个点到最近质心的距离
            distances = np.min(cdist(X, centroids[:k], 'euclidean'), axis=1)
            
            # 距离平方作为概率权重
            probs = distances ** 2
            probs /= probs.sum()
            
            # 轮盘赌选择下一个质心
            next_idx = np.random.choice(n_samples, p=probs)
            centroids[k] = X[next_idx]
        
        return centroids


def create_similarity_features(
    etf_df: pd.DataFrame,
    constituents_dict: dict
) -> pd.DataFrame:
    """
    创建用于聚类的相似度特征
    
    Parameters
    ----------
    etf_df : pd.DataFrame
        ETF基础信息DataFrame
    constituents_dict : dict
        指数成分股字典
    
    Returns
    -------
    pd.DataFrame
        特征矩阵
    """
    clustering = ETFIndexClustering()
    similarity_matrix = clustering.build_similarity_matrix(constituents_dict)
    
    # 计算成分股相似度作为特征
    similarity = clustering.compute_similarity(similarity_matrix.values)
    
    return pd.DataFrame(
        similarity,
        index=similarity_matrix.index,
        columns=similarity_matrix.index
    )


def cluster_indices_by_constituents(
    constituents_dict: dict,
    n_clusters: Optional[int] = None,
    **kwargs
) -> Tuple[pd.DataFrame, dict]:
    """
    对指数进行聚类分析
    
    Parameters
    ----------
    constituents_dict : dict
        指数成分股字典
    n_clusters : int, optional
        聚类数量
    **kwargs : dict
        其他聚类参数
    
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (聚类结果, 聚类摘要)
    """
    clustering = ETFIndexClustering(n_clusters=n_clusters, **kwargs)
    
    # 构建相似度矩阵
    similarity_matrix = clustering.build_similarity_matrix(constituents_dict)
    
    # 计算相似度特征
    features = clustering.compute_similarity(similarity_matrix.values)
    
    # 聚类
    labels = clustering.fit_predict(features)
    
    # 获取聚类信息
    cluster_result = clustering.get_cluster_info()
    summary = clustering.get_cluster_summary()
    
    return cluster_result, summary


def evaluate_clustering_quality(
    X: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    评估聚类质量
    
    Parameters
    ----------
    X : np.ndarray
        特征矩阵
    labels : np.ndarray
        聚类标签
    
    Returns
    -------
    dict
        评估指标
    """
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    
    metrics = {}
    
    try:
        # 轮廓系数
        if len(set(labels)) >= 2 and len(X) >= 2:
            metrics['silhouette'] = silhouette_score(X, labels)
    except Exception:
        metrics['silhouette'] = None
    
    try:
        # Calinski-Harabasz指数
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    except Exception:
        metrics['calinski_harabasz'] = None
    
    try:
        # Davies-Bouldin指数
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    except Exception:
        metrics['davies_bouldin'] = None
    
    return metrics


# ==================== 测试函数 ====================
if __name__ == '__main__':
    print("测试ETF指数聚类模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_indices = 30
    
    # 模拟成分股数据
    constituents_dict = {}
    index_codes = [f'00000{i}.SH' for i in range(n_indices)]
    
    for idx_code in index_codes:
        np.random.seed(hash(idx_code) % 2**32)
        n_stocks = np.random.randint(20, 50)
        stocks = []
        
        for j in range(n_stocks):
            stocks.append({
                'con_code': f'60000{j}.SH',
                'con_name': f'股票{j}',
                'weight': np.random.uniform(0.01, 0.1)
            })
        
        df = pd.DataFrame(stocks)
        df['weight'] = df['weight'] / df['weight'].sum()
        constituents_dict[idx_code] = df
    
    # 执行聚类
    print("\n执行指数聚类...")
    cluster_result, summary = cluster_indices_by_constituents(
        constituents_dict,
        random_state=42
    )
    
    print(f"\n聚类结果:")
    print(cluster_result.head(10))
    
    print(f"\n聚类摘要:")
    print(f"- 聚类数量: {summary['n_clusters']}")
    print(f"- 轮廓系数: {summary['silhouette_score']:.4f}")
    
    for cluster_id, info in summary['indices_per_cluster'].items():
        print(f"- 聚类{cluster_id}: {info['count']}只指数")
