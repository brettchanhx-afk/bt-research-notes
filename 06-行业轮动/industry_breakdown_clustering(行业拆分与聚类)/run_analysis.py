import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

from source.data_fetcher import get_sw_industry_returns, SW_INDUSTRIES
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('output', exist_ok=True)

industry_returns = get_sw_industry_returns()
industry_returns = industry_returns.dropna(axis=1, how='all')
print(f'行业数: {industry_returns.shape[1]}')

corr_matrix = industry_returns.corr()
print('相关系数矩阵计算完成')

distance_matrix = 1 - corr_matrix
np.fill_diagonal(distance_matrix.values, 0)
dist_condensed = squareform(distance_matrix.values, checks=False)
linkage_matrix = linkage(dist_condensed, method='ward')
print('层次聚类完成')

G = nx.Graph()
industries = corr_matrix.columns.tolist()
for i in range(len(industries)):
    for j in range(i+1, len(industries)):
        weight = corr_matrix.iloc[i, j]
        G.add_edge(industries[i], industries[j], weight=weight)

mst = nx.maximum_spanning_tree(G)
mst_edges = pd.DataFrame([
    {'行业1': u, '行业2': v, '相关系数': d['weight']}
    for u, v, d in mst.edges(data=True)
])
mst_edges.to_csv('output/mst_edges.csv', index=False)
print('最大生成树计算完成，边数:', len(mst_edges))

fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(linkage_matrix, labels=industries, ax=ax, leaf_rotation=45)
ax.set_title('申万行业层次聚类树状图', fontsize=14)
ax.set_xlabel('行业', fontsize=12)
ax.set_ylabel('距离', fontsize=12)
plt.tight_layout()
plt.savefig('output/clustering_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print('树状图已保存')

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(industries)))
ax.set_yticks(range(len(industries)))
ax.set_xticklabels(industries, rotation=45, ha='right')
ax.set_yticklabels(industries)
ax.set_title('申万行业相关系数矩阵', fontsize=14)
plt.colorbar(im, ax=ax, label='相关系数')
plt.tight_layout()
plt.savefig('output/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('热力图已保存')

fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(mst, seed=42, k=2)
edges = mst.edges(data=True)
weights = [d['weight'] for u, v, d in edges]
nx.draw_networkx_nodes(mst, pos, ax=ax, node_size=800, node_color='lightblue')
nx.draw_networkx_labels(mst, pos, ax=ax, font_size=8)
nx.draw_networkx_edges(mst, pos, ax=ax, edge_color='gray', width=[w*3 for w in weights])
edge_labels = {}
for u, v, d in edges:
    edge_labels[(u, v)] = '{:.2f}'.format(d['weight'])
nx.draw_networkx_edge_labels(mst, pos, edge_labels, font_size=6, ax=ax)
ax.set_title('申万行业最大生成树', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig('output/maximum_spanning_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print('最大生成树图已保存')

cluster_summary = pd.DataFrame({
    '行业': industries,
    '英文名': [SW_INDUSTRIES.get(ind, ind) for ind in industries]
})
cluster_summary.to_csv('output/cluster_summary.csv', index=False)
print('聚类摘要已保存')

print('\n所有分析完成!')
print('\n输出文件:')
for f in os.listdir('output'):
    print(f'  - output/{f}')