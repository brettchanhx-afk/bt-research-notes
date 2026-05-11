# 行业拆分与聚类量化研究项目

## 项目概述

本项目复现了华泰证券金工研究报告《中观基本面轮动系列之一：确立研究对象-行业拆分与聚类》(2020-03-03) 的核心方法论。

### 原始研报核心结论

1. **行业拆分**：将食品饮料拆分为酒类、饮料、食品；将非银行金融拆分为证券、保险、多元金融
2. **行业聚类**：将33个细分行业聚类为五大风格、八大主题板块

## 项目结构

```
industry_breakdown_clustering/
├── source/                      # 源代码模块
│   ├── __init__.py             # 包初始化
│   ├── config.py               # 配置文件
│   ├── data_fetcher.py         # 数据获取模块
│   ├── industry_split.py        # 行业拆分模块
│   ├── industry_cluster.py      # 行业聚类模块
│   ├── evaluation.py            # 评估模块
│   └── visualization.py         # 可视化模块
├── ipynb/                       # Jupyter notebooks
│   └── industry_clustering_analysis.ipynb  # 完整分析流程
├── output/                      # 输出结果目录
├── 参考研报/                    # 原始研报PDF
└── README.md                    # 项目说明文档
```

## 核心方法论

### 行业拆分方法

#### 收益率分化度刻画

1. **多空累计收益法**：将行业内个股按收益率分层，计算多空收益差
2. **回归拟合优度法**：计算个股对行业指数的回归R²均值
3. **平均相关系数法**：计算行业内个股收益率的相关系数均值

通过蒙特卡洛模拟（1000次）得到稳健的分化度排名。

#### 基本面分化度刻画

从三个维度进行评估：
- **估值维度**：PE、PB
- **盈利维度**：ROE、ROA、ROIC、净利率、毛利率
- **营运维度**：资产负债率、总资产周转率、存货周转率

#### 拆分结论

| 原行业 | 拆分后子行业 | 拆分效果 |
|--------|-------------|---------|
| 食品饮料 | 酒类、饮料、食品 | 酒类行业内相关系数从0.31提升至0.44 |
| 非银行金融 | 证券、保险、多元金融 | 证券行业内相关系数从0.49提升至0.60 |

### 行业聚类方法

#### 聚类流程

1. 对行业收益率序列进行K-means聚类（k=5）
2. 重复1000次蒙特卡洛模拟
3. 计算每两个行业被归为一类的概率
4. 基于最大生成树算法（Kruskal）剪枝

#### 聚类结果

| 风格 | 主题板块 | 细分行业 |
|------|---------|---------|
| **周期** | 上游资源 | 石油石化、煤炭、有色金属 |
|  | 中游材料 | 钢铁、建材、基础化工 |
|  | 中游制造 | 机械、电力设备及新能源、国防军工 |
| **消费** | 可选消费 | 汽车、家电、酒类 |
|  | 必须消费 | 食品、饮料、纺织服装、医药、农林牧渔、消费者服务、商贸零售、轻工制造 |
| **金融** | 大金融 | 银行、证券、保险、多元金融、综合金融、房地产 |
| **成长** | TMT | 计算机、电子、传媒、通信 |
| **稳定** | 公共产业 | 电力及公用事业、交通运输、建筑 |

## 依赖库

```
pandas>=1.0.0
numpy>=1.18.0
scipy>=1.4.0
scikit-learn>=0.22.0
matplotlib>=3.2.0
networkx>=2.5.0 (可选，用于网络图绘制)
tushare>=1.2.0
```

## 安装与使用

### 1. 安装依赖

```bash
pip install pandas numpy scipy scikit-learn matplotlib networkx tushare
```

### 2. 配置Tushare Token

本项目使用tushare获取数据，请在 `source/data_fetcher.py` 中配置您的token：

```python
TOKEN = "your_token_here"
pro = ts.pro_api(TOKEN)
pro._DataApi__token = TOKEN
pro._DataApi__http_url = "http://jiaoch.site"
```

### 3. 运行分析

```bash
cd ipynb
jupyter notebook industry_clustering_analysis.ipynb
```

或者在Python中直接使用模块：

```python
from source.industry_cluster import IndustryClustering, MonteCarloKMeans

# 初始化聚类器
clustering = IndustryClustering(n_clusters=5, n_simulations=1000)

# 拟合数据
clustering.fit(returns_df)

# 获取聚类结果
cluster_labels = clustering.get_predefined_cluster_labels()
```

## 核心API说明

### 数据获取模块 (data_fetcher.py)

```python
# 获取行业成分股
get_industry_stocks(industry_name, trade_date)

# 获取行业指数日线数据
get_industry_index_daily(industry_code, start_date, end_date)

# 获取个股日线数据
get_stock_daily(ts_code, start_date, end_date)

# 批量获取个股数据
batch_get_stock_daily(ts_codes, start_date, end_date)
```

### 行业拆分模块 (industry_split.py)

```python
# 收益率分化度计算器
return_divergence = IndustryReturnDivergence(n_simulations=1000)

# 计算分化度
divergence = return_divergence.monte_carlo_simulation(stock_returns, industry_returns)

# 行业拆分决策
splitter = IndustrySplitter()
subindustries = splitter.get_split_subindustries('食品饮料')
```

### 行业聚类模块 (industry_cluster.py)

```python
# 蒙特卡洛K-means
mc_kmeans = MonteCarloKMeans(n_clusters=5, n_simulations=1000)
similarity_matrix = mc_kmeans.calculate_similarity_matrix(returns_df)

# 最大生成树
mst = MaximumSpanningTree(similarity_matrix)
mst_edges = mst.build_mst()
clusters = mst.get_clusters()

# 完整聚类流程
clustering = IndustryClustering(n_clusters=5, n_simulations=1000)
clustering.fit(returns_df)
```

### 评估模块 (evaluation.py)

```python
# 综合评估器
evaluator = ComprehensiveEvaluator()

# 评估拆分效果
result = evaluator.evaluate_industry_split(
    returns_df, fundamental_df,
    '食品饮料', ['酒类', '饮料', '食品'],
    original_stocks, subindustry_stocks_dict,
    n_industries_new=33, n_industries_old=29
)

# 生成评估报告
report = evaluator.generate_evaluation_report(result)
```

### 可视化模块 (visualization.py)

```python
# 分化度可视化
div_viz = DivergenceVisualizer()
div_viz.plot_return_divergence_ranking(divergence_df)
div_viz.plot_split_comparison(eval_result)

# 聚类可视化
cluster_viz = ClusterVisualizer()
cluster_viz.plot_similarity_matrix(similarity_matrix)
cluster_viz.plot_mst_network(mst_edges, cluster_labels)
```

## 输出文件

| 文件名 | 说明 |
|--------|------|
| `cluster_result.csv` | 聚类结果表 |
| `industry_divergence_ranking.csv` | 行业分化度排名 |
| `similarity_matrix.csv` | 行业相似度矩阵 |
| `mst_edges.csv` | 最大生成树边列表 |
| `split_evaluation.csv` | 拆分效果评估结果 |
| `industry_network.png` | 行业关联网络图 |

## 数据说明

### 中信一级行业列表（30个）

能源: 石油石化、煤炭
材料: 有色金属、钢铁、基础化工、建材
制造: 机械、电力设备及新能源、国防军工
消费: 汽车、家电、食品饮料、纺织服装、医药、农林牧渔、商贸零售、轻工制造、消费者服务
金融: 银行、非银行金融、房地产
成长: 计算机、电子、传媒、通信
稳定: 交通运输、电力及公用事业、建筑
其他: 综合

### 拆分后行业列表（33个）

在原30个一级行业基础上：
- 食品饮料 → 酒类、饮料、食品（+2）
- 非银行金融 → 证券、保险、多元金融（+2）
- 剔除综合行业（-1）

## 注意事项

1. **API限制**: tushare有API调用频率限制，大批量数据获取需要分批进行
2. **数据延迟**: 财务数据可能有季度延迟，需要注意数据完整性
3. **计算耗时**: 蒙特卡洛模拟1000次可能耗时较长，可根据需要调整
4. **NetworkX可选**: 如不需要绘制网络图，可不安装networkx库

## 风险提示

1. 模型基于历史数据建模，历史规律可能失效
2. 行业拆分与聚类结果反映的是历史产业链关系
3. 随着经济结构转型升级，行业分类可能需要调整
4. 实际使用时请结合市场环境和主观判断

## 参考资料

- 华泰证券研究所，《中观基本面轮动系列之一：确立研究对象-行业拆分与聚类》，2020-03-03
- 中信证券行业分类标准
- tushare数据接口文档

## 许可

本项目仅供学习研究使用，不构成投资建议。
