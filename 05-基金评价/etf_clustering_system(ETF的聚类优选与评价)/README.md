# ETF聚类优选系统

基于民生证券研报《ETF的聚类优选与热点趋势策略构建》的量化投资策略实现。

## 项目概述

本项目实现了一套完整的ETF聚类优选框架，包括：

1. **K-means++聚类分析**：根据成分股相似度对ETF跟踪指数进行分类
2. **多维指数评价**：从基本面和业绩角度评价指数
3. **ETF产品筛选**：从费率和流动性角度筛选最优ETF产品
4. **回测验证**：验证策略有效性

## 目录结构

```
etf_clustering_system/
├── config.py              # 配置文件
├── main.py                # 主程序
├── README.md              # 项目说明
├── source/                # 源代码模块
│   ├── __init__.py        # 模块初始化
│   ├── data_loader.py     # 数据加载模块
│   ├── clustering.py       # K-means++聚类
│   ├── index_evaluator.py # 指数评价
│   ├── etf_evaluator.py   # ETF筛选
│   └── plot.py            # 可视化
├── ipynb/                 # Jupyter Notebook
│   └── etf_clustering_demo.ipynb  # 演示notebook
├── data/                  # 数据目录
└── output/                # 输出目录
```

## 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install tushare akshare efinance baostock
```

## 快速开始

### 方法1：运行主程序

```bash
cd etf_clustering_system
python main.py
```

### 方法2：使用Jupyter Notebook

```bash
cd etf_clustering_system
jupyter notebook ipynb/etf_clustering_demo.ipynb
```

## 核心方法论

### 1. K-means++聚类

根据指数成分股相似度进行分类：

```
arg min_C J(C) = Σ Σ ||x(i) - μ(k)||²
```

K-means++改进：
- 第一个质心随机选择
- 后续质心按距离平方概率分布选择
- 确保质心分散，避免局部最优

### 2. 多维指数评价

| 维度 | 权重 | 说明 |
|------|------|------|
| 估值贡献 | 25% | 成分股估值变化对指数的影响 |
| 集中度 | 25% | 前十大权重股占比、前五行业涨跌影响 |
| 盈利能力 | 25% | ROE_TTM |
| 成长性 | 25% | 营收同比增速 |

筛选步骤：
1. 根据财务指标剔除后20%
2. 选择长短期夏普比率前50%

### 3. ETF产品筛选

| 指标 | 权重 | 说明 |
|------|------|------|
| 费率 | 40% | 管理费+托管费 |
| 流动性 | 20% | 近一月日均成交额 |
| 规模 | 20% | 基金规模 |
| 跟踪误差 | 10% | 与指数的跟踪偏差 |
| 信息比率 | 10% | 超额收益/跟踪误差 |

## 数据来源

优先级：efinance > tushare > akshare > baostock

- **ETF基础信息**：fund_basic, fund_etf_hist_em
- **指数成分股**：index_weight, index_weight_cons
- **历史净值**：fund_nav, get_quote_history
- **财务数据**：fina_indicator

## 输出文件

- `cluster_result.csv`: 聚类结果
- `index_evaluation.csv`: 指数评价结果
- `selected_etfs.csv`: 筛选后的ETF列表
- `backtest_result.csv`: 回测结果
- `*.png`: 可视化图表

## 风险提示

本报告仅对基金产品进行定量不定性分析，不做任何推荐建议。基金历史业绩不代表未来业绩，基金投资有风险，投资者需谨慎决策。

## 参考研报

民生证券研究所 - 《ETF的聚类优选与热点趋势策略构建》
- 分析师：叶尔乐、关舒丹
- 日期：2025年02月28日
