# 行业配置策略：景气度视角 - 量化复现项目

## 项目概述

本项目复现华泰证券研报《基本面轮动系列之八：行业配置策略-景气度视角》（2020年11月05日），基于Python实现完整的行业景气度轮动策略框架。

### 原始研报核心内容

1. **基于业绩和一致预期数据的行业景气度指标构建**
   - 定期业绩指标（七大类财务指标：盈利能力、成长能力、现金流量、资本结构、偿债能力、营运能力）
   - 即时业绩指标（业绩预告、业绩快报、正式财报）
   - 行业一致预期指标
   - 个股合成一致预期指标
   - 关注度指标

2. **复合景气度指标构建**
   - 筛选高收益、低相关性指标
   - 多指标叠加构建复合指标
   - 最终复合指标包含7个业绩指标和10个一致预期指标

3. **行业轮动策略回测结果**
   - 五行业多头组合年化超额收益率：13.07%
   - 调仓胜率：68.22%
   - 策略在各行业上信号频率相近，无明显偏向性

4. **基于行业基本面数据的修正**
   - 纳入行业基本面数据可提高策略时效性
   - 修正后年化收益率从19.26%提高到20.05%

## 目录结构

```
industry_prosperity_rotation_v1/
├── source/                          # 源代码模块
│   ├── __init__.py                  # 包初始化，导出所有模块
│   ├── data_loader.py               # 数据获取模块（tushare）
│   ├── indicators.py                # 景气度指标计算
│   ├── composite_indicator.py       # 复合指标构建
│   ├── backtest.py                  # 回测引擎
│   ├── performance.py               # 绩效评估
│   ├── visualization.py             # 可视化
│   ├── strategy.py                  # 策略逻辑
│   └── utils.py                     # 工具函数
├── ipynb/                           # Jupyter notebooks
│   └── prosperity_rotation_reproduction.ipynb  # 复现notebook
├── output/                          # 输出目录
│   ├── data_cache/                  # 数据缓存
│   └── results/                     # 回测结果
├── 参考研报/                        # 原始研报
│   └── 2020-11-05_华泰证券_基本面轮动系列之八：行业配置策略-景气度视角.pdf
├── README.md                        # 项目说明
└── 项目结构.md                       # 本文件
```

## 依赖库

```bash
pip install tushare pandas numpy matplotlib seaborn scipy tqdm
```

### 核心依赖说明

- **tushare**: 数据获取（已配置自定义API地址）
- **pandas**: 数据处理
- **numpy**: 数值计算
- **matplotlib/seaborn**: 可视化
- **scipy**: 科学计算
- **tqdm**: 进度条

## 快速开始

### 1. 配置Tushare Token

在代码中已配置：
```python
token = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
pro._DataApi__http_url = "http://jiaoch.site"
```

### 2. 运行复现Notebook

```bash
cd ipynb
jupyter notebook prosperity_rotation_reproduction.ipynb
```

### 3. 模块调用示例

```python
from source import (
    DataLoader,
    ProsperityIndicator,
    CompositeIndicatorBuilder,
    BacktestEngine,
    PerformanceAnalyzer,
    Visualizer
)

# 初始化数据加载器
loader = DataLoader()

# 获取行业列表
sw_list = loader.get_sw_industry_list(level=1)

# 计算景气度指标
indicator = ProsperityIndicator()

# 构建复合指标
builder = CompositeIndicatorBuilder()

# 运行回测
engine = BacktestEngine(initial_capital=1000000)

# 绩效评估
analyzer = PerformanceAnalyzer()
```

## 核心模块说明

### data_loader.py
数据获取模块，支持：
- 申万行业列表
- 行业历史行情
- 交易日历
- 财务指标数据
- 一致预期数据

### indicators.py
景气度指标计算：
- TTM指标计算
- 同比/环比增长率
- 行业聚合指标
- 信号生成

### composite_indicator.py
复合指标构建：
- 相关性分析
- 增量选择法
- 多方法加权

### backtest.py
回测引擎：
- 持仓管理
- 交易模拟
- 佣金滑点计算

### performance.py
绩效评估：
- 年化收益率/波动率
- Sharpe/Calmar/Sortino比率
- 最大回撤
- 胜率统计

### visualization.py
可视化：
- 净值曲线
- 回撤分析
- 行业排名
- 相关性矩阵

## 复现要点

### 1. 数据获取限制
- 原始研报使用Wind数据
- 本项目使用Tushare作为替代
- 部分数据可能存在差异

### 2. 指标计算
研报中定义的指标类型：
- **定期业绩指标**: TTM环比增量、资产负债率同比增量等
- **即时业绩指标**: 业绩预告、快报、财报综合
- **一致预期指标**: 行业/个股一致预期EPS、ROE等

### 3. 回测设置
- 调仓频率: 月频
- 持仓数量: 5个行业
- 基准: 行业等权
- 手续费: 0.03%

### 4. 复合指标构建方法
1. 计算单项指标回测表现
2. 筛选高收益指标
3. 计算指标间相关性
4. 选择低相关性指标纳入复合
5. 等权或加权平均

## 数据需求

如需更完整复现，建议补充以下数据：

### 财务数据
- 申万一级行业成分股财务数据
- 盈利能力指标（ROE、净利率、毛利率等）
- 成长能力指标（营收增速、利润增速等）
- 现金流量指标（经营现金流等）
- 资本结构/偿债能力指标

### 一致预期数据
- 行业一致预期EPS、ROE
- 个股一致预期数据汇总
- 关注度指标（调高/调低家数占比）

### 行业基本面数据
- 钢铁: 铁矿石价格、钢材价格、产能利用率
- 煤炭: 煤炭价格、库存、产量
- 石油石化: 原油价格、炼油毛利
- 基础化工: 化工品价格、开工率

## 风险提示

1. 历史规律可能失效
2. 市场出现超预期波动可能导致拥挤交易
3. 数据质量和完整性影响策略表现
4. 实际交易需考虑流动性冲击

## 研报原文结论

> 基于业绩和一致预期数据构建的景气度指标展现出了有效的行业选择能力。在总计五类190个单项景气度指标中，有167个指标的多头组合年化收益率可以超越行业等权基准。将多个景气度指标复合可以增厚收益，构建更稳健可靠的行业轮动策略。

## 联系方式

如有问题或建议，请联系项目维护者。

## 参考资料

1. 华泰证券《基本面轮动系列之八：行业配置策略-景气度视角》（2020-11-05）
2. Tushare文档: http://tushare.org/
3. 申万行业分类标准
