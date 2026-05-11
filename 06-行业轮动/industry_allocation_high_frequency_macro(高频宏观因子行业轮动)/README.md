# 行业配置策略：高频宏观因子

基于华泰证券研报《行业配置策略：高频宏观因子》(2023-06-10) 的工程化复现项目

## 项目概述

本项目复现了华泰金工团队提出的基于高频宏观因子的行业配置策略，核心思想包括：

1. **Factor Mimicking**: 采用自上而下的方法构建7个高频宏观因子
2. **宏观风险配置模型**: 将宏观观点转化为行业配置比例
3. **戴维斯双击策略**: 预测行业盈利能力变化(Δg)和估值水平变化(ΔPB)，捕捉双击机会

### 回测绩效

- **回测区间**: 2016-04-30 至 2023-05-31
- **年化超额收益**: 10.06% (相对行业等权基准)

## 目录结构

```
industry_allocation_high_frequency_macro/
├── config/
│   └── settings.py          # 配置文件 (Tushare token, 参数设置)
├── source/
│   ├── data_fetcher.py     # 数据获取模块
│   ├── factors.py          # 高频宏观因子计算
│   ├── allocation.py       # 行业配置模型
│   └── backtest.py         # 回测引擎
├── ipynb/
│   └── 行业配置策略_高频宏观因子复现.ipynb  # 主notebook
├── output/                  # 回测结果输出
├── data/                    # 数据缓存
├── main.py                  # 主程序入口
└── README.md                # 项目说明文档
```

## 核心模块说明

### 1. 数据获取 (source/data_fetcher.py)

使用Tushare Pro API获取市场数据：

```python
from config.settings import PRO
from source.data_fetcher import DataFetcher

fetcher = DataFetcher(pro=PRO)
```

**功能**:
- 获取申万行业指数日频数据
- 获取上市公司财务数据 (ROE_TTM等)
- 获取市场交易数据 (市值、PB等)
- 获取宏观指标 (CPI, PPI, GDP)

### 2. 高频宏观因子 (source/factors.py)

使用Factor Mimicking方法构建7个高频宏观因子：

| 因子名称 | 描述 | 代理资产 |
|---------|------|---------|
| 增长因子 | 经济景气度 | 恒生指数、CRB工业现货、南华沪铜 (多头); 7-10年国债 (空头) |
| 生活端通胀 | 消费品价格 | 南华生猪指数 |
| 生产端通胀 | 工业品价格 | 布伦特原油、南华螺纹钢、南华动力煤 |
| 无风险利率 | 货币松紧 | 1-3年国债财富指数 |
| 信用利差 | 信用风险 | 企业债AA-国开债 |
| 期限利差 | 利率期限结构 | 7-10年国债-1-3年国债 |
| 汇率 | 人民币汇率 | 伦敦金现 |

**构建步骤**:
1. 滚动4周移动平均降噪
2. 计算周度环比和年度同比收益率
3. 滚动3年计算权重 (标准差倒数加权)
4. 加权得到因子组合净值

### 3. 行业配置 (source/allocation.py)

#### 宏观风险配置模型

将宏观观点转化为行业配置比例：

```python
from source.allocation import MacroRiskAllocator

allocator = MacroRiskAllocator(lambda_param=0.3)
allocator.fit_exposure_matrix(factor_returns, asset_returns)
weights = allocator.get_recommended_weights(macro_views)
```

**强复苏情景宏观观点示例**:
- 增长: ↑上行
- 生活端通胀: ↓下行
- 生产端通胀: ↓下行
- 无风险利率: ↓下行 (宽松)
- 信用利差: ↑收窄
- 期限利差: ↑走阔
- 汇率: ↑人民币升值

#### 戴维斯双击策略

预测Δg和ΔPB，捕捉行业双击机会：

```python
from source.allocation import DavisDoubleHitStrategy

strategy = DavisDoubleHitStrategy()
delta_g_pred = strategy.fit_macro_delta_g_mapping(factor_returns, delta_g)
delta_pb_pred = strategy.fit_macro_delta_pb_mapping(factor_returns, delta_pb)
composite = strategy.calculate_composite_factor(delta_g_pred, delta_pb_pred)
top_industries = strategy.select_top_industries(composite, top_n=10)
```

### 4. 回测 (source/backtest.py)

```python
from source.backtest import DavisDoubleHitBacktest

backtest_engine = DavisDoubleHitBacktest()
results = backtest_engine.run_strategy_backtest(factor_returns, industry_returns)
```

## 复现要点

### 1. 数据依赖

**已使用Tushare获取**:
- 申万行业指数 (SW 30个一级行业)
- 部分市场数据

**待补充数据** (需Wind/Choice等):
- 国债财富指数序列 (CBA00621.CS, CBA00652.CS等)
- 南华商品指数 (南华生猪、南华螺纹钢、南华沪铜等)
- 布伦特原油期货数据
- 伦敦金现价格

### 2. 关键参数

| 参数 | 值 | 说明 |
|-----|-----|-----|
| 因子滚动窗口 | 52周 | 约1年 |
| 权重计算窗口 | 156周 | 约3年 |
| 宏观-行业回归窗口 | 52周 | 约1年 |
| 调仓频率 | 季度/月度 | 可配置 |
| λ (跟踪误差权重) | 0-0.5 | 可配置 |

### 3. 注意事项

- 研报强调宏观-行业映射关系是动态变化的
- 财务数据存在延迟发布特点，需利用此特征进行实时预测
- 不同宏观环境下应配置的行业没有固定答案

## 使用方法

### 1. 安装依赖

```bash
pip install tushare pandas numpy matplotlib scipy
```

### 2. 配置Tushare Token

在 `config/settings.py` 中配置您的Tushare Token

### 3. 运行主程序

```bash
python main.py
```

### 4. 运行Jupyter Notebook

```bash
jupyter notebook ipynb/行业配置策略_高频宏观因子复现.ipynb
```

## 风险提示

1. 高频宏观因子构建前提是相对稳定的宏观-大类资产映射关系
2. 历史规律可能失效
3. 宏观环境对行业的影响可能随行业生命周期变化而改变
4. 报告中涉及的具体行业不代表任何投资意见

## 参考研报

- 华泰证券《行业配置策略：高频宏观因子》(2023-06-10)
- 华泰证券《行业配置策略：宏观因子视角》(2020-08-04)
- 华泰证券《行业配置策略：投资时钟视角》(2021-07-06)
- 华泰证券《行业配置策略：中观景气视角(2)》(2022-07-18)

## 依赖库

- tushare (数据获取)
- pandas (数据处理)
- numpy (数值计算)
- scipy (优化)
- matplotlib (可视化)

## 版本

- v1.0 (2026-05-05): 初始版本
