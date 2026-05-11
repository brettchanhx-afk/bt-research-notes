# 北向资金量化策略项目

## 项目概述

本项目复现华泰证券金工深度研究报告《析精剖微：机构拆解看北向资金》（2022年10月27日）。

### 研报核心内容

该研报从机构拆解角度入手，对北向资金信息进行全方位剖析，主要内容包括：

1. **北向持仓机构全面画像**：从持仓规模、换手特征、收益情况及配置选股能力四个视角，对北向机构进行全面画像
2. **北向资金情绪指数**：利用13项事件指标合成情绪指数，构建样本外年化超额收益均在10%以上的择时策略
3. **行业配置策略**：多角度构建360个北向资金指标，构建周频和双周频复合因子，形成年化超额收益在10%以上的行业轮动策略

## 项目结构

```
industry_allocation_northbound_capital/
├── README.md                      # 项目说明文档
├── source/                       # 源代码目录
│   ├── __init__.py
│   ├── config/                    # 配置模块
│   │   ├── __init__.py
│   │   └── config.py             # 项目配置参数
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   ├── time_utils.py         # 时间工具函数
│   │   └── data_utils.py         # 数据处理工具函数
│   ├── data/                      # 数据获取模块
│   │   ├── __init__.py
│   │   ├── tushare_client.py     # Tushare API客户端
│   │   ├── northbound_data.py    # 北向资金数据获取
│   │   └── market_data.py        # 市场数据获取
│   ├── factors/                   # 因子模块
│   │   ├── __init__.py
│   │   ├── position_factor.py    # 持仓市值因子
│   │   ├── flow_factor.py        # 资金流向因子
│   │   ├── active_weight_factor.py  # 主动权重因子
│   │   ├── institution_score_factor.py  # 机构打分因子
│   │   └── sentiment_index.py    # 情绪指数构建
│   ├── strategies/                # 策略模块
│   │   ├── __init__.py
│   │   ├── timing_strategy.py     # 择时策略
│   │   └── allocation_strategy.py  # 行业配置策略
│   ├── backtest/                  # 回测模块
│   │   ├── __init__.py
│   │   └── backtest_engine.py    # 回测引擎
│   └── visualization/            # 可视化模块
│       ├── __init__.py
│       └── plots.py              # 绘图函数
├── ipynb/                        # Jupyter notebooks
│   └── northbound_strategy_demo.ipynb  # 策略演示notebook
├── output/                       # 输出目录（回测结果、图表等）
└── data/                         # 数据目录（本地缓存数据）
```

## 安装依赖

### Python版本

- Python 3.8+

### 依赖库

```bash
pip install pandas numpy matplotlib seaborn tushare akshare
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据获取

```python
from source.data import fetch_northbound_data, fetch_market_data
from source.config import BACKTEST_START_DATE, BACKTEST_END_DATE

# 获取北向资金数据
northbound_data = fetch_northbound_data(
    start_date=BACKTEST_START_DATE,
    end_date=BACKTEST_END_DATE
)

# 获取市场数据
market_data = fetch_market_data(
    start_date=BACKTEST_START_DATE,
    end_date=BACKTEST_END_DATE
)
```

### 2. 因子构建

```python
from source.factors import (
    PositionMarketValueFactor,
    CapitalFlowFactor,
    ActiveWeightFactor,
    InstitutionScoreFactor,
)

# 持仓市值因子
position_factor = PositionMarketValueFactor().calculate(
    northbound_holding, market_cap_data, industry_mapping, construction='yoy'
)

# 资金流向因子
flow_factor = CapitalFlowFactor().calculate(
    northbound_flow, industry_turnover, construction='raw'
)

# 主动权重因子
weight_factor = ActiveWeightFactor().calculate(
    northbound_weight, benchmark_weight, construction='yoy'
)

# 机构打分因子
score_factor = InstitutionScoreFactor().calculate(
    institution_flow_data, construction='raw'
)
```

### 3. 情绪指数构建

```python
from source.factors import SentimentIndexBuilder

sentiment_builder = SentimentIndexBuilder()

# 计算各事件指标
counter_market = sentiment_builder.calculate_counter_market_flow(flow_data, price_data)
large_flow = sentiment_builder.calculate_large_flow(flow_data)
abnormal_flow = sentiment_builder.calculate_abnormal_flow(flow_data)
divergence = sentiment_builder.calculate_divergence_flow(flow_data, price_data)

# 构建情绪指数
sentiment_index = sentiment_builder.build_sentiment_index(
    event_data={
        'counter_market_signal': counter_market,
        'large_flow_signal': large_flow,
        'abnormal_flow_signal': abnormal_flow,
        'divergence_signal': divergence,
    },
    selected_events=['counter_market_signal', 'large_flow_signal']
)
```

### 4. 策略运行

```python
from source.strategies import (
    SentimentTimingStrategy,
    IndustryAllocationStrategy,
    CompositeFactorStrategy,
)
from source.backtest import run_backtest

# 行业配置策略
strategy = IndustryAllocationStrategy(n_industries=3, frequency='weekly')
signals = strategy.generate_signals(factor_data, industry_returns)
metrics = strategy.get_performance_metrics(benchmark_returns)

# 回测
results = run_backtest(
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    initial_capital=1000000
)
```

### 5. 可视化

```python
from source.visualization import BacktestPlotter, plot_backtest_results

# 绘制回测结果
figures = plot_backtest_results(
    strategy_return=strategy_returns,
    benchmark_return=benchmark_returns,
    output_dir='output'
)
```

## 核心因子说明

### 1. 持仓市值因子

**定义**：北向资金持仓在中信/申万一级行业的流通市值，除以全部A股在该行业的流通市值

**构造方式**：
- 原始值（raw）
- 同比（yoy）- 与去年同期相比
- 环比（qoq）- 与上期相比

### 2. 资金流向因子

**定义**：使用成交额对北向资金流（增减持）进行归一化

**计算方式**：北向资金在特定行业的成交额 / 全部A股在该行业的成交额

### 3. 主动权重因子

**定义**：相比基准指数权重（沪深300），北向资金行业配置权重的偏配

**计算方式**：北向资金行业配置权重 - 基准指数行业权重

### 4. 机构打分因子

**定义**：根据行业净流入的机构数目，对行业进行打分

**计算方式**：净流入某行业的机构数目 / 总机构数目

### 5. 情绪指数

**构建方法**：
1. 构建13项二级事件指标（逆市流入/流出、大额流入/流出、反常态流入/流出、背离等）
2. 根据事件分析结果选取6项有效指标
3. 合成北向资金情绪指数

## 复现要点

### 研报关键结论

1. **机构画像**：
   - 外资银行：资金规模大，擅长行业配置，换手率较低
   - 外资券商：累计收益高，同时擅长行业配置和选股，换手率高
   - 内资机构：行业配置和选股能力一般

2. **情绪指数有效性**：
   - 基于6项事件指标合成的情绪指数
   - 样本外年化超额收益在10%以上（回测区间：2021年10月至2022年9月）

3. **行业配置策略**：
   - 推荐周频和双周频因子
   - 持仓市值因子推荐同比或环比构造
   - 资金流向因子推荐原始值
   - 年化超额收益在10%以上（回测区间：2017年12月至2022年9月）

### 数据说明

**重要提示**：研报中使用的按机构类型（外资银行、外资券商、内资银行、内资券商）分类的详细持仓数据需要从港交所获取，这部分数据通过免费API无法完全获取。本项目提供了数据获取接口，但在真实使用时可能需要：

1. 购买港交所机构持仓数据
2. 或使用其他数据源（如Wind终端）
3. 或联系券商获取相关数据

当前代码使用了模拟数据进行演示，请替换为真实数据后使用。

## 业绩指标说明

| 指标 | 说明 |
|------|------|
| 总收益 | 策略在整个回测期间的总收益率 |
| 年化收益率 | 平均每年的复合收益率 |
| 年化波动率 | 收益率的年化标准差 |
| 夏普比率 | (年化收益率 - 无风险利率) / 年化波动率 |
| 最大回撤 | 从最高点到最低点的最大跌幅 |
| 胜率 | 正收益天数 / 总天数 |
| 信息比率 | 超额收益 / 跟踪误差 |

## 注意事项

1. **数据质量**：免费API数据可能存在延迟和缺失，建议使用专业数据源
2. **交易成本**：实际策略需要考虑佣金、印花税、滑点等成本
3. **过拟合风险**：参数优化可能导致过拟合，需进行样本外验证
4. **市场变化**：历史规律可能失效，需定期更新模型

## 风险提示

- 模型根据历史规律总结，历史规律可能失效
- 市场出现超预期波动，可能导致拥挤交易
- 资金流向指标存续时间较短，策略有效性有待长期观察验证
- 本项目仅供研究参考，不构成投资建议

## 致谢

本项目基于华泰证券研究报告《析精剖微：机构拆解看北向资金》（2022年10月27日）复现。

## 许可证

MIT License
