# 行业配置策略：拥挤度视角

基于华泰证券研报《基本面轮动系列之九：行业配置策略-拥挤度视角》的量化策略复现项目。

## 项目概述

本项目复现了研报中提出的拥挤度指标构建方法和基于拥挤度的行业配置策略。核心思想是通过监测行业交易过热风险，规避拥挤状态行业，获取超额收益。

### 研报核心内容

1. **拥挤度指标构建**：从行业指数时序特征和成分股特征两个角度构建17项拥挤度指标
2. **指标有效性验证**：采用门限回归方法筛选出3个有效指标
3. **策略构建**：月度空头轮动、日度风险监控、大盘择时、景气拥挤复合策略

## 项目结构

```
industry_allocation_crowdedness/
├── source/                    # 源代码模块
│   ├── __init__.py
│   ├── data_fetcher.py       # 数据获取模块
│   ├── crowdedness.py        # 拥挤度指标计算
│   ├── rotation_strategy.py  # 行业轮动策略
│   ├── backtest.py           # 回测引擎
│   ├── risk_management.py    # 风控模块
│   └── visualization.py      # 可视化模块
├── ipynb/                    # Jupyter notebooks
│   └── crowdedness_strategy_reproduction.ipynb
├── output/                    # 输出结果目录
├── data/                      # 数据存储目录
├── config/                    # 配置目录
└── README.md
```

## 核心拥挤度指标

研报筛选出3个通过有效性检验的拥挤度指标：

| 指标名称 | 说明 | 推荐阈值 |
|---------|------|---------|
| comp_turn_kurtosis_10 | 成分股10日收益率峰度 | 95% |
| turn_20 | 过去20日平均换手率 | 90% |
| corr_amount_close_40 | 40日成交额收盘价相关系数 | 95% |

## 实现的策略

1. **月度空头行业轮动策略**：每月做空拥挤行业
2. **日度行业风险监控策略**：每日监测并及时清仓拥挤行业
3. **大盘择时策略**：根据全市场拥挤行业数量进行择时
4. **景气度+拥挤度复合策略**：高景气+低拥挤行业配置

## 安装依赖

```bash
pip install tushare pandas numpy matplotlib scipy jupyter
```

## 使用方法

### 1. 数据获取

```python
from source.data_fetcher import get_all_industries_data

# 获取申万行业指数数据
industry_data = get_all_industries_data(
    start_date='20100101',
    end_date='20231231',
    save=True
)
```

### 2. 拥挤度计算

```python
from source.crowdedness import CrowdednessIndicator

indicator = CrowdednessIndicator()
composite = indicator.calculate_composite_crowdedness(indicators)
```

### 3. 策略运行

```python
from source.rotation_strategy import IndustryRotationStrategy

rotation = IndustryRotationStrategy(industry_data, crowdedness_signals)
strat_returns, positions = rotation.strategy_two_daily_risk_monitor()
```

### 4. 回测

```python
from source.backtest import run_backtest

results = run_backtest(
    strategy_returns,
    benchmark_returns,
    initial_capital=1000000,
    strategy_name='策略名称'
)
```

### 5. 可视化

```python
from source.visualization import plot_equity_curves, plot_strategy_comparison

plot_equity_curves({'策略': equity_curve}, title='策略净值')
plot_strategy_comparison(all_results)
```

## 复现要点

1. **拥挤度指标核心逻辑**：采用历史分位数形式表征行业拥挤度
2. **门限回归验证**：拥挤度指标只需在门限值之上对指数下行风险起预测作用
3. **复合拥挤度**：三个指标任意一个触发即视为拥挤
4. **数据处理**：使用tushare获取申万行业指数日频数据

## 数据来源

- **tushare**: 行业指数日频行情数据
- **token配置**: 使用指定的tushare token和接口地址

## 重要提示

1. 本项目为量化策略研究学习之用，不构成投资建议
2. 历史回测结果仅供参考，不代表未来收益
3. 市场出现超预期波动时，拥挤交易可能导致更大损失
4. 研报原始数据来自Wind，本项目使用tushare替代

## 缺失数据说明

1. **景气度指标原始数据**：研报中使用的景气度指标来自前期报告《行业配置策略：景气度视角》(2020-11-05)，本项目使用简化代理指标替代
2. **完整历史数据**：由于数据接口限制，回测区间可能短于研报中的2010-2020年
3. **成分股数据**：成分股层面的拥挤度指标需要个股数据，本项目主要实现行业指数层面的指标

## 依赖库

- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- scipy>=1.7.0
- tushare>=1.2.0
- jupyter>=1.0.0

## 作者

基于华泰证券研报《基本面轮动系列之九：行业配置策略-拥挤度视角》复现

原始研报作者：林晓明、李聪、韩晳、王佳星

## 免责声明

本项目仅供学习研究使用，不构成任何投资建议。投资者应理性看待，自行承担投资风险。