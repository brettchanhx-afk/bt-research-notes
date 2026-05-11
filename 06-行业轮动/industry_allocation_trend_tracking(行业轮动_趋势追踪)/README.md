# 趋势追踪行业配置策略

## 项目概述

本项目复现华泰证券研报《基本面轮动系列之七：行业配置策略-趋势追踪视角》(2020-08-31) 的核心内容，实现完整的趋势追踪策略框架。

### 研报核心观点

1. **趋势追踪指标体系**: 整理了市场上41个主流趋势追踪指标，涵盖MA、EMA、MACD、DPO、TSI等多种类型
2. **蒙特卡洛模拟验证**: 基于AR(1)模型和几何布朗运动模拟虚拟价格序列，分析资产风险收益特性与策略表现关系
3. **时序动量 vs 截面动量**: 大类资产配置适合时序动量策略，行业配置适合截面动量策略
4. **CSCV过拟合检验**: 引入组合对称交叉验证框架评估策略过拟合风险

## 目录结构

```
industry_allocation_trend_tracking/
├── source/                      # 核心模块
│   ├── __init__.py              # 包初始化
│   ├── data_loader.py           # 数据获取模块 (tushare)
│   ├── indicators.py            # 趋势追踪指标计算
│   ├── backtest.py              # 回测引擎
│   ├── monte_carlo.py           # 蒙特卡洛模拟
│   ├── cscv.py                  # CSCV过拟合检验
│   └── strategy_builder.py      # 策略构建框架
├── ipynb/                       # Jupyter notebooks
│   └── trend_following_strategy.ipynb
├── output/                       # 输出结果目录
├── requirements.txt             # 依赖库
└── README.md                    # 本文件
```

## 核心模块说明

### 1. data_loader.py - 数据获取

使用tushare pro API获取市场数据:

- `DataLoader`: 主数据获取类
- `get_index_daily()`: 获取指数日线数据
- `get_industry_daily()`: 获取行业指数数据
- `calculate_assets_statistics()`: 计算资产统计信息
- `calculate_correlation_matrix()`: 计算相关系数矩阵

**预定义资产代码**:

```python
INDEX_CODES = {
    "上证50": "000016.SH",
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "恒生指数": "HSI.HI",
    "标普500": "SPX.GI",
    ...
}

INDUSTRY_CODES = {
    "石油石化": "CI005001.WI",
    "煤炭": "CI005002.WI",
    ...
}
```

### 2. indicators.py - 趋势追踪指标

实现了研报中的41个趋势追踪指标，包括:

- **单参数指标**: MA, EMA, WMA, ROC, DPO, CMO, PSY等
- **双参数指标**: OSC, TSI, INVVOL, SHARP_MOM, MACD等
- **多参数指标**: PMO, DBCD, COPP, PPO等

**指标分类**:

- 类别1: 原始信号与0比较
- 类别2: 快慢线比较
- 类别3: 正负收益比较

### 3. backtest.py - 回测引擎

`BacktestEngine`: 核心回测类

- `time_series_momentum_backtest()`: 时序动量策略回测
- `cross_section_momentum_backtest()`: 截面动量策略回测
- `risk_parity_allocation()`: 风险平价权重分配

**回测指标**:

- 年化收益率
- 年化波动率
- 夏普比率
- 最大回撤
- 月度胜率

### 4. monte_carlo.py - 蒙特卡洛模拟

`MonteCarloSimulator`: 蒙特卡洛模拟器

- `generate_ar1_series()`: 基于AR(1)模型生成单资产虚拟序列
- `generate_multi_asset_gbm()`: 基于几何布朗运动生成多资产序列

### 5. cscv.py - 过拟合检验

`CSCVTest`: 组合对称交叉验证框架

- `split_data()`: 数据分割
- `compute_overfitting_probability()`: 计算过拟合概率

**核心公式**:
$$P_{overfit} = \frac{P(Sharpe_{IS} > 0 \ AND \ Sharpe_{OS} < 0)}{P(Sharpe_{IS} > 0)}$$

### 6. strategy_builder.py - 策略构建框架

`TrendFollowingStrategyBuilder`: 策略构建器

完整流程:

1. **指标评估**: 对所有指标进行回测，筛选表现好的指标
2. **过拟合检验**: 使用CSCV框架剔除过拟合概率>50%的指标
3. **指标复合**: 两两搭配构建复合策略

## 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖**:

- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- tushare>=1.2.0
- scipy>=1.7.0
- statsmodels>=0.13.0

## 使用示例

### 1. 数据获取

```python
from source import DataLoader

loader = DataLoader()

# 获取指数数据
index_data = loader.get_multiple_indices(
    start_date='20100101',
    end_date='20200731'
)

# 获取行业数据
industry_data = loader.get_multiple_industries(
    start_date='20100101',
    end_date='20200731'
)

# 计算统计信息
stats = loader.calculate_assets_statistics(index_data)
corr = loader.calculate_correlation_matrix(index_data)
```

### 2. 指标计算

```python
from source import TrendIndicatorCalculator

calculator = TrendIndicatorCalculator()

# 生成单一指标
ema_20 = calculator.generate_signal(data, 'EMA', {'n': 20})

# 生成全部指标
all_signals = calculator.generate_all_signals(data)
```

### 3. 策略回测

```python
from source import BacktestEngine

engine = BacktestEngine(
    rebalance_freq='monthly',
    commission_rate=0.001
)

# 时序动量回测
result = engine.time_series_momentum_backtest(prices, signals)

# 截面动量回测
result = engine.cross_section_momentum_backtest(
    prices, signals, n_select=5
)
```

### 4. CSCV过拟合检验

```python
from source import CSCVTest

cscv = CSCVTest(n_splits=100)

result = cscv.run_cscv_analysis(signal, prices)

print(f"过拟合概率: {result['overfitting_probability']:.4f}")
```

### 5. 完整策略构建

```python
from source import (
    StrategyConfig,
    TrendFollowingStrategyBuilder
)

config = StrategyConfig(
    name="大类资产配置策略",
    asset_type="asset_allocation",
    strategy_type="time_series",
    target_volatility=7.5
)

builder = TrendFollowingStrategyBuilder(config)
pipeline_results = builder.run_full_pipeline(prices, data)
```

## 复现要点

### 研报核心结论验证

1. **趋势追踪指标共性**: 不同指标在趋势签名图上表现相似，核心区别在于历史价格加权方式
2. **资产风险收益匹配**: 高收益、低波动资产更适合趋势追踪策略
3. **时序vs截面**: 大类资产低相关性场景用时序动量，行业指数高相关性场景用截面动量
4. **过拟合控制**: CSCV过拟合概率<50%的指标更可靠

### 与研报差异说明

1. **数据来源**: 本项目使用tushare获取数据，可能与研报使用的Wind数据存在差异
2. **参数简化**: 为提高计算效率，部分参数设置可能与研报有差异
3. **资产覆盖**: 由于数据获取限制，部分大类资产(如南华商品指数)可能无法完整获取

## 数据需求

如需完整复现研报结果，建议补充以下数据:

1. **债券指数**: 中债-国债总财富(3-5年)指数、中债-国债总净价(7-10年)指数
2. **商品指数**: 南华螺纹钢指数、南华豆粕指数、南华PTA指数、南华郑糖指数
3. **海外资产**: 标普500、恒生指数、伦敦金现、布伦特原油期货、LME铜

## 运行Notebook

```bash
cd ipynb
jupyter notebook trend_following_strategy.ipynb
```

## 风险提示

1. 本项目仅供研究学习，不构成投资建议
2. 历史规律可能失效，市场波动可能导致策略表现不及预期
3. 报告中涉及的资产不代表任何投资意见

## 许可证

MIT License

## 参考研报

华泰证券研究所 - 《基本面轮动系列之七：行业配置策略-趋势追踪视角》
2020年8月31日