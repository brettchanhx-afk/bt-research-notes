# 行业配置策略：资金流向视角

基于华泰证券研报《行业配置策略：资金流向视角》(2021-11-05) 复现的量化投资策略项目。

## 项目概述

本项目复现了华泰证券金工团队的深度研究报告，通过资金流向指标构建行业轮动策略。

### 核心策略逻辑

- **资金流向是股市涨跌的重要驱动力**
- **持续的资金流入是股价上涨的核心推动力**
- **资金流向指标包含不同市场参与者的观点**

## 目录结构

```
industry_allocation_capital_flow_perspective/
├── config/
│   ├── __init__.py
│   └── settings.py          # 项目配置参数
├── source/
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载模块
│   ├── northbound.py        # 北向资金模块
│   ├── margin.py            # 两融资金模块
│   ├── etf.py               # ETF资金模块
│   ├── industrial_capital.py # 产业资本模块
│   ├── indicators.py        # 指标计算模块
│   ├── strategy.py          # 策略回测模块
│   └── composite.py         # 复合指标模块
├── ipynb/
│   └── backtest.ipynb       # 回测展示notebook
├── output/
│   ├── data/                # 数据输出目录
│   └── charts/              # 图表输出目录
├── 参考研报/
│   └── 2021-11-05_华泰证券_行业配置策略：资金流向视角.txt
└── README.md
```

## 功能模块

### 1. 数据获取模块 (data_loader.py)

- 北向资金数据 (沪股通+深股通)
- 两融资金数据 (融资余额、融资买入额等)
- ETF份额变化数据
- 产业资本数据 (定增、回购、限售解禁、增减持)
- 行业日线行情数据

### 2. 资金流向指标

#### 北向资金指标
- `north_change_amount_{freq}` - 北向资金净流入
- `north_holdings_float_{freq}` - 北向资金持股占比
- 支持频率: W(周度)、M(月度)
- 支持处理: 原始值、同比(yoy)、环比(qoq)

#### 两融资金指标
- `margin_tr_balance_orig_{freq}_yoy` - 融资余额同比变化
- `margin_borrow_amount_{freq}` - 融资买入额
- 归一化方式: 除以成交额、除以流通市值、原始值

#### ETF资金指标
- 全市场ETF资金流
- 主题行业ETF资金流
- 趋势型ETF资金流 (过滤避险资金)

#### 产业资本指标
- 定向增发 (AShareSEO)
- 限售解禁 (AShareFreeFloatCalendar)
- 股票回购 (AShareRepurchase)
- 股东增减持 (MjrHolderTrade)

### 3. 策略评估方法

1. **分层回测**: 将行业按资金流向指标分为5组
2. **阈值回测**: 在90%、70%、50%分位数阈值上构建多空组合
3. **行业偏向性检验**: 确保指标不过度偏向某些行业
4. **事件有效性分析**: 验证个股资金流事件对股价的影响

## 依赖库

```python
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
tushare>=1.2.0
```

## 安装

```bash
pip install pandas numpy matplotlib tushare
```

## 使用方法

### 1. 初始化

```python
import sys
sys.path.append('..')

from config import settings
from source import DataLoader, NorthboundFunds, MarginFunds
from source import ETFFunds, IndustrialCapital, IndicatorCalculator
from source import IndustryRotationStrategy, CompositeIndicator

dl = DataLoader()
ic = IndicatorCalculator(dl)
north = NorthboundFunds(dl)
margin = MarginFunds(dl)
etf = ETFFunds(dl)
ic_capital = IndustrialCapital(dl)
strategy = IndustryRotationStrategy(dl, ic)
composite = CompositeIndicator(dl, ic)
```

### 2. 获取数据

```python
north_df = north.get_northbound_net_inflow()
margin_df = margin.get_margin_summary()
industry_list = ic.get_industry_list(level='L1')
```

### 3. 构建指标

```python
north_indicators = north.build_north_indicators(freq='W')
margin_indicators = margin.build_margin_indicators(freq='W')
```

### 4. 运行回测

```python
industry_returns = strategy.get_industry_returns(code_list)
strat_results = strategy.run_stratification_test(merged_df, 'indicator')
```

### 5. 构建复合指标

```python
combined = composite.combine_indicators(indicator_list)
```

## 研报复现要点

### 核心发现

1. **北向资金效果最好**: 北向资金用于行业轮动策略效果最好
2. **两融指标次之**: 融资余额同比指标表现稳定
3. **ETF资金需筛选**: 趋势型ETF资金流指标优于原始指标
4. **产业资本有效**:
   - 定向增预案公告对股价有提振作用
   - 限售解禁、股东减持会拖累行业表现
5. **复合策略更优**: 多空头年化超额收益均在10%以上

### 推荐指标

研报筛选出的9个代表性指标:
1. `north_change_amount_W_orig` - 北向资金周度净流入
2. `north_holdings_float_W_yoy` - 北向资金持股占比周度同比
3. `margin_tr_balance_orig_M_yoy` - 融资余额月度同比
4. `margin_tr_balance_orig_W_yoy` - 融资余额周度同比
5. `allETFselect_amount_M_yoy` - 全市场ETF资金月度同比
6. `AShareSEO_preplan_recent_orig_W` - 定增预案周度原始
7. `AShareFreeFloatCalendar_listdt_next_orig_M` - 限售解禁月度原始
8. `MjrHolderTrade_under_recent_amount_W` - 股东减持周度原始
9. `AShareRepurchase_recent_orig` - 股票回购原始

### 策略叠加

研报还展示了将资金流向策略与行业景气度策略叠加的效果:
- 单一景气度策略: 多头年化超额收益 9.55%
- 单一资金流策略: 多头年化超额收益 11.96%
- **复合策略**: 多头年化超额收益 15.40%

## 数据限制说明

本项目使用`tushare`获取数据，部分研报中使用的数据可能存在以下限制:

1. **北向资金行业归属**: tushare主要提供持股明细，需要额外处理归因到中信行业
2. **两融行业明细**: 主要提供全市场汇总，行业层面数据需要通过成分股计算
3. **ETF持仓明细**: 需要ETF持仓明细才能精确归因到行业
4. **产业资本详细事件**: 定向增发、限售解禁等事件的完整数据
5. **行业景气度数据**: 用于策略叠加的财务和一致预期数据

如需完整复现研报结果，建议补充:
- Wind终端数据
- Choice金融终端
- 聚源数据库

## 注意事项

1. 本项目仅供学习参考，不构成投资建议
2. 历史规律可能失效，市场存在超预期波动风险
3. 资金流向指标存续时间较短，策略有效性有待长期验证
4. 报告中涉及的具体行业不代表任何投资意见

## 参考研报

- 华泰证券《行业配置策略：资金流向视角》(2021-11-05)
- 研究员: 林晓明、王佳星
- SAC No. S0570516010001

## License

MIT License