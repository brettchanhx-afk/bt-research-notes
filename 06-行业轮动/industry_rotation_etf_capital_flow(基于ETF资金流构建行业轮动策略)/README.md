# ETF行业轮动策略 - 华泰证券研报复现

基于华泰证券金工深度研究《基于ETF资金流构建行业轮动策略》的策略复现项目。

## 项目概述

本项目复现了研报中描述的ETF行业轮动策略，核心逻辑是：

> **当行业ETF资金净流出位于历史高点时，未来短期内预期收益通常为正。**

### 研报核心结论

| 指标 | 预期表现 |
|------|----------|
| 年化收益率 | > 20% |
| Sharpe比率 | > 1 |
| Calmar比率 | ≈ 1 |
| 月度胜率 | ≈ 70% |
| 盈亏比 | > 1.4 |

## 目录结构

```
industry_rotation_etf_capital_flow/
├── config/
│   └── settings.py           # 配置文件（行业-指数映射、策略参数）
├── source/
│   ├── __init__.py
│   ├── data_fetcher.py      # 数据获取模块（Tushare）
│   ├── etf_flow.py          # ETF资金流计算模块
│   ├── industry_rotation.py # 行业轮动策略模块
│   ├── backtest.py          # 回测引擎模块
│   └── utils.py             # 工具函数模块
├── ipynb/
│   └── backtest_demo.ipynb  # Jupyter Notebook演示
├── data/                    # 数据缓存目录
├── output/                  # 输出结果目录
├── run.py                   # 主运行脚本
└── README.md               # 项目说明文档
```

## 依赖库

```bash
pip install pandas numpy matplotlib tushare akshare
```

或使用requirements.txt:

```bash
pip install -r requirements.txt
```

## 数据需求说明

### 当前状态

**Tushare不提供ETF份额日度变动数据**，因此本项目存在以下限制：

1. **ETF资金流数据使用模拟数据**：由于无法获取真实的ETF申赎数据，当前使用随机生成的模拟数据
2. **指数数据部分可用**：Tushare可获取部分行业指数数据，但部分特殊指数代码格式可能不支持

### 需要补充的数据

要完整复现研报结果，需要以下数据：

| 数据项 | 说明 | 推荐数据源 |
|--------|------|-----------|
| ETF每日份额变动 | 用于计算资金净流入 | Wind/Choice金融终端 |
| ETF净值数据 | 份额×净值=资金流入 | Wind/天天基金 |
| 真实ETF列表 | 行业ETF分类 | Wind/AkShare |

### 数据计算公式

```
资金净流入 = 份额增加 × ETF净值
历史分位数 = rank(当期净流入 / 历史净流入) / 总期数
```

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib tushare
```

### 2. 配置Tushare Token

在 `config/settings.py` 中已配置token，如需更换：

```python
TUSHARE_TOKEN = "your_token_here"
TUSHARE_API_URL = "http://jiaoch.site"  # 使用第三方接口
```

### 3. 运行主脚本

```bash
python run.py
```

### 4. 运行Jupyter Notebook

```bash
jupyter notebook ipynb/backtest_demo.ipynb
```

## 策略逻辑

### 核心原理

根据研报，ETF资金流是一个**有效的反向指标**：

1. **价格压力假说**：非信息性的大额交易会造成短期供需失衡
2. 当ETF资金大量净流出时，短期内价格可能被低估
3. 定价恢复理性后，价格会出现反弹

### 策略规则

```
做多信号：当ETF资金净流入历史分位数 <= 阈值（如5%或10%）
做空信号：当ETF资金净流入历史分位数 >= 阈值（如95%或90%）
```

### 参数设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| long_threshold | 0.10 | 做多阈值（分位数<=10%） |
| short_threshold | 0.90 | 做空阈值（分位数>=90%） |
| rolling_window_years | 2 | 历史分位数滚动窗口（年） |
| rebalance_freq | weekly | 调仓频率 |

## 模块说明

### source/data_fetcher.py

数据获取模块，基于Tushare API：

```python
from source.data_fetcher import TushareDataFetcher

fetcher = TushareDataFetcher()
etf_df = fetcher.get_etf_basic_info()  # 获取ETF列表
index_data = fetcher.get_index_daily('399975.SZ', '20230101', '20231231')  # 获取指数数据
```

### source/etf_flow.py

ETF资金流计算模块：

```python
from source.etf_flow import ETFFlowCalculator

calculator = ETFFlowCalculator(industry_etf_flow)
weekly_flow = calculator.calculate_weekly_net_flow(method='natural_week')
percentile_data = calculator.calculate_rolling_percentile(weekly_flow, window_years=2)
```

### source/industry_rotation.py

行业轮动策略模块：

```python
from source.industry_rotation import IndustryRotationStrategy

strategy = IndustryRotationStrategy(
    long_threshold=0.10,
    short_threshold=0.90,
    rolling_window_years=2
)
results = strategy.run_backtest(percentile_data, returns_data)
```

### source/backtest.py

回测引擎模块：

```python
from source.backtest import BacktestEngine, StrategyBacktester, plot_equity_curve

backtester = StrategyBacktester(strategy, data_provider)
results = backtester.run(start_date, end_date, signals, prices)
metrics = backtester.calculate_metrics(results)
plot_equity_curve(results)
```

## 重要提示

1. **数据限制**：本项目使用模拟数据，**不能直接用于实盘交易**
2. **参数过拟合风险**：研报中阈值参数通过全样本参数寻优得到，可能存在过拟合
3. **交易成本**：实际策略需考虑佣金、印花税、滑点等交易成本
4. **空头策略可行性**：融券费用年化约10%且券源紧张，空头策略实盘可行性低

## 复现要点

### 研报关键步骤

1. **ETF产品分类**：按行业分类ETF，筛选行业ETF产品
2. **资金流计算**：`资金净流入 = 份额变化 × 净值`
3. **滚动分位数**：计算N年滚动历史分位数
4. **信号生成**：当分位数低于阈值时开仓
5. **回测验证**：在样本区间内验证策略表现

### 行业-指数映射（21个行业）

| 行业 | 指数代码 | 指数名称 |
|------|----------|----------|
| 非银金融 | 399975.SZ | 中证全指证券公司指数 |
| 计算机 | 930651.CSI | 中证计算机主题指数 |
| 电子 | 980017.CNI | 国证半导体芯片 |
| 食品饮料 | 000815.CSI | 中证细分食品饮料产业主题指数 |
| 医药生物 | 399989.SZ | 中证医疗指数 |
| ... | ... | ... |

## 风险提示

- 策略模型根据历史规律总结，历史规律可能失效
- 本报告不涉及证券投资基金评价，不涉及对具体基金产品的投资建议
- 择时策略参数均通过全样本参数寻优得到，可能存在过拟合风险

## 参考研报

- **华泰证券**，《金工深度研究：基于ETF资金流构建行业轮动策略》
- **研究员**：林晓明、刘依苇、何康
- **发布日期**：2024年10月15日

## License

本项目仅供学习研究使用，不构成投资建议。
