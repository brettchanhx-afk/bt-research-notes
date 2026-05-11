# 多ETF轮动 + 趋势震荡双策略回测系统

本项目将聚宽平台的量化回测代码重构为本地可运行的量化项目，包含完整的回测框架、策略逻辑、数据分析和可视化功能。

## 项目结构

```
项目根目录/
├── source/                # 源代码目录
│   ├── __init__.py       # 包初始化文件
│   ├── config.py         # 配置文件
│   ├── data_loader.py    # 数据加载模块
│   ├── indicators.py     # 技术指标计算模块
│   ├── strategy.py       # 策略逻辑模块
│   ├── backtest.py       # 回测引擎模块
│   └── analysis.py       # 分析和可视化模块
├── ipynb/                # Jupyter Notebook目录
│   └── backtest_demo.ipynb  # 回测演示Notebook
├── output/               # 输出目录（自动生成）
├── data/                 # 数据目录（自动生成）
├── logs/                 # 日志目录（自动生成）
├── requirements.txt      # 依赖包列表
└── README.md            # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 方法1：使用Jupyter Notebook（推荐）

1. 启动Jupyter Notebook:
```bash
jupyter notebook
```

2. 打开 `ipynb/backtest_demo.ipynb`

3. 按顺序执行代码单元格

### 方法2：直接使用Python代码

```python
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from source import (
    DataLoader, ETFRotationStrategy, BacktestEngine, 
    PerformanceAnalyzer, ETF_POOL, 
    BACKTEST_CONFIG, STRATEGY_CONFIG
)

# 1. 加载数据
data_loader = DataLoader()
etf_data = data_loader.load_all_data(
    etf_codes=list(ETF_POOL.keys()),
    start_date='2021-01-01',
    end_date='2024-06-30'
)
trading_dates = data_loader.get_trading_dates('2021-01-01', '2024-06-30')

# 2. 初始化策略
strategy = ETFRotationStrategy(ETF_POOL, STRATEGY_CONFIG)

# 3. 运行回测
backtest = BacktestEngine(etf_data, trading_dates, strategy, BACKTEST_CONFIG)
results = backtest.run()

# 4. 分析结果
analyzer = PerformanceAnalyzer(results)
metrics = analyzer.calculate_metrics()
analyzer.generate_report()

print(metrics)
```

## 模块说明

### 1. config.py
配置文件，包含：
- ETF池配置
- 回测参数（初始资金、手续费等）
- 策略参数（持仓数量、最小/最大仓位等）
- 技术指标参数

### 2. data_loader.py
数据加载模块，使用akshare获取：
- ETF历史行情数据
- 交易日历
- 数据缓存功能

### 3. indicators.py
技术指标计算模块，包含：
- RSI（相对强弱指标）
- MA/EMA（移动平均）
- ATR（平均真实波幅）
- VaR/ES（风险指标）
- 波动率、增长率等

### 4. strategy.py
策略逻辑模块，包含：
- 多ETF轮动算法
- RSI择时信号
- ATR网格信号
- 风险控制机制
- 仓位计算和优化

### 5. backtest.py
回测引擎模块，包含：
- 回测执行框架
- 订单执行（含手续费和滑点）
- 资金管理
- 仓位管理
- 每日定投

### 6. analysis.py
分析和可视化模块，包含：
- 回测指标计算（收益率、夏普比率、最大回撤等）
- 净值曲线图
- 回撤曲线图
- 月度/年度收益率图
- 报告生成

## 策略特点

### 核心功能
1. **多ETF轮动**：基于风险收益特征动态选择ETF
2. **双策略融合**：RSI择时 + ATR网格
3. **风险控制**：VaR/ES、涨跌幅限制、仓位控制
4. **定投策略**：每日注入资金
5. **平滑调仓**：每月定期调仓，避免频繁交易

### 技术指标
- **RSI**：用于判断超买超卖
- **ATR**：用于网格交易和止损
- **MA/EMA**：趋势判断
- **VaR/ES**：风险评估

## 数据源说明

本项目使用 **akshare** 作为数据源：
- 提供A股ETF历史行情数据
- 提供指数历史数据
- 所有数据均为真实市场数据
- 支持自动数据缓存

## 输出内容

运行回测后，`output/` 目录将生成：
1. 回测报告（txt文件）
2. 净值曲线图（png）
3. 回撤曲线图（png）
4. 月度收益率图（png）
5. 年度收益率图（png）

## 注意事项

1. **数据获取**：首次运行需要从akshare下载数据，可能需要几分钟
2. **网络连接**：获取数据需要网络连接
3. **参数优化**：可根据需要调整策略参数
4. **历史局限性**：回测结果不代表未来表现，仅供参考

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。

## 更新日志

### v1.0 (2024-05-01)
- 完成项目基本框架搭建
- 实现完整的回测系统
- 实现策略逻辑和技术指标
- 实现数据分析和可视化功能
