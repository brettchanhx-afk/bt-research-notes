# 基于动量效应改进风险平价策略

本项目复现国泰君安研报《基于动量效应改进风险平价策略》的核心方法论。

## 项目概述

风险平价策略由于其简单易行，业界应用较广，但在实际使用中面临诸多问题：
1. 资产数量较多时，求解精度、难度和复杂度指数级上升
2. 风险平价策略不受资产收益率方向的影响
3. 风险平价策略在金融危机发生时往往产生难以承受的最大回撤

本项目引入**层次风险平价策略**和**动量风险预算策略**，通过聚类分层思想提升求解精度和速度，同时利用资产动量效应改进风险预算策略，提高投资收益。

## 目录结构

```
momentum_enhanced_risk_parity/
├── source/                     # Python源代码模块
│   ├── __init__.py             # 包初始化文件
│   ├── config.py               # 配置文件（参数、资产列表）
│   ├── data_fetcher.py         # 数据获取模块（tushare）
│   ├── risk_parity.py          # 传统风险平价实现
│   ├── momentum_risk_budget.py # 动量风险预算核心策略
│   ├── hierarchical_risk_parity.py  # 层次风险平价
│   ├── backtest.py             # 回测框架
│   └── visualization.py        # 可视化模块
├── ipynb/                      # Jupyter notebooks
│   └── momentum_risk_parity_backtest.ipynb  # 复现演示notebook
├── output/                     # 输出结果目录
│   ├── cumulative_returns.png  # 累计收益图
│   ├── drawdown.png            # 回撤分析图
│   └── metrics_comparison.png  # 指标对比图
├── data/                       # 数据缓存目录
├── 参考研报/                   # 原始研报PDF
└── README.md                   # 项目说明文档

```

## 核心策略说明

### 1. 传统风险平价 (Risk Parity)

风险平价策略的核心思想是让每个资产对组合风险的贡献相等。

**数学表达：**
```
minimize: Σ(RC_i - 1/n)²
subject to: Σw_i = 1, w_i ≥ 0
```

其中 RC_i 是资产i的风险贡献。

### 2. 动量风险预算 (Momentum Risk Budget)

基于研报核心公式，根据预测夏普比率设定风险预算：

```
b_i = (1 + ln(E_k × IR_i + 1))²
```

其中：
- IR_i: 资产i的预测夏普比率
- k: 动量参数（越大动量效应越强）
- b_i: 归一化后的风险预算

**参数k的影响：**
| k值 | 收益 | 风险 | 适用场景 |
|-----|------|------|----------|
| 0.1 | 较低 | 最低 | 保守型投资者 |
| 0.5 | 中等 | 中等 | 稳健型投资者 |
| 1.0 | 较高 | 中等 | 平衡型投资者 |
| 1.5 | 高 | 较高 | 进取型投资者 |
| 2.0 | 最高 | 最高 | 激进型投资者 |

### 3. 层次风险平价 (Hierarchical Risk Parity)

通过层次聚类将资产分组，在每个组内应用风险平价，最后在组间进行风险分配。

## 资产配置

本项目使用以下10种资产进行回测：

| 序号 | 资产名称 | 代码 | 类型 |
|------|----------|------|------|
| 1 | 沪深300 | 000300.SH | 股票指数 |
| 2 | 恒生指数 | HSI.HK | 股票指数 |
| 3 | 日经225 | N225.JP | 股票指数 |
| 4 | 标普500 | SPX.GI | 股票指数 |
| 5 | COMEX黄金 | GC00Y.NYM | 商品期货 |
| 6 | ICE布油 | BZ00Y.NYM | 商品期货 |
| 7 | SHFE铜 | CU00Y.SHF | 商品期货 |
| 8 | 美国国债7-10年ETF | IEF.US | 债券ETF |
| 9 | 中债国债总财富指数 | CBA00603.CI | 债券指数 |
| 10 | 中债企业债AAA指数 | CBA00701.CI | 债券指数 |

## 安装与依赖

### 必要依赖

```bash
pip install numpy pandas matplotlib scipy tushare
```

### 可选依赖

```bash
pip install jupyter notebook
```

### 数据获取

本项目优先使用 **tushare** 获取数据。需要设置有效的tushare token：

```python
import tushare as ts
token = "your_tushare_token"
pro = ts.pro_api(token)
```

**注意：** 本项目配置了指定的tushare API地址，如需使用请确保网络可访问。

## 快速开始

### 1. 克隆项目

```bash
git clone <repository_url>
cd momentum_enhanced_risk_parity
```

### 2. 安装依赖

```bash
pip install numpy pandas matplotlib scipy tushare
```

### 3. 运行notebook

```bash
cd ipynb
jupyter notebook momentum_risk_parity_backtest.ipynb
```

### 4. 代码示例

```python
import sys
sys.path.append('..')

from source import (
    DataFetcher, Backtest, create_sample_data,
    plot_cumulative_returns, plot_metrics_comparison
)

# 获取数据
fetcher = DataFetcher(use_cache=True)
all_data, failed = fetcher.fetch_all_assets()

# 初始化回测
bt = Backtest(fetcher.get_daily_returns(all_data))

# 运行策略回测
bt.backtest_risk_parity(name='RiskParity')
bt.backtest_momentum_risk_budget(k=1.0, name='Momentum_k1')

# 查看结果
print(bt.get_metrics_summary())

# 可视化
plot_cumulative_returns(bt, show=True)
```

## 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 回测期间 | 2007-01-01 至 2024-09-30 | 约17年 |
| 调仓频率 | 月频 | 每月月底 |
| 交易成本 | 双边0.05% | 万分之五 |
| 协方差估计窗口 | 6个月 | 126个交易日 |
| 初始资金 | 1亿 | - |

## 复现要点

### 研报核心发现

1. **动量风险预算策略(k=1)**：
   - 年化收益：5.31%
   - 最大回撤：3.69%
   - 夏普比率：2.24

2. **传统风险平价对比**：
   - 年化收益：4.56%
   - 最大回撤：6.15%
   - 夏普比率：1.95

3. **关键改进**：
   - 动量风险预算策略在回撤控制上显著优于传统风险平价
   - 夏普比率和卡玛比率均有明显提升

### 注意事项

1. **数据限制**：部分海外资产（恒生指数、日经225、标普500、黄金、布油、铜、美国国债ETF）数据可能无法通过tushare获取，项目会自动回退到示例数据进行演示。

2. **真实数据获取**：如需获取完整真实数据，建议：
   - 使用Wind终端导出数据
   - 使用Bloomberg数据接口
   - 使用efinance、yfinance等其他数据源补充

3. **参数优化**：研报中的最优参数（k=1）是基于历史数据得出，实际使用需结合自身风险偏好和市场环境进行调整。

## 模块说明

### source/config.py
- 配置参数定义
- tushare token设置
- 资产列表定义
- 回测参数设置

### source/data_fetcher.py
- DataFetcher类：数据获取和缓存
- 支持指数、ETF、期货、债券指数数据
- 自动缓存到本地

### source/risk_parity.py
- RiskParity类：传统风险平价实现
- HierarchicalRiskParity类：层次风险平价
- 风险贡献计算函数

### source/momentum_risk_budget.py
- MomentumRiskBudget类：核心动量风险预算策略
- MomentumRiskBudgetStrategy类：多参数批量回测
- 预测夏普比率计算
- 风险预算到权重的映射

### source/backtest.py
- Backtest类：完整回测框架
- 支持多种策略同时回测
- 自动计算交易成本

### source/visualization.py
- 累计收益曲线绘制
- 回撤分析图
- 权重热力图
- 指标对比图

## 致谢

本项目基于国泰君安证券研究报告《基于动量效应改进风险平价策略》复现。

原始研报作者：
- 张雪杰（分析师）
- 朱惠东（研究助理）
- 张涵（研究助理）

## 免责声明

本项目仅供学习研究之用，不构成任何投资建议。历史业绩不代表未来表现，投资有风险，入市需谨慎。
