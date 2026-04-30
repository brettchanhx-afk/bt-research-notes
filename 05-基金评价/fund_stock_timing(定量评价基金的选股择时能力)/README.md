# 基金选股择时能力定量评价模型
## T-M 模型 / H-M 模型 / C-L 模型

> 华泰金工研究 | 2020-08-21 | 复现项目

---

## 一、项目简介

本项目复现华泰金工研报《定量评价基金的选股择时能力》中的三种经典量化模型，用于评估基金经理的**选股能力（Stock Selection）**和**择时能力（Market Timing）**。

### 三种模型概述

| 模型 | 全称 | 年份 | 核心思想 |
|------|------|------|----------|
| **T-M** | Treynor-Mazuy | 1966 | 在詹森α模型中加入二次项 `(Rm-Rf)^2` |
| **H-M** | Henriksson-Merton | 1981 | 引入虚拟变量 D 区分牛熊市 |
| **C-L** | Chang-Lewellen | 1984 | 区分多头/空头市场，分别估计Beta |

### 核心公式（摘自研报原文）

**T-M 模型：**
```
Rp - Rf = α + β1(Rm - Rf) + β2(Rm - Rf)^2 + ε
```
- α > 0 且显著 → 基金经理有选股能力
- β2 > 0 且显著 → 基金经理有择时能力

**H-M 模型：**
```
Rp - Rf = α + β1(Rm - Rf) + β2(Rm - Rf)·D + ε
```
- D = 1（牛市，Rm > Rf）；D = 0（熊市，Rm ≤ Rf）
- 牛市Beta = β1 + β2，熊市Beta = β1
- β2 > 0 且显著 → 有择时能力

**C-L 模型：**
```
Rp - Rf = α + β1(Rm - Rf)·D1 + β2(Rm - Rf)·D2 + ε
```
- 多头市场（Rm > Rf）：D1=0, D2=1
- 空头市场（Rm ≤ Rf）：D1=1, D2=0
- β2 - β1 > 0 → 有择时能力（多头Beta > 空头Beta）

---

## 二、目录结构

```
fund_stock_timing/
├── config.py          # 配置文件（数据源、分析区间、参数）
├── main.py             # 命令行主程序
├── README.md           # 本文件
│
├── source/             # 核心模块
│   ├── __init__.py
│   ├── data_loader.py  # 多数据源数据获取（efinance/akshare/baostock）
│   ├── factor.py       # T-M / H-M / C-L 模型实现
│   ├── backtest.py     # 滚动窗口回测、业绩归因
│   ├── plot.py         # 可视化（仪表盘、时序图、散点图）
│   └── utils.py        # 工具函数
│
├── ipynb/              # Jupyter Notebook 复现
│   └── 研报复现.ipynb
│
├── data/               # 原始数据缓存目录
├── output/             # 分析结果输出目录
│   └── {fund_code}/
│       ├── {fund_code}_timing_dashboard.png   # 四合一仪表盘
│       ├── {fund_code}_timing_results.json     # 结构化结果
│       ├── {fund_code}_rolling_TM.csv          # T-M滚动回归CSV
│       └── {fund_code}_rolling_HM.csv         # H-M滚动回归CSV
│
└── 研报原文.txt        # pdfplumber 提取的研报文本
```

---

## 三、快速开始

### 环境要求

```bash
pip install pandas numpy matplotlib seaborn statsmodels efinance akshare baostock
```

### 命令行使用

```bash
# 基本分析（中欧价值精选混合A，021181）
python main.py --fund 021181

# 自定义区间
python main.py --fund 021181 --start 2021-01-01 --end 2026-04-28

# 启用滚动回测（1年窗口，月度步长）
python main.py --fund 021181 --rolling --window 252 --step 21

# 仅运行指定模型
python main.py --fund 021181 --rolling
```

### Python 脚本使用

```python
from source.data_loader import load_all_data
from source.factor import StockTimingEvaluator

# 加载数据
fund_returns, bench_returns = load_all_data(
    fund_code='021181',
    start_date='2021-01-01',
    end_date='2026-04-28',
)

# 运行三模型分析
evaluator = StockTimingEvaluator(fund_returns, bench_returns)
results = evaluator.evaluate()

# 打印汇总表
print(evaluator.get_summary())
```

### Jupyter Notebook

```bash
jupyter notebook ipynb/研报复现.ipynb
```

---

## 四、配置说明

编辑 `config.py` 或在 `main.py` 中传入参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `START_DATE` | 2018-01-01 | 分析开始日期 |
| `END_DATE` | 2026-04-28 | 分析结束日期 |
| `BENCHMARK` | 000300.SH | 基准指数（沪深300） |
| `RISK_FREE_RATE` | 0.015 | 年化无风险利率（1.5%） |
| `ROLLING_WINDOW` | 252 | 滚动窗口（交易日，1年） |
| `ROLLING_STEP` | 21 | 滚动步长（交易日，月度） |

---

## 五、数据源说明

数据按以下优先级自动拉取（用户指定）：

1. **efinance** — 基金净值、持仓数据（优先）
2. **akshare** — A股、期货、基金、宏观经济
3. **baostock** — A股历史行情、指数
4. **mootdx** — 通达信数据接口
5. **yfinance** — 美股、国际期货
6. **bondpy** — 债券数据
7. **fundata** — 基金数据
8. **ifind-MCP** — 同花顺iFind接口

> 所有数据均为**真实市场数据**，不编造、不使用模拟数据。

---

## 六、输出说明

### 1. 仪表盘图（`_timing_dashboard.png`）

四合一图表：
- 基金 vs 基准超额收益散点图（含回归线）
- 三种模型 alpha/beta2 系数对比柱状图
- R-squared 对比柱状图
- 汇总表格（显著性标注）

### 2. 滚动回测图（`_rolling_{model}.png`）

- Alpha 时序图（选股能力变化）
- Beta2 时序图（择时能力变化）
- R-squared 时序图（模型解释力）

### 3. JSON 结果（`_timing_results.json`）

包含所有模型系数、p值、显著性判断、R2等，可直接被后续程序读取。

---

## 七、关键实现要点

### 7.1 超额收益率计算

```python
# 日频无风险利率 = 年化利率 / 252
daily_rf = RISK_FREE_RATE / 252
excess_fund  = fund_returns  - daily_rf   # Rp - Rf
excess_bench = bench_returns - daily_rf   # Rm - Rf
```

### 7.2 T-M 模型回归变量

```python
X = pd.DataFrame({
    'bench':    excess_bench,          # (Rm - Rf)
    'bench_sq': excess_bench ** 2,     # (Rm - Rf)^2
})
X = sm.add_constant(X)  # 添加常数项 α
```

### 7.3 H-M 模型虚拟变量

```python
D = (excess_bench > 0).astype(int)  # D=1牛市，D=0熊市
X = pd.DataFrame({
    'bench':   excess_bench,
    'bench_D': excess_bench * D,     # (Rm - Rf) * D
})
```

### 7.4 C-L 模型虚拟变量

```python
D1 = (excess_bench <= 0).astype(int)  # 空头市场
D2 = (excess_bench >  0).astype(int)   # 多头市场
X = pd.DataFrame({
    'bench_D1': excess_bench * D1,   # (Rm - Rf) * D1
    'bench_D2': excess_bench * D2,   # (Rm - Rf) * D2
})
```

---

## 八、结果解读指南

### 择时能力判断标准

| 模型 | 指标 | 条件 | 含义 |
|------|------|------|------|
| T-M | β2 | β2 > 0 且 p < 0.05 | 基金经理能预判市场涨跌 |
| H-M | β2 | β2 > 0 且 p < 0.05 | 牛市仓位高于熊市 |
| C-L | β2 - β1 | 差值 > 0 | 多头市场Beta大于空头市场Beta |

### 选股能力判断标准

| 模型 | 指标 | 条件 | 含义 |
|------|------|------|------|
| 所有模型 | α | α > 0 且 p < 0.05 | 基金经理有选股超额收益 |

### 综合判断

- **3/3模型显著**：择时/选股能力**确认**
- **2/3模型显著**：择时/选股能力**较强确认**
- **1/3模型显著**：择时/选股能力**弱确认**
- **0/3模型显著**：择时/选股能力**未确认**

---

## 九、注意事项

1. **数据量要求**：每个模型至少需要30个以上数据点，滚动回测建议1年以上数据
2. **无风险利率**：本项目使用存款基准利率1.5%，实际可用Shibor或国债收益率替代
3. **基准选择**：主动基金建议使用沪深300，债券基金建议使用中债综合指数
4. **模型局限性**：
   - 适用于日频数据，月频数据需调整窗口长度
   - 基金经理更换时，分析结果可能失真
   - 模型假设市场收益与基金收益线性关系

---

## 十、参考文献

1. Treynor, J., & Mazuy, K. (1966). "Can Mutual Funds Outguess the Market?" Harvard Business Review.
2. Henriksson, R., & Merton, R. (1981). "On Market Timing and Investment Performance." Journal of Business.
3. Chang, E., & Lewellen, W. (1984). "Market Timing and Mutual Fund Investment Performance." Journal of Business.
4. 华泰金工研究 (2020-08-21). 《定量评价基金的选股择时能力》.
