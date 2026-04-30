# Fama-French多因子归因分析

基于华泰证券研报《Fama模型基于净值数据对基金收益做多因子回归分析》的完整复现项目。

## 项目概述

本项目实现Fama-French五因子模型，用于对基金收益进行多因子回归分析，分解基金超额收益来源。

## Fama-French五因子模型

### 核心公式

```
R = α + b×R_M + s×SMB + h×HML + r×RMW + c×CMA + ε
```

其中：
- **R**: 基金相对于无风险的超额收益
- **R_M**: 市场因子（市场相对于无风险的超额收益）
- **SMB**: 市值因子（小市值股票相对于大市值股票的收益）
- **HML**: 价值因子（账面价值比）
- **RMW**: 盈利水平因子
- **CMA**: 投资水平因子（股票所在企业的生产再投资水平）
- **α**: 常数项，即主动管理中所追求的Alpha
- **ε**: 残差项，不能由公共因子解释的部分

## 项目结构

```
fama-factor-attribution/
├── data/               # 数据存放目录
├── ipynb/              # Jupyter复现主文件
│   └── fama_attribution.ipynb
├── source/             # 核心模块化代码
│   ├── __init__.py
│   ├── data_loader.py  # 数据获取：基金净值、因子数据
│   ├── factor.py       # 因子计算：SMB/HML/RMW/CMA构建
│   ├── backtest.py     # 回测逻辑：多因子回归、归因分析
│   ├── plot.py         # 可视化：因子暴露、收益分解
│   └── utils.py        # 工具函数：数据清洗、统计检验
├── output/             # 输出结果
└── README.md           # 项目说明
```

## 数据来源

- **基金净值数据**: efinance（优先）、akshare
- **无风险利率**: 中债国债到期收益率（1年期）
- **市场因子**: 中证800指数收益率
- **SMB/HML/RMW/CMA因子**: 基于A股全市场股票数据构建

## 使用方法

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib seaborn efinance akshare baostock
```

### 2. 运行分析

```python
from source.data_loader import FundDataLoader
from source.factor import FamaFrenchFactorBuilder
from source.backtest import FamaFrenchAttribution

# 加载数据
loader = FundDataLoader()
fund_data = loader.get_fund_nav('019888', '2022-01-01', '2024-12-31')

# 构建Fama-French五因子
factor_builder = FamaFrenchFactorBuilder()
factors = factor_builder.build_five_factors('2022-01-01', '2024-12-31')

# 多因子归因
attribution = FamaFrenchAttribution()
results = attribution.run_regression(fund_data, factors)
```

### 3. 查看结果

运行结果将保存在`output/`目录下：
- `fama_factor_exposure.csv`: 因子暴露系数
- `fama_attribution_results.csv`: 归因结果汇总
- `factor_exposure_chart.png`: 因子暴露可视化
- `return_decomposition.png`: 收益分解图

## 因子构建方法

### SMB（市值因子）
- 按市值将股票分为Small（小市值）和Big（大市值）两组
- SMB = Small组合收益率 - Big组合收益率

### HML（价值因子）
- 按账面市值比（B/M）将股票分为High（高B/M）、Medium、Low（低B/M）三组
- HML = High组合收益率 - Low组合收益率

### RMW（盈利水平因子）
- 按盈利能力（ROE）将股票分为Robust（高盈利）、Medium、Weak（低盈利）三组
- RMW = Robust组合收益率 - Weak组合收益率

### CMA（投资水平因子）
- 按投资水平（资产增长率）将股票分为Conservative（低投资）、Medium、Aggressive（高投资）三组
- CMA = Conservative组合收益率 - Aggressive组合收益率

## 复现要点

1. **数据频率**: 月度数据（与研报一致）
2. **回测区间**: 2022-01-01 至 2024-12-31（示例）
3. **回归方法**: OLS线性回归
4. **显著性检验**: t检验，p值<0.05认为显著
5. **业绩指标**: Alpha、R-squared、各因子暴露系数

## 参考文献

华泰证券《Fama模型基于净值数据对基金收益做多因子回归分析》，2020年8月21日
