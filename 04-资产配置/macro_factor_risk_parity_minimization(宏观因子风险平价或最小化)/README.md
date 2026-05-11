# 宏观风险配置方法复现项目

## 项目概述

本项目复现国泰君安研报《宏观风险配置方法思考：以风险平价和风险最小化为例——大类资产配置量化模型研究系列之八》（2024.05.29）。

### 研报核心内容

研报借鉴桥水全天候策略的底层思想，基于国内宏观六因子模型，提出了量化宏观风险配置的思路，并对以下三种策略进行了回测：

| 策略 | 核心理念 | 回测年化收益 | 回测夏普比 |
|------|---------|-------------|-----------|
| 资产风险平价（基准） | 各资产风险贡献相等 | 5.78% | 1.23 |
| 宏观风险平价 | 各宏观因子风险贡献相等 | 6.28% | 1.48 |
| 宏观风险最小化 | 组合宏观风险暴露最小化 | 6.81% | 1.47 |

回测区间：2010年1月 - 2024年4月

## 项目结构

```
macro_factor_risk_parity_minimization/
├── source/
│   ├── __init__.py           # 包初始化，导出核心接口
│   ├── config.py             # 项目配置（资产列表、因子体系、回测参数）
│   ├── data_loader.py        # 多数据源市场数据加载（tushare/yfinance/akshare）
│   ├── macro_factors.py      # 宏观因子构建（mimicking portfolio + PCA）
│   ├── risk_attribution.py   # Boudt & Benedict (2013) 风险归因
│   ├── optimization.py       # 风险平价/风险最小化优化求解器
│   ├── performance.py        # 绩效指标计算
│   ├── backtest.py           # 回测引擎
│   └── plotting.py            # 可视化模块
├── ipynb/
│   └── macro_risk_allocation.ipynb  # 主notebook（完整复现流程）
├── output/                   # 回测图表输出目录
├── 参考研报/                 # 原始研报PDF及文本
├── requirements.txt
└── README.md
```

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### tushare Token配置

本项目使用自定义tushare API端点。在 `source/config.py` 中已预配置：
- Token: `1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb`
- API URL: `http://jiaoch.site`

### 数据源说明

| 数据类型 | 主数据源 | 备选数据源 |
|---------|---------|-----------|
| A股指数（沪深300、中证1000）| tushare | baostock |
| 海外指数（标普500、纳斯达克）| yfinance | - |
| 国内债券指数 | tushare | akshare |
| 商品（原油、黄金）| yfinance | akshare |
| 南华商品指数 | akshare | - |
| 宏观因子原始数据 | tushare/akshare | - |

## 快速开始

### 在Jupyter Notebook中运行

```bash
cd d:\Documents\trae_projects\macro_factor_risk_parity_minimization
jupyter notebook ipynb/macro_risk_allocation.ipynb
```

### 在Python脚本中运行

```python
from source.backtest import run_full_backtest
results = run_full_backtest()
```

### 模块化调用示例

```python
from source import *

# 1. 初始化tushare
init_tushare()

# 2. 加载数据
bt = MacroRiskBacktester()
bt.load_data()

# 3. 构建因子
bt.build_factors(use_pca=True)
bt.prepare_factor_data(lookback=36)

# 4. 运行策略
bt.run_all_strategies()
bt.compare_all_strategies()

# 5. 生成图表
plot_all(bt)
```

## 核心模块说明

### config.py
- 定义10种标的资产及其数据源
- 定义宏观六因子体系及其mimicking portfolio配置
- 回测参数（区间、调仓频率）

### data_loader.py
- `init_tushare()`: 初始化tushare（使用自定义API端点）
- `load_all_asset_prices()`: 加载所有资产月度价格
- `load_macro_indicators()`: 加载宏观指标（CPI/PPI/PMI等）
- 支持缓存，避免重复下载

### macro_factors.py
- `build_macro_factors()`: 基于mimicking portfolio + PCA构建因子
- `compute_factor_exposures()`: 计算滚动因子暴露度（B矩阵）

### risk_attribution.py
实现 Boudt & Benedict (2013) 风险归因框架：

$$\text{FRC}_i = \frac{\gamma_i \cdot (\theta \gamma)_i / \sqrt{\gamma^T \theta \gamma}}{\sum_j \gamma_j \cdot (\theta \gamma)_j / \sqrt{\gamma^T \theta \gamma}}$$

- `compute_portfolio_factor_risk_contribution()`: 计算组合的因子风险贡献
- `compute_mrc_frc_series()`: 计算时序MRC/FRC

### optimization.py
- `asset_risk_parity()`: 资产风险平价优化
- `macro_risk_parity()`: 宏观风险平价优化
- `macro_risk_minimization()`: 宏观风险最小化优化
- 使用scipy.optimize SLSQP求解器

### backtest.py
- `MacroRiskBacktester`: 主回测引擎类
- `run_asset_risk_parity()`: 运行资产风险平价
- `run_macro_risk_parity()`: 运行宏观风险平价
- `run_macro_risk_minimization()`: 运行宏观风险最小化

## 数据限制说明

**⚠️ 重要**：研报使用的部分数据目前无法通过免费数据源获取，可能导致复现结果与研报存在差异。

### 无法直接获取的数据

| 数据项 | 研报用途 | 限制原因 |
|-------|---------|---------|
| PMI同比差分 | 增长因子原始指标 | 官方PMI数据需Wind/iFinD |
| CRB工业原料指数 | 增长因子mimicking | 需Bloomberg/Wind |
| 南华沪铜 | 增长因子mimicking | 需南华期货API |
| 猪肉价格 | 通胀因子mimicking | akshare可能受限 |
| 普钢螺纹 | 通胀因子mimicking | 需期货数据 |
| 信用利差(AA中票-国开债) | 信用因子 | 需Wind/iFinD |
| M2同比、社融同比 | 流动性因子 | tushare受限 |
| 房地产开发行业指数 | 增长因子mimicking | 需Wind/同花顺 |

### 当前替代方案

1. **PCA因子提取**: 当mimicking portfolio数据缺失时，使用PCA从资产收益率中提取补充因子
2. **Mimicking Portfolio近似**: 使用可获取的资产收益率（如南华金属、沪深300等）模拟因子收益
3. **单资产代理**: 用单一资产直接代理复杂因子（如用中债国债收益率直接代理利率因子）

### 建议补充数据

如需精确复现研报结果，建议获取：
- Wind终端数据
- iFinD（同花顺）数据
- 宏观因子历史数据库

## 核心结果解读

### 宏观风险平价 vs 资产风险平价

| 指标 | 资产风险平价 | 宏观风险平价 | 差异 |
|------|------------|------------|-----|
| 年化收益 | ~5.78% | ~6.28% | +0.47% |
| 年化波动 | ~2.50% | ~2.41% | -0.09% |
| 最大回撤 | -4.92% | -3.72% | -1.20% |
| 夏普比 | 1.23 | 1.48 | +0.25 |

**结论**：宏观风险平价在风险控制方面效果显著，将最大回撤降低了约1.2个百分点。

### 宏观风险最小化

| 指标 | 资产风险平价 | 宏观风险最小化 | 差异 |
|------|------------|--------------|-----|
| 年化收益 | ~5.78% | ~6.81% | +0.97% |
| 年化波动 | ~2.50% | ~2.78% | +0.28% |
| 最大回撤 | -4.92% | -4.38% | -0.54% |
| 夏普比 | 1.23 | 1.47 | +0.24 |

**结论**：宏观风险最小化在承担更多波动的同时获得了更高收益，风险调整后表现更优。

## 依赖库

- **numpy, scipy**: 数值计算与优化
- **pandas**: 数据处理
- **scikit-learn**: PCA因子提取
- **matplotlib, seaborn**: 可视化
- **tushare**: A股数据
- **yfinance**: 美股及大宗商品数据
- **akshare**: 中国宏观数据
- **baostock**: 备选A股数据

## 参考文献

1. 国泰君安《宏观风险配置方法思考：以风险平价和风险最小化为例》（2024.05.29）
2. Boudt & Benedict (2013). "Asset allocation with risk factors"
3. Bridgewater "All Weather Strategy" (1996)
4. 钱恩平 (2005). Risk Parity / Equal Risk Contribution

## License

本项目仅供学习研究参考，不构成投资建议。
