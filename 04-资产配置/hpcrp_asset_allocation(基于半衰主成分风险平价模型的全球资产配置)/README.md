# 半衰主成分风险平价模型 (HPCRP) 全球资产配置策略研究

> 研报复现 - 天风证券 2017-09-18

## 项目概述

本项目复现天风证券研报《基于半衰主成分风险平价模型的全球资产配置策略研究》中的核心内容，实现了7种资产配置模型并进行全球资产配置回测。

## 研报核心内容

### 资产配置模型

| 模型 | 英文名 | 说明 |
|------|--------|------|
| 等权重 | Equal Weight (EW) | 每个资产配置相同权重 |
| 等波动率 | Equal Volatility (EV) | 波动率倒数加权 |
| 最小方差 | Minimum Variance (MV) | 最小化组合方差 |
| 最大分散化 | Maximum Diversification (MD) | 最大化分散化比率 |
| 风险平价 | Risk Parity (RP) | 各资产风险贡献相等 |
| 主成分风险平价 | PCRP | 对主成分因子进行风险均衡 |
| **半衰主成分风险平价** | **HPCRP** | 引入半衰加权到PCRP |

### 全球指数

- **A股**: 沪深300指数 (CSI300), 中证500指数 (CSI500)
- **港股**: 恒生指数 (HSI), 恒生国企指数 (HSCEI)
- **美股**: 标普500 (SPX), 纳斯达克 (NDAQ)
- **欧洲**: 富时100 (FTSE), 巴黎CAC40, 德国DAX

### 关键参数

- 调仓频率: 季度调仓
- 协方差估计窗口: 240个交易日
- 半衰期: 120个交易日 (HPCRP模型)

## 项目结构

```
hpcrp_asset_allocation/
├── source/
│   ├── data_loader.py    # 数据获取模块
│   ├── models.py        # 资产配置模型
│   ├── backtest.py      # 回测引擎
│   └── plot.py         # 可视化模块
├── ipynb/
│   └── reproduce.ipynb  # 复现notebook
├── output/             # 输出结果
│   ├── nav_curve.png
│   ├── annual_returns.png
│   └── correlation_matrix.png
├── data/               # 数据缓存
├── config.py           # 配置文件
├── main.py             # 主程序
└── README.md
```

## 安装依赖

```bash
pip install pandas numpy matplotlib scipy
pip install yfinance efinance akshare
```

## 快速开始

### 方式1: 运行主程序

```bash
python main.py
```

### 方式2: Jupyter Notebook

```bash
jupyter notebook ipynb/reproduce.ipynb
```

## 核心函数

### 数据获取

```python
from source.data_loader import fetch_global_index_data
returns = fetch_global_index_data()
```

### 模型计算

```python
from source.models import get_model_weights

# HPCRP模型
weights = get_model_weights('HPCRP', returns, half_life=120)
```

### 回测

```python
from source.backtest import run_backtest, calculate_metrics

result = run_backtest(returns, weights_func, 'HPCRP', 
                     rebalance_freq='quarterly', window=240)
metrics = calculate_metrics(result['returns'], result['nav'])
```

## 回测指标说明

| 指标 | 说明 |
|------|------|
| 年化收益率 | (1+累计收益)^(252/交易日数) - 1 |
| 年化波动率 | 日收益标准差 * sqrt(252) |
| 夏普比率 | (年化收益 - 无风险利率) / 年化波动率 |
| 最大回撤 | (净值 - 历史最高) / 历史最高 的最大值 |
| Calmar比率 | 年化收益 / 最大回撤绝对值 |

## 参考

- 天风证券 (2017-09-18). 基于半衰主成分风险平价模型的全球资产配置策略研究
- Partovi M H, Caputo M. Principal portfolios: Recasting the efficient frontier[J]. Economics Bulletin, 2004

## 注意事项

1. 数据来源: 使用 yfinance, efinance, akshare 等开源库获取真实市场数据
2. 研报原始数据区间为2009-2017年，当前复现使用更长期数据
3. 回测结果仅供参考，不构成投资建议
