# 不同协方差矩阵估计方法对比分析 - 大类资产配置量化模型研究系列之五

## 项目简介

本项目完整复现了国泰君安证券《不同协方差矩阵估计方法对比分析》研报的核心内容，包含：
- 多种协方差矩阵估计方法实现
- 最低波动组合、目标波动组合构建
- Black-Litterman模型和风险平价策略回测
- 完整的绩效评估与可视化分析

## 项目结构

```
multi_asset_cov_estimate_research/
├── source/                    # 源代码目录
│   ├── __init__.py           # 包初始化
│   ├── data_fetcher.py       # 数据获取模块 (tushare)
│   ├── covariance_estimators.py  # 协方差估计方法
│   ├── evaluation.py         # 组合绩效评估
│   ├── portfolio_builders.py  # 组合构建器
│   ├── backtest.py           # 回测引擎
│   └── strategies.py         # BL模型 & 风险平价策略
├── ipynb/                    # Jupyter Notebooks
│   ├── 协方差矩阵估计方法对比分析.ipynb
│   └── 真实数据回测.ipynb
├── configs/                  # 配置文件
│   └── asset_config.py       # 资产配置参数
├── output/                   # 输出目录
└── 参考研报/                # 原始研报资料
```

## 协方差估计方法

本项目实现了以下8种协方差矩阵估计方法：

| 方法 | 说明 |
|------|------|
| `sample_cov` | 样本协方差 |
| `ledoit_wolf_constant_variance` | Ledoit-Wolf等方差压缩估计 |
| `ledoit_wolf_single_factor` | Ledoit-Wolf单因子压缩估计 |
| `ledoit_wolf_constant_correlation` | Ledoit-Wolf等相关压缩估计 |
| `random_matrix` | 随机矩阵滤波 |
| `risk_metrics` | RiskMetrics EWMA (λ=0.94) |
| `ccc_garch` | CCC-GARCH模型 |
| `dcc_garch` | DCC-GARCH模型 |

## 安装依赖

```bash
pip install numpy pandas scipy matplotlib seaborn tushare akshare arch
```

## 快速开始

### 1. 使用tushare配置（重要）

在 `data_fetcher.py` 中已配置好您的tushare token：
```python
token = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
```

### 2. 运行Jupyter Notebook

```bash
cd d:\Documents\trae_projects\multi_asset_cov_estimate_research
jupyter notebook ipynb/协方差矩阵估计方法对比分析.ipynb
```

### 3. 代码示例

```python
from source.data_fetcher import DataFetcher
from source.covariance_estimators import CovarianceEstimator
from source.evaluation import PortfolioEvaluator
from source.portfolio_builders import PortfolioBuilder
from source.backtest import BacktestEngine

# 初始化模块
data_fetcher = DataFetcher()
cov_estimator = CovarianceEstimator()
portfolio_builder = PortfolioBuilder()
evaluator = PortfolioEvaluator()
backtest_engine = BacktestEngine(initial_capital=1000000)

# 获取数据
asset_config = {
    '沪深300': '000300.SH',
    '标普500': 'SPX.GI',
    '恒生指数': 'HSI.HK',
}
returns_df = data_fetcher.fetch_asset_returns(asset_config, '20170101', '20231231')

# 计算协方差矩阵
cov_matrix = cov_estimator.get_covariance(returns_df, method='ledoit_wolf_single_factor')

# 构建最低波动组合
weights = portfolio_builder.minimum_variance_portfolio(cov_matrix, allow_short=False)

# 运行回测
result = backtest_engine.run_rolling_backtest(
    returns=returns_df,
    cov_estimator=cov_estimator,
    portfolio_builder=portfolio_builder,
    method='sample_cov',
    lookback_period=252,
    rebalance_freq='monthly',
    allow_short=False,
    portfolio_type='min_variance'
)

# 评估绩效
metrics = evaluator.evaluate_portfolio(weights, returns_df)
print(f"年化收益率: {metrics['annualized_return']:.4f}")
print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
```

## 研报核心结论复现

### 最低波动组合
- 对于大类资产，压缩估计(`ledoit_wolf_single_factor`)和DCC-GARCH表现较好
- 限制卖空条件下，样本协方差仍具竞争力

### 目标波动组合
- 压缩估计和CCC-GARCH模型表现较好
- 随机矩阵方法在资产类别较少时表现不佳

### Black-Litterman策略
- 不同协方差估计方法效果差异不显著
- 推荐使用样本协方差简化计算

### 风险平价策略
- 除`ledoit_wolf_constant_variance`外，其他方法效果接近
- 推荐使用样本协方差或等相关系数压缩估计

## 数据获取说明

### 使用tushare
优先使用tushare获取A股和债券指数数据，已在代码中配置。

### 使用akshare
对于tushare无法获取的数据，可以使用akshare：
```python
import akshare as ak
df = ak.index_zh_a_hist(symbol='000300', period='daily', start_date='20170101', end_date='20231231')
```

### 模拟数据
Notebook中包含模拟数据生成，确保在无数据时也能运行演示。

## 输出说明

运行Notebook后，图表将保存到 `output/` 目录：
- `covariance_matrices_comparison.png` - 协方差矩阵热力图
- `min_variance_comparison.png` - 最低波动组合对比
- `target_volatility_comparison.png` - 目标波动组合对比
- `bl_strategy_comparison.png` - BL策略对比
- `risk_parity_comparison.png` - 风险平价策略对比

## 注意事项

1. **数据获取限制**：部分指数（如南华商品、中债指数）可能需要专业数据源
2. **GARCH模型**：使用`arch`库，计算较慢，可使用较短历史窗口加速
3. **卖空限制**：默认限制卖空，符合国内市场规则

## 参考文献

[1] Ledoit, O., & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance*, 10(5), 603-621.

[2] Engle, R. (2002). Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models. *Journal of Business and Economic Statistics*, 20(3), 339-350.

[3] 国泰君安证券. (2023). 不同协方差矩阵估计方法对比分析 - 大类资产配置量化模型研究系列之五.

## 项目作者

本项目为金融工程量化研究复现项目，仅供学习交流使用。

## 免责声明

本项目仅用于量化研究和学习，不构成任何投资建议。投资有风险，入市需谨慎。