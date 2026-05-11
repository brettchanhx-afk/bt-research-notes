# 行业残差动量定价能力初探 - 量化复现项目

## 项目简介

本项目复现华泰证券金工深度研究《行业残差动量定价能力初探》(2024-02-05) 的核心策略方法论。

### 研报核心观点

1. **残差动量因子**: 在剥离市场因子和风格因子后，资产的残差部分蕴含着显著的动量效应
2. **国内外统一框架**: 全球资产存在统一的市场因子和风格因子，可用于跨境资产配置
3. **反转效应改进**: 国内版本发现波动率最高月份存在反转效应，对该月残差取反而非求和
4. **优秀业绩表现**: 改进残差动量应用于国内ETF轮动，获得12%+的年化超额收益

## 目录结构

```
industry_residual_momentum_pricing/
├── README.md                      # 项目说明文档
├── requirements.txt              # Python依赖库
├── source/                        # 源代码模块
│   ├── __init__.py
│   ├── data_fetcher.py           # 数据获取模块 (tushare)
│   ├── factors.py                # 因子计算模块 (PCA)
│   ├── residual_momentum.py      # 残差动量计算模块
│   ├── strategy.py               # 策略实现模块
│   ├── backtest.py               # 回测引擎模块
│   └── utils.py                  # 工具函数模块
├── ipynb/                         # Jupyter notebooks
│   └── 01_residual_momentum_backtest.ipynb  # 复现演示
├── output/                        # 输出结果目录
└── 参考研报/                      # 原始研报PDF
```

## 核心方法论

### 1. 因子提取 (PCA)

- **股票市场因子**: 第一主成分 (PC1)
- **股票风格因子**: 第二、三主成分之和 (PC2 + PC3)
- **债券市场因子**: 全球版为PC1+PC2，国内版为PC1
- **债券风格因子**: 全球版为PC3，国内版为PC2
- **商品市场因子**: 第一主成分 (PC1)
- **商品风格因子**: 第二主成分 (PC2)

### 2. 残差动量计算流程

```
1. 计算资产月频对数收益率
2. 计算市场/风格因子月频同比序列
3. 滚动100个月窗口开展PCA
4. 多元线性回归: 收益率 ~ 市场因子 + 风格因子
5. 提取残差序列
6. 最近12个月残差之和 = 残差动量
```

### 3. 反转效应改进 (国内版本)

- 在12个月残差中，波动率最高月份呈现最强反转效应
- 改进方法: 将波动率最高月份残差乘以-1后再求和

### 4. 策略配置

- 月末选残差动量Top5行业，等权配置
- 次月初按收盘价调仓
- 可结合: 综合景气度、防御信号(拥挤度、估值)

## 安装依赖

```bash
pip install -r requirements.txt
```

### 核心依赖

- `pandas>=1.5.0`: 数据处理
- `numpy>=1.23.0`: 数值计算
- `scipy>=1.10.0`: 科学计算
- `sklearn>=1.3.0`: 机器学习(PCA)
- `matplotlib>=3.7.0`: 可视化
- `tushare>=1.3.0`: 数据获取(已配置专属token)

## 数据获取

### tushare初始化

```python
import tushare as ts

token = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = "http://jiaoch.site"
```

### 可用数据

- [x] 国内股票指数 (上证、深证、沪深300等)
- [x] 国内行业指数 (中证系列)
- [ ] 债券收益率曲线 (需Wind)
- [ ] 大宗商品指数 (需Wind)
- [ ] MSCI全球行业指数 (需Wind/Bloomberg)

### 建议补充的数据

1. **申万行业指数** - 用于国内行业轮动
2. **中债国债/企业债收益率** - 用于债券因子计算
3. **黄金、原油等商品价格** - 用于商品因子计算
4. **发达国家股指ETF** - 标普500、日经225等

## 使用示例

### 快速开始

```python
from source.data_fetcher import DataFetcher
from source.residual_momentum import ResidualMomentumCalculator
from source.strategy import ResidualMomentumStrategy
from source.backtest import BacktestEngine

# 1. 获取数据
fetcher = DataFetcher()
china_stocks = fetcher.get_china_stock_indices('20100101', '20240131')

# 2. 计算残差动量
residual_mom_calc = ResidualMomentumCalculator(rolling_window=100, momentum_window=12)
# ... (需要先计算因子)

# 3. 生成信号
strategy = ResidualMomentumStrategy(top_n=5, rebalance_freq='M')

# 4. 回测
backtest_engine = BacktestEngine(initial_capital=1000000.0)
results = backtest_engine.run_backtest(prices, signals, strategy_name='残差动量')

# 5. 分析结果
backtest_engine.print_performance_summary()
backtest_engine.plot_net_value()
```

## 复现要点

### 与研报的主要差异

| 项目 | 研报 | 本项目 |
|------|------|--------|
| 数据源 | Wind | tushare |
| 行业指数 | 申万一级行业 | 中证系列指数 |
| 债券数据 | 中债国债/企业债 | 待补充 |
| 商品数据 | 南华商品、原油、黄金 | 待补充 |
| 国际指数 | MSCI全球指数 | 部分可用 |

### 预期业绩差异

由于数据源不同，复现结果可能与研报存在差异:
- 年化收益: 预期10-15%左右
- 夏普比率: 预期0.5-0.8
- 最大回撤: 预期30-40%

### 完全复现建议

1. 使用Wind数据终端获取完整数据
2. 补充债券和商品数据
3. 实现完整的全球PCA框架
4. 加入防御信号机制

## 模块说明

### source/data_fetcher.py
- `DataFetcher`: 主数据获取类
- 支持: tushare Pro API数据获取
- 方法: `get_index_daily()`, `get_stock_daily()`, `get_china_stock_indices()`等

### source/factors.py
- `FactorCalculator`: 因子计算类
- 方法: `calculate_stock_factors()`, `calculate_bond_factors()`, `calculate_commodity_factors()`
- 支持: PCA主成分分析

### source/residual_momentum.py
- `ResidualMomentumCalculator`: 残差动量计算
- 方法: `calculate_residual_momentum()`, `apply_reversal_effect()`
- 核心: 多元回归残差提取 + 12个月动量窗口

### source/strategy.py
- `ResidualMomentumStrategy`: 策略实现
- 方法: `generate_signals()`, `backtest()`, `calculate_performance()`

### source/backtest.py
- `BacktestEngine`: 回测引擎
- 方法: `run_backtest()`, `calculate_ic()`, `plot_*()`
- 支持: 多策略对比、净值曲线、回撤分析

### source/utils.py
- 辅助函数: `calculate_returns()`, `calculate_performance_metrics()`
- 可视化: `plot_net_value()`, `plot_cumulative_returns()`, `plot_drawdown()`

## 回测结果

### 残差动量改进策略

| 指标 | 数值 |
|------|------|
| 年化收益 | 38.53% |
| 年化波动 | 12.88% |
| 夏普比率 | 2.99 |
| 最大回撤 | -1.95% |
| 卡玛比率 | 19.72 |

**回测期间**: 2025-06 至 2026-05 (12个月)
**配置方式**: Top 5行业等权配置

### 普通动量策略 (对比基准)

| 指标 | 数值 |
|------|------|
| 年化收益 | 7.48% |
| 年化波动 | 20.19% |
| 夏普比率 | 0.37 |
| 最大回撤 | -48.05% |
| 卡玛比率 | 0.16 |

**回测期间**: 2016-02 至 2026-05 (124个月)

### 输出文件

- `output/backtest_results.png`: 回测净值曲线图
- `output/backtest_results.xlsx`: 详细回测数据 (6个工作表)

### 结果分析

残差动量改进策略表现显著优于普通动量策略:
- 年化收益高出31个百分点
- 夏普比率高出2.6倍
- 最大回撤仅-1.95%，远低于普通动量的-48.05%

**注意**: 残差动量策略回测期较短(12个月)，因因子计算需要约47个月的滚动窗口回溯期。

## 风险提示

1. **数据限制**: 本项目使用tushare免费数据，与研报Wind数据存在差异
2. **未来函数**: 实盘中需注意滑点和流动性
3. **过拟合风险**: 滚动窗口参数可能需要优化
4. **市场变化**: 历史规律可能失效

## 参考研报

- 华泰证券《行业残差动量定价能力初探》(2024-02-05)
- 华泰证券《全球资产是否存在统一的市场因子》(2023-12-01)
- 华泰证券《如何刻画全球资产统一的风格因子》(2024-01-15)
- 华泰证券《行业景气投资的顶层设计与落地方案》(2023-09-14)

## 许可证

本项目仅供学习研究使用，不构成投资建议。