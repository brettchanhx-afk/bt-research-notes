# 基于宏观因子的大类资产配置框架

本项目复现国泰金工研报《基于宏观因子的大类资产配置框架-大类资产配置量化模型研究系列之四》(2023.06.14)

## 项目概述

本项目实现了一套基于宏观因子的量化资产配置框架，主要功能包括：

1. **宏观因子构建**：使用PCA和资产组合方法从市场数据中提取宏观因子
2. **因子暴露度计算**：使用带先验信息的LASSO回归计算资产对各因子的敏感度
3. **资产配置优化**：基于Blyth和Greenberg框架进行权重优化
4. **风险分析与归因**：将组合风险分解到各因子
5. **回测与评估**：对因子偏离策略进行历史回测

## 项目结构

```
macro_factor_asset_alloc/
├── data/                          # 数据文件夹
│   ├── seven_assets_price_2013_PCA.csv    # 7种资产日频价格
│   ├── original_macro_factor_2013.csv      # 原始宏观因子数据
│   └── high_frequency_macro_factor_portfolio.csv  # 高频宏观因子
├── source/                        # 源代码
│   ├── config.py                  # 配置参数
│   ├── data_fetcher.py            # 数据获取模块
│   ├── macro_factors.py           # 宏观因子构建
│   ├── factor_exposure.py         # 因子暴露度计算
│   ├── portfolio_optimizer.py     # 资产配置优化
│   ├── risk_analysis.py           # 风险分析模块
│   ├── backtest.py                # 回测引擎
│   ├── main.py                    # 主程序入口
│   ├── run_test.py                # 测试运行脚本
│   ├── run_with_real_data.py      # 真实数据运行脚本
│   ├── run_with_real_data_v2.py   # 真实数据运行脚本v2
│   ├── plot_figures.py            # 图表绘制脚本(模拟数据)
│   └── plot_figures_real_data.py  # 图表绘制脚本(真实数据)
├── ipynb/                         # Jupyter Notebook
│   └── macro_factor_allocation.ipynb
├── output/                        # 输出结果
│   ├── figure_*.png               # 生成的图表
│   └── factor_strategy_summary*.csv  # 策略绩效汇总
└── README.md                      # 项目说明文档
```

## 数据说明

### 1. seven_assets_price_2013_PCA.csv
7种资产2013年至今的日频收盘价序列：
- 沪深300 (000300.SH)
- 中证500 (000905.SH)
- 中债国债 (CBA00601.CB)
- 中债企业债 (CBA02001.CB)
- 南华商品 (NHCI.SL)
- 布伦特原油 (BRN0Y.ICE)

### 2. original_macro_factor_2013.csv
国君量化配置团队的原始宏观因子体系：
- 制造业PMI
- 固定资产投资完成额累计同比
- 社会消费品零售总额当月同比
- 进出口金额当月同比
- CPI当月同比
- PPI当月同比
- 中债国债到期收益率(10年)
- 中债中短期票据到期收益率(AA, 3年)
- 中债国开债到期收益率(3年)
- 美元指数
- M2同比
- 社会融资规模增量

### 3. high_frequency_macro_factor_portfolio.csv
高频化宏观因子所需数据（资产组合法）：
- 恒生指数
- CRB现货指数
- 南华铜指数
- 申万行业指数(房地产开发)
- 生猪价格
- 布伦特原油期货
- 螺纹钢现货
- 中债国债总指数
- 中债信用债总指数
- 美元指数
- 申万大盘指数市盈率
- 申万小盘指数市盈率

## 核心模块

### config.py
配置项目参数，包括：
- 资产列表及权重
- 宏观因子定义
- 回测时间段设置
- tushare API配置

### macro_factors.py
宏观因子构建模块：
- `MacroFactorBuilder`: 主因子构建类
- `construct_all_factors()`: 构建全部宏观因子
- 支持PCA降维和资产组合法

### factor_exposure.py
因子暴露度计算模块：
- `FactorExposureWithPrior`: 带先验信息的LASSO回归
- `fit()`: 拟合计息暴露度
- `transform()`: 转换新数据

### portfolio_optimizer.py
资产配置优化模块：
- `PortfolioAllocator`: 组合配置器
- `blyth_optimize()`: Blyth最优化框架
- `greenberg_optimize()`: Greenberg最优化框架
- `risk_parity()`: 风险平价模型

### risk_analysis.py
风险分析模块：
- `RiskAnalyzer`: 风险分析器
- `compute_factor_covariance()`: 计算因子协方差
- `compute_heterogeneous_variance()`: 计算异质风险方差
- `decompose_portfolio_risk()`: 分解组合风险

### backtest.py
回测引擎模块：
- `BacktestEngine`: 回测引擎
- `run_factor_deviation_backtest()`: 运行因子偏离回测
- `BacktestResultAnalyzer`: 结果分析器

## 依赖库

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tushare>=1.3.0
scipy>=1.9.0
```

安装依赖：
```bash
pip install pandas numpy scikit-learn matplotlib tushare scipy
```

## 使用方法

### 1. 运行完整回测（模拟数据）

```bash
cd source
python run_test.py
```

### 2. 运行完整回测（真实数据）

```bash
cd source
python run_with_real_data_v2.py
```

### 3. 生成图表

模拟数据图表：
```bash
cd source
python plot_figures.py
```

真实数据图表：
```bash
cd source
python plot_figures_real_data.py
```

### 4. Jupyter Notebook演示

```bash
cd ipynb
jupyter notebook macro_factor_allocation.ipynb
```

## 研报复现要点

### 1. 宏观因子体系
研报采用六因子系统：
- **增长因子**: 反映经济增长水平
- **通胀因子**: 反映物价水平变化
- **利率因子**: 反映利率环境变化
- **信用因子**: 反映信用利差变化
- **汇率因子**: 反映汇率变化
- **流动性因子**: 反映流动性环境

### 2. 四步配置框架
1. **选取合适的因子**: 使用PCA降维或宏观指标构造
2. **计算资产的因子暴露**: 使用LASSO回归+先验信息
3. **确定因子的目标暴露**: 基准暴露+因子偏离
4. **匹配因子的目标暴露**: Blyth/Greenberg优化框架

### 3. 风险归因
基于Boudt(2013)方法，将组合风险分解为：
- 因子共同风险
- 资产特异性风险

## 输出结果

### 图表文件
- `figure_02_factor_allocation_flowchart.png`: 因子配置流程图
- `figure_pca_variance_explained.png`: PCA方差解释率
- `figure_asset_returns_heatmap.png`: 资产收益热力图
- `figure_factor_correlation.png`: 因子相关性热力图
- `figure_exposure_matrix.png`: 因子暴露矩阵
- `figure_cumulative_returns.png`: 累计收益对比图
- `figure_factor_returns.png`: 因子收益图
- `figure_risk_decomposition.png`: 风险分解图

### 数据文件
- `factor_strategy_summary.csv`: 策略绩效汇总
- `factor_strategy_summary_real_data.csv`: 真实数据策略绩效

## 注意事项

1. 数据来源：真实数据需从tushare等数据源获取，部分数据可能存在缺失
2. 回测区间：默认2013-06至2023-05
3. 再平衡频率：默认月度再平衡
4. 因子偏离度：默认0.05（可调整）

## 参考文献

1. 国泰君安证券研究所 - 《基于宏观因子的大类资产配置框架》
2. Boudt, K., et al. (2013). "Quantifying Gains from Active Risk Factor Timing"
3. PCA降维与因子提取相关文献