# 中观景气度之上游资源/中游材料

## 项目概述

本项目复现了华泰证券金工研究报告《中观景气度之上游资源中游材料》(2021-10-14) 的核心方法论，使用 **Nowcasting模型** 构建上游资源和中游材料板块的6个周期行业景气度指数。

**原始研报作者**: 林晓明 (SAC No. S0570516010001)

## 目录结构

```
meso_prosperity_upstream_midstream/
├── README.md                    # 项目说明文档
├── source/                      # 源代码模块
│   ├── __init__.py             # 包初始化文件
│   ├── data_fetcher.py         # 数据获取模块
│   ├── nowcasting_model.py      # Nowcasting模型核心
│   ├── industry_indicators.py   # 行业指标库配置
│   ├── preprocessing.py         # 数据预处理
│   ├── evaluation.py           # 评价指标计算
│   ├── industry_analyzer.py     # 行业景气度分析器
│   └── visualization.py        # 可视化模块
├── ipynb/                      # Jupyter notebooks
│   └── meso_prosperity_reproduction.ipynb  # 研报复现notebook
├── output/                     # 输出结果目录
│   └── (分析结果和图表将保存在此)
└── data/                       # 数据目录
    └── (本地数据文件)
```

## 功能特性

### 核心功能

1. **Nowcasting模型实现**
   - 状态空间模型框架
   - PCA初始化 + OLS估计的简化求解方法
   - 支持缺失值处理
   - 可预测未来因子值

2. **行业覆盖**
   - 上游资源: 石油石化、煤炭、有色金属
   - 中游材料: 钢铁、基础化工、建材

3. **评价体系**
   - ROE复现度(R²)
   - 方向预测准确率(最新一期/下期预测)

4. **数据获取**
   - 支持tushare、baostock、akshare等数据源
   - 宏观指标、行业数据、商品价格等

## 快速开始

### 环境要求

```bash
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
statsmodels>=0.13.0
matplotlib>=3.4.0
tushare>=1.2.0
```

### 安装依赖

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib tushare
```

### 基本使用

```python
# 导入模块
from source import (
    IndustrySentimentAnalyzer,
    init_tushare,
    NowcastingModel
)

# 初始化tushare
init_tushare()

# 创建行业分析器
analyzer = IndustrySentimentAnalyzer(
    industry_name='石油石化',
    start_date='20150101',
    end_date='20211231'
)

# 加载数据
analyzer.load_roe_data()
analyzer.load_all_indicators()

# 构建景气度指数
analyzer.build_sentiment_indices()

# 评估指数
analyzer.evaluate_indices()

# 生成报告
print(analyzer.generate_report())
```

## 模块说明

### 1. data_fetcher.py - 数据获取模块

从各数据源获取市场数据，优先使用tushare。

**主要函数:**
- `init_tushare()`: 初始化tushare pro接口
- `get_industry_roe_ttm()`: 获取行业ROE_TTM数据
- `get_macro_ppi()`: 获取PPI数据
- `get_market_indicator()`: 获取宏观经济指标
- `get_commodity_price()`: 获取大宗商品价格
- `get_index_daily()`: 获取指数日线数据

### 2. nowcasting_model.py - Nowcasting模型核心

实现Nowcasting状态空间模型及简化求解算法。

**主要类:**
- `NowcastingModel`: Nowcasting模型类
- `SentimentIndexBuilder`: 景气度指数构建器

**模型方程:**
```
1) y_i,t = b_i*f_t + e_i,t (if w_i,t = 1) else 0
2) f_t = a_1*f_{t-1} + a_2*f_{t-2} + δ_t
3) e_i,t = h_i1*e_i,t-1 + h_i2*e_i,t-2 + φ_i,t
```

### 3. industry_indicators.py - 行业指标库

定义6个行业的代理指标及预期载荷系数。

**包含行业:**
- 石油石化: 18个代理指标
- 煤炭: 24个代理指标
- 有色金属: 32个代理指标
- 钢铁: 31个代理指标
- 基础化工: 28个代理指标
- 建材: 18个代理指标

### 4. preprocessing.py - 数据预处理

处理指标数据的季节性、趋势和口径转换。

**主要类:**
- `IndicatorPreprocessor`: 指标预处理器
- `IndicatorSelector`: 指标选择器

**处理方法:**
- 平稳性检验(ADF检验)
- 去趋势处理
- 异常值处理(n倍标准差)
- 缺失值填充
- 同比/环比转换
- 标准化/归一化

### 5. evaluation.py - 评价指标

计算ROE复现度和方向预测准确率。

**主要函数:**
- `calculate_roe_reproduction()`: 计算ROE复现度(R²)
- `calculate_direction_accuracy()`: 计算方向准确率
- `evaluate_sentiment_index()`: 综合评价

**评价指标:**
- ROE复现度(R²): 衡量景气度指数对ROE_TTM的解释程度
- 最新一期方向准确率: 基于因子动量
- 下期预测方向准确率: 基于短期预测

### 6. industry_analyzer.py - 行业景气度分析器

整合各模块，提供完整的行业景气度分析流程。

**主要类:**
- `IndustrySentimentAnalyzer`: 单一行业分析器
- `MultiIndustrySentimentAnalyzer`: 多行业分析器

### 7. visualization.py - 可视化模块

绘制景气度指数及相关分析图表。

**主要类/函数:**
- `SentimentIndexVisualizer`: 景气度指数可视化器
- `plot_industry_chain()`: 绘制产业链结构图

**图表类型:**
- 景气度指数与ROE对比图
- 方向信号对比图
- 因子载荷图
- 多行业对比图
- 产业链结构图

## 研报复现要点

### 研报核心结论

| 行业 | 实时ROE复现度 | 全局ROE复现度 | 最新一期方向准确率 | 下期预测方向准确率 |
|------|--------------|--------------|------------------|------------------|
| 石油石化 | 57% | 60% | 69.2% | 76.9% |
| 煤炭 | 25% | 65% | 82.1% | 82.1% |
| 有色金属 | 38% | 67% | 71.8% | 71.8% |
| 钢铁 | 37% | 58% | 82.1% | 84.6% |
| 基础化工 | 67% | 80% | 66.7% | 66.7% |
| 建材 | 5% | 36% | 84.6% | 84.6% |

### 简化求解方法流程

1. **PCA初始化**: 取全体代理指标均有有效值的截面，开展PCA作为景气度指数初始值
2. **OLS拟合**: 使用最小二乘法拟合隐含状态方程，得到各指标载荷系数和特质因子
3. **状态转移拟合**: 分别拟合景气度指数和特质因子状态转移方程
4. **缺失值预测**: 基于状态转移方程预测缺失值
5. **再次PCA**: 使用预测值填补后再次PCA，得到最终景气度指数

### 方向信号定义

**最新一期方向 (侧重因子动量):**
```
trend_t = sign[(f_t + f_{t-1} + f_{t-2}) - (f_{t-12} + f_{t-13} + f_{t-14})]
```

**下期预测方向 (侧重短期预测):**
```
trend_t = sign[(f_{t+1}预测 + f_t + f_{t-1}) - (f_{t-11} + f_{t-12} + f_{t-13})]
```

## 数据说明

### 可获取的数据

通过开源库可以获取:
- 交易所行情数据(股票、期货、指数)
- 部分宏观经济指标(PPI、CPI、PMI、M0/M1/M2等)
- 交易日历
- 行业分类信息

### 难以获取的数据(需专业数据源)

以下数据需要Wind、Bloomberg等专业数据源才能获取:
- 申万行业详细ROE_TTM
- 南华商品价格指数(沪燃油指数、能化指数等)
- Myspic钢材价格指数
- 中国煤炭价格指数
- 各港口货物运量
- 详细进出口数据(平均单价)
- 铁路/公路货运量分项数据

**注意**: 本项目使用模拟数据演示方法论，实际使用时建议接入专业数据源获取真实数据。

## 复现局限性

1. **数据限制**: 部分原始数据无法通过开源库获取，使用模拟数据替代
2. **指标完整性**: 研报使用220+指标构建石油石化行业景气度，开源数据仅能覆盖部分
3. **时效性**: 研报基于2021年数据，近期数据可能存在差异
4. **参数优化**: 未进行参数的敏感性分析和优化

## 后续改进建议

1. **数据层面**
   - 接入Wind/Bloomberg等专业数据源
   - 扩展更多代理指标
   - 实现日频/周频数据获取

2. **模型层面**
   - 添加参数敏感性分析
   - 实现自适应参数选择
   - 加入行业特异性调整

3. **应用层面**
   - 开发行业轮动策略
   - 构建实时监控系统
   - 扩展到其他行业板块

## 参考研报

1. 《中观行业景气度：Nowcasting初探》(2021-09-26)
2. 《行业配置策略：景气度视角》(2020-11)
3. 《行业配置策略：投资时钟视角》(2021-07)
4. 《确立研究对象，行业拆分与聚类》(2020-03-03)

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。原始研报版权归华泰证券所有，本项目不对其内容的准确性做任何保证。投资者应自行承担投资风险。

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
