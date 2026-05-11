# Nowcasting行业景气度预测项目

本项目复现华泰证券金工研报《中观行业景气度：Nowcasting初探》，实现了基于动态因子模型的行业景气度实时预测系统。

## 项目概述

**研报核心观点：**
- Nowcasting模型实时预测行业景气度，把握经济系统同源性和经济周期内生性
- 动态因子模型(DFM)包含三个方程：隐含状态方程、隐含因子状态转移方程、特质因子状态转移方程
- 使用EM算法求解模型参数
- 五大优势：内生预测、支持数据缺失、支持混频数据、支持重叠指标、不丢失信息

## 数据源集成

本项目集成了多种金融数据源，按优先级自动切换：

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | akshare | 沪深300指数、宏观（PPI/GDP）等 |
| 2 | efinance | 东方财富网数据，股票/指数K线数据 |
| 3 | baostock | BaoStock开源数据，股票/指数历史数据 |
| 4 | tushare | Tushare Pro数据（需token） |
| 后备 | 模拟数据 | 数据源均不可用时自动生成 |

**注意**：由于申万行业指数（801010.SI）难以获取，当前使用沪深300（000300.SH）作为市场代表进行演示。宏观指标使用PPI和GDP。

**自动切换机制**：优先尝试高优先级数据源，失败时自动降级到下一优先级，最终后备模拟数据。

## 目录结构

```
meso_industry_prosperity_nowcasting/
├── source/                     # 核心源代码模块
│   ├── __init__.py            # 包初始化
│   ├── config.py              # 配置文件
│   ├── utils.py               # 工具函数
│   ├── data_fetcher.py        # 多数据源数据获取
│   ├── dfm_model.py           # 动态因子模型
│   ├── nowcasting_model.py    # Nowcasting模型
│   ├── sentiment_index.py     # 行业景气度指数
│   └── backtest.py            # 回测评价模块
├── ipynb/                      # Jupyter notebooks
│   └── nowcasting_demo.ipynb  # 复现演示notebook
├── output/                     # 输出目录
│   ├── sentiment_index.csv     # 景气度指数
│   ├── sentiment_index.png     # 景气度指数图
│   ├── indicator_weights.png   # 指标权重图
│   └── backtest_results.png    # 回测结果图
├── data/                       # 数据缓存目录
├── tests/                      # 测试目录
├── 参考研报/
│   ├── 2021-09-26_华泰证券_...pdf
│   └── 2021-09-26_华泰证券_...txt
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖库
└── README.md                   # 项目说明文档
```

## 核心模块说明

### 1. config.py - 配置模块
- Tushare API配置
- 模型参数（隐含因子数、指标数、EM算法参数）
- 文件路径配置

### 2. data_fetcher.py - 多数据源数据获取
- `MultiSourceDataFetcher`: 多数据源获取器
  - 自动检测并初始化各数据源
  - 按优先级自动切换数据源
  - 获取股票/指数/宏观/行业/财务数据
- `SteelIndustryDataFetcher`: 钢铁行业专用
  - 获取钢铁行业指数
  - 构建行业指标矩阵
- `DataCache`: 数据缓存

### 3. dfm_model.py - 动态因子模型
- `DynamicFactorModel`: DFM实现
  - E步：卡尔曼滤波估计隐含因子
  - M步：最大似然估计参数
- `DFMSentimentIndex`: DFM景气度指数

### 4. nowcasting_model.py - Nowcasting模型
- `NowcastingModel`: Nowcasting核心
  - 内生预测景气度方向
  - 支持数据缺失和混频数据
  - 滚动Nowcasting
- `IndicatorSelector`: 代理指标筛选
  - 解释度阈值：20%
  - 平稳性p值：10%
  - 最少指标数：15个

### 5. sentiment_index.py - 行业景气度指数
- `SteelIndustrySentimentIndex`: 钢铁行业景气度
- `SentimentIndexComparison`: 指数对比分析

### 6. backtest.py - 回测评价
- `IndustryTimingBacktest`: 行业择时回测
- `GodViewBacktest`: "上帝视角"回测
- `TimingComparison`: 择时效果对比

### 7. utils.py - 工具函数
- `check_stationarity()`: ADF平稳性检验
- `standardize_series()`: Z-score标准化
- `calculate_direction_accuracy()`: 方向准确率

## 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖：**
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.5.0
- statsmodels>=0.13.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- tushare>=1.3.0
- efinance>=0.13
- akshare>=1.10.0
- baostock>=0.8.8

## 使用方法

### 方式1：运行主程序
```bash
python main.py
```

### 方式2：Jupyter Notebook
```bash
jupyter notebook ipynb/nowcasting_demo.ipynb
```

### 方式3：代码中调用
```python
from source import (
    SteelIndustryDataFetcher,
    SteelIndustrySentimentIndex,
    IndustryTimingBacktest
)

# 数据获取
data_fetcher = SteelIndustryDataFetcher()
indicator_data = data_fetcher.get_steel_indicators()

# 构建景气度指数
sentiment_builder = SteelIndustrySentimentIndex()
sentiment_index = sentiment_builder.build_sentiment_index(indicator_data)

# 回测分析
backtester = IndustryTimingBacktest()
result = backtester.run_backtest(sentiment_index, price_series)
```

## 研报复现要点

### 1. 代理指标筛选标准
- 隐含因子对指标的解释度 > 20%
- 指标序列在10%显著性水平下平稳
- 选用代理指标数目不少于15个

### 2. 方向预测准确率
- Nowcasting最新一期方向预测准确率
- 下期方向预测准确率
- 研报结果：分别高达82.1%和84.6%

### 3. 择时效果评价
- 对比景气度指数择时 vs "上帝视角"择时
- 评估指标：总收益、年化收益、夏普比率、最大回撤、胜率

## 数据说明

**数据获取优先级：**
1. akshare - 沪深300指数、PPI/GDP宏观数据
2. efinance - 股票/指数K线数据
3. baostock - 股票/指数历史数据
4. tushare - 股票/指数/财务数据

**实际使用数据：**
- 市场指数：沪深300（000300.SH）日频数据
- 宏观指标：PPI月度同比、GDP季度同比
- 行业指标：收盘价、收益率、成交量、移动平均线、波动率等

**注意：** 数据源可能因网络或API限制不稳定，项目提供模拟数据作为后备保证运行。

## 注意事项

1. **Tushare Token**: config.py中已配置，如过期请更新
2. **网络要求**: 需保证网络畅通以便访问各数据源
3. **模型参数**: 可根据实际情况调整n_factors、em_max_iter等参数
4. **回测风险**: 历史业绩不代表未来，请谨慎投资

## 项目特色

- **多数据源集成**: 四大数据源自动切换，永不掉线
- **真实市场数据**: 通过akshare实时拉取沪深300和PPI/GDP数据
- **模块化设计**: 每个功能独立成模块，低耦合高内聚
- **完整流程**: 从数据获取到回测评价的完整闭环
- **可视化输出**: 生成图表便于分析

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。报告中涉及的具体行业不代表任何投资意见，请投资者谨慎、理性地看待。模型根据历史规律总结，历史规律可能失效。
