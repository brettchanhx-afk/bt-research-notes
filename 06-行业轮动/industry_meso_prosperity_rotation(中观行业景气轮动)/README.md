# 中观景气视角行业轮动策略

基于华泰证券金工深度研究报告《行业配置策略：中观景气视角》的量化策略复现项目。

## 项目概述

本项目复现了华泰证券研报中提出的中观行业景气度建模框架和行业轮动策略，核心方法包括：

1. **行业指标库构建**：为16个中信行业构建相关指标库
2. **指标预处理**：总量类、价格类、同比类指标分别处理
3. **指标评价筛选**：使用时差拟合精度和DTW算法评价指标有效性
4. **Simple-Nowcasting建模**：生成中观行业景气指数
5. **行业轮动策略**：基于景气指数构建行业轮动策略

## 目录结构

```
industry_meso_prosperity_rotation/
├── source/                 # 源代码模块
│   ├── __init__.py        # 包初始化
│   ├── config.py          # 配置参数
│   ├── utils.py           # 工具函数
│   ├── data_fetcher.py    # 数据获取模块
│   ├── indicator_lib.py   # 行业指标库
│   ├── preprocessing.py   # 指标预处理
│   ├── nowcasting.py      # Simple-Nowcasting模型
│   ├── strategy.py        # 行业轮动策略
│   └── backtest.py        # 回测评估
├── ipynb/                 # Jupyter笔记本
│   └── meso_prosperity_strategy.ipynb
├── output/                # 输出结果
├── README.md
└── requirements.txt
```

## 核心模块说明

### 1. config.py - 配置模块
- tushare token和API配置
- 中信行业列表和代码映射
- 回测参数配置

### 2. data_fetcher.py - 数据获取
使用tushare Pro API获取：
- 行业指数日/月线数据
- 财务指标数据
- 宏观数据（PPI、PMI等）

### 3. indicator_lib.py - 行业指标库
为16个中信行业预设的指标库：
- 石油石化、煤炭、有色金属、钢铁
- 基础化工、建材、机械、电力设备及新能源
- 国防军工、汽车、家电、酒类
- 饮料、食品、房地产、电子

### 4. preprocessing.py - 指标预处理
- 总量类指标：同比变换
- 价格类指标：同比变换
- 同比类指标：原始值或差分
- 扩散类指标：同比变换
- 离群值处理、缺失值填充

### 5. nowcasting.py - Simple-Nowcasting模型
核心算法流程：
1. 指标评价：计算R2、时差相关系数、DTW距离
2. 指标筛选：综合评分排序选取Top-K指标
3. 景气指数生成：代理指标加权合成

### 6. strategy.py - 行业轮动策略
两种策略模式：
- **纯中观视角**：基于中观景气度打分选行业
- **宏观+中观视角**：宏观景气度与中观景气度结合

因子模式：
- orig：景气相对位置
- mom：景气月度变化
- moma3：景气3月变化
- qoq：景气环比变化

### 7. backtest.py - 回测评估
- 回测引擎：支持调仓、交易成本
- 业绩指标：年化收益、夏普比率、最大回撤、卡玛比率
- 可视化：净值曲线、回撤图、夏普比率滚动图

## 复现要点

### 数据获取
本项目使用tushare Pro获取真实市场数据：
- 行业指数：CITIC行业指数
- 财务数据：ROE_TTM、营收增速等
- 宏观数据：PPI、PMI等

### 关键参数
- 滚动窗口：60个月
- 最小有效序列：36个月
- 代理指标数量：每行业6-10个
- 策略持仓：Top 4行业
- 回测区间：2016-04至2022-06

### 预期结果
根据研报，实证结果：
- 中观景气组合年化收益：~19%
- 超额年化收益：~13%
- 夏普比率：~0.84
- 最大回撤：~-26%

## 依赖库

```
tushare>=1.4.0
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 初始化数据
```python
from source.data_fetcher import DataFetcher

fetcher = DataFetcher()
dates = fetcher.get_month_end_dates("2016-04-01", "2022-06-30")
```

### 2. 行业景气建模
```python
from source.indicator_lib import IndicatorLibrary
from source.preprocessing import IndicatorPreprocessor
from source.nowcasting import NowcastingModel

lib = IndicatorLibrary()
preprocessor = IndicatorPreprocessor(rolling_window=60)
model = NowcastingModel(rolling_window=60, top_k_indicators=6)

prosperity_index, selected_indicators, scores = model.fit(
    indicators, financial_reference
)
```

### 3. 运行回测
```python
from source.strategy import IndustryRotationStrategy
from source.backtest import Backtester, PerformanceAnalyzer

strategy = IndustryRotationStrategy(top_n=4)
backtester = Backtester(initial_capital=1000000)
analyzer = PerformanceAnalyzer()

# ... 运行回测逻辑
metrics = analyzer.calculate_metrics(returns)
```

### 4. Jupyter笔记本
打开 `ipynb/meso_prosperity_strategy.ipynb` 查看完整复现流程。

## 注意事项

1. **数据限制**：部分宏观和中观数据可能需要Wind等付费数据源，本项目使用tushare替代
2. **指标缺失**：部分原始研报中的指标在开源数据中不可得，已用相似指标替代
3. **模型简化**：Simple-Nowcasting模型已简化以提高可运行性
4. **回测风险**：历史回测结果不代表未来收益

## 研报原文参考

- 《行业配置策略：中观景气视角（1）》- 2022年1月18日
- 《行业配置策略：中观景气视角（2）》- 2022年7月18日

## 免责声明

本项目仅供学习研究使用，不构成任何投资建议。投资者应自主决策，自担风险。
