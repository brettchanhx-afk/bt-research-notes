# 投资时钟策略项目

基于华泰证券《行业配置策略：投资时钟视角》研报的量化策略复现

## 项目概述

本项目复现了华泰证券金工团队2021年7月发布的研报核心策略，该策略构建了**增长-通胀**和**信用-货币**双轮驱动投资时钟，用于大类资产配置和行业轮动。

### 研报核心结论

| 策略 | 年化收益 | 夏普比率 | 最大回撤 |
|------|---------|---------|---------|
| 大类资产投资时钟策略 | 7.39% | 1.69 | -6.11% |
| 行业轮动投资时钟策略 | 16.08% | 0.59 | -51.93% |

## 目录结构

```
industry_clock_allocation/
├── config.py                      # 项目配置文件
├── README.md                      # 项目说明文档
├── source/                        # 源代码模块
│   ├── __init__.py
│   ├── data_fetcher.py           # 数据获取模块 (tushare)
│   ├── preprocessing.py           # 宏观指标预处理
│   ├── indicator_selector.py      # 领先指标筛选
│   ├── factor_synthesis.py        # 因子合成 (OECD法)
│   ├── asset_mapping.py           # 宏观-资产映射
│   ├── factor_predictor.py        # 因子预测 (相位/动量)
│   ├── asset_strategy.py          # 大类资产配置策略
│   ├── industry_strategy.py       # 行业轮动策略
│   └── backtest.py                # 回测引擎
├── ipynb/                         # Jupyter Notebook演示
│   └── investment_clock_demo.ipynb
└── output/                        # 输出目录
    ├── config/
    ├── logs/
    └── results/
```

## 核心模块说明

### 1. 数据获取 (data_fetcher.py)
- 使用tushare Pro API获取市场数据
- 支持：宏观指标、指数数据、行业数据、国债收益率等
- tushare初始化已配置自定义API地址

### 2. 指标预处理 (preprocessing.py)
- **变频处理**：统一到月频
- **缺失值填充**：线性插值、向前向后填充
- **季节性调整**：X-11季节调整方法
- **HP滤波**：趋势-周期分解（月度参数λ=129600）
- **同比计算**：统一到同比增长率口径

### 3. 领先指标筛选 (indicator_selector.py)
四种定量筛选方法：
- **时差相关系数**：评估指标领先/滞后关系
- **K-L信息量**：衡量信息增益
- **拐点匹配率**：Bry-Boschan算法识别拐点
- **DTW距离**：动态时间规整评估形态相似性

### 4. 因子合成 (factor_synthesis.py)
三种合成方法：
- **OECD法**：标准化偏差加权求和（推荐）
- **PCA法**：主成分分析提取第一主成分
- **扩散指数法**：扩张指标占比

### 5. 因子预测 (factor_predictor.py)
- **相位判断法**：基于42个月周期，判断上行/下行趋势
- **因子动量法**：比较当期与过去3期均值
- **复合策略**：结合两者观点

### 6. 资产映射 (asset_mapping.py)
构建双轮投资时钟：
- **增长-通胀时钟**：复苏、过热、滞胀、衰退
- **信用-货币时钟**：宽货币+宽信用等四种状态

### 7. 配置策略
- **AssetStrategy**：大类资产配置（目标波动率控制）
- **IndustryStrategy**：行业轮动（等权+动量增强）

### 8. 回测引擎 (backtest.py)
- 支持大类资产和行业轮动回测
- 计算完整绩效指标
- 支持结果可视化

## 复现要点

### 数据需求（需补充）

研报使用了超过500个宏观指标，以下是需要获取的关键数据：

**增长因子领先指标（11个）**：
- 发电量、铝材、硫酸、乙烯、空调、汽车产量
- 叉车销量、房屋新开工面积、房地产开发投资
- 货物周转量、税收收入

**通胀因子领先指标（5个）**：
- 猪肉价格、CRB油脂现货指数
- 螺纹钢价格、布伦特原油、MyIpic矿价指数

**信用因子**：
- M1、M2、社会融资规模存量
- 金融机构贷款余额、企业存款余额

**货币因子**：
- 一年期国债到期收益率

**资产数据**：
- 沪深300、中证500、创业板指
- 中债-国债总净价指数
- 南华工业品指数、黄金现货

**行业数据**：
- 中信一级行业指数（剔除综合、综合金融）

### 环境配置

```bash
pip install pandas numpy scipy scikit-learn matplotlib tushare
pip install dtaidistance  # DTW距离计算
```

### 使用示例

```python
from source.data_fetcher import DataFetcher
from source.factor_synthesis import FactorSynthesis
from source.factor_predictor import FactorPredictor
from source.backtest import BacktestEngine

# 初始化
fetcher = DataFetcher()
synthesizer = FactorSynthesis()
predictor = FactorPredictor(cycle_period=42)
backtester = BacktestEngine()

# 获取数据
pmi_df = fetcher.get_PMI(start_date='20080101', end_date='20210630')

# 合成因子
growth_factor = synthesizer.synthesize_growth_factor(leading_indicators)

# 预测因子方向
views = predictor.predict_factor_direction(growth_factor, method='combined')

# 运行回测
results = backtester.run_asset_backtest(strategy, asset_returns, ...)
```

## 研报结论与复现差异

### 研报结论
1. 增长-通胀时钟与美林时钟规律一致
2. 领先因子比基准指标（PMI）有更好的收益区分度
3. 复合策略表现优于单一方法

### 复现说明
- 由于数据获取限制，本项目使用模拟数据演示核心逻辑
- 完整复现需要获取研报中的完整宏观指标数据库
- 建议使用Wind终端或Baidu Mofan API获取专业数据

## 风险提示

1. 模型基于历史规律，可能失效
2. 宏观-资产映射基于长期统计，与短期表现可能有出入
3. 实际投资需考虑交易成本、滑点等

## 依赖库

```
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tushare>=1.3.0
dtaidistance>=2.3.0
```

## 参考文献

1. 华泰证券《行业配置策略：投资时钟视角》（2021年7月）
2. Bry-Boschan (1971). Cyclical analysis of time series
3. Hodrick-Prescott (1997). Post-war U.S. business cycles
4. OECD (1978). Leading indicators

## License

MIT License
