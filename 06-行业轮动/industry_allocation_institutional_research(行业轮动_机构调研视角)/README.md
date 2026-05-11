# 行业配置策略：机构调研视角

基于华泰证券研报的量化策略复现项目

## 项目简介

本项目复现华泰证券2021年3月发布的金工研报《行业配置策略：机构调研视角》的核心策略逻辑和回测框架。

### 研报核心观点

1. **机构调研信息能反映机构投资者的关注方向**，可用于构建有效的量化策略
2. **机构调研数据和股票收益率存在关联关系**：被调研越多的个股超额收益越高
3. **可构建三类策略**：
   - 事件驱动策略：年化超额收益 >12%
   - 定期选股策略：年化超额收益 >18%
   - 行业轮动策略：年化超额收益 >7%

## 项目结构

```
industry_allocation_institutional_research/
├── README.md                    # 项目说明文档
├── requirements.txt              # Python依赖
├── source/                      # 源代码模块
│   ├── __init__.py
│   ├── data_loader.py           # 数据获取模块
│   ├── data_preprocessor.py     # 数据预处理模块
│   ├── analysis.py              # 数据分析模块
│   ├── backtest.py              # 回测引擎
│   └── strategies/              # 策略模块
│       ├── __init__.py
│       ├── event_driven.py      # 事件驱动策略
│       ├── regular_stock.py     # 定期选股策略
│       └── industry_rotation.py # 行业轮动策略
├── ipynb/                       # Jupyter notebooks
│   ├── 01_数据获取与预处理.ipynb
│   ├── 02_策略回测.ipynb
│   └── 03_综合分析报告.ipynb
└── output/                      # 输出目录
    ├── figures/                 # 生成的图表
    ├── results/                 # 回测结果
    └── data/                   # 处理后的数据
```

## 核心依赖库

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
tushare>=1.2.0        # 主要数据源(需token)
akshare>=1.10.0       # 备用数据源
baostock>=0.8.8       # 备用数据源
efinance>=0.15.0      # 金融数据
jupyter>=1.0.0
ipykernel>=6.0.0
scikit-learn>=0.24.0
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据获取

使用tushare获取数据（需设置token）：

```python
from source.data_loader import DataLoader

# 初始化（使用项目提供的token）
dl = DataLoader(token="1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb")

# 获取指数数据
index_data = dl.get_index_data(index_code='000985.SH', start_date='20150101', end_date='20210228')

# 获取股票数据
stock_data = dl.get_daily_stock_data('000001.SZ', '20150101', '20210228')
```

### 2. 策略回测

```python
from source.strategies.event_driven import EventDrivenStrategy
from source.strategies.regular_stock import RegularStockStrategy
from source.strategies.industry_rotation import IndustryRotationStrategy

# 事件驱动策略
event_strategy = EventDrivenStrategy()
results = event_strategy.run_strategy2(survey_df, price_df)

# 定期选股策略
regular_strategy = RegularStockStrategy()
results = regular_strategy.run_typical_strategy(survey_df, price_df, num_stocks=20)

# 行业轮动策略
industry_strategy = IndustryRotationStrategy()
results = industry_strategy.run_typical_strategy(industry_survey_df, industry_price_df)
```

## 策略详情

### 事件驱动策略

**策略逻辑**：当机构调研次数超过阈值时买入，持有一定天数后卖出

**推荐参数**：
- 策略1：回看1日，持仓200日，阈值50次 → 年化超额收益12.43%
- 策略2：回看60日，持仓100日，阈值50次 → 年化超额收益12.48%

### 定期选股策略

**策略逻辑**：定期持有机构关注度最高的股票

**推荐参数**：
- 回看天数：120日
- 调仓频率：周频
- 持股数：20只
- 年化超额收益：18.68%

### 行业轮动策略

**策略逻辑**：基于行业调研次数Z-score进行轮动

**推荐参数**：
- 平滑窗口：250日
- 持有行业数：5个
- 调仓频率：月频
- 年化超额收益：约7%

## 重要数据说明

### 缺失数据

**机构调研数据（核心）**：本研报使用的机构调研数据来源于Wind商业数据库（ASHAREINSTITUTIONALACTIVITY、ASHAREINSTITUTIONALPARTICIPANT表），目前免费数据接口无法获取完整历史数据。

### 数据获取途径

1. **Wind终端**：使用WD函数获取
2. **同花顺iFinD**：使用iFinD终端
3. **聚源数据**：使用聚源终端
4. **Tushare Pro付费版**：可能提供部分数据

### 数据补充方法

获取机构调研数据后，将其保存为CSV文件并修改 `source/data_loader.py` 中的 `get_all_survey_data()` 方法。

## 复现要点

1. **回测区间**：2015年1月1日至2021年2月28日
2. **基准指数**：中证全指（000985.SH）
3. **手续费**：双边千一
4. **数据处理**：
   - 剔除公告日晚于实际调研日5个交易日以上的数据
   - 剔除涨跌停、ST、上市不满180天的股票

## 风险提示

1. 模型根据历史规律总结，历史规律可能失效
2. 市场出现超预期波动，可能导致拥挤交易
3. 机构调研数据存在发布延迟
4. 策略主要靠盈亏比赚钱，月度胜率50%-60%

## 免责声明

本项目仅供学习研究参考，不构成任何投资建议。报告中涉及的具体行业或股票不代表任何投资意见，请投资者谨慎、理性看待。

## 研报来源

华泰证券研究所
- 研究员：林晓明、李聪、韩晳
- 发布日期：2021年3月28日
- 报告类型：深度研究

## License

MIT License