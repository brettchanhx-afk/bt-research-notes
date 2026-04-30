# 债券基金久期与信用配置风格分析

> 复现华泰证券研报《基于净值数据对债券基金久期和信用配置风格进行估计的方法》(2020-08-21)

## 项目简介

本项目通过债券基金净值收益率对中债指数收益率的回归分析，估计债券基金的：

1. **久期配置风格** - 衡量利率风险暴露
2. **信用配置风格** - 衡量信用风险暴露

## 核心方法

### 久期估计

通过回归模型估计基金的久期暴露：

```
R_fund = α + β₁×R_index1 + β₂×R_index2 + ... + βₙ×R_indexn
D_fund = α + β₁×D_index1 + β₂×D_index2 + ... + βₙ×D_indexn
```

其中：
- R_fund: 基金日收益率
- R_index: 债券指数日收益率
- D_index: 债券指数久期
- β: 回归系数（因子暴露）

### 信用估计

同样的回归框架，将久期 D 替换为信用评分 C：

```
C_fund = α + β₁×C_index1 + β₂×C_index2 + ... + βₙ×C_indexn
```

信用评分标准（人民银行2006）：
- AAA: 16.5分
- AA+: 15分
- AA: 14分
- AA-: 12.5分

## 项目结构

```
bond_style_regression/
├── data/              # 数据目录
├── ipynb/             # Jupyter notebook
├── output/            # 输出结果
├── source/            # 核心模块
│   ├── __init__.py
│   ├── data_loader.py # 数据加载
│   ├── factor.py      # 因子计算
│   ├── backtest.py    # 回测模块
│   ├── plot.py        # 可视化
│   └── utils.py       # 工具函数
├── main.py            # 主程序
└── README.md
```

## 使用方法

### 命令行运行

```bash
# 分析指定基金
python main.py 000012

# 指定日期范围
python main.py 000012 --start 20230101 --end 20231231

# 自定义滚动窗口
python main.py 000012 --window 90 --step 30
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| fund_code | 基金代码 | 000012 |
| --start | 开始日期 (YYYYMMDD) | 一年前 |
| --end | 结束日期 (YYYYMMDD) | 今天 |
| --window | 滚动窗口大小（交易日） | 60 |
| --step | 滚动步长（交易日） | 20 |
| --output | 输出目录 | output/ |

### Python API

```python
from source.data_loader import BondDataLoader
from source.factor import BondStyleEstimator

# 加载数据
loader = BondDataLoader()
fund_df = loader.get_fund_nav("000012", "20230101", "20231231")
index_data = loader.load_all_index_data("2023-01-01", "2023-12-31")

# 风格估计
estimator = BondStyleEstimator(loader.get_index_info())
estimator.fit(X, y)

# 获取结果
duration = estimator.estimate_duration()
credit = estimator.estimate_credit()
style_box = estimator.get_style_box()
```

## 中债指数体系

本项目使用以下中债指数作为风格因子：

| 指数类型 | 代码 | 久期 | 信用评分 |
|----------|------|------|----------|
| 国债1-3年 | CBA00161.CS | 2.0 | 0 |
| 国债3-5年 | CBA00162.CS | 4.0 | 0 |
| 国债5-7年 | CBA00163.CS | 6.0 | 0 |
| 国债7-10年 | CBA00164.CS | 8.5 | 0 |
| 金融债1-3年 | CBA01221.CS | 2.0 | 16 |
| 金融债3-5年 | CBA01222.CS | 4.0 | 16 |
| 金融债5-7年 | CBA01223.CS | 6.0 | 16 |
| 高信用等级 | CBA01921.CS | 4.0 | 15.5 |
| 企业债AAA | CBA04221.CS | 4.0 | 16.5 |
| 企业债AA+ | CBA04121.CS | 4.0 | 15 |
| 企业债AA | CBA04021.CS | 4.0 | 14 |

## 风格箱定义

### 久期风格

| 分类 | 久期范围 |
|------|----------|
| 短久期 | < 3.5年 |
| 中久期 | 3.5 - 6年 |
| 长久期 | > 6年 |

### 信用风格

| 分类 | 信用评分 |
|------|----------|
| 高信用 | >= 16 (AAA) |
| 中信用 | 14 - 16 (AA~AA+) |
| 低信用 | < 14 (AA及以下) |

## 输出结果

运行后将在 `output/<fund_code>/` 目录生成：

1. `<fund_code>_style_result.json` - 风格估计结果
2. `<fund_code>_rolling_results.csv` - 滚动回测结果
3. `<fund_code>_style_evolution.png` - 风格演变图
4. `<fund_code>_style_box.png` - 风格箱定位图
5. `<fund_code>_factor_exposure.png` - 因子暴露图
6. `<fund_code>_report_*.txt` - 分析报告

## 依赖库

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
scipy>=1.9.0
efinance>=0.5.0
akshare>=1.10.0
```

## 参考文献

- 华泰证券研究所. 《基于净值数据对债券基金久期和信用配置风格进行估计的方法》. 2020-08-21.
- Sharpe, W. F. (1992). Asset Allocation: Management Style and Performance Measurement.
- 中债金融估值中心. 中债指数编制说明书.

## License

MIT
