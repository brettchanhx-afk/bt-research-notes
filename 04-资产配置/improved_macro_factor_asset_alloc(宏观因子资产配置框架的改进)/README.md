# 宏观因子资产配置框架改进

本项目复现国泰君安研报《大类资产配置量化模型研究系列之七：宏观因子资产配置框架的改进》的量化策略实现。

## 项目概述

本项目实现了一个基于6大宏观因子的资产配置框架，主要改进包括：

1. **宏观因子生成改进**：使用波动率倒数加权合成高频因子，使用黄金组合构造汇率因子
2. **因子暴露计算改进**：使用带半衰期的多远线性回归替代Lasso回归
3. **宏观打分规则**：将主观宏观观点转化为标准化的因子偏离值

## 项目结构

```
improved_macro_factor_asset_alloc/
├── source/                     # 源代码模块
│   ├── __init__.py
│   ├── config.py              # 项目配置（资产列表、因子配置、回测参数等）
│   ├── data_loader.py         # 数据获取模块（tushare等）
│   ├── factor_generator.py    # 宏观因子生成模块
│   ├── factor_exposure.py    # 因子暴露计算模块
│   ├── portfolio_optimizer.py # 组合优化模块（风险平价等）
│   ├── macro_scoring.py       # 宏观打分规则模块
│   ├── backtest.py            # 回测引擎
│   └── visualization.py       # 可视化模块
├── ipynb/                      # Jupyter notebooks
│   └── macro_factor_asset_allocation.ipynb
├── output/                     # 输出结果
│   ├── data/                  # 缓存数据
│   ├── results/              # 回测结果
│   └── logs/                 # 日志文件
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖库
└── README.md                  # 项目说明文档
```

## 安装依赖

```bash
pip install pandas numpy scipy scikit-learn matplotlib plotly tushare yfinance
```

## 数据源

本项目使用以下数据源获取市场数据：

- **tushare Pro**（首选）：股票、债券、期货、宏观指标等
- **yfinance**：国际商品数据（如COMEX黄金）

### tushare配置

在 `source/config.py` 中配置您的tushare token：

```python
TUSHARE_TOKEN = "your_token_here"
TUSHARE_HTTP_URL = "http://jiaoch.site"  # 或使用默认地址
```

## 使用方法

### 方法1：运行主程序

```bash
python main.py
```

### 方法2：使用Jupyter Notebook

```bash
jupyter notebook ipynb/macro_factor_asset_allocation.ipynb
```

### 方法3：在代码中调用

```python
from source import Backtest, Visualizer, RESULTS_DIR, BACKTEST_CONFIG

# 初始化回测
backtest = Backtest(
    start_date="2015-01-01",
    end_date="2024-02-29",
    initial_capital=100000000.0,
)

# 运行回测
results = backtest.run_backtest(use_macro_view=False)

# 获取基准回测
benchmark = backtest.run_benchmark_backtest()

# 生成可视化报告
visualizer = Visualizer()
visualizer.generate_backtest_report(
    backtest_results=results,
    portfolio_values=results['portfolio_values'],
    benchmark_values=benchmark,
)
```

## 六大宏观因子体系

| 因子 | 名称 | 原始因子 | 高频因子 |
|------|------|----------|----------|
| Growth | 增长因子 | PMI、固定资产投资、社消、进出口 | CRB工业原料、南华沪铜、房地产开发 |
| Inflation | 通胀因子 | CPI、PPI | 猪肉、布伦特原油、螺纹钢 |
| IntRate | 利率因子 | 10年期国债收益率 | 中债国债总净价指数 |
| Credit | 信用因子 | 信用利差 | 中债企业债AA-中债国开债 |
| ExchRate | 汇率因子 | 美元兑人民币 | 沪金-COMEX黄金 |
| Liquidity | 流动性因子 | M2-社融 | 申万大盘-小盘市盈率 |

## 配置资产

- 沪深300 (CSI300)
- 中证1000 (CSI1000)
- 恒生指数 (HSI)
- 中债国债 (GOV_BOND)
- 中债企业债 (CORP_BOND)
- 中证转债 (CSI_CONVERT)
- 南华商品 (COMMODITY)
- 沪金 (SH_GOLD)

## 核心模块说明

### config.py
- 项目全局配置
- 资产列表和因子配置
- 回测参数设置

### data_loader.py
- tushare API初始化和数据获取
- 支持缓存机制减少重复请求

### factor_generator.py
- 生成6大宏观因子（原始因子和高频因子）
- 波动率倒数加权方法
- 黄金组合汇率因子

### factor_exposure.py
- 基于先验信息的多元线性回归
- 半衰期加权机制
- 滚动窗口计算

### portfolio_optimizer.py
- 风险平价组合优化
- 因子暴露约束
- 换手率控制

### macro_scoring.py
- 宏观观点打分规则（5档：+2, +1, 0, -1, -2）
- 因子偏离值生成
- 目标暴露计算

### backtest.py
- 完整回测引擎
- 支持宏观观点集成
- 绩效指标计算

### visualization.py
- 支持matplotlib和plotly
- 生成策略净值、回撤、权重分配等图表

## 重要提示

1. **数据依赖**：本项目需要有效的tushare Pro API token才能获取完整数据
2. **数据缺失**：部分历史数据或高频数据可能存在缺失，代码中已做相应处理
3. **复现限制**：由于数据源差异和参数设置，实际结果可能与研报有所差异

## 研报复现要点

### 1. 宏观因子生成改进
- 使用波动率倒数加权合成高频增长和通胀因子
- 使用黄金组合（沪金做多 + COMEX黄金做空）代替美元指数构造汇率因子

### 2. 因子暴露计算改进
- 将Lasso回归改为带半衰期的多远线性回归
- 回归窗口期从10年缩短至5年
- 半衰期设为1年，及时反映短期关系变化

### 3. 宏观打分规则
- 将宏观观点分为5档（+2, +1, 0, -1, -2）
- 每档对应不同调整系数
- 目标因子暴露 = 基准暴露 + 调整系数 × 历史标准差

## 风险提示

量化模型基于历史数据构建，历史规律存在失效风险。本项目仅供研究参考，不构成投资建议。

## 许可证

MIT License

## 参考资料

- 国泰君安《大类资产配置量化模型研究系列之七：宏观因子资产配置框架的改进》(2024-03-28)

## 📝 说明
1. 数据使用情况 ：
   
   - 资产价格数据：7个资产（沪深300、中证500、南华商品、国债、企业债、布伦特原油）
   - 高频因子数据：使用资产组合构建的高频宏观因子
   - 原始宏观因子：PMI、FAI、CPI、PPI等
2. 策略说明 ：
   
   - 当前策略为纯风险平价策略（未加入宏观因子观点）
   - 基准为等权配置
   - 再平衡频率：月度
3. 注意事项 ：
   
   - 策略收益较低可能是因为未使用宏观因子观点进行主动配置
   - 可以通过设置 use_macro_view=True 来启用宏观因子观点