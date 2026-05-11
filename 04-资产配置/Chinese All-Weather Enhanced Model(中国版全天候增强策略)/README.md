
# 中国版全天候增强策略

本项目基于华泰金工研报《从资产配置走向因子配置：中国版全天候增强策略》实现完整的量化回测系统。

## 项目结构

```
Chinese All-Weather Enhanced Model/
├── source/                      # 源代码目录
│   ├── __init__.py             # 模块初始化
│   ├── data_loader.py          # 数据加载模块
│   ├── risk_calculator.py      # 风险计算模块
│   ├── strategy_builder.py     # 策略构建模块
│   ├── backtest_engine.py      # 回测引擎模块
│   └── visualizer.py           # 可视化模块
├── ipynb/                      # Jupyter Notebook目录
│   └── allweather_backtest.ipynb
├── output/                     # 输出目录（自动生成）
├── config.py                   # 配置文件
├── requirements.txt            # 依赖库
├── main.py                     # 主程序
└── README.md                   # 项目说明
```

## 策略概述

### 1. 传统资产风险平价
- 所有资产直接进行风险平价配置
- 过度依赖低波动的债券资产

### 2. 全天候基准策略
- 将资产划分到四个宏观象限：
  - 增长超预期：股票、商品
  - 增长不及预期：债券、黄金
  - 通胀超预期：商品、黄金
  - 通胀不及预期：债券、黄金、高股息股票
- 对四象限进行风险平价，象限内等权
- 避免过度依赖债券资产

### 3. 全天候增强策略
- 在基准策略基础上引入预期共振动量
- 使用位移路径比动量判断宏观趋势
- 在增长和通胀维度各选择一个象限进行配置
- 进一步提升策略收益

## 资产池

| 类别 | 代码 | 名称 |
|------|------|------|
| 股票 | 510300.SH | 沪深300ETF |
| 股票 | 512100.SH | 中证1000ETF |
| 高股息 | 512890.SH | 红利低波ETF |
| 债券 | 511260.SH | 十年国债ETF |
| 债券 | 511090.SH | 三十年国债ETF |
| 商品 | 159980.SZ | 有色ETF |
| 商品 | 159981.SZ | 能化ETF |
| 商品 | 159985.SZ | 豆粕ETF |
| 黄金 | 518880.SH | 黄金ETF |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主程序

```bash
python main.py
```

### 3. 使用Jupyter Notebook

```bash
jupyter notebook ipynb/allweather_backtest.ipynb
```

## 配置说明

在 `config.py` 中可以调整以下参数：

- `DATA_CONFIG`：数据起止日期、是否使用本地数据
- `STRATEGY_CONFIG`：回看窗口、是否使用半协方差、动量参数
- `BACKTEST_CONFIG`：初始资金、调仓频率、交易成本
- `OUTPUT_CONFIG`：输出目录、是否保存图表和数据

## 模块说明

### data_loader.py
- 使用akshare获取ETF历史数据
- 构建四象限等权组合
- 支持数据的保存和加载

### risk_calculator.py
- 计算EWMA协方差/半协方差矩阵
- 实现风险平价优化
- 计算绩效指标（年化收益、波动率、夏普比率、最大回撤等）
- 计算位移路径比动量

### strategy_builder.py
- 构建全天候基准策略
- 构建全天候增强策略
- 构建传统资产风险平价策略（用于对比）

### backtest_engine.py
- 月频调仓回测
- 考虑交易成本
- 对比不同策略表现

### visualizer.py
- 绘制净值曲线
- 绘制回撤曲线
- 绘制仓位演变
- 绘制绩效对比图
- 绘制年度收益

## 回测参数

- 回测区间：2013-12-31 至 2025-04-30
- 调仓频率：月频
- 交易成本：单边万分之五
- 无风险利率：0%

## 注意事项

1. **数据获取**：首次运行需要从网络获取数据，请确保网络连接正常
2. **预期共振动量**：本项目使用简化的价格动量替代研报中的买方/卖方预期动量，如需要更精确的实现，请补充宏观数据
3. **历史不代表未来**：回测结果基于历史数据，不构成投资建议

## 参考文献

- 华泰金工研报《从资产配置走向因子配置：中国版全天候增强策略》
- Shahidi A. Balanced asset allocation: How to profit in any economic climate[M]. Hoboken, NJ: John Wiley &amp; Sons, 2014.
- Ang A. Asset management: A systematic approach to factor investing[M]. Oxford: Oxford University Press, 2014.

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。

