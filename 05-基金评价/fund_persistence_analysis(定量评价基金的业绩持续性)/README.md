# 基金业绩持续性量化评价

基于华泰金工研报《定量评价基金的业绩持续性》的工程化复现项目。

## 项目结构

```
fund_persistence_analysis/
├── config.py              # 全局配置文件
├── main.py                # 主程序入口
├── source/
│   ├── data_loader.py     # 数据获取模块（多数据源支持）
│   ├── factor.py          # 因子计算模块（三种持续性评价方法）
│   ├── backtest.py         # 回测模块
│   ├── plot.py             # 可视化模块
│   └── utils.py            # 工具函数
├── ipynb/
│   └── 研报复现.ipynb      # 复现Notebook
├── data/                  # 基金净值数据存放目录
├── output/                # 分析结果输出目录
└── README.md              # 本文件
```

## 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scipy akshare efinance baostock
```

## 使用方法

### 1. 命令行分析

```bash
# 分析单只基金
python main.py --fund 000628 --start 2018-01-01 --end 2023-12-31

# 分析基金池
python main.py --pool --codes 000628,008404,021181

# 运行回测
python main.py --backtest --codes 000628,008404,021181
```

### 2. Jupyter Notebook

打开 `ipynb/研报复现.ipynb`，按步骤运行即可。

## 三种业绩持续性评价方法

### 1. 横截面分析法

- **原理**：将样本期划分为两个等长子期间，检验评价期超额收益与持有期超额收益的正相关性
- **公式**：α₂ᵢ = α + β × α₁ᵢ
- **判断**：β 显著为正 → 业绩有持续性

### 2. 交叉积比率法 (CPR)

- **原理**：将样本期分为多个等长期间，比较赢家和输家的转换概率
- **公式**：CPR = (WW × LL) / (WL × LW)
- **判断**：
  - CPR ≈ 1：业绩不具有持续性
  - CPR > 1：业绩有持续性
  - CPR < 1：业绩有反转倾向

### 3. Hurst指数法

- **原理**：研究时间序列历史取值对未来取值的影响力（长记忆性）
- **公式**：log((R/S)ₙ) = log(c) + H × log(n)
- **判断**：
  - 0.5 < H < 1：业绩有正向持续性
  - H = 0.5：收益随机波动
  - 0 < H < 0.5：业绩有反转倾向

## 数据源说明

项目支持多个数据源，按优先级自动选择：

1. **efinance** - 基金净值、持仓等
2. **akshare** - A股、期货、基金等
3. **baostock** - A股历史行情
4. **mootdx** - 通达信数据
5. **yfinance** - 美股、国际市场

## 输出结果

分析结果保存在 `output/` 目录：

- `XXXXXX_persistence_results.json` - JSON格式详细结果
- `XXXXXX_persistence_dashboard.png` - 综合分析仪表盘
- `fund_pool_hurst_distribution.png` - Hurst指数分布图
- `backtest_results.png` - 回测结果图

## 研报来源

华泰金工研报《定量评价基金的业绩持续性（横截面分析法、交叉积比率法、Hurst指数法）》
- 发布日期：2020年08月21日
- 研究团队：华泰金融工程团队
