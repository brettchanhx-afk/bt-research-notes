# 基于净值的债券基金风险因子归因模型

基于华泰金工研报《基于债券基金净值的归因模型分析利率、信用、可转债等风险因子的暴露情况》复现。

## 📚 研报来源

华泰证券研究所 - 金工研究/深度研究 | 2020年08月21日

## 🎯 模型原理

与多因子模型类似，基于净值的归因模型通过提取债券相关风险因子，计算风险因子暴露情况来分解收益。

### 三大风险因子

1. **利率风险因子**
   - 久期（Duration）
   - 凸性（Convexity）
   - 反映利率变动对债券价格的影响

2. **信用风险因子**
   - 信用债指数 - 国债指数
   - 反映信用利差变化的影响

3. **可转债风险因子**
   - 可转债指数
   - 反映转债市场的系统性风险

### 模型框架

```
R_fund = α + β_duration × R_duration + β_convexity × R_convexity + β_credit × R_credit + β_convertible × R_convertible + ε
```

其中：
- R_fund：债券基金收益率
- β_duration：久期因子暴露（利率风险）
- β_convexity：凸性因子暴露
- β_credit：信用因子暴露
- β_convertible：转债因子暴露

### 因子共线性处理

各风险因子之间可能存在共线性，需要通过：
- 方差膨胀因子（VIF）检验
- 主成分分析（PCA）
- 正交化处理

## 📁 项目结构

```
bond_factor_attribution/
├── data/                    # 数据目录
├── source/                  # 核心源码
│   ├── __init__.py
│   ├── data_loader.py       # 数据获取（基金净值、指数数据）
│   ├── factor_builder.py    # 风险因子构建
│   ├── factor_model.py      # 多因子回归模型
│   ├── collinearity.py      # 共线性诊断与处理
│   └── plot.py              # 可视化
├── output/                  # 输出结果
├── ipynb/                   # Jupyter演示
├── config.py                # 配置文件
├── main.py                  # 主程序
└── README.md
```

## 🔧 数据源

- **tushare** - 基金净值、债券指数
- **efinance** - 基金行情
- **akshare** - 债券指数、可转债指数

## 🚀 快速开始

```bash
python main.py
```

## 📊 输出结果

- 各风险因子暴露系数
- 因子贡献度分解
- 滚动因子暴露时序图
- 共线性诊断报告
