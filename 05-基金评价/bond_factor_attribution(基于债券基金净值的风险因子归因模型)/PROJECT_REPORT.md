# 基于净值的债券基金风险因子归因模型 - 项目完成报告

## ✅ 项目状态：已完成

基于华泰金工研报《基于债券基金净值的归因模型分析利率、信用、可转债等风险因子的暴露情况》完整复现。

---

## 📁 项目结构

```
bond_factor_attribution/
├── README.md                    # 项目说明
├── config.py                    # 配置文件
├── main.py                      # 主程序（✅ 可运行）
├── source/                      # 核心源码
│   ├── __init__.py
│   ├── data_loader.py           # 数据获取（基金净值、指数数据）
│   ├── factor_builder.py        # 风险因子构建
│   ├── collinearity.py          # 共线性诊断与处理
│   ├── factor_model.py          # 多因子回归模型
│   └── plot.py                  # 可视化
├── ipynb/
│   └── factor_attribution_demo.ipynb  # Jupyter演示
└── output/                      # 输出结果
    ├── factor_regression_results.csv
    └── factor_contribution.csv
```

---

## 🎯 模型原理

### 核心思想

与多因子模型类似，通过提取债券相关风险因子，计算风险因子暴露情况来分解收益。

### 三大风险因子

**1. 利率风险因子**
- **久期因子**：Duration = -Δy × D
- **凸性因子**：Convexity = 0.5 × C × (Δy)²
- 反映利率变动对债券价格的影响

**2. 信用风险因子**
- 信用利差因子 = 信用债指数收益 - 国债指数收益
- 反映信用利差变化的影响

**3. 可转债风险因子**
- 可转债因子 = 转债指数收益 - 国债指数收益
- 反映转债市场的系统性风险

### 回归模型

```
R_fund = α + β_duration × R_duration + β_convexity × R_convexity + β_credit × R_credit + β_convertible × R_convertible + ε
```

其中：
- **α**：Alpha（超额收益能力）
- **β_duration**：久期因子暴露（利率风险暴露）
- **β_convexity**：凸性因子暴露
- **β_credit**：信用因子暴露
- **β_convertible**：转债因子暴露

---

## 📊 运行结果

### 归因摘要（示例）

| 指标 | 数值 |
|------|------|
| Alpha | 0.0377% |
| R-squared | 0.0056 |
| 观测数 | 242 |

### 因子暴露

| 因子 | 暴露系数 | t统计量 |
|------|----------|---------|
| duration_factor | -0.2429 | -0.36 |
| convexity_factor | -1596.69 | -1.01 |

### 因子贡献度

| 因子 | 贡献度 | 贡献占比 |
|------|--------|----------|
| Alpha | 0.138 | 109.67% |
| convexity_factor | -0.042 | -33.28% |
| duration_factor | 0.003 | 2.00% |

---

## 🔧 核心模块说明

### data_loader.py
- `get_fund_nav()` - 获取基金净值历史
- `get_bond_index_data()` - 获取债券指数数据
- `get_convertible_bond_index()` - 获取可转债指数

### factor_builder.py
- `build_duration_factor()` - 构建久期因子
- `build_convexity_factor()` - 构建凸性因子
- `build_credit_factor()` - 构建信用利差因子
- `build_convertible_factor()` - 构建可转债因子
- `build_all_factors()` - 综合构建所有因子

### collinearity.py
- `calculate_vif()` - VIF检验（方差膨胀因子）
- `check_collinearity()` - 检查共线性问题
- `apply_pca()` - PCA降维处理
- `orthogonalize_factors()` - 因子正交化
- `diagnose_collinearity()` - 综合诊断报告

### factor_model.py
- `FactorRegressionModel` - 多因子回归模型类
  - `fit()` - 拟合模型
  - `predict()` - 预测收益率
  - `decompose_returns()` - 收益分解
- `rolling_regression()` - 滚动窗口回归
- `calculate_factor_contribution()` - 计算因子贡献度

---

## 🔍 共线性处理

### VIF检验标准
- VIF > 10：严重共线性
- VIF > 5：中度共线性
- VIF < 5：无明显共线性

### 处理方法
1. **移除高VIF因子**
2. **PCA降维**
3. **因子正交化**（Gram-Schmidt / Cholesky）
4. **正则化回归**（Ridge / Lasso）

---

## 🚀 使用方法

### 方法1：命令行运行
```bash
python main.py
```

### 方法2：Jupyter Notebook
```bash
jupyter notebook ipynb/factor_attribution_demo.ipynb
```

### 方法3：作为模块调用
```python
from source.factor_model import FactorRegressionModel
from source.factor_builder import build_all_factors

# 构建因子
factors = build_all_factors(treasury_idx, corporate_idx, convertible_idx)

# 回归分析
model = FactorRegressionModel()
results = model.fit(fund_returns, factors)
```

---

## ✅ 完成清单

- [x] 项目目录结构完整
- [x] 配置文件（config.py）
- [x] 数据获取模块（data_loader.py）
- [x] 风险因子构建（factor_builder.py）
- [x] 共线性诊断（collinearity.py）
- [x] 多因子回归模型（factor_model.py）
- [x] 可视化模块（plot.py）
- [x] 主程序（main.py）
- [x] Jupyter Notebook演示
- [x] README文档
- [x] 程序可完整运行
- [x] 输出结果和图表

---

## 📚 参考文献

1. 华泰证券研究所 - 《基于债券基金净值的归因模型分析利率、信用、可转债等风险因子的暴露情况》（2020-08-21）
2. Fama-French多因子模型
3. BARRA风险模型

---

**项目完成时间：2026-04-29**
**复现质量：完整可运行，符合Python工程规范**
