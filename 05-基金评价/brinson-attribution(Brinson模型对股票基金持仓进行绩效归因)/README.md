# Brinson模型绩效归因复现项目

## 项目概述

本项目复现华泰证券研究所2020年8月发布的《Brinson模型基于持仓数据从类别配置、个券选择、交互作用进行绩效归因》研报核心内容。

## Brinson模型简介

Brinson模型由Brinson和Fachler于1985年提出，是基金绩效归因的经典模型。该模型将基金的超额收益分解为三个部分：

1. **类别配置收益 (Allocation Effect)**: 基金经理在行业/资产类别配置上的能力
2. **个券选择收益 (Selection Effect)**: 基金经理在个股选择上的能力  
3. **交互作用收益 (Interaction Effect)**: 类别配置与个券选择的联合作用

## 核心公式

### 单期Brinson模型

**四象限矩阵:**

|  | 实际组合资产类别j收益 | 基准组合资产类别j收益 |
|--|---------------------|---------------------|
| **实际组合资产类别j权重** | Q4 = ∑wp·rp (实际组合) | Q2 = ∑wp·rb (类别配置组合) |
| **基准组合资产类别j权重** | Q3 = ∑wb·rp (股票选择组合) | Q1 = ∑wb·rb (基准组合) |

**收益分解:**

- **总超额收益**: R = Q4 - Q1
- **类别配置收益**: R_AA = Q2 - Q1 = ∑(wp - wb)·rb
- **个券选择收益**: R_SS = Q3 - Q1 = ∑(rp - rb)·wb
- **交互作用收益**: R_I = R - R_AA - R_SS = ∑(rp - rb)·(wp - wb)

### 多期Brinson模型

累计收益计算公式:

```
R_T = ∏(1 + rt) - 1
```

其中:
- R_p^T: 实际投资组合的T期累计收益率
- R_b^T: 基准组合的T期累计收益率
- R_AA^T: 类别配置组合的T期累计收益率
- R_SS^T: 股票选择组合的T期累计收益率

## 项目结构

```
brinson-attribution/
├── data/                       # 数据存放目录
│   ├── fund_holdings.csv      # 基金持仓数据
│   ├── benchmark_weights.csv  # 基准权重数据
│   └── sector_returns.csv     # 行业收益率数据
├── source/                     # 核心模块化代码
│   ├── data_loader.py         # 数据获取与加载
│   ├── factor.py              # Brinson模型计算
│   ├── backtest.py            # 回测逻辑
│   ├── plot.py                # 可视化
│   └── utils.py               # 工具函数
├── ipynb/                      # 复现主文件
│   └── brinson_attribution.ipynb  # 完整复现流程
├── output/                     # 输出结果
│   ├── attribution_results.csv
│   └── attribution_charts.png
└── README.md                   # 项目说明
```

## 依赖库

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
efinance>=0.4.0
akshare>=1.9.0
```

## 使用方法

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 运行Jupyter Notebook:
```bash
jupyter notebook ipynb/brinson_attribution.ipynb
```

3. 或使用Python脚本:
```bash
python source/main.py
```

## 复现要点

1. **数据准备**: 使用efinance/akshare获取基金持仓、行业分类、基准指数数据
2. **单期归因**: 按月/季度计算Brinson四象限和收益分解
3. **多期归因**: 使用几何链接法计算累计收益贡献
4. **可视化**: 绘制归因贡献柱状图、累计收益曲线

## 注意事项

- 行业分类采用申万或中信行业分类标准
- 基准指数根据基金类型选择（如沪深300、中证500等）
- 持仓数据频率通常为季度（基金季报披露）
- 多期归因时需注意权重和收益率的时间对齐

## 参考

- Brinson, G.P. and Fachler, N. (1985). Measuring non-US equity portfolio performance.
- 华泰证券研究所. (2020). Brinson模型基于持仓数据从类别配置、个券选择、交互作用进行绩效归因.
