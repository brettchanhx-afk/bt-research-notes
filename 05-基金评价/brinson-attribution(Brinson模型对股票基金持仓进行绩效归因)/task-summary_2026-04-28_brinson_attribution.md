# 任务记录: Brinson模型绩效归因复现项目

## 时间
2026-04-28 16:30

## 任务
按照标准Python量化项目结构，复现华泰证券《Brinson模型基于持仓数据从类别配置、个券选择、交互作用进行绩效归因》研报

## 研报核心内容

### Brinson模型简介
Brinson模型由Brinson和Fachler于1985年提出，是基金绩效归因的经典模型。将基金超额收益分解为三部分：

1. **类别配置收益 (Allocation Effect)**: 基金经理在行业/资产类别配置上的能力
2. **个券选择收益 (Selection Effect)**: 基金经理在个股选择上的能力
3. **交互作用收益 (Interaction Effect)**: 类别配置与个券选择的联合作用

### 核心公式

**四象限矩阵:**

|  | 实际组合资产类别j收益 | 基准组合资产类别j收益 |
|--|---------------------|---------------------|
| **实际组合资产类别j权重** | Q4 = ∑wp·rp | Q2 = ∑wp·rb |
| **基准组合资产类别j权重** | Q3 = ∑wb·rp | Q1 = ∑wb·rb |

**收益分解:**
- 总超额收益: R = Q4 - Q1
- 类别配置收益: R_AA = Q2 - Q1 = ∑(wp - wb)·rb
- 个券选择收益: R_SS = Q3 - Q1 = ∑(rp - rb)·wb
- 交互作用收益: R_I = R - R_AA - R_SS = ∑(rp - rb)·(wp - wb)

**多期Brinson模型:**
- 累计收益: R_T = ∏(1 + rt) - 1

## 项目结构

```
brinson-attribution/
├── README.md                   # 项目说明文档
├── source/                     # 核心模块化代码
│   ├── data_loader.py         # 数据获取与加载 (efinance/akshare)
│   ├── factor.py              # Brinson模型核心计算
│   ├── backtest.py            # 回测逻辑与滚动归因
│   ├── plot.py                # 可视化绘图
│   └── utils.py               # 工具函数
├── ipynb/                      # 复现主文件
│   └── brinson_attribution.ipynb  # 完整复现流程
└── output/                     # 输出结果目录
```

## 核心模块功能

### 1. data_loader.py
- `FundDataLoader` 类: 基金数据加载器
  - `get_fund_holdings()`: 获取基金持仓数据
  - `get_sector_index_returns()`: 获取申万行业指数收益
  - `get_benchmark_returns()`: 获取基准指数收益
  - `aggregate_holdings_by_sector()`: 按行业聚合持仓
- `DataProcessor` 类: 数据对齐与归一化

### 2. factor.py
- `BrinsonAttribution` 数据类: 归因结果存储
- `SinglePeriodBrinson` 类: 单期Brinson模型
  - `calculate_four_quadrants()`: 计算四象限
  - `calculate_attribution()`: 计算归因（支持三因素/两因素分解）
  - `calculate_sector_contribution()`: 计算各行业贡献
- `MultiPeriodBrinson` 类: 多期Brinson模型
  - `geometric_link()`: 几何链接计算累计收益
  - `calculate_multi_period_attribution()`: 多期累计归因
- `BrinsonAttributionAnalyzer` 类: 归因分析器整合

### 3. backtest.py
- `BacktestConfig` 数据类: 回测配置
- `BrinsonBacktest` 类: 回测引擎
  - `run_backtest()`: 运行回测
  - `get_multi_period_attribution()`: 获取多期归因
  - `calculate_performance_metrics()`: 计算绩效指标
  - `analyze_attribution_stability()`: 归因稳定性分析
- `BrinsonAnalysisReport` 类: 报告生成器

### 4. plot.py
- `BrinsonVisualizer` 类: 可视化器
  - `plot_attribution_waterfall()`: 归因瀑布图
  - `plot_sector_contribution()`: 行业贡献图
  - `plot_time_series_attribution()`: 时间序列图
  - `plot_attribution_summary()`: 归因摘要图
  - `plot_heatmap()`: 归因热力图
- `create_full_report()`: 生成完整图表报告

### 5. utils.py
- 数据验证、格式转换、回撤计算等工具函数
- `print_attribution_summary()`: 打印归因摘要

## 数据源

按优先级使用以下开源库：
1. **efinance**: 基金净值、持仓数据
2. **akshare**: A股、指数、行业数据
3. **模拟数据**: 当数据源不可用时自动生成测试数据

## 复现要点

1. **单期归因**: 完整实现四象限矩阵和收益分解
2. **多期归因**: 使用几何链接法计算累计收益
3. **可视化**: 瀑布图、行业贡献图、时间序列图
4. **工程化**: 模块化设计、类型注解、完整文档

## 产出文件

| 文件 | 路径 |
|------|------|
| README.md | `C:\Users\chenh\.qclaw\workspace\brinson-attribution\README.md` |
| data_loader.py | `C:\Users\chenh\.qclaw\workspace\brinson-attribution\source\data_loader.py` |
| factor.py | `C:\Users\chenh\.qclaw\workspace\brinson-attribution\source\factor.py` |
| backtest.py | `C:\Users\chenh\.qclaw\workspace\brinson-attribution\source\backtest.py` |
| plot.py | `C:\Users\chenh\.qclaw\workspace\brinson-attribution\source\plot.py` |
| utils.py | `C:\Users\chenh\.qclaw\workspace\brinson-attribution\source\utils.py` |
| brinson_attribution.ipynb | `C:\Users\chenh\.qclaw\workspace\brinson-attribution\ipynb\brinson_attribution.ipynb` |

## 使用方法

```bash
# 进入项目目录
cd brinson-attribution

# 安装依赖
pip install pandas numpy matplotlib seaborn efinance akshare

# 运行Jupyter Notebook
jupyter notebook ipynb/brinson_attribution.ipynb
```

## 参考

- Brinson, G.P. and Fachler, N. (1985). Measuring non-US equity portfolio performance.
- 华泰证券研究所. (2020). Brinson模型基于持仓数据从类别配置、个券选择、交互作用进行绩效归因.
