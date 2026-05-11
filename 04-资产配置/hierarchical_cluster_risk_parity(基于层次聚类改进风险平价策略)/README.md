# 基于层次聚类改进风险平价策略

本项目复现国泰君安金融工程研报《基于层次聚类改进风险平价策略》(2024-10-31)

## 项目概述

本项目实现了层次风险平价(Hierarchical Risk Parity, HRP)策略及其变种，并与传统风险平价策略进行对比分析。

### 核心策略

| 策略名称 | 描述 |
|---------|------|
| 层次风险平价(HRP) | 基于层次聚类，通过递归二分法分配风险权重 |
| 朴素层次风险平价 | 按树杈顺序逐层均衡分配风险 |
| 基于波动率HRP | 使用波动率倒数替代方差倒数进行权重分配 |
| 传统风险平价 | 基于波动率倒数的风险平价组合 |

## 项目结构

```
hierarchical_cluster_risk_parity/
├── README.md                      # 项目说明文档
├── requirements.txt               # 依赖库列表
├── pdf_parser.py                  # PDF解析脚本
├── source/
│   ├── __init__.py               # 包初始化
│   ├── data_loader.py            # 数据获取模块
│   ├── hrp_strategy.py            # HRP核心算法
│   ├── backtest.py               # 回测引擎
│   └── performance.py             # 绩效评估
├── ipynb/
│   └── hierarchical_risk_parity_demo.ipynb  # 演示notebook
├── output/                        # 输出结果目录
└── 参考研报/
    ├── 国泰君安_基于层次聚类改进风险平价策略_2024-10-31.pdf
    └── pdf_content.txt            # PDF文本内容
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖库：
- `numpy`: 数值计算
- `pandas`: 数据处理
- `scipy`: 科学计算（层次聚类）
- `matplotlib`: 数据可视化
- `tushare`: 国内金融数据（主要数据源）
- `yfinance`: 国际金融数据

## 数据说明

本项目使用10个资产进行回测：

| 序号 | 资产名称 | 代码 | 数据源 |
|------|---------|------|--------|
| 1 | 沪深300 | 000300.SH | tushare |
| 2 | 恒生指数 | HSI.HI | tushare |
| 3 | 日经225 | N225.GI | yfinance |
| 4 | 标普500 | SPX.GI | yfinance |
| 5 | COMEX黄金 | GC.CMX | yfinance |
| 6 | ICE布油 | B.IPE | yfinance |
| 7 | SHFE铜 | CU.SHF | tushare |
| 8 | 美国国债7-10年ETF | IEF.O | yfinance |
| 9 | 中债-国债总财富(5-7年) | CBA00641.CS | tushare |
| 10 | 中债-企业债AAA | CBA04201.CS | tushare |

## 核心算法

### 层次风险平价(HRP)三步骤

1. **层次聚类**: 根据资产相关性定义距离矩阵，进行层次聚类
2. **拟对角化**: 重新排列协方差矩阵，将相关性高的资产聚集到对角线附近
3. **循环对切法**: 逐层二分分配风险权重

### 距离矩阵计算

$$D_{i,j} = \sqrt{\frac{1-\rho_{i,j}}{2}}$$

其中 $\rho_{i,j}$ 为资产 $i$ 和 $j$ 的相关系数。

### 权重分配公式

对于分割后的两个子集 $C_1$ 和 $C_2$：

$$w_1 = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}, \quad w_2 = \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}$$

## 回测参数

- **调仓频率**: 月频（每月月底）
- **交易费率**: 双边万分之五
- **协方差估计**: 过去6个月日收益率滚动估计
- **回测区间**: 2007年1月 - 2024年9月

## 使用方法

### 1. 数据加载

```python
from source import DataLoader

loader = DataLoader(start_date='20070101', end_date='20240930')
loader.load_all_data()
monthly_returns = loader.get_returns(freq='monthly')
```

### 2. 运行回测

```python
from source import Backtest, HierarchicalRiskParity, RiskParity

backtest = Backtest(
    returns_df=monthly_returns,
    transaction_cost=0.0005,
    rebalance_freq='monthly'
)
results = backtest.run_all_strategies()
```

### 3. 绩效评估

```python
from source import PerformanceEvaluator

evaluator = PerformanceEvaluator(backtest.results)
metrics = evaluator.evaluate_all()
```

### 4. 可视化

```python
# 净值曲线
backtest.plot_nav()

# 指标对比
evaluator.plot_metrics_comparison()

# 回撤对比
evaluator.plot_drawdown()
```

## 输出文件

| 文件 | 描述 |
|-----|------|
| `output/nav_comparison.png` | 策略净值曲线对比图 |
| `output/correlation_matrix.png` | 资产相关性矩阵热力图 |
| `output/dendrogram.png` | 层次聚类树状图 |
| `output/metrics_comparison.png` | 绩效指标对比图 |
| `output/drawdown_comparison.png` | 回撤对比图 |
| `output/yearly_returns_heatmap.png` | 年度收益热力图 |
| `output/backtest_results.csv` | 回测详细结果 |
| `output/performance_report.csv` | 绩效评估报告 |

## 研报关键结论

根据原研报结论：
1. 层次风险平价策略在运算速度上较传统风险平价缩短90%
2. 年化收益从4.56%提升到4.67%
3. 最大回撤从6.15%下降到4.34%

## 注意事项

1. **数据源**: 国外资产（恒生指数、日经225、标普500、黄金、布油、铜、美债ETF）使用yfinance获取，可能存在数据缺失
2. **债券数据**: 中债指数数据可能需要额外数据源支持
3. **tushare配置**: 请确保正确配置tushare token以获取国内数据

## 数据缺失说明

如遇部分资产数据获取失败，可能原因：
1. **tushare权限**: 部分指数需要更高权限
2. **yfinance连接**: 国际网络访问问题
3. **数据覆盖期**: 部分资产在2007年前可能无数据

建议补充数据源：
- `akshare`: 丰富的国内金融数据
- `baostock`: 免费A股数据
- `efinance`: 东方财富数据

## 参考资料

- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample. *Journal of Portfolio Management*, 43(1), 15-25.
- Pfitzinger, J., & Katzke, N. (2019). A Hierarchical Risk Parity Approach. *South African Journal of Economics*.

## License

MIT License
