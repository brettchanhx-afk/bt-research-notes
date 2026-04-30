# DEPI 基金绩效归因分析

## 项目概述

复现华泰金工研报《DEA模型基于净值构建DEPI指标，横向比较同类基金》（2020-08-21）

### 研报核心方法

- **DEA模型**：CCR模型（Charnes-Cooper-Rhodes），通过线性规划探究最大化产出/投入比
- **DEPI指标**：Murthi 1997年提出，衡量基金在单位成本投入下的最大收益获取能力
- **公式**：

```
DEPI_j = R_j / Σ(w_i · x_ij)

其中：
  R_j  = 基金j的超额收益（产出指标）
  x_ij = 基金j的第i个投入指标
  w_i  = 第i个投入指标的最优权重
```

### 投入指标

| 指标 | 说明 |
|------|------|
| volatility | 年化收益率标准差（风险成本） |
| fee_rate | 基金费率（管理费+托管费，年化） |
| timing_alpha | C-L模型 alpha（选股择时能力） |
| timing_beta | C-L模型 beta2 - beta1（择时能力） |

### 求解方法

CCR模型 → Charnes-Cooper变换 → 线性规划（LP）

## 目录结构

```
depi_fund_analysis/
├── config.py          # 配置文件
├── main.py            # 主程序
├── README.md
├── source/
│   ├── __init__.py
│   ├── data_loader.py # 数据获取（efinance/akshare/baostock）
│   ├── factor.py      # 因子计算
│   ├── backtest.py    # DEPI回测引擎
│   ├── plot.py        # 可视化
│   └── utils.py       # 工具函数
├── ipynb/
│   └── depi_reproduction.ipynb  # Jupyter复现
├── data/               # 数据输出
└── output/             # 图表输出
```

## 使用方法

```bash
cd depi_fund_analysis
pip install efinance akshare baostock pandas numpy matplotlib seaborn scipy
python main.py
```

## 数据来源

- 基金净值：efinance
- 基金费率：akshare
- 基准指数：baostock（沪深300）
