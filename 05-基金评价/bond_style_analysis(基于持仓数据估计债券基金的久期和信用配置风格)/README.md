# 债券基金风格分析

> 基于华泰证券研报《基于持仓数据估计债券基金的久期和信用配置风格》(2020-08-21) 的 Python 完整复现项目

---

## 项目结构

```
bond_style_analysis/
├── data/                           # 数据缓存目录
├── ipynb/
│   └── bond_style_analysis.ipynb   # Jupyter 复现主文件
├── output/                          # 分析结果输出目录
├── source/                          # 核心模块
│   ├── __init__.py
│   ├── data_loader.py               # 数据获取（efinance → akshare）
│   ├── factor.py                    # 久期/信用风格因子计算
│   ├── backtest.py                  # 回测引擎、绩效指标
│   ├── plot.py                      # 可视化（风格箱、净值曲线等）
│   └── utils.py                     # 工具函数、导出
├── main.py                          # 命令行主程序
└── README.md
```

---

## 核心算法（研报定义）

### 1. 加权平均久期（Duration Style）

$$\bar{D} = \frac{\sum_{i=1}^{n} W_i \cdot D_i}{\sum_{i=1}^{n} W_i}$$

其中 $W_i$ 为债券 i 的持仓市值权重，$D_i$ 为修正久期

### 2. 加权平均信用评分（Credit Style）

$$\bar{C} = \frac{\sum_{i=1}^{n} W_i \cdot C_i}{\sum_{i=1}^{n} W_i}$$

其中 $C_i$ 为债券 i 的信用评分

### 3. 信用评级评分表（来源：中国人民银行 2006 年规范）

| 评级 | 评分 | 评级 | 评分 | 评级 | 评分 |
|------|------|------|------|------|------|
| AAA+ | 17.0 | BBB+ | 11.5 | BB+ | 9.5 |
| AAA  | 16.5 | BBB  | 11.0 | BB  | 9.0 |
| AA+  | 15.5 | BBB- | 10.5 | BB- | 8.5 |
| AA   | 15.0 | A+   | 13.5 | B+  | 7.5 |
| AA-  | 14.5 | A    | 13.0 | B   | 7.0 |
|      |      | A-   | 12.5 | ... | ... |

### 4. 久期风格分类

| 标签 | 范围 | 说明 |
|------|------|------|
| short（短期） | D < 3.5 年 | 利率风险低 |
| mid（中期） | 3.5 <= D < 6.0 年 | 利率风险中 |
| long（长期） | D >= 6.0 年 | 利率风险高 |

### 5. 信用风格分类

| 标签 | 范围 | 说明 |
|------|------|------|
| high（高等级） | C >= 14.0 | AAA ~ AA |
| medium（中等级） | 11.0 <= C < 14.0 | A ~ BBB |
| low（低等级） | C < 11.0 | BB 及以下 |

---

## 快速开始

### 安装依赖

```bash
pip install efinance akshare pandas numpy matplotlib seaborn openpyxl
```

### 命令行运行

```bash
# 演示模式（使用示例数据）
python main.py 000012

# 真实API模式（需网络连接）
python main.py 000012 --real

# 分析多期
python main.py 000084 --real --n-periods 4
```

### Jupyter Notebook 运行

```bash
jupyter notebook ipynb/bond_style_analysis.ipynb
```

---

## 数据源说明

| 数据类型 | 第一优先级 | 第二优先级 | 第三优先级 |
|----------|-----------|-----------|-----------|
| 基金净值 | efinance | akshare | baostock |
| 基金持仓 | akshare | efinance | - |
| 债券信息 | akshare | bondpy | - |
| 评级数据 | akshare | - | - |

> **注意**：部分债券久期和信用评级数据需要专业数据接口（如同花顺 iFind、Wind），本项目使用 AKShare 接口获取，数据缺失时使用估算值。

---

## 示例债券基金

| 基金代码 | 基金名称 |
|----------|----------|
| 000012 | 华夏债券 A |
| 000084 | 博时裕祥 A |
| 000355 | 景顺长城优选 |
| 001001 | 华夏希望债券 A |
| 020003 | 国泰金龙债券 A |

---

## 输出文件

运行后在 `output/{fund_code}/` 目录生成：

- `{fund_code}_style_summary_YYYYMMDD.csv` - 风格分析结果
- `{fund_code}_style_box.png` - 风格箱可视化
- `{fund_code}_holdings_pie.png` - 持仓结构图
- `{fund_code}_credit_dist.png` - 信用评级分布
- `{fund_code}_style_evolution.png` - 风格演变图
- `{fund_code}_report_YYYYMMDD.txt` - 文本分析报告

---

## 研报原文摘要

> **方法**：通过基金定期报告披露的重仓债券列表，加权平均各债券的久期和信用评分，估计基金的整体久期风格和信用风格。
>
> **优势**：解决了债券基金不披露完整持仓的问题，只需季报中的前 5-10 大重仓债券即可。
>
> **局限**：季报存在时滞（约 45 天）；部分债券（如私募债）缺乏评级数据。

---

## 依赖库

```
pandas       >= 1.5
numpy        >= 1.24
matplotlib   >= 3.7
seaborn      >= 0.12
efinance     >= 0.85
akshare      >= 1.12
openpyxl     >= 3.1
```

---

*作者：QClaw Agent | 基于华泰证券 2020 年金工研报*
