# Barra 公募基金风格分析

基于 **efinance** + **baostock** 的 Barra 风格分析框架，支持基金净值获取、风格因子构建、回归分析与可视化。

---

## 项目结构

```
barra-fund-analysis/
├── barra/                    # 核心模块包
│   ├── __init__.py
│   ├── data.py              # 数据加载 (efinance + baostock)
│   ├── factors.py           # 因子构建
│   ├── regression.py        # 回归分析
│   └── visualization.py     # 可视化
├── notebooks/               # Jupyter Notebook 教程
│   ├── 01_data_explore.ipynb
│   ├── 02_factor_analysis.ipynb
│   └── 03_fund_report.ipynb
├── data/                    # 数据缓存目录
├── output/                  # 报告输出目录
├── config.py                # 全局配置
├── main.py                  # 命令行入口
├── requirements.txt         # 依赖
└── README.md                # 本文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 命令行一键分析

```bash
# 分析默认基金 (config.py 中配置)
python main.py

# 分析指定基金
python main.py --fund 000628

# 强制重新下载数据
python main.py --force
```

### 3. Jupyter Notebook 交互分析

```bash
jupyter notebook notebooks/
```

按顺序运行：
1. `01_data_explore.ipynb` — 数据探索
2. `02_factor_analysis.ipynb` — 因子分析
3. `03_fund_report.ipynb` — 生成报告

---

## Barra 因子定义

| 因子 | 定义 | 代理指数 |
|------|------|---------|
| **market** | 市场因子 (Beta) | 沪深300 |
| **size** | 规模因子 | 中证1000 − 沪深300 |
| **value** | 价值因子 | 沪深300价值 − 沪深300成长 |
| **momentum** | 动量因子 | 中证红利 |

---

## 配置说明

编辑 `config.py`：

```python
FUND_CODE = "000628"          # 基金代码
FUND_NAME = "大成高鑫股票A"    # 基金名称

START_DATE = "2020-01-01"     # 分析起始日期
END_DATE   = "2026-04-25"     # 分析结束日期

ROLLING_WINDOW = 60           # 滚动窗口 (交易日)
```

---

## 输出文件

运行后生成：

```
output/
├── report_000628_20250425_230817.png   # 可视化报告图
├── report_000628.txt                    # 文本报告
├── exposures_000628.csv                 # 风格暴露度
└── rolling_exposures_000628.csv         # 滚动暴露时序
```

---

## 示例结果

**大成高鑫股票A (000628)** 分析结果：

| 指标 | 数值 |
|------|------|
| 累计收益 | +162.18% |
| 年化收益 | 17.18% |
| 夏普比率 | 1.08 |
| 最大回撤 | -25.52% |

**风格暴露度：**
- 市场暴露: +0.69 (中等Beta)
- 规模暴露: +0.10 (略偏小盘)
- 价值暴露: +0.01 (价值成长均衡)
- 动量暴露: 0.00 (无明显动量)

**模型质量：** R² = 0.68 (强解释力度)

---

## 扩展开发

### 添加新因子

编辑 `barra/factors.py`：

```python
# 在 build() 方法中添加
factors['new_factor'] = self._get_col('new_factor')
```

### 批量分析多只基金

编辑 `config.py`：

```python
FUND_LIST = [
    {"code": "000628", "name": "大成高鑫股票A"},
    {"code": "110011", "name": "易方达中小盘"},
]
```

然后运行：
```bash
python main.py --list
```

---

## 依赖库

- **efinance** — 基金净值数据
- **baostock** — 指数行情数据
- **pandas / numpy** — 数据处理
- **statsmodels** — 回归分析
- **matplotlib / seaborn** — 可视化

---

## License

MIT
