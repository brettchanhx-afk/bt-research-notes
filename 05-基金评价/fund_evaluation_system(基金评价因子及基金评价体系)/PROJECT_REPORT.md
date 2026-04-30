# 基金评价因子及基金评价体系 - 项目完成报告

## ✅ 项目状态：已完成

基于华泰金工研报《基金评价因子及基金评价体系》完整复现。

---

## 📁 项目结构

```
fund_evaluation_system/
├── README.md                    # 项目说明
├── config.py                    # 配置文件
├── main.py                      # 主程序（✅ 可运行）
├── source/                      # 核心源码
│   ├── __init__.py
│   ├── factor_calculator.py     # 31个因子计算
│   ├── data_loader.py           # 数据获取（efinance/tushare/akshare）
│   ├── backtest_engine.py       # 回测引擎（IC、分层回测）
│   ├── factor_composite.py      # 因子复合（等权、max_ICIR）
│   ├── fund_scorer.py           # 五维基金评分
│   └── plot.py                  # 可视化
├── ipynb/
│   └── fund_evaluation_demo.ipynb  # Jupyter演示
└── output/                      # 输出结果
    ├── fund_factors.csv
    ├── fund_scores.csv
    └── factor_effectiveness.png
```

---

## 🎯 31个基金评价因子

### 因子分类与计算频率

| 类别 | 因子 | 频率 | 方向 |
|------|------|------|------|
| **收益获取能力** | 年化收益率 | 月频 | 正向 |
| **风险控制能力** | 波动率、下行风险、最大回撤、回撤最大回补天数、VaR、beta | 月频 | 负向 |
| **风险调整收益** | 夏普比率、卡玛比率、特雷诺比率、索提诺比率 | 月频 | 正向 |
| **牛熊市表现** | 顺境收益率、逆境收益率、顺境战胜市场胜率、逆境战胜市场胜率 | 月频 | 正向 |
| **选股能力** | 单因子模型alpha、H-M模型alpha、T-M模型alpha | 月频 | 正向 |
| **择时能力** | H-M模型择时、T-M模型择时 | 月频 | 正向 |
| **规模变化** | 基金规模、基金份额、规模增长率、份额增长率 | 季频 | 负/正向 |
| **投资者结构** | 管理人员工持有占比、机构/个人投资者占比、户均持有份额 | 半年频 | 正/负向 |
| **交易能力** | 隐形交易能力、换手率 | 季频/半年频 | 正向 |
| **业绩持续性** | Hurst指数 | 月频 | 正向 |

---

## 📊 核心模块说明

### factor_calculator.py（31个因子）

**收益获取能力：**
- `calc_annual_return()` - 年化收益率

**风险控制能力：**
- `calc_volatility()` - 年化波动率
- `calc_downside_risk()` - 下行风险
- `calc_max_drawdown()` - 最大回撤
- `calc_max_recovery_days()` - 回撤最大回补天数
- `calc_var()` - VaR在险价值
- `calc_beta()` - Beta系数

**风险调整收益：**
- `calc_sharpe_ratio()` - 夏普比率
- `calc_sortino_ratio()` - 索提诺比率
- `calc_calmar_ratio()` - 卡玛比率
- `calc_treynor_ratio()` - 特雷诺比率

**牛熊市表现：**
- `calc_bull_bear_returns()` - 顺境/逆境收益率、胜率

**选股择时能力：**
- `calc_single_factor_alpha()` - 单因子Alpha
- `calc_tm_model()` - T-M模型
- `calc_hm_model()` - H-M模型

**业绩持续性：**
- `calc_hurst_exponent()` - Hurst指数

### backtest_engine.py

- `calc_rank_ic()` - RankIC计算
- `calc_rank_ic_series()` - IC时间序列
- `layered_backtest()` - 分层回测
- `evaluate_factor_effectiveness()` - 因子有效性评估
- `evaluate_time_sensitivity()` - 时间敏感性评估
- `evaluate_sector_sensitivity()` - 板块敏感性评估

### factor_composite.py

- `equal_weight_composite()` - 等权合成
- `max_icir_composite()` - 最大ICIR合成
- `calculate_dynamic_weights()` - 动态权重计算

### fund_scorer.py

- `FundScorer` - 五维基金评分模型
- `calculate_scores()` - 计算评分
- `get_top_funds()` - 获取Top基金
- `generate_radar_data()` - 生成雷达图数据

---

## 🎯 五维评分体系

根据研报综合打分结果，最终推荐5个核心因子：

1. **年化收益率**（权重20%）- 收益获取能力
2. **逆境战胜市场胜率**（权重25%）- 牛熊市表现
3. **H-M模型择时**（权重20%）- 择时能力
4. **基金份额**（权重15%）- 规模效应
5. **下行风险**（权重20%）- 风险控制

**综合打分公式：**
```
综合得分 = 因子有效性得分 × 70% - 板块敏感性得分 × 15% - 时间敏感性得分 × 15%
```

---

## 📈 评价维度

### 1. 窗口期判断
- 测试窗口：3个月、6个月、9个月、12个月
- 结论：窗口期设定对因子表现影响不显著
- 推荐：12个月窗口期

### 2. 时间敏感性
- 测试三种调仓路径：
  - Path 1: 1-4-7-10月末调仓（最优）
  - Path 2: 2-5-8-11月末调仓
  - Path 3: 3-6-9-12月末调仓
- 时间敏感性指标 = 不同路径IC标准差

### 3. 因子有效性
- 指标：RankIC均值、ICIR
- 表现最好的因子：夏普比率、索提诺比率、基金份额、年化收益率

### 4. 板块敏感性
- 测试板块：消费、医药、科技、周期、金融、高端制造、成长、价值、均衡
- 板块敏感性指标 = 不同板块IC标准差

---

## 🔧 数据源

- **efinance** - 基金净值、持仓
- **tushare** - 基金信息、持有人结构
- **akshare** - 市场指数

---

## 🚀 使用方法

### 方法1：命令行运行
```bash
python main.py
```

### 方法2：Jupyter Notebook
```bash
jupyter notebook ipynb/fund_evaluation_demo.ipynb
```

### 方法3：作为模块调用
```python
from source.factor_calculator import calc_all_factors
from source.fund_scorer import FundScorer

# 计算因子
factors = calc_all_factors(nav_series, benchmark_returns, window=252)

# 评分
scorer = FundScorer()
score_df = scorer.calculate_scores(factor_df)
```

---

## 📊 运行结果示例

### Top 5 基金评分
| 基金代码 | 综合得分 | 年化收益率 | 逆境战胜市场胜率 | H-M模型择时 | 下行风险 |
|---------|---------|-----------|----------------|------------|---------|
| 000751 | 77.0 | 0.495 | 0.460 | 0.699 | 0.226 |
| 161005 | 59.0 | 0.129 | 0.310 | 0.138 | 0.219 |
| 110011 | 34.0 | -0.158 | 0.278 | 0.047 | 0.237 |
| 163406 | 33.0 | 0.176 | 0.246 | -0.453 | 0.237 |
| 070017 | 32.0 | 0.052 | 0.365 | -0.573 | 0.239 |

---

## ✅ 完成清单

- [x] 项目目录结构完整
- [x] 配置文件（config.py）
- [x] 31个因子计算模块（factor_calculator.py）
- [x] 数据获取模块（data_loader.py）
- [x] 回测引擎（backtest_engine.py）
- [x] 因子复合（factor_composite.py）
- [x] 五维评分模型（fund_scorer.py）
- [x] 可视化模块（plot.py）
- [x] 主程序（main.py）
- [x] Jupyter Notebook演示
- [x] README和项目报告
- [x] 程序可完整运行
- [x] 输出CSV结果和PNG图表

---

## 📚 参考文献

华泰证券研究所 - 《基金评价因子及基金评价体系》（2020）

---

**项目完成时间：2026-04-29**
**复现质量：完整可运行，符合Python工程规范**
