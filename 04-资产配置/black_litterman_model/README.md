# Black-Litterman 模型 - 大类资产配置量化实现

> **文献来源**: 国泰君安证券研究 | 大类资产配置量化模型研究系列之二  
> **报告标题**: 《手把手教你实现 Black-Litterman 模型》  
> **核心作者**: 廖静池、张雪杰 | 2023.04.05  
> **代码实现**: 本项目

---

## 📁 项目结构

```
black_litterman_model/
├── config.py              # 全局配置 (数据路径、策略参数、资产列表)
├── main.py                # 一键运行脚本
├── README.md              # 本文件
│
├── source/
│   ├── __init__.py        # 包初始化
│   ├── data_loader.py     # 数据获取模块 (tushare / akshare / yfinance)
│   ├── bl_model.py        # BL核心模型 (先验 → 观点 → 后验 → MVO)
│   ├── optimizer.py       # 带约束组合优化器 (cvxopt)
│   ├── backtest.py        # 回测引擎 (月频调仓、多策略对比)
│   └── plot.py            # 可视化模块 (7张图)
│
├── data/                  # 原始数据缓存 (CSV)
├── output/                # 回测结果输出 (CSV + 图片)
│
└── ipynb/
    └── main.ipynb         # Jupyter Notebook 交互式复现
```

---

## 🎯 研报核心结论

| 指标 | BL策略1 | BL策略2 | MVO基准 | 固定权重 |
|------|--------|--------|--------|---------|
| 年化收益 | ~6.58% | ~6.59% | ~5.82% | ~5.41% |
| 最大回撤 | ~3.13% | ~2.96% | ~3.86% | ~3.55% |
| 夏普比率 | ~2.15 | ~2.31 | ~1.93 | ~1.36 |
| 收益回撤比 | ~2.10 | ~2.23 | ~1.51 | ~1.52 |

**结论**: BL模型整体优于MVO和固定权重策略，尤其在2020年、2022年市场大幅波动时回撤控制显著更好。

---

## 🔬 模型原理 (四步法)

### Step 1: 先验分布 (CAPM逆向优化)
$$\Pi = \lambda \Sigma w_{market}$$

通过风险厌恶系数、协方差矩阵和市场均衡权重逆向求解均衡收益。

### Step 2: 主观观点 (P, Q, Ω)
观点矩阵 P (k×n)、观点收益 Q (k×1)、信心水平矩阵 Ω (k×k)

本项目使用**一个月动量**作为主观观点: $Q = R_{t-1}$ (过去一个月收益率)

### Step 3: 后验分布 (贝叶斯)
$$\hat{\Pi} = \Pi + \tau\Sigma P^T[P\tau\Sigma P^T+\Omega]^{-1}[Q - P\Pi]$$

$$\hat{\Sigma}_r = \Sigma + \hat{\Sigma}_\pi$$

### Step 4: 均值-方差优化
$$\max_w \quad \lambda w^T\hat{\Pi} - \frac{1}{2}w^T\hat{\Sigma}_r w$$

---

## 📊 六大类资产

| 代码 | 名称 | 类别 | 数据来源 |
|------|------|------|---------|
| CSI300 | 沪深300 | 股票 | tushare |
| SP500 | 标普500 | 股票 | yfinance |
| HSI | 恒生指数 | 股票 | yfinance |
| CR_GOV | 中债国债总财富 | 债券 | akshare / tushare |
| CR_CORP | 中债企业债总财富 | 债券 | akshare |
| NHCI | 南华商品指数 | 商品 | akshare |

---

## ⚙️ 使用说明

### 环境依赖

```bash
pip install numpy pandas matplotlib seaborn scipy
pip install cvxopt          # 凸优化求解器
pip install akshare yfinance tushare
pip install tqdm
```

### 一键运行

```bash
cd black_litterman_model
python main.py
```

### Notebook 交互式复现

```bash
jupyter notebook ipynb/main.ipynb
```

---

## ⚠️ 已知数据限制

1. **中债指数数据**: akshare/tushare 债券指数接口不稳定，若获取失败，回测将跳过债券类资产或使用 `NHCI` 商品指数替代
2. **恒生指数**: yfinance 可能返回延迟数据，建议确认 `^HSI` 的数据完整性
3. **南华商品指数**: akshare 的 `futures_nh_commodity_index` 接口偶有缺失

**处理方案**: 已在 `data_loader.py` 中实现多重 fallback 链 (tushare → akshare → yfinance)，若全部失败会打印 ERROR 并跳过该资产，最终用 `force_reload=True` 强制重新拉取。

---

## 📈 图表清单

1. `fig1_cumulative_returns.png` - 累计收益曲线对比 (对标研报图4)
2. `fig2_drawdown.png` - 回撤对比 (对标研报图5)
3. `fig3_weights_{strat}.png` - 权重堆叠面积图 (对标研报图6/7)
4. `fig4_stats_bar.png` - 绩效指标柱状图
5. `fig5_yearly_heatmap.png` - 年度收益热力图
6. `fig6_prior_posterior.png` - 先验/后验收益对比
7. `fig7_correlation_matrix.png` - 资产相关性热力图

---

## 📚 参考文献

1. Black, F., & Litterman, R. (1992). Global portfolio optimization. *FAJ*, 48(5), 28-43.
2. Idzorek, T. (2007). A step-by-step guide to the Black-Litterman model.
3. Walters, J. (2009). The Black-Litterman model in detail.
4. Meucci, A. (2010). The black-litterman approach: Original model and extensions.
5. 杨朝军等 (2021). 资产配置理论与实证前沿问题研究. 经济管理出版社.
