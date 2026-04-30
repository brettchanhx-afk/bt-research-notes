# -*- coding: utf-8 -*-
"""
配置文件 - 基金选股择时能力定量评价模型
T-M 模型、H-M 模型、C-L 模型
华泰金工研究 | 2020-08-21

依赖库: pandas, numpy, matplotlib, seaborn, efinance, akshare, baostock, statsmodels
"""

import os

# ==================== 数据源配置 ====================
# 数据源优先级（按用户指定顺序）
DATA_SOURCES = [
    'efinance',    # 基金净值、持仓数据
    'akshare',     # A股、期货、基金
    'baostock',    # A股历史行情
    'mootdx',      # 通达信数据
    'yfinance',    # 美股
    'bondpy',      # 债券数据
    'fundata',     # 基金数据
]

# ==================== 分析区间 ====================
# 默认分析区间
START_DATE = '2018-01-01'   # 开始日期
END_DATE   = '2026-04-28'   # 截止日期

# ==================== 基准配置 ====================
BENCHMARK = '000300.SH'     # 沪深300作为市场基准（沪深300指数代码）

# ==================== 无风险利率 ====================
# 1年期存款基准利率（年化），日频数据需除以252
# 若需要日内无风险利率 = RISK_FREE_RATE / 252
RISK_FREE_RATE = 0.015     # 1.5% 年化（存款基准利率）

# ==================== 回归配置 ====================
OLS_REGRESS_KWGS = {
    'missing': 'drop',     # 缺失值处理：删除
}

# ==================== 输出配置 ====================
# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

# ==================== 可视化配置 ====================
PLT_STYLE     = 'seaborn-v0_8-whitegrid'
CHINESE_FONT  = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
FIGURE_DPI    = 120

# ==================== 模型参数 ====================
# 调仓频率（单位：交易日）
# 若月度调仓，设置为约21个交易日
REBALANCE_FREQ = 21        # 默认月度调仓

# 回归显著性水平
SIGNIFICANCE_LEVEL = 0.05
