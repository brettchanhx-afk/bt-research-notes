# -*- coding: utf-8 -*-
"""
DEPI 基金绩效归因分析 - 配置文件
复现研报：《DEA模型基于净值构建DEPI指标，横向比较同类基金》
来源：华泰金工，2020-08-21
"""
import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent
DATA_DIR       = WORKSPACE_ROOT / 'data'
OUTPUT_DIR     = WORKSPACE_ROOT / 'output'
NOTEBOOK_DIR   = WORKSPACE_ROOT / 'ipynb'
for _d in [DATA_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---- 回测区间 ----
BACKTEST_START = '2019-01-01'
BACKTEST_END   = '2024-12-31'

# ---- 无风险利率（年化，3% 国内5年期国债均值）----
RISK_FREE_RATE = 0.03

# ---- 调仓频率 ----
# Q = 每季度末，M = 每月末
REBALANCE_FREQ = 'Q'

# ---- DEA/DEPI 投入指标（与研报一致）----
# 1. volatility   年化收益率标准差（风险成本）
# 2. fee_rate     基金费率（管理费+托管费，年化）
# 3. timing_alpha C-L模型 alpha（选股择时能力）
# 4. timing_beta  C-L模型 beta2 - beta1（择时能力）
INPUT_INDICATORS = ['volatility', 'fee_rate', 'timing_alpha', 'timing_beta']

# ---- 基金池 ----
FUND_TYPE            = '股票型'
MIN_ESTABLISH_MONTHS = 12   # 剔除成立不足1年的新基金
MIN_POOL_SIZE        = 10   # 同类对比最少样本数
SAMPLE_SIZE          = 50   # 每次横评抽样数

# ---- 基准指数 ----
BENCHMARK_CODE = '000300'
BENCHMARK_NAME = '沪深300'

# ---- 可视化 ----
PLOT_DPI     = 120
PLOT_STYLE   = 'seaborn-v0_8-whitegrid'
CHINESE_FONT = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']

# ---- 数据源优先级（用户指定）----
DATA_SOURCE_PRIORITY = ['efinance', 'akshare', 'baostock']
