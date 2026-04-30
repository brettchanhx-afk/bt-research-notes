# -*- coding: utf-8 -*-
"""
全局配置文件
"""

# ── 基金配置 ──────────────────────────────────────────────
FUND_CODE = "008404"          # 华泰紫金泰盈混合A
FUND_NAME = "华泰紫金泰盈混合A"

# 支持批量分析多只基金
FUND_LIST = [
    {"code": "000628", "name": "大成高鑫股票A"},
    # {"code": "110011", "name": "易方达中小盘"},
    # {"code": "161725", "name": "招商中证白酒"},
]

# ── 时间范围 ──────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2026-04-25"

# ── Barra 因子指数代码 (baostock) ─────────────────────────
FACTOR_INDEXES = {
    "market":   "sh.000300",   # 沪深300  → 市场因子
    "size":     "sh.000852",   # 中证1000 → 规模因子 (小盘代理)
    "value":    "sh.000919",   # 沪深300价值
    "growth":   "sh.000918",   # 沪深300成长
    "momentum": "sh.000862",   # 中证红利  → 动量代理
}

# ── 滚动窗口 ──────────────────────────────────────────────
ROLLING_WINDOW = 60           # 滚动回归窗口（交易日）

# ── 路径 ─────────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── 图表样式 ──────────────────────────────────────────────
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
FONT_FAMILY = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
DPI         = 150
