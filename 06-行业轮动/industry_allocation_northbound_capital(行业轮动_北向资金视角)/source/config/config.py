"""
配置文件 - 项目配置参数
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

TUSHARE_TOKEN = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
TUSHARE_API_URL = "http://jiaoch.site"

BACKTEST_START_DATE = "20171201"
BACKTEST_END_DATE = "20220930"
IN_SAMPLE_END_DATE = "20210930"
OUT_OF_SAMPLE_START_DATE = "20211001"

INDUSTRY_LEVEL = "sw_l1"
BENCHMARK = "000300.SH"

INSTITUTION_TYPES = {
    "all": "全体北向机构",
    "foreign_bank": "外资银行",
    "foreign_broker": "外资券商",
    "domestic_bank": "内资银行",
    "domestic_broker": "内资券商",
    "selected": "优选机构",
}

SELECTED_INSTITUTIONS = [
    "瑞信香港",
    "摩根大通经纪",
    "高盛",
    "摩根士丹利",
    "富瑞金融",
    "法兴证券香港",
]

SENTIMENT_THRESHOLDS = {
    "large_flow_pct": 90,
    "counter_market_days": 2,
    "abnormal_flow_window": 10,
    "abnormal_flow_pct": 50,
}

FACTOR_FREQUENCIES = ["daily", "weekly", "biweekly", "monthly", "bimonthly"]
FACTOR_CONSTRUCTION_METHODS = ["raw", "yoy", "qoq"]

COMPOSITE_FACTORS = {
    "weekly": {
        "institution": "foreign_broker",
        "factors": ["position_market_value", "capital_flow", "institution_score"],
        "construction": ["yoy", "raw", "yoy"],
    },
    "biweekly": {
        "institution": "all",
        "factors": ["position_market_value", "capital_flow", "institution_score"],
        "construction": ["yoy", "raw", "yoy"],
    },
}

RISK_FREE_RATE = 0.03

PLOT_STYLE = {
    "figure.figsize": (12, 6),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
