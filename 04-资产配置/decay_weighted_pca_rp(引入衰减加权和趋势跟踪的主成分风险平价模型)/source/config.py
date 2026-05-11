"""
Configuration module for the decay-weighted PCA risk parity project.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_API_URL = "http://jiaoch.site"

RISK_FREE_RATE = 0.03

ASSET_CLASSES = {
    "sh000001": {"name": "上证指数", "type": "index"},
    "sh000300": {"name": "沪深300", "type": "index"},
    "sh000016": {"name": "上证50", "type": "index"},
    "sh000010": {"name": "上证180", "type": "index"},
    "sh000009": {"name": "上证380", "type": "index"},
    "sh000905": {"name": "中证500", "type": "index"},
    "sz399001": {"name": "深证成指", "type": "index"},
    "sz399006": {"name": "创业板指", "type": "index"},
    "sh000012": {"name": "上证国债", "type": "bond"},
    "sh000013": {"name": "上证企债", "type": "bond"},
    "sh000300": {"name": "沪深300", "type": "index"},
    "CL00Y": {"name": "WTI原油", "type": "commodity"},
    "GC00Y": {"name": "黄金", "type": "commodity"},
    "HG00Y": {"name": "铜", "type": "commodity"},
    "NG00Y": {"name": "天然气", "type": "commodity"},
    "AL00Y": {"name": "铝", "type": "commodity"},
}

DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20171117"

HALF_LIFE_DAYS = {
    "volatility": 30,
    "correlation": 60,
}

TREND_LOOKBACK_PERIOD = 20

DECAY_WEIGHTING_PARAMS = {
    "volatility_half_life": 30,
    "correlation_half_life": 60,
}

TF_PARAMS = {
    "short_ma": 5,
    "long_ma": 20,
    "lookback_period": 20,
}