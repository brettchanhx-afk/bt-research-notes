"""
Configuration module for macro factor risk parity project.
"""
import os

PROJECT_ROOT = r"d:\Documents\trae_projects\macro_factor_risk_parity_minimization"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SOURCE_DIR = os.path.join(PROJECT_ROOT, "source")
REPORT_DIR = os.path.join(PROJECT_ROOT, "参考研报")

BACKTEST_START = "2013-01-01"
BACKTEST_END = "2024-04-30"
REBALANCE_FREQ = "monthly"

RISK_FREE_RATE = 0.0

ASSET_COLS_RAW = {
    "沪深300": "000300.SH",
    "南华商品": "NHCI.SL",
    "中证500": "000905.SH",
    "中债国债": "CBA00601.CB",
    "中债企业债": "CBA02001.CB",
    "布伦特原油": "BRN0Y.ICE",
}

ASSET_NAME_MAP = {
    "000300.SH": "沪深300",
    "NHCI.SL": "南华商品",
    "000905.SH": "中证500",
    "CBA00601.CB": "中债国债",
    "CBA02001.CB": "中债企业债",
    "BRN0Y.ICE": "布伦特原油",
}

MACRO_FACTOR_COLS = {
    "增长因子": {
        "name_cn": "增长因子",
        "mimicking": {
            "CRB现货指数:工业": {"weight": 0.61, "direction": 1},
            "南华铜指数": {"weight": 0.24, "direction": 1},
            "申万行业指数:房地产开发": {"weight": 0.15, "direction": 1},
        },
    },
    "通胀因子": {
        "name_cn": "通胀因子",
        "mimicking": {
            "市场价:生猪(外三元):全国均价": {"weight": 0.35, "direction": 1},
            "期货收盘价(连续):布伦特原油:ICE": {"weight": 0.22, "direction": 1},
            "现货价:螺纹钢": {"weight": 0.43, "direction": 1},
        },
    },
    "利率因子": {
        "name_cn": "利率因子",
        "mimicking": {
            "中债国债总指数(总值)净价指数": {"weight": 1.0, "direction": -1},
        },
    },
    "信用因子": {
        "name_cn": "信用因子",
        "mimicking": {
            "中债信用债总指数(3-5年)财富指数": {"weight": 1.0, "direction": 1},
            "中债3-5年期国债指数(总值)财富指数": {"weight": 1.0, "direction": -1},
        },
    },
    "汇率因子": {
        "name_cn": "汇率因子",
        "mimicking": {
            "美国:美元指数": {"weight": 1.0, "direction": 1},
        },
    },
    "流动性因子": {
        "name_cn": "流动性因子",
        "mimicking": {
            "市盈率(成份股计算):申万大盘指数": {"weight": 1.0, "direction": 1},
            "市盈率(成份股计算):申万小盘指数": {"weight": 1.0, "direction": -1},
        },
    },
}

FACTOR_NAMES = list(MACRO_FACTOR_COLS.keys())

FACTOR_COLS_IN_HIGHFREQ = {
    "增长因子": ["CRB现货指数:工业", "南华铜指数", "申万行业指数:房地产开发"],
    "通胀因子": ["市场价:生猪(外三元):全国均价", "期货收盘价(连续):布伦特原油:ICE", "现货价:螺纹钢"],
    "利率因子": ["中债国债总指数(总值)净价指数"],
    "信用因子": ["中债信用债总指数(3-5年)财富指数", "中债3-5年期国债指数(总值)财富指数"],
    "汇率因子": ["美国:美元指数"],
    "流动性因子": ["市盈率(成份股计算):申万大盘指数", "市盈率(成份股计算):申万小盘指数"],
}

ORIGINAL_MACRO_COLS = {
    "增长因子": ["制造业PMI", "固定资产投资(不含农户)完成额:累计同比",
                 "社会消费品零售 总额:当月同比", "进出口金额(人民币计价):中国:当月同比"],
    "通胀因子": ["CPI:当月同比", "PPI: 月同比"],
    "利率因子": ["中债国债到期收益率:10年:月:平均值"],
    "信用因子": ["中债中短期票据到期收益率(AA):3年:月:平均值",
                "中债国开债到期收益率:3年:月:平均值"],
    "汇率因子": ["美国:美元指数:月:平均值"],
    "流动性因子": ["M2(货币和准货币):同比", "社会融资规模增量:当月同比"],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
