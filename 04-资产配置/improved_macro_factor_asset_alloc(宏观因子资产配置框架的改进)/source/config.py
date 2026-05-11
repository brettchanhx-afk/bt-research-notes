import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

for dir_path in [OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

ASSETS_CONFIG = {
    "CSI300": {
        "name": "沪深300",
        "code": "000300.SH",
        "type": "index",
        "csv_col": "000300.SH",
    },
    "CSI500": {
        "name": "中证500",
        "code": "000905.SH",
        "type": "index",
        "csv_col": "000905.SH",
    },
    "NHCI": {
        "name": "南华商品",
        "code": "NHCI.SL",
        "type": "commodity",
        "csv_col": "NHCI.SL",
    },
    "GOV_BOND": {
        "name": "中债国债",
        "code": "CBA00601.CB",
        "type": "bond",
        "csv_col": "CBA00601.CB",
    },
    "CORP_BOND": {
        "name": "中债企业债",
        "code": "CBA02001.CB",
        "type": "bond",
        "csv_col": "CBA02001.CB",
    },
    "BRENT": {
        "name": "布伦特原油",
        "code": "BRN0Y.ICE",
        "type": "commodity",
        "csv_col": "BRN0Y.ICE",
    },
}

FACTOR_CONFIG = {
    "Growth": {
        "name": "增长因子",
        "raw_factors": ["PMI", "FAI", "RetailSales", "ExportImport"],
        "high_freq_assets": ["CRBIndustrial", "SouthwestCopper", "RealEstate"],
        "weights": {"CRBIndustrial": 0.61, "SouthwestCopper": 0.24, "RealEstate": 0.15},
    },
    "Inflation": {
        "name": "通胀因子",
        "raw_factors": ["CPI", "PPI"],
        "high_freq_assets": ["PorkPrice", "BrentOil", "SteelRebar"],
        "weights": {"PorkPrice": 0.35, "BrentOil": 0.22, "SteelRebar": 0.43},
    },
    "IntRate": {
        "name": "利率因子",
        "raw_factors": ["TenYearYield"],
        "high_freq_assets": ["ChinaGovBond"],
    },
    "Credit": {
        "name": "信用因子",
        "raw_factors": ["CreditSpread"],
        "high_freq_assets": ["CorpBondAA", "GovBond"],
    },
    "ExchRate": {
        "name": "汇率因子",
        "raw_factors": ["USDIndex"],
        "high_freq_assets": ["HSI"],
    },
    "Liquidity": {
        "name": "流动性因子",
        "raw_factors": ["M2_SocialFin"],
        "high_freq_assets": ["SWLargeCapPE", "SWSmallCapPE"],
    },
}

RAW_FACTOR_COLUMNS = {
    "PMI": "制造业PMI",
    "FAI": "固定资产投资(不含农户)完成额:累计同比",
    "RetailSales": "社会消费品零售总额:当月同比",
    "ExportImport": "进出口金额(人民币计价):中国:当月同比",
    "CPI": "CPI:当月同比",
    "PPI": "PPI:当月同比",
    "TenYearYield": "中债国债到期收益率:10年:月:平均值",
    "CreditSpread": "AA3Y_GovBond3Y",
    "USDIndex": "美国:美元指数:月:平均值",
    "M2YoY": "M2(货币和准货币):同比",
    "SocialFinYoY": "社会融资规模增量:当月同比",
}

HIGH_FREQ_FACTOR_COLUMNS = {
    "HSI": "恒生指数",
    "CRBIndustrial": "CRB现货指数:工业",
    "SouthwestCopper": "南华铜指数",
    "RealEstate": "申万行业指数:房地产开发",
    "PorkPrice": "市场价:生猪(外三元):全国均价",
    "BrentOil": "期货收盘价(连续):布伦特原油:ICE",
    "SteelRebar": "现货价:螺纹钢",
    "ChinaGovBond": "中债国债总指数(总值)净价指数",
    "CorpBondAA": "中债信用债总指数(3-5年)财富指数",
    "ChinaGovBond3Y": "中债3-5年期国债指数(总值)财富指数",
    "USDIndex": "美国:美元指数",
    "SWLargeCapPE": "市盈率(成份股计算):申万大盘指数",
    "SWSmallCapPE": "市盈率(成份股计算):申万小盘指数",
}

REGRESSION_CONFIG = {
    "window_years": 5,
    "half_life_years": 1,
    "min_window_years": 3,
}

PORTFOLIO_CONFIG = {
    "risk_parity_method": "ewma_volatility",
    "volatility_window": 60,
    "rebalance_freq": "monthly",
    "max_leverage": 1.0,
    "min_weight": 0.0,
    "max_weight": 1.0,
}

MACRO_SCORING_RULES = {
    "score_levels": [-2, -1, 0, 1, 2],
    "coefficient_mapping": {
        -2: -1.0,
        -1: -0.5,
        0: 0.0,
        1: 0.5,
        2: 1.0,
    },
}

BACKTEST_CONFIG = {
    "start_date": "2013-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000000.0,
    "commission_rate": 0.0003,
    "stamp_tax": 0.001,
}

DATE_CONFIG = {
    "factor_update_freq": "monthly",
    "rebalance_freq": "monthly",
    "trading_calendar": "SSE",
}
