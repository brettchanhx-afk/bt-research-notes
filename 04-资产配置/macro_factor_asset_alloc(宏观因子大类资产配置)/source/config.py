"""
配置文件 - 宏观因子资产配置框架
包含资产列表、参数设置、数据源配置等
"""

TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_URL = "http://jiaoch.site"

ASSET_CONFIG = {
    "stock_indices": {
        "沪深300": {"ts_code": "000300.SH", "source": "tushare"},
        "中证500": {"ts_code": "000905.SH", "source": "tushare"},
        "恒生指数": {"ts_code": "HSI.HI", "source": "tushare"},
    },
    "bonds": {
        "中债国债": {"index_code": "CBA00201.CS", "source": "efinance"},
        "中债企业债": {"index_code": "CBA20121.CS", "source": "efinance"},
        "中证转债": {"ts_code": "000832.SH", "source": "tushare"},
    },
    "commodities": {
        "南华工业品": {"index_code": "NHCI.NHF", "source": "efinance"},
        "南华农产品": {"index_code": "NHCA.NHF", "source": "efinance"},
        "布伦特原油": {"commodity_code": "布伦特原油", "source": "akshare"},
        "沪金": {"commodity_code": "沪金", "source": "akshare"},
    },
    "forex": {
        "美元兑人民币": {"symbol": "USDCNY", "source": "akshare"},
    }
}

ALL_ASSETS = [
    "沪深300", "中证500", "中债国债", "中债企业债", "中证转债",
    "南华工业品", "南华农产品", "布伦特原油", "沪金", "美元兑人民币", "恒生指数"
]

MACRO_FACTORS = ["增长", "通胀", "利率", "信用", "汇率", "流动性"]

BACKTEST_CONFIG = {
    "start_date": "2010-02-01",
    "end_date": "2023-05-31",
    "rebalance_freq": "monthly",
    "factor_deviation": 0.05,
}

OPTIMIZER_CONFIG = {
    "lambda_param": 0.1,
    "weight_bounds": (0, 1),
    "deviation_bounds": (-1, 1),
}

PCA_PARAMS = {
    "n_components": 6,
    "whiten": False,
}

LASSO_PARAMS = {
    "alpha": 0.01,
    "max_iter": 5000,
    "random_state": 42,
}

RISK_FREE_RATE = 0.03

OUTPUT_DIR = "d:/Documents/trae_projects/macro_factor_asset_alloc/output"
DATA_DIR = "d:/Documents/trae_projects/macro_factor_asset_alloc/data"
