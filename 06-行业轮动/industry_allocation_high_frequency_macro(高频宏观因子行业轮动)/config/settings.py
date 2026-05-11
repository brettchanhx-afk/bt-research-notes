import tushare as ts

TOKEN = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
TUSHARE_API_URL = "http://jiaoch.site"

def init_tushare():
    pro = ts.pro_api(TOKEN)
    pro._DataApi__token = TOKEN
    pro._DataApi__http_url = TUSHARE_API_URL
    return pro

PRO = init_tushare()

DATA_DIR = "data"
OUTPUT_DIR = "output"

START_DATE = "20160430"
END_DATE = "20230531"

INDUSTRY_CODES = [
    "CI005001.WI", "CI005002.WI", "CI005107.WI", "CI005188.WI", "CI005005.WI",
    "CI005006.WI", "CI005008.WI", "CI005004.WI", "CI005024.WI", "CI005007.WI",
    "CI005010.WI", "CI005011.WI", "CI005012.WI", "CI005013.WI", "CI005016.WI",
    "CI005156.WI", "CI005823.WI", "CI005822.WI", "CI005020.WI", "CI005014.WI",
    "CI005015.WI", "CI005017.WI", "CI005009.WI", "CI005018.WI", "CI005025.WI",
    "CI005026.WI", "CI005027.WI", "CI005028.WI", "CI005021.WI", "CI005022.WI",
    "CI005023.WI"
]

FACTOR_CONFIG = {
    "growth": {
        "name": "增长",
        "long_assets": ["HSI.HI", "CRBRI.RB", "NH0012.NHF"],
        "short_assets": ["CBA00652.CS"],
        "description": "经济增长因子"
    },
    "life_inflation": {
        "name": "生活端通胀",
        "assets": ["NH0056.NHF"],
        "description": "生活端通胀因子"
    },
    "production_inflation": {
        "name": "生产端通胀",
        "assets": ["B00.IPE", "NH0016.NHF", "NH0030.NHF"],
        "description": "生产端通胀因子"
    },
    "risk_free_rate": {
        "name": "无风险利率",
        "long_assets": ["CBA00621.CS"],
        "description": "无风险利率因子"
    },
    "credit_spread": {
        "name": "信用利差",
        "long_assets": ["CBA00621.CS"],
        "short_assets": ["CBA02501.CS"],
        "description": "信用利差因子"
    },
    "term_spread": {
        "name": "期限利差",
        "long_assets": ["CBA00651.CS"],
        "short_assets": ["CBA00621.CS"],
        "description": "期限利差因子"
    },
    "exchange_rate": {
        "name": "汇率",
        "long_assets": ["AU9999.SGE"],
        "description": "汇率因子"
    }
}

SW_INDUSTRY_CODES = [
    "801010.SI", "801020.SI", "801030.SI", "801040.SI", "801050.SI",
    "801060.SI", "801080.SI", "801110.SI", "801120.SI", "801130.SI",
    "801140.SI", "801150.SI", "801160.SI", "801170.SI", "801180.SI",
    "801200.SI", "801210.SI", "801230.SI", "801710.SI", "801720.SI",
    "801730.SI", "801740.SI", "801750.SI", "801760.SI", "801770.SI",
    "801780.SI", "801790.SI", "801880.SI", "801890.SI"
]

AKSHARE_PROXY_ASSETS = {
    'HSI.HI': 'HSI',
    'CRBRI.RB': 'CRB',
    'NH0012.NHF': 'NHSC',
    'CBA00652.CS': None,
    'NH0056.NHF': 'pig',
    'B00.IPE': 'CL',
    'NH0016.NHF': 'RB',
    'NH0030.NHF': 'HC',
    'CBA00621.CS': None,
    'CBA02501.CS': None,
    'CBA00651.CS': None,
    'AU9999.SGE': 'XAU'
}
