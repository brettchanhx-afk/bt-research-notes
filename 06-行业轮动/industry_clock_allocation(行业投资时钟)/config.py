"""
配置文件 - 投资时钟策略配置
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
CONFIG_DIR = os.path.join(OUTPUT_DIR, 'config')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

TUSHARE_TOKEN = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
TUSHARE_API_URL = "http://jiaoch.site"

BACKTEST_START_DATE = "20110101"
BACKTEST_END_DATE = "20210630"

RISK_FREE_RATE = 0.04
TARGET_VOLATILITY = 0.05
MAX_LEVERAGE = 2.0
TRANSACTION_COST = 0.002

CYCLE_PERIOD = 42

FACTOR_MOMENTUM_WINDOW = 3
FACTOR_MOMENTUM_CONSECUTIVE = 2

PHASE_TOP_RANGE = (1.047, 2.094)
PHASE_BOTTOM_RANGE = (4.712, 5.759)

ASSET_UNIVERSE = {
    'stock': ['000300.SH', '000905.SH', '399006.SZ'],
    'bond': ['H11001.CSI'],
    'commodity': ['NH0100.NHF', 'NH0500.NHF', 'SC.FUTURE', 'AU9999.SGE'],
    'cash': []
}

INDUSTRY_UNIVERSE = [
    '801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI',
    '801080.SI', '801110.SI', '801120.SI', '801130.SI', '801140.SI',
    '801150.SI', '801160.SI', '801170.SI', '801180.SI', '801200.SI',
    '801210.SI', '801230.SI', '801710.SI', '801720.SI', '801730.SI',
    '801740.SI', '801750.SI', '801760.SI', '801770.SI', '801780.SI',
    '801790.SI', '801880.SI', '801890.SI'
]

LEADING_INDICATORS = {
    'growth': {
        'benchmark': 'M0017126',
        'indicators': [
            'S0027012', 'S0027571', 'S0027103', 'S0027159', 'S0028202',
            'S0027907', 'S6001740', 'S0029669', 'S0029656', 'S0036018', 'M0024057'
        ]
    },
    'inflation': {
        'benchmark': 'M0000705',
        'indicators': [
            'S0066840', 'S0031507', 'S5711190', 'S0031525', 'S5705040'
        ]
    },
    'credit': {
        'indicators': ['M0001382', 'M0001384', 'M5525755', 'M0009969', 'M0043410']
    },
    'monetary': {
        'indicators': ['S0059744']
    }
}
