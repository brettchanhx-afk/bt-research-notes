import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'ipynb')

TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_API_URL = "http://jiaoch.site"

ASSETS = {
    'CSI300': {
        'name': '沪深300',
        'ts_code': '000300.SH',
        'type': 'index',
    },
    'HSI': {
        'name': '恒生指数',
        'ts_code': 'HSI.HK',
        'type': 'index',
    },
    'Nikkei225': {
        'name': '日经225',
        'ts_code': 'N225.JP',
        'type': 'index',
    },
    'SP500': {
        'name': '标普500',
        'ts_code': 'SPX.GI',
        'type': 'index',
    },
    'Gold': {
        'name': 'COMEX黄金',
        'ts_code': 'GC00Y.NYM',
        'type': 'future',
    },
    'BrentOil': {
        'name': 'ICE布油',
        'ts_code': 'BZ00Y.NYM',
        'type': 'future',
    },
    'Copper': {
        'name': 'SHFE铜',
        'ts_code': 'CU00Y.SHF',
        'type': 'future',
    },
    'USTBond': {
        'name': '美国国债7-10年ETF',
        'ts_code': 'IEF.US',
        'type': 'etf',
    },
    'CNTBond': {
        'name': '中债-国债总财富(5-7年)指数',
        'ts_code': 'CBA00603.CI',
        'type': 'bond_index',
    },
    'CNCorpBond': {
        'name': '中债-企业债AAA财富指数',
        'ts_code': 'CBA00701.CI',
        'type': 'bond_index',
    },
}

RISK_PARITY_PARAMS = {
    'lookback_months': 6,
    'transaction_cost': 0.0005,
    'rebalance_frequency': 'monthly',
}

MOMENTUM_PARAMS = {
    'k_values': [0.1, 0.5, 1.0, 1.5, 2.0],
    'default_k': 1.0,
    'sharpe_lookback_l1': {
        'CSI300': 89,
        'HSI': 69,
        'Nikkei225': 30,
        'SP500': 5,
        'Gold': 114,
        'BrentOil': 25,
        'Copper': 35,
        'USTBond': 64,
        'CNTBond': 10,
        'CNCorpBond': 5,
    },
    'sharpe_lookback_l2': {
        'CSI300': 15,
        'HSI': 45,
        'Nikkei225': 99,
        'SP500': 60,
        'Gold': 15,
        'BrentOil': 124,
        'Copper': 15,
        'USTBond': 79,
        'CNTBond': 15,
        'CNCorpBond': 15,
    },
}

BACKTEST_PARAMS = {
    'start_date': '20070101',
    'end_date': '20240930',
    'initial_capital': 100000000.0,
}
