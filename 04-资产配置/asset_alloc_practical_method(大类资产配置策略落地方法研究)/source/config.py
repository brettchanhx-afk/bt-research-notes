import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_URL = "http://jiaoch.site"

ASSET_CONFIG = {
    'equity': {
        'index': '000300.SH',
        'name': '沪深300'
    },
    'bond': {
        'index': '000016.SH',
        'name': '上证50国债'
    },
    'commodity': {
        'index': '000998.SH',
        'name': '中证商品'
    },
    'gold': {
        'index': '518880.SH',
        'name': '黄金ETF'
    }
}

BACKTEST_CONFIG = {
    'start_date': '20210101',
    'end_date': '20240111',
    'rebalance_freq': 'monthly',
    'transaction_cost': 0.0005
}

FIT_CONFIG = {
    'max_tracking_error': 0.01,
    'max_drawdown': 0.05,
    'lookback_period': 60
}

MODEL_CONFIG = {
    'bl_confidence': 0.5,
    'risk_parity_target_risk': 0.1,
    'macro_factor_lambda': 0.1
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)