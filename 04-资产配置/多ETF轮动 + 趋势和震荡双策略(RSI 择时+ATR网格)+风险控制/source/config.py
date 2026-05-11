import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

ETF_POOL = {
    '515520': 10,
    '161907': 10,
    '512890': 10,
    '515300': 10,
    '510050': 10,
    '159916': 10,
    '512910': 10,
    '510310': 10,
    '512260': 10,
    '512090': 10,
    '515000': 10,
    '512040': 10,
    '510900': 10,
    '501021': 10,
    '513050': 10,
    '518880': 4
}

BOND_ETF = '163210'
REFERENCE_INDEX = '000001'

BACKTEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'initial_cash': 1000000,
    'daily_injection': 400,
    'commission': 0.0003,
    'min_commission': 5,
    'slippage': 0.002
}

STRATEGY_CONFIG = {
    'stock_count': 5,
    'min_ratio': 0.10,
    'max_ratio': 0.30,
    'min_amount': 2000,
    'confidence_level': 0.02,
    'reference_cycle': 250,
    'max_range': 30,
    'consolidation': 5,
    'weekday': 4,
    'net_rate': 3,
    'min_avg_money': 2000000,
    'min_list_days': 300
}

INDICATOR_CONFIG = {
    'rsi_period': 7,
    'rsi_avg_period': 14,
    'atr_period': 21,
    'ema_long_period': 60,
    'volatility_period': 120
}
