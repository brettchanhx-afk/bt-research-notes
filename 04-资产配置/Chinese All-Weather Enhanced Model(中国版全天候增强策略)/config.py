
"""
中国版全天候增强策略配置文件
"""

# 数据配置
DATA_CONFIG = {
    'start_date': '20131231',
    'end_date': '20250430',
    'use_saved_data': True
}

# 策略配置
STRATEGY_CONFIG = {
    'lookback_window': 252,
    'use_semicovariance': True,
    'lambda_param': 0.94,
    'momentum_lookback': 60
}

# 回测配置
BACKTEST_CONFIG = {
    'initial_capital': 1.0,
    'rebalance_freq': 'M',
    'fee_rate': 0.0005,
    'risk_free_rate': 0.0
}

# 资产池配置
ASSET_POOL = {
    'stocks': {
        '510300.SH': '沪深300ETF',
        '512100.SH': '中证1000ETF'
    },
    'high_dividend': {
        '512890.SH': '红利低波ETF'
    },
    'bonds': {
        '511260.SH': '十年国债ETF',
        '511090.SH': '三十年国债ETF'
    },
    'commodities': {
        '159980.SZ': '有色ETF',
        '159981.SZ': '能化ETF',
        '159985.SZ': '豆粕ETF'
    },
    'gold': {
        '518880.SH': '黄金ETF'
    }
}

# 四象限资产配置
QUADRANT_ASSETS = {
    'growth_above': ['510300.SH', '512100.SH', '159980.SZ', '159981.SZ', '159985.SZ'],
    'growth_below': ['511260.SH', '511090.SH', '518880.SH'],
    'inflation_above': ['159980.SZ', '159981.SZ', '159985.SZ', '518880.SH'],
    'inflation_below': ['511260.SH', '511090.SH', '518880.SH', '512890.SH']
}

# 输出配置
OUTPUT_CONFIG = {
    'output_dir': 'output',
    'save_plots': True,
    'save_data': True
}

