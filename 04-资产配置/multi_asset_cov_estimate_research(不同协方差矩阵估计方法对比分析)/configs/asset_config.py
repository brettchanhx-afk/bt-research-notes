ASSET_CONFIG = {
    '沪深300': '000300.SH',
    '中证1000': '000852.SH',
    '恒生指数': 'HSI.HK',
    '标普500': 'SPX.GI',
    '中债-国债总财富': 'CBA00101.CI',
    '中债-企业债总财富': 'CBA00201.CI',
    '南华商品指数': 'NH0100.NH',
    'COMEX黄金': 'GC00Y.GC',
    'ICE布油': 'CL00Y.NYM',
    '美元指数': 'DX0001.DXY',
}

BL_ASSET_CONFIG = {
    '沪深300': '000300.SH',
    '标普500': 'SPX.GI',
    '恒生指数': 'HSI.HK',
    '中债-国债总财富': 'CBA00101.CI',
    '中债-企业债总财富': 'CBA00201.CI',
    '南华商品指数': 'NH0100.NH',
}

RISK_PARITY_CONFIG = {
    '沪深300': '000300.SH',
    '标普500': 'SPX.GI',
    '恒生指数': 'HSI.HK',
    '中债-企业债总财富': 'CBA00201.CI',
    '南华商品指数': 'NH0100.NH',
    'COMEX黄金': 'GC00Y.GC',
}

CITIC_INDUSTRY_CODES = {
    '农林牧渔': '801010.SI',
    '采掘': '801020.SI',
    '化工': '801030.SI',
    '钢铁': '801040.SI',
    '有色金属': '801050.SI',
    '电子': '801080.SI',
    '汽车': '801110.SI',
    '家用电器': '801110.SI',
    '食品饮料': '801120.SI',
    '纺织服装': '801130.SI',
    '轻工制造': '801140.SI',
    '医药生物': '801150.SI',
    '公用事业': '801160.SI',
    '交通运输': '801170.SI',
    '房地产': '801180.SI',
    '商业贸易': '801200.SI',
    '休闲服务': '801210.SI',
    '银行': '801780.SI',
    '非银金融': '801790.SI',
    '建筑材料': '801710.SI',
    '建筑装饰': '801720.SI',
    '电气设备': '801730.SI',
    '国防军工': '801740.SI',
    '计算机': '801750.SI',
    '传媒': '801760.SI',
    '通信': '801770.SI',
    '机械设备': '801780.SI',
    '综合': '801230.SI',
}

COVARIANCE_METHODS = [
    'sample_cov',
    'ledoit_wolf_constant_variance',
    'ledoit_wolf_single_factor',
    'ledoit_wolf_constant_correlation',
    'random_matrix',
    'risk_metrics',
    'ccc_garch',
    'dcc_garch',
]

LOOKBACK_PERIODS = {
    '126': 126,
    '252': 252,
    '504': 504,
    '756': 756,
    '1008': 1008,
    '1260': 1260,
}

BACKTEST_PARAMS = {
    'start_date': '20070101',
    'end_date': '20230228',
    'rebalance_freq': 'monthly',
    'allow_short': False,
    'target_volatility': 0.05,
}

INITIAL_CAPITAL = 1000000