"""
配置文件
项目参数设置
"""

# 数据相关
DATA_CONFIG = {
    # 指数代码映射 (用于数据获取)
    'INDEX_CODES': {
        'CSI300': '000300',  # 沪深300
        'CSI500': '000905',  # 中证500
        'HSI': '^HSI',     # 恒生指数
        'HSCEI': '^HSCEI', # 恒生国企指数
        'SPX': '^GSPC',   # 标普500
        'NDAQ': '^IXIC',  # 纳斯达克
        'FTSE': '^UKX',   # 富时100
        'CAC40': '^FCHI',  # 巴黎CAC40
        'DAX': '^GDAXI',  # 德国DAX
    },
    
    # 数据获取参数
    'START_DATE': '2008-01-01',
    'END_DATE': '2025-12-31',
}

# 回测相关
BACKTEST_CONFIG = {
    # 回测时间范围 (用于回测验证)
    'BACKTEST_START': '2009-01-01',
    'BACKTEST_END': '2025-12-31',
    
    # 协方差估计窗口 (研报中使用240个交易日)
    'WINDOW': 240,
    
    # 调仓频率 ('monthly', 'quarterly', 'yearly')
    'REBALANCE_FREQ': 'quarterly',
    
    # 半衰期 (研报中默认120个交易日)
    'HALF_LIFE': 120,
}

# 模型相关
MODEL_CONFIG = {
    # 要测试的模型列表
    'MODELS': [
        'EW',    # 等权重
        'EV',    # 等波动率
        'MV',    # 最小方差
        'MD',    # 最大分散化
        'RP',    # 风险平价
        'PCRP',  # 主成分风险平价
        'HPCRP', # 半衰主成分风险平价
    ],
    
    # 模型中文名称
    'MODEL_NAMES_CN': {
        'EW': '等权重',
        'EV': '等波动率',
        'MV': '最小方差',
        'MD': '最大分散化',
        'RP': '风险平价',
        'PCRP': '主成分风险平价',
        'HPCRP': '半衰主成分风险平价',
    }
}

# 可视化相关
PLOT_CONFIG = {
    # 图表配色
    'COLORS': {
        'EW': '#1f77b4',
        'EV': '#ff7f0e',
        'MV': '#2ca02c',
        'MD': '#d62728',
        'RP': '#9467bd',
        'PCRP': '#8c564b',
        'HPCRP': '#e377c2',
    },
    
    # 输出 DPI
    'DPI': 150,
}

# 输出相关
OUTPUT_CONFIG = {
    # 输出目录
    'OUTPUT_DIR': 'output',
    'DATA_DIR': 'data',
    
    # 中英文
    'LANGUAGE': 'CN',
    
    # 无风险利率 (年化)
    'RISK_FREE_RATE': 0.03,
}