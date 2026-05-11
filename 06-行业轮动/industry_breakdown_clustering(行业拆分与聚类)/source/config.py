"""
配置文件 - 项目配置参数 (申万行业数据)
"""

SW_INDUSTRIES = {
    '农林牧渔': 'Agriculture',
    '基础化工': 'Chemicals',
    '钢铁': 'Steel',
    '有色金属': 'Nonferrous Metals',
    '电子': 'Electronics',
    '汽车': 'Automobile',
    '家用电器': 'Home Appliances',
    '食品饮料': 'Food & Beverage',
    '纺织服饰': 'Textile & Apparel',
    '轻工制造': 'Light Manufacturing',
    '医药生物': 'Pharmaceutical',
    '公用事业': 'Utilities',
    '交通运输': 'Transportation',
    '房地产': 'Real Estate',
    '商贸零售': 'Retail',
    '社会服务': 'Consumer Services',
    '银行': 'Banking',
    '非银金融': 'Non-bank Finance',
    '综合': 'Conglomerate',
    '建筑材料': 'Building Materials',
    '建筑装饰': 'Building Decoration',
    '电力设备': 'Power Equipment',
    '机械设备': 'Machinery',
    '国防军工': 'Defense',
    '计算机': 'Computer',
    '传媒': 'Media',
    '通信': 'Communication',
    '煤炭': 'Coal',
    '石油石化': 'Petroleum',
    '环保': 'Environmental'
}

SW_INDUSTRY_CODES = {
    '农林牧渔': '801010',
    '基础化工': '801030',
    '钢铁': '801040',
    '有色金属': '801050',
    '电子': '801080',
    '汽车': '801880',
    '家用电器': '801110',
    '食品饮料': '801120',
    '纺织服饰': '801130',
    '轻工制造': '801140',
    '医药生物': '801150',
    '公用事业': '801160',
    '交通运输': '801170',
    '房地产': '801180',
    '商贸零售': '801200',
    '社会服务': '801210',
    '银行': '801780',
    '非银金融': '801790',
    '综合': '801230',
    '建筑材料': '801710',
    '建筑装饰': '801720',
    '电力设备': '801730',
    '机械设备': '801890',
    '国防军工': '801740',
    '计算机': '801750',
    '传媒': '801760',
    '通信': '801770',
    '煤炭': '801950',
    '石油石化': '801960',
    '环保': '801970'
}

CITIC_INDUSTRIES = SW_INDUSTRIES
CITIC_INDUSTRY_CODES = SW_INDUSTRY_CODES

CLUSTER_RESULT = {
    '能源': {
        '石油石化': ['石油石化'],
        '煤炭': ['煤炭']
    },
    '材料': {
        '钢铁有色': ['钢铁', '有色金属'],
        '化工建材': ['基础化工', '建筑材料', '建筑装饰']
    },
    '制造': {
        '电力设备': ['电力设备'],
        '机械设备': ['机械设备'],
        '国防军工': ['国防军工']
    },
    '消费': {
        '汽车家电': ['汽车', '家用电器'],
        '食品纺织': ['食品饮料', '纺织服饰', '轻工制造'],
        '医药生物': ['医药生物'],
        '商贸零售': ['商贸零售', '社会服务'],
        '农林牧渔': ['农林牧渔']
    },
    '金融': {
        '银行': ['银行'],
        '非银金融': ['非银金融'],
        '房地产': ['房地产']
    },
    '成长': {
        'TMT': ['计算机', '电子', '传媒', '通信']
    },
    '稳定': {
        '公用事业': ['公用事业'],
        '交通运输': ['交通运输']
    },
    '综合': {
        '综合': ['综合']
    }
}

DATA_CONFIG = {
    'start_date': '20140101',
    'end_date': '20201231',
    'monte_carlo_simulations': 1000,
    'min_days': 750,
    'n_clusters': 5
}

FILE_PATHS = {
    'output': 'd:\\Documents\\trae_projects\\industry_breakdown_clustering\\output\\',
    'reference_report': 'd:\\Documents\\trae_projects\\industry_breakdown_clustering\\参考研报\\',
    'data': 'd:\\Documents\\trae_projects\\industry_breakdown_clustering\\data\\'
}

OUTPUT_FILES = {
    'divergence_ranking': 'industry_divergence_ranking.csv',
    'cluster_result': 'cluster_result.csv',
    'similarity_matrix': 'similarity_matrix.csv',
    'mst_edges': 'mst_edges.csv',
    'split_evaluation': 'split_evaluation.csv'
}
