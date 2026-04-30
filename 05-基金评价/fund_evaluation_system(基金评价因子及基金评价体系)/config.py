# -*- coding: utf-8 -*-
"""
基金评价体系 - 配置文件
"""
import os

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tushare配置
TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_API_URL = "http://jiaoch.site"

# 回测区间
BACKTEST_START = '2016-12-31'
BACKTEST_END = '2022-05-31'

# 调仓路径
REBALANCE_PATHS = {
    'path_1': [1, 4, 7, 10],   # 1/4/7/10月末
    'path_2': [2, 5, 8, 11],   # 2/5/8/11月末
    'path_3': [3, 6, 9, 12],   # 3/6/9/12月末
}

# 因子窗口期（月）
WINDOWS = [3, 6, 9, 12]

# 板块划分
SECTORS = ['消费', '医药', '科技', '周期', '金融', '高端制造', '成长', '价值', '均衡', '全市场']

# 复合因子（推荐5个）
COMPOSITE_FACTORS = [
    '年化收益率',
    '逆境战胜市场胜率',
    'H-M模型择时',
    '基金份额',
    '下行风险',
]

# 五维评分体系
SCORE_DIMENSIONS = {
    '年化收益率': {'weight': 0.2, 'direction': 1},
    '逆境战胜市场胜率': {'weight': 0.25, 'direction': 1},
    'H-M模型择时': {'weight': 0.2, 'direction': 1},
    '基金份额': {'weight': 0.15, 'direction': 1},
    '下行风险': {'weight': 0.2, 'direction': -1},  # 负向，越小越好
}

# 筛选条件
SCREENING_RULES = {
    'min_scale': 1e8,           # 规模>1亿
    'min_age': 365,            # 成立>1年
    'min_manager_tenure': 365, # 基金经理任职>1年
    'max_holder_ratio': 0.8,    # 机构占比<80%
}

# max_ICIR权重约束
MAX_ICIR_WEIGHT_MIN = 0.10
MAX_ICIR_WEIGHT_MAX = 0.30
MAX_ICIR_WINDOW = 6  # 月

# 可视化
import matplotlib.pyplot as plt

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120

COLORS = {
    'layer1': '#2E86AB', 'layer2': '#A23B72', 'layer3': '#F18F01',
    'ic_positive': '#28A745', 'ic_negative': '#DC3545',
}