# -*- coding: utf-8 -*-
"""
基于净值的债券基金风险因子归因模型 - 配置文件
"""
import os

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

for d in [DATA_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 数据源配置
# ============================================================
DATA_SOURCE_PRIORITY = ['tushare', 'efinance', 'akshare']

# Tushare配置
TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_API_URL = "http://jiaoch.site"


# ============================================================
# 风险因子配置
# ============================================================
# 利率风险因子
INTEREST_RATE_FACTORS = {
    'duration': '久期因子',
    'convexity': '凸性因子',
}

# 信用风险因子（信用债指数 - 国债指数）
CREDIT_FACTORS = {
    'credit_spread': '信用利差因子',
}

# 可转债风险因子
CONVERTIBLE_FACTORS = {
    'convertible': '可转债因子',
}

# 指数代码映射
INDEX_CODES = {
    # 国债指数
    'treasury': '000012',        # 国债指数
    # 信用债指数
    'corporate_bond': '000013',  # 企业债指数
    # 可转债指数
    'convertible': '000832',     # 中证转债指数
    # 国开债指数
    'cdb': 'CBA00101',           # 国开债指数
}


# ============================================================
# 债券基金池
# ============================================================
BOND_FUND_POOL = [
    '110017',  # 易方达增强回报A
    '050011',  # 博时信用债券A
    '240001',  # 宝康债券
    '161716',  # 融通添利A
    '519667',  # 银河银泰理财
]


# ============================================================
# 模型参数
# ============================================================
# 回归窗口（交易日）
REGRESSION_WINDOW = 60

# 滚动步长
ROLLING_STEP = 20

# VIF阈值（共线性诊断）
VIF_THRESHOLD = 10.0

# 显著性水平
SIGNIFICANCE_LEVEL = 0.05


# ============================================================
# 可视化配置
# ============================================================
import matplotlib.pyplot as plt

def setup_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120

# 颜色方案
COLORS = {
    'duration': '#2E86AB',       # 久期因子 - 蓝色
    'convexity': '#A23B72',      # 凸性因子 - 紫红色
    'credit': '#F18F01',         # 信用因子 - 橙色
    'convertible': '#28A745',    # 转债因子 - 绿色
    'alpha': '#6C757D',          # Alpha - 灰色
    'residual': '#DC3545',       # 残差 - 红色
}
