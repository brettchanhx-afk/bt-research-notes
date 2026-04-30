# -*- coding: utf-8 -*-
"""
Campisi 债券基金归因模型 - 配置文件
"""
import os

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# 数据子目录
BOND_HOLDINGS_DIR = os.path.join(DATA_DIR, 'bond_holdings')
YIELD_CURVES_DIR = os.path.join(DATA_DIR, 'yield_curves')

# 确保目录存在
for d in [DATA_DIR, OUTPUT_DIR, BOND_HOLDINGS_DIR, YIELD_CURVES_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 数据源配置
# ============================================================
# 数据源优先级
DATA_SOURCE_PRIORITY = ['tushare', 'efinance', 'akshare', 'baostock']

# Tushare配置（根据MEMORY.md中的配置）
TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_API_URL = "http://jiaoch.site"


# ============================================================
# 债券类型与评级映射
# ============================================================
# 债券类型
BOND_TYPES = {
    '国债': 'treasury',
    '地方政府债': 'local_gov',
    '金融债': 'financial',
    '企业债': 'corporate',
    '公司债': 'corporate',
    '中期票据': 'mtm',
    '短期融资券': 'cp',
    '超短期融资券': 'scp',
    '定向债务融资工具': 'ppn',
    '资产支持证券': 'abs',
    '可转债': 'convertible',
}

# 信用评级映射（标准化）
CREDIT_RATING_MAP = {
    'AAA': 'AAA',
    'AAA-': 'AAA',
    'AA+': 'AA',
    'AA': 'AA',
    'AA-': 'AA',
    'A+': 'A',
    'A': 'A',
    'A-': 'A',
    'BBB+': 'BBB',
    'BBB': 'BBB',
    'BBB-': 'BBB',
    'BB+': 'BB',
    'BB': 'BB',
    'BB-': 'BB',
    'B+': 'B',
    'B': 'B',
    'B-': 'B',
    'CCC': 'CCC',
    'CC': 'CC',
    'C': 'C',
    'D': 'D',
}

# 国开债作为无风险利率基准
RISK_FREE_BENCHMARK = '国开债'


# ============================================================
# 归因分析参数
# ============================================================
# 付息频率（年付息次数）
COUPON_FREQUENCY = {
    'treasury': 2,      # 国债：半年付息
    'local_gov': 2,     # 地方债：半年付息
    'financial': 1,     # 金融债：年付息
    'corporate': 1,     # 企业债：年付息
    'mtm': 1,           # 中票：年付息
    'cp': 0,            # 短融：到期一次还本付息
    'scp': 0,           # 超短融：到期一次还本付息
}

# 默认付息频率
DEFAULT_COUPON_FREQUENCY = 1

# 收益率曲线插值方法
YIELD_CURVE_INTERPOLATION = 'cubic'  # 'linear', 'cubic', 'spline'


# ============================================================
# 可视化配置
# ============================================================
import matplotlib
import matplotlib.pyplot as plt

# 中文字体配置
def setup_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120

# 颜色方案
COLORS = {
    'coupon': '#2E86AB',           # 票息效应 - 蓝色
    'treasury': '#A23B72',         # 国债利率效应 - 紫红色
    'credit': '#F18F01',           # 信用利差效应 - 橙色
    'total': '#C73E1D',            # 总收益 - 红色
    'positive': '#28A745',         # 正贡献 - 绿色
    'negative': '#DC3545',         # 负贡献 - 红色
}


# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
