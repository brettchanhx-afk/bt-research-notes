# -*- coding: utf-8 -*-
"""
配置文件：ETF聚类优选系统
基于民生证券研报《ETF的聚类优选与热点趋势策略构建》
"""

import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ==================== 聚类参数 ====================
CLUSTERING_CONFIG = {
    'method': 'kmeans_plusplus',  # k-means++算法
    'n_clusters': None,  # 自动计算为n/5
    'init': 'k-means++',  # k-means++初始化
    'n_init': 10,  # 运行10次取最优
    'max_iter': 300,
    'random_state': 42,
    'update_frequency': 'semi_annual',  # 每半年度更新
}

# ==================== 指数评价参数 ====================
INDEX_EVALUATION_CONFIG = {
    'valuation_contribution_weight': 0.25,  # 估值贡献影响
    'concentration_weight': 0.25,  # 集中度
    'profitability_weight': 0.25,  # 盈利能力ROE_TTM
    'growth_weight': 0.25,  # 成长性营收同比
    'sharpe_period_all': 'all',  # 全期年化夏普
    'sharpe_period_1y': '1y',  # 近一年夏普
    'sharpe_top_percent': 0.5,  # 选择前50%
    'financial_period': '6m',  # 半年期
}

# ==================== ETF筛选参数 ====================
ETF_SELECTION_CONFIG = {
    'fee_score_weight': 0.40,  # 费率得分
    'liquidity_score_weight': 0.20,  # 成交额得分
    'scale_score_weight': 0.20,  # 规模得分
    'tracking_error_weight': 0.10,  # 跟踪误差
    'info_ratio_weight': 0.10,  # 信息比率
    'max_etf_per_index': 2,  # 每个指数最多保留2只ETF
    'min_etf_threshold': 5,  # 超过5只ETF才保留前2只
    'min_scale': 0.5,  # 最小规模（亿元）
    'min_liquidity': 0.1,  # 最小日均成交额（亿元）
}

# ==================== 数据源优先级 ====================
DATA_SOURCE_PRIORITY = ['efinance', 'tushare', 'akshare', 'baostock']

# ==================== Tushare配置 ====================
TUSHARE_CONFIG = {
    'token': '1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb',
    'api_url': 'http://jiaoch.site',
}

# ==================== 基准配置 ====================
BENCHMARK_CONFIG = {
    'index_code': '000300.SH',  # 沪深300
    'name': '沪深300',
}

# ==================== 回测参数 ====================
BACKTEST_CONFIG = {
    'start_date': '2014-01-01',
    'end_date': '2025-02-21',
    'rebalance_frequency': 'semi_annual',
    'select_top_ratio': 0.5,
}

# ==================== 可视化配置 ====================
PLOT_CONFIG = {
    'style': 'seaborn-v0_8-whitegrid',
    'fig_size': (12, 6),
    'dpi': 150,
    'font_size': 10,
    'title_font_size': 14,
}

# ==================== ETF类型分类 ====================
ETF_TYPE_MAPPING = {
    'wide_base': ['规模风格', '宽基'],
    'industry': ['行业'],
    'bond': ['债券'],
    'commodity': ['商品'],
    'cross_border': ['跨境'],
}
