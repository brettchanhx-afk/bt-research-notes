"""
数据获取模块 - 使用本地申万行业数据
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

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

_industry_data_cache = {}
_returns_cache = None

def _load_industry_data(industry_name):
    """加载单个行业数据"""
    if industry_name in _industry_data_cache:
        return _industry_data_cache[industry_name]

    file_path = os.path.join(DATA_DIR, f'industry_{industry_name}.csv')
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            _industry_data_cache[industry_name] = df
            return df
        except Exception as e:
            print(f"加载{industry_name}数据失败: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def _load_all_returns():
    """加载所有行业收益率数据"""
    global _returns_cache
    if _returns_cache is not None:
        return _returns_cache

    returns_path = os.path.join(DATA_DIR, 'industry_returns.csv')
    if os.path.exists(returns_path):
        try:
            _returns_cache = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            return _returns_cache
        except Exception as e:
            print(f"加载收益率数据失败: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_industry_stocks(industry_name, trade_date=None):
    """
    获取申万一级行业成分股

    Parameters:
    -----------
    industry_name : str
        行业名称
    trade_date : str
        交易日期，格式YYYYMMDD (暂不支持)

    Returns:
    --------
    pd.DataFrame
        包含成分股代码和名称的DataFrame
    """
    return pd.DataFrame({
        'industry': [industry_name],
        'note': '使用申万行业分类，成分股信息需另行获取'
    })

def get_industry_index_daily(industry_code, start_date, end_date):
    """
    获取申万行业指数日线数据

    Parameters:
    -----------
    industry_code : str
        行业指数代码
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD

    Returns:
    --------
    pd.DataFrame
        包含日期、开盘、收盘、最高、最低、成交量等
    """
    industry_map = {v: k for k, v in SW_INDUSTRY_CODES.items()}
    industry_name = industry_map.get(industry_code, None)

    if industry_name is None:
        industry_name = industry_code

    df = _load_industry_data(industry_name)

    if not df.empty and isinstance(df.index, pd.DatetimeIndex):
        start_dt = pd.to_datetime(start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:])
        end_dt = pd.to_datetime(end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:])
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    return df

def get_stock_daily(ts_code, start_date, end_date):
    """
    获取个股日线数据

    Parameters:
    -----------
    ts_code : str
        股票代码，格式XXXXXX.XSHS
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD

    Returns:
    --------
    pd.DataFrame
        包含日期、开盘、收盘、最高、最低、成交量等
    """
    return pd.DataFrame()

def get_stock_financial_data(ts_code, start_year, end_year):
    """
    获取个股财务数据

    Parameters:
    -----------
    ts_code : str
        股票代码
    start_year : int
        开始年份
    end_year : int
        结束年份

    Returns:
    --------
    pd.DataFrame
        包含各类财务指标
    """
    return pd.DataFrame()

def get_all_industry_members():
    """
    获取所有申万一级行业

    Returns:
    --------
    dict
        行业名称列表
    """
    return {k: [] for k in SW_INDUSTRIES.keys() if k != '综合'}

def get_industry_price_data(start_date, end_date):
    """
    获取所有申万一级行业的价格数据

    Parameters:
    -----------
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD

    Returns:
    --------
    dict
        行业名称到价格数据的映射
    """
    price_data = {}
    for industry in SW_INDUSTRIES.keys():
        if industry == '综合':
            continue
        df = _load_industry_data(industry)
        if not df.empty:
            if isinstance(df.index, pd.DatetimeIndex):
                start_dt = pd.to_datetime(start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:])
                end_dt = pd.to_datetime(end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:])
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if not df.empty:
                price_data[industry] = df

    return price_data

def calculate_returns_from_prices(price_data):
    """
    从价格数据计算收益率

    Parameters:
    -----------
    price_data : dict
        行业名称到价格数据的映射

    Returns:
    --------
    pd.DataFrame
        收益率矩阵
    """
    returns_dict = {}
    for industry, df in price_data.items():
        if 'close' in df.columns:
            returns_dict[industry] = df['close'].pct_change().dropna()

    if returns_dict:
        returns_df = pd.DataFrame(returns_dict)
        return returns_df
    return pd.DataFrame()

def get_sw_industry_returns(start_date=None, end_date=None):
    """
    获取申万行业收益率矩阵

    Parameters:
    -----------
    start_date : str, optional
        开始日期，格式YYYY-MM-DD
    end_date : str, optional
        结束日期，格式YYYY-MM-DD

    Returns:
    --------
    pd.DataFrame
        申万行业收益率矩阵
    """
    returns_df = _load_all_returns()

    if not returns_df.empty:
        if start_date:
            returns_df = returns_df[returns_df.index >= start_date]
        if end_date:
            returns_df = returns_df[returns_df.index <= end_date]

    return returns_df

def get_sw_industry_price_data(industry_name):
    """
    获取单个申万行业的价格数据

    Parameters:
    -----------
    industry_name : str
        行业名称

    Returns:
    --------
    pd.DataFrame
        行业价格数据
    """
    return _load_industry_data(industry_name)

def get_index_weights(index_code, trade_date):
    """
    获取指数成分股权重 (暂不支持)

    Parameters:
    -----------
    index_code : str
        指数代码
    trade_date : str
        交易日期

    Returns:
    --------
    pd.DataFrame
        空数据框
    """
    return pd.DataFrame()

def batch_get_stock_daily(ts_codes, start_date, end_date):
    """
    批量获取个股日线数据 (暂不支持)

    Parameters:
    -----------
    ts_codes : list
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    dict
        股票代码到数据的映射
    """
    return {}
