"""
数据获取模块 - 从各数据源获取市场数据
优先从本地CSV文件读取数据，支持tushare作为备用
"""

import pandas as pd
import numpy as np
import tushare as ts
import os
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'd:/Documents/trae_projects/meso_prosperity_upstream_midstream/data'

_roe_df_cache = None
_industry_indicator_cache = {}

def _get_roe_df():
    """获取ROE数据的缓存DataFrame"""
    global _roe_df_cache
    if _roe_df_cache is None:
        roe_file = os.path.join(DATA_DIR, '行业ROE_TTM历史数据.csv')
        _roe_df_cache = pd.read_csv(roe_file, encoding='gbk')
        _roe_df_cache['date'] = pd.to_datetime(_roe_df_cache['date'])
    return _roe_df_cache

def _get_industry_indicator_df(industry_name):
    """获取行业代理指标数据的缓存DataFrame"""
    global _industry_indicator_cache
    if industry_name not in _industry_indicator_cache:
        indicator_file_map = {
            '石油石化': '石油石化行业中观景气度代理指标.csv',
            '煤炭': '煤炭行业中观景气度代理指标.csv',
            '有色金属': '有色金属行业中观景气度代理指标.csv',
            '钢铁': '钢铁行业中观景气度代理指标.csv',
            '基础化工': '基础化工行业中观景气度代理指标.csv',
            '建材': '建材行业中观景气度代理指标.csv'
        }
        filename = indicator_file_map.get(industry_name)
        if filename:
            indicator_file = os.path.join(DATA_DIR, filename)
            _industry_indicator_cache[industry_name] = pd.read_excel(indicator_file)
        else:
            _industry_indicator_cache[industry_name] = pd.DataFrame()
    return _industry_indicator_cache[industry_name]

def load_roe_from_csv(industry_name: str, start_date: str = '20100101',
                       end_date: str = '20211231') -> pd.DataFrame:
    """
    从CSV文件加载行业ROE_TTM数据

    Parameters:
    -----------
    industry_name : str
        行业名称，如 '石油石化'、'煤炭'、'有色金属'、'钢铁'、'基础化工'、'建材'
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'roe_ttm']
    """
    industry_col_map = {
        '石油石化': '石油石化ROE_TTM',
        '煤炭': '煤炭ROE_TTM',
        '有色金属': '有色金属ROE_TTM',
        '钢铁': '钢铁ROE_TTM',
        '基础化工': '基础化工ROE_TTM',
        '建材': '建材ROE_TTM'
    }

    col_name = industry_col_map.get(industry_name)
    if not col_name:
        print(f"未知的行业: {industry_name}")
        return pd.DataFrame(columns=['trade_date', 'roe_ttm'])

    df = _get_roe_df()

    if col_name not in df.columns:
        print(f"ROE数据中未找到列: {col_name}")
        return pd.DataFrame(columns=['trade_date', 'roe_ttm'])

    result = df[['date', col_name]].copy()
    result.columns = ['trade_date', 'roe_ttm']
    result = result.dropna(subset=['roe_ttm'])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    result = result[(result['trade_date'] >= start_dt) & (result['trade_date'] <= end_dt)]

    result = result.sort_values('trade_date')
    return result.reset_index(drop=True)

def load_indicators_from_csv(industry_name: str, start_date: str = '20100101',
                              end_date: str = '20211231') -> Dict[str, pd.Series]:
    """
    从CSV文件加载行业代理指标数据

    Parameters:
    -----------
    industry_name : str
        行业名称
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD

    Returns:
    --------
    Dict[str, pd.Series] - 指标名称到指标数据的映射
    """
    df = _get_industry_indicator_df(industry_name)

    if df.empty:
        return {}

    result = {}

    date_col = '指标名称' if '指标名称' in df.columns else 'date'

    if date_col not in df.columns:
        return {}

    df[date_col] = pd.to_datetime(df[date_col])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    for col in df.columns:
        if col == date_col:
            continue
        series = df[[date_col, col]].copy()
        series = series[(series[date_col] >= start_dt) & (series[date_col] <= end_dt)]
        series = series.dropna(subset=[col])
        if len(series) > 0:
            series = series.set_index(date_col)[col]
            series.index.name = 'trade_date'
            result[col] = series.sort_index()

    return result

def load_all_roe_from_csv() -> pd.DataFrame:
    """
    从CSV文件加载所有行业的ROE_TTM数据

    Returns:
    --------
    pd.DataFrame with columns: ['date', '石油石化ROE_TTM', '煤炭ROE_TTM', ...]
    """
    return _get_roe_df()

# Tushare初始化
TOKEN = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
PRO = None

def init_tushare():
    """初始化tushare pro接口（使用用户提供的token）"""
    global PRO
    if PRO is None:
        pro = ts.pro_api(TOKEN)
        pro._DataApi__token = TOKEN
        pro._DataApi__http_url = "http://jiaoch.site"
        PRO = pro
    return PRO

def get_industry_roe_ttm(industry_code: str, start_date: str = '20100101',
                         end_date: str = '20211231',
                         use_alternative: bool = False) -> pd.DataFrame:
    """
    获取行业ROE_TTM数据

    Parameters:
    -----------
    industry_code : str
        行业代码，如 '石油石化'、'煤炭'、'有色金属'、'钢铁'、'基础化工'、'建材'
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD
    use_alternative : bool
        是否使用替代方案获取真实数据（通过个股财务数据聚合），
        默认为False直接返回模拟数据，因为API可能不稳定

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'roe_ttm']
    """
    pro = init_tushare()

    industry_map = {
        '石油石化': '石油石化',
        '煤炭': '煤炭开采和洗选业',
        '有色金属': '有色金属矿采选业',
        '钢铁': '黑色金属冶炼及压延加工业',
        '基础化工': '化学原料及化学制品制造业',
        '建材': '非金属矿物制品业'
    }

    code = industry_map.get(industry_code, industry_code)

    try:
        df = pro.sw_industry_daily(
            start_date=start_date,
            end_date=end_date,
            industry=code
        )
        if df is not None and len(df) > 0:
            df = df.sort_values('date')
            df['trade_date'] = pd.to_datetime(df['date'])
            return df[['trade_date', 'roe_ttm']].rename(columns={'date': 'trade_date'})
    except Exception as e:
        print(f"sw_industry_daily接口不可用: {e}")

    if use_alternative:
        try:
            df = _get_industry_roe_alternative(industry_code, start_date, end_date)
            if df is not None and len(df) > 0:
                print(f"使用替代方案获取到{len(df)}条真实数据")
                return df
        except Exception as e2:
            print(f"替代方案失败: {e2}")

    print(f"无法获取{industry_code}行业ROE数据，使用模拟数据")
    return _generate_simulated_roe(industry_code, start_date, end_date)


def _get_industry_roe_alternative(industry_code: str, start_date: str,
                                  end_date: str) -> pd.DataFrame:
    """
    替代方案：通过个股财务数据聚合计算行业ROE

    Parameters:
    -----------
    industry_code : str
        行业代码
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'roe_ttm']
    """
    pro = init_tushare()

    sw_to_tushare_industry = {
        '石油石化': ['石油加工', '石油开采', '石油贸易'],
        '煤炭': ['煤炭开采', '焦炭加工'],
        '有色金属': ['铜', '铝', '小金属', '铅锌', '黄金'],
        '钢铁': ['普钢', '特种钢', '钢加工'],
        '基础化工': ['化工原料', '化学制药', '农药化肥', '化纤', '塑料', '橡胶'],
        '建材': ['水泥', '玻璃', '其他建材', '矿物制品']
    }

    target_industries = sw_to_tushare_industry.get(industry_code, [])

    if not target_industries:
        return pd.DataFrame()

    df_stocks = pro.stock_basic(exchange='SSE', list_status='L',
                                fields='ts_code,name,industry')

    target_stocks = df_stocks[df_stocks['industry'].isin(target_industries)]

    if len(target_stocks) == 0:
        return pd.DataFrame()

    stock_codes = target_stocks['ts_code'].tolist()[:5]

    all_roe_data = []
    for code in stock_codes:
        try:
            df_fina = pro.fina_indicator(
                ts_code=code,
                start_date=start_date,
                end_date=end_date
            )
            if df_fina is not None and len(df_fina) > 0:
                df_fina = df_fina[['end_date', 'roe']].dropna()
                df_fina['ts_code'] = code
                all_roe_data.append(df_fina)
        except:
            continue

    if not all_roe_data:
        return pd.DataFrame()

    df_roe = pd.concat(all_roe_data, ignore_index=True)

    df_roe['trade_date'] = pd.to_datetime(df_roe['end_date'], format='%Y%m%d')
    df_roe = df_roe.groupby('trade_date')['roe'].mean().reset_index()
    df_roe.columns = ['trade_date', 'roe_ttm']
    df_roe = df_roe.sort_values('trade_date')

    return df_roe


def _generate_simulated_roe(industry_code: str, start_date: str,
                            end_date: str) -> pd.DataFrame:
    """
    生成模拟ROE数据（当无法获取真实数据时）

    Parameters:
    -----------
    industry_code : str
        行业代码
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'roe_ttm']
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    dates = pd.date_range(start=start, end=end, freq='Q')

    np.random.seed(hash(industry_code) % (2**32))

    base_roe = {
        '石油石化': 8.0,
        '煤炭': 10.0,
        '有色金属': 7.0,
        '钢铁': 6.0,
        '基础化工': 9.0,
        '建材': 5.0
    }.get(industry_code, 7.0)

    trend = np.linspace(0, 2, len(dates))
    seasonal = np.sin(np.arange(len(dates)) * np.pi / 4) * 1.5
    noise = np.random.randn(len(dates)) * 1.5

    roe_values = base_roe + trend + seasonal + noise
    roe_values = np.clip(roe_values, 1.0, 25.0)

    df = pd.DataFrame({
        'trade_date': dates,
        'roe_ttm': roe_values
    })

    print(f"为{industry_code}生成{len(df)}个季度的模拟ROE数据")
    return df

def get_macro_ppi(start_date: str = '20100101', end_date: str = '20211231') -> pd.DataFrame:
    """
    获取PPI数据

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'ppi']
    """
    pro = init_tushare()

    try:
        df = pro.cn_ppi(start_month=start_date[:6], end_month=end_date[:6])
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['month'], format='%Y%m')
            return df[['trade_date', 'ppi']]
    except Exception as e:
        print(f"获取PPI数据失败: {e}")

    return pd.DataFrame(columns=['trade_date', 'ppi'])

def get_market_indicator(indicator_code: str, start_date: str = '20100101',
                         end_date: str = '20211231') -> pd.DataFrame:
    """
    获取宏观经济指标数据

    Parameters:
    -----------
    indicator_code : str
        指标代码，如 'M0', 'M1', 'M2', 'GDP', 'PMI'等
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
    """
    pro = init_tushare()

    macro_map = {
        'M0': 'M0',
        'M1': 'M1',
        'M2': 'M2',
        'GDP': 'GDP',
        'GDP_CPI': 'GDP',
        'PMI': 'PMI',
        'PPI': 'PPI',
        'CPI': 'CPI'
    }

    try:
        if indicator_code in ['M0', 'M1', 'M2']:
            df = pro.cn_m(indicator=indicator_code, start_date=start_date, end_date=end_date)
        elif indicator_code == 'GDP':
            df = pro.cn_gdp(start_q='2010q1', end_q='2021q4')
        elif indicator_code == 'PMI':
            df = pro.cn_pmi(start_month=start_date[:6], end_month=end_date[:6])
        elif indicator_code in ['PPI', 'CPI']:
            df = pro.cn_ppi(start_month=start_date[:6], end_month=end_date[:6])
        else:
            return pd.DataFrame()

        if df is not None and len(df) > 0:
            if 'month' in df.columns:
                df['trade_date'] = pd.to_datetime(df['month'], format='%Y%m')
            elif 'date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['date'])
            elif 'quarter' in df.columns:
                df['trade_date'] = pd.to_datetime(df['quarter'])
            return df
    except Exception as e:
        print(f"获取{indicator_code}指标数据失败: {e}")

    return pd.DataFrame()

def get_commodity_price(commodity_name: str, start_date: str = '20100101',
                        end_date: str = '20211231') -> pd.DataFrame:
    """
    获取大宗商品价格数据

    Parameters:
    -----------
    commodity_name : str
        商品名称，如 '铜', '铝', '原油', '煤炭'等

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'price']
    """
    pro = init_tushare()

    try:
        if commodity_name in ['铜', '铝', '锌', '铅', '锡', '镍']:
            df = pro.metal(commodity=commodity_name, start_date=start_date, end_date=end_date)
        elif commodity_name == '原油':
            df = pro.cn_oil(indicator='布伦特原油', start_date=start_date, end_date=end_date)
        else:
            return pd.DataFrame()

        if df is not None and len(df) > 0:
            if 'date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        print(f"获取{commodity_name}价格数据失败: {e}")

    return pd.DataFrame()

def get_futures_data(contract: str, start_date: str = '20100101',
                     end_date: str = '20211231') -> pd.DataFrame:
    """
    获取期货数据

    Parameters:
    -----------
    contract : str
        期货合约代码

    Returns:
    --------
    pd.DataFrame
    """
    pro = init_tushare()

    try:
        df = pro.fut_daily(ts_code=contract, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date')
            return df
    except Exception as e:
        print(f"获取期货{contract}数据失败: {e}")

    return pd.DataFrame()

def get_stock_industry_classification() -> pd.DataFrame:
    """
    获取申万行业分类

    Returns:
    --------
    pd.DataFrame
    """
    pro = init_tushare()

    try:
        df = pro.sws_classification(level='L1', src='SW2021')
        return df
    except Exception as e:
        print(f"获取申万行业分类失败: {e}")

    return pd.DataFrame()

def get_index_daily(index_code: str, start_date: str = '20100101',
                    end_date: str = '20211231') -> pd.DataFrame:
    """
    获取指数日线数据

    Parameters:
    -----------
    index_code : str
        指数代码，如 '000300.SH' (沪深300)

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'close', 'pct_chg']
    """
    pro = init_tushare()

    try:
        df = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date')
            return df[['trade_date', 'close', 'pct_chg']]
    except Exception as e:
        print(f"获取指数{index_code}数据失败: {e}")

    return pd.DataFrame()

def get_trade_calendar(start_date: str = '20100101', end_date: str = '20211231') -> pd.DataFrame:
    """
    获取交易日历

    Returns:
    --------
    pd.DataFrame with columns: ['trade_date', 'is_open']
    """
    pro = init_tushare()

    try:
        df = pro.trade_cal(start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['cal_date'], format='%Y%m%d')
            return df[['trade_date', 'is_open']]
    except Exception as e:
        print(f"获取交易日历失败: {e}")

    return pd.DataFrame()

def resample_to_weekly(df: pd.DataFrame, date_col: str = 'trade_date',
                        value_col: str = 'close') -> pd.DataFrame:
    """
    将数据重采样为周频

    Parameters:
    -----------
    df : pd.DataFrame
        原始数据
    date_col : str
        日期列名
    value_col : str or list
        值列名

    Returns:
    --------
    pd.DataFrame
    """
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()

    df = df.set_index(date_col)

    if isinstance(value_col, str):
        value_col = [value_col]

    result = df[value_col].resample('W').last()
    result = result.reset_index()
    result[date_col] = result[date_col].dt.tz_localize(None)

    return result

def resample_to_monthly(df: pd.DataFrame, date_col: str = 'trade_date',
                         value_col: str = 'close') -> pd.DataFrame:
    """
    将数据重采样为月频

    Parameters:
    -----------
    df : pd.DataFrame
        原始数据
    date_col : str
        日期列名
    value_col : str or list
        值列名

    Returns:
    --------
    pd.DataFrame
    """
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()

    df = df.set_index(date_col)

    if isinstance(value_col, str):
        value_col = [value_col]

    result = df[value_col].resample('M').last()
    result = result.reset_index()
    result[date_col] = result[date_col].dt.tz_localize(None)

    return result

class IndustryDataLoader:
    """行业数据加载器 - 从本地CSV文件加载数据"""

    def __init__(self, industry_name: str):
        self.industry_name = industry_name
        self.data_cache = {}

    def load_roe_data(self, start_date: str = '20100101',
                      end_date: str = '20211231') -> pd.DataFrame:
        """从CSV加载行业ROE_TTM数据"""
        key = f'roe_{self.industry_name}_{start_date}_{end_date}'
        if key not in self.data_cache:
            self.data_cache[key] = load_roe_from_csv(
                self.industry_name, start_date, end_date
            )
        return self.data_cache[key]

    def load_all_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有相关指标数据

        Returns:
        --------
        Dict[str, pd.DataFrame]
        """
        indicators = {}
        indicators['roe_ttm'] = self.load_roe_data()
        return indicators

    def load_indicator_data(self, start_date: str = '20100101',
                            end_date: str = '20211231') -> Dict[str, pd.Series]:
        """
        加载行业代理指标数据

        Returns:
        --------
        Dict[str, pd.Series]
        """
        key = f'indicators_{self.industry_name}_{start_date}_{end_date}'
        if key not in self.data_cache:
            self.data_cache[key] = load_indicators_from_csv(
                self.industry_name, start_date, end_date
            )
        return self.data_cache[key]

if __name__ == '__main__':
    print("测试数据获取模块...")

    init_tushare()

    print("\n1. 测试获取交易日历:")
    calendar = get_trade_calendar()
    print(f"交易日历数据量: {len(calendar)}")

    print("\n2. 测试获取沪深300指数数据:")
    index_data = get_index_daily('000300.SH')
    print(f"指数数据量: {len(index_data)}")

    print("\n3. 测试行业ROE数据加载:")
    for industry in ['石油石化', '煤炭', '有色金属', '钢铁', '基础化工', '建材']:
        loader = IndustryDataLoader(industry)
        roe_data = loader.load_roe_data()
        print(f"{industry}: {len(roe_data)} 条记录")
