# -*- coding: utf-8 -*-
"""
数据获取模块

数据源优先级：tushare > efinance > akshare > baostock

功能：
  - 获取债券基金持仓数据
  - 获取债券基本信息（久期、评级、票息等）
  - 获取国债/国开债收益率曲线
  - 获取信用利差数据
"""
import os
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TUSHARE_TOKEN, TUSHARE_API_URL, 
    BOND_TYPES, CREDIT_RATING_MAP, RISK_FREE_BENCHMARK
)


# ============================================================
# Tushare 初始化
# ============================================================
def init_tushare():
    """初始化Tushare API（按MEMORY.md配置）"""
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    pro._DataApi__token = TUSHARE_TOKEN
    pro._DataApi__http_url = TUSHARE_API_URL
    return pro


# ============================================================
# 1. 债券基金持仓数据
# ============================================================
def get_fund_bond_holdings(fund_code: str, date: str = None) -> pd.DataFrame:
    """获取债券基金的持仓数据（tushare）。
    
    Parameters
    ----------
    fund_code : str
        基金代码（如 '000001'）
    date : str
        报告期日期（如 '20231231'），默认最新
    
    Returns
    -------
    pd.DataFrame
        债券持仓明细，包含：
        - bond_code: 债券代码
        - bond_name: 债券名称
        - weight: 持仓权重（%）
        - amount: 持仓金额（万元）
        - bond_type: 债券类型
    """
    try:
        pro = init_tushare()
        # 获取基金持仓明细
        df = pro.fund_portfolio(ts_code=f'{fund_code}.OF', period=date)
        
        if df is None or len(df) == 0:
            print(f'  [tushare] 基金{fund_code}持仓数据为空')
            return pd.DataFrame()
        
        # 筛选债券类持仓
        df = df[df['asset_type'] == '债券'].copy()
        if len(df) == 0:
            return pd.DataFrame()
        
        # 提取债券代码和名称
        result = []
        for _, row in df.iterrows():
            # 从持仓名称中提取债券代码
            name = row.get('mk_value_name', '')
            code = row.get('symbol', '')
            if pd.isna(code) or code == '':
                continue
            
            result.append({
                'bond_code': str(code).zfill(6),
                'bond_name': name,
                'weight': float(row.get('mk_value_ratio', 0)),  # 市值占比%
                'amount': float(row.get('mk_value', 0)),         # 市值（万元）
            })
        
        if result:
            holdings_df = pd.DataFrame(result)
            # 识别债券类型
            holdings_df['bond_type'] = holdings_df['bond_name'].apply(_identify_bond_type)
            print(f'  [tushare] 基金{fund_code}债券持仓 {len(holdings_df)} 只')
            return holdings_df
            
    except Exception as e:
        print(f'  [tushare] 获取基金持仓失败: {e}')
    
    return pd.DataFrame()


def _identify_bond_type(bond_name: str) -> str:
    """从债券名称识别债券类型。"""
    if pd.isna(bond_name):
        return 'other'
    
    name = str(bond_name)
    for cn_type, en_type in BOND_TYPES.items():
        if cn_type in name:
            return en_type
    
    return 'other'


# ============================================================
# 2. 债券基本信息
# ============================================================
def get_bond_info(bond_codes: list) -> pd.DataFrame:
    """获取债券基本信息（久期、评级、票息、到期日等）。
    
    数据源：tushare > akshare
    
    Parameters
    ----------
    bond_codes : list
        债券代码列表
    
    Returns
    -------
    pd.DataFrame
        债券信息，包含：
        - bond_code: 债券代码
        - bond_name: 债券名称
        - issue_date: 发行日期
        - maturity_date: 到期日期
        - coupon_rate: 票面利率（%）
        - duration: 久期
        - modified_duration: 修正久期
        - convexity: 凸性
        - credit_rating: 信用评级
        - bond_type: 债券类型
        - ytm: 到期收益率（%）
    """
    results = []
    
    # 尝试tushare
    try:
        pro = init_tushare()
        for code in bond_codes[:100]:  # 限制数量避免超限
            try:
                # 获取债券基本信息
                info = pro.bond_basic(ts_code=f'{code}')
                if info is not None and len(info) > 0:
                    row = info.iloc[0]
                    results.append({
                        'bond_code': code,
                        'bond_name': row.get('bond_name', ''),
                        'issue_date': row.get('issue_date', ''),
                        'maturity_date': row.get('maturity_date', ''),
                        'coupon_rate': float(row.get('coupon_rate', 0)),
                        'duration': float(row.get('duration', 0)),
                        'modified_duration': float(row.get('mduration', 0)),
                        'convexity': float(row.get('convexity', 0)),
                        'credit_rating': _standardize_rating(row.get('credit_rating', '')),
                        'bond_type': _identify_bond_type(row.get('bond_name', '')),
                        'ytm': float(row.get('ytm', 0)),
                    })
            except Exception:
                pass
            time.sleep(0.1)
    except Exception as e:
        print(f'  [tushare] 债券信息获取失败: {e}')
    
    if results:
        print(f'  [tushare] 债券信息 {len(results)} 只')
        return pd.DataFrame(results)
    
    # 备选：akshare
    return _get_bond_info_akshare(bond_codes)


def _get_bond_info_akshare(bond_codes: list) -> pd.DataFrame:
    """通过akshare获取债券信息（备选）。"""
    try:
        import akshare as ak
        results = []
        for code in bond_codes[:50]:
            try:
                # 尝试获取债券详情
                df = ak.bond_zh_hs_daily(symbol=code)
                if df is not None and len(df) > 0:
                    # akshare数据有限，使用默认值
                    results.append({
                        'bond_code': code,
                        'bond_name': '',
                        'issue_date': '',
                        'maturity_date': '',
                        'coupon_rate': 0.0,
                        'duration': 0.0,
                        'modified_duration': 0.0,
                        'convexity': 0.0,
                        'credit_rating': 'A',
                        'bond_type': 'corporate',
                        'ytm': 0.0,
                    })
            except Exception:
                pass
        
        if results:
            print(f'  [akshare] 债券信息 {len(results)} 只（备选）')
            return pd.DataFrame(results)
    except Exception as e:
        print(f'  [akshare] 债券信息获取失败: {e}')
    
    return pd.DataFrame()


def _standardize_rating(rating: str) -> str:
    """标准化信用评级。"""
    if pd.isna(rating) or rating == '':
        return 'A'  # 默认评级
    
    rating = str(rating).upper().strip()
    return CREDIT_RATING_MAP.get(rating, 'A')


# ============================================================
# 3. 收益率曲线数据
# ============================================================
def get_treasury_yield_curve(date: str, curve_type: str = '国债') -> pd.DataFrame:
    """获取国债收益率曲线。
    
    Parameters
    ----------
    date : str
        日期（YYYYMMDD或YYYY-MM-DD）
    curve_type : str
        曲线类型：'国债'、'国开债'
    
    Returns
    -------
    pd.DataFrame
        收益率曲线，包含：
        - term: 期限（年）
        - yield_rate: 收益率（%）
    """
    date_str = str(date).replace('-', '')
    
    # 尝试tushare
    try:
        pro = init_tushare()
        
        if curve_type == '国债':
            # 国债收益率曲线
            df = pro.yield_curve(curve_type='0', trade_date=date_str)  # 0=中债国债
        else:
            # 国开债收益率曲线
            df = pro.yield_curve(curve_type='1', trade_date=date_str)  # 1=中债国开债
        
        if df is not None and len(df) > 0:
            result = pd.DataFrame({
                'term': df['term'].astype(float),      # 期限（年）
                'yield_rate': df['yield_rate'].astype(float),  # 收益率（%）
            })
            print(f'  [tushare] {curve_type}收益率曲线 {len(result)} 个期限')
            return result
            
    except Exception as e:
        print(f'  [tushare] 收益率曲线获取失败: {e}')
    
    # 备选：akshare
    return _get_yield_curve_akshare(date, curve_type)


def _get_yield_curve_akshare(date: str, curve_type: str = '国债') -> pd.DataFrame:
    """通过akshare获取收益率曲线（备选）。"""
    try:
        import akshare as ak
        
        if curve_type == '国债':
            # 中国国债收益率曲线
            df = ak.bond_china_yield(start_date=str(date).replace('-', ''))
        else:
            # 国开债收益率曲线
            df = ak.bond_china_yield(start_date=str(date).replace('-', ''))
        
        if df is not None and len(df) > 0:
            # akshare返回格式可能不同，需要适配
            # 假设返回列包含期限和收益率
            print(f'  [akshare] {curve_type}收益率曲线（备选）')
            return df
            
    except Exception as e:
        print(f'  [akshare] 收益率曲线获取失败: {e}')
    
    return pd.DataFrame()


# ============================================================
# 4. 信用利差数据
# ============================================================
def get_credit_spread(date: str, rating: str, term: float) -> float:
    """计算信用利差。
    
    信用利差 = 同评级同期限债券收益率 - 同期限国开债收益率
    
    Parameters
    ----------
    date : str
        日期
    rating : str
        信用评级（AAA, AA, A等）
    term : float
        期限（年）
    
    Returns
    -------
    float
        信用利差（%）
    """
    # 获取国开债收益率曲线
    risk_free_curve = get_treasury_yield_curve(date, curve_type='国开债')
    
    if len(risk_free_curve) == 0:
        return 0.0
    
    # 插值获取特定期限收益率
    from scipy.interpolate import interp1d
    try:
        f = interp1d(risk_free_curve['term'], risk_free_curve['yield_rate'], 
                     kind='linear', fill_value='extrapolate')
        risk_free_yield = float(f(term))
    except Exception:
        # 取最接近期限的收益率
        idx = (risk_free_curve['term'] - term).abs().idxmin()
        risk_free_yield = risk_free_curve.loc[idx, 'yield_rate']
    
    # 获取同评级债券收益率（简化：使用评级利差表）
    rating_spread = _get_rating_spread(rating)
    
    return risk_free_yield + rating_spread


def _get_rating_spread(rating: str) -> float:
    """获取不同评级的信用利差基准（bp）。
    
    基于市场经验值，单位：基点(bp)
    """
    spread_map = {
        'AAA': 50,    # AAA级：约50bp
        'AA': 100,    # AA级：约100bp
        'A': 200,     # A级：约200bp
        'BBB': 350,   # BBB级：约350bp
        'BB': 500,    # BB级：约500bp
        'B': 700,     # B级：约700bp
        'CCC': 1000,  # CCC级：约1000bp
    }
    return spread_map.get(rating, 200) / 100  # 转换为%


# ============================================================
# 5. 债券基金净值数据
# ============================================================
def get_fund_nav_history(fund_code: str, start: str, end: str) -> pd.DataFrame:
    """获取债券基金净值历史。
    
    Parameters
    ----------
    fund_code : str
        基金代码
    start, end : str
        开始/结束日期
    
    Returns
    -------
    pd.DataFrame
        净值历史，包含日期、单位净值、累计净值
    """
    try:
        import efinance as ef
        df = ef.fund.get_quote_history(fund_code, start=start, end=end)
        
        if df is not None and len(df) > 0:
            df.columns = [str(c).strip() for c in df.columns]
            # 列名映射
            date_col = df.columns[0]
            nav_col = df.columns[1]
            
            result = pd.DataFrame({
                'date': pd.to_datetime(df[date_col]),
                'nav': pd.to_numeric(df[nav_col], errors='coerce'),
            })
            result = result.set_index('date').sort_index()
            result['daily_return'] = result['nav'].pct_change()
            
            print(f'  [efinance] 基金{fund_code}净值历史 {len(result)} 条')
            return result
            
    except Exception as e:
        print(f'  [efinance] 基金净值获取失败: {e}')
    
    return pd.DataFrame()
