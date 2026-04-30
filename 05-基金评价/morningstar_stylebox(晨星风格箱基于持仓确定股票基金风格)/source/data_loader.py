# -*- coding: utf-8 -*-
"""
晨星风格箱 - 数据获取模块
数据源优先级: efinance → akshare → baostock

核心功能:
1. 获取全A股票实时行情（含总市值）
2. 获取基金持仓数据（季报重仓股）
3. 获取个股财务指标（用于计算价值/成长因子）
4. 计算市值门槛值（MST/LMT）
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


# ==================== 全A股票行情与市值 ====================

def get_all_stock_market_cap() -> pd.DataFrame:
    """
    获取全A股票实时行情（含总市值）
    
    优先级: efinance → akshare
    
    Returns:
        DataFrame with columns: [code, name, market_cap, close_price]
        market_cap单位: 元
    """
    for source in ['efinance', 'akshare']:
        try:
            if source == 'efinance':
                return _get_market_cap_efinance()
            elif source == 'akshare':
                return _get_market_cap_akshare()
        except Exception as e:
            print(f"[{source}] 获取市值数据失败: {e}")
            continue
    raise RuntimeError("所有数据源均获取失败，无法获取股票市值数据")


def _get_market_cap_efinance() -> pd.DataFrame:
    """通过efinance获取全A股票市值"""
    import efinance as ef
    
    # 获取全A股票实时行情
    df = ef.stock.get_realtime_quotes()
    
    if df is None or df.empty:
        raise RuntimeError("efinance返回空数据")
    
    # 统一列名
    result = pd.DataFrame()
    result['code'] = df['股票代码'].values
    result['name'] = df['股票名称'].values
    result['close_price'] = pd.to_numeric(df['最新价'], errors='coerce').values
    result['market_cap'] = pd.to_numeric(df['总市值'], errors='coerce').values
    
    # 过滤: 仅保留沪深A股（6/0/3开头）
    result = result[result['code'].str.match(r'^[036]')]
    
    # 过滤: 去除ST、退市
    result = result[~result['name'].str.contains('ST|退', na=False)]
    
    # 过滤: 去除市值为0或NaN
    result = result[result['market_cap'] > 0].dropna(subset=['market_cap'])
    
    result = result.reset_index(drop=True)
    print(f"[efinance] 成功获取 {len(result)} 只A股市值数据")
    return result


def _get_market_cap_akshare() -> pd.DataFrame:
    """通过akshare获取全A股票市值"""
    import akshare as ak
    
    df = ak.stock_zh_a_spot_em()
    
    if df is None or df.empty:
        raise RuntimeError("akshare返回空数据")
    
    result = pd.DataFrame()
    result['code'] = df['代码'].values
    result['name'] = df['名称'].values
    result['close_price'] = pd.to_numeric(df['最新价'], errors='coerce').values
    
    # 总市值: akshare单位通常是元
    if '总市值' in df.columns:
        result['market_cap'] = pd.to_numeric(df['总市值'], errors='coerce').values
    else:
        # 用 流通市值 * 1.2 近似
        result['market_cap'] = pd.to_numeric(df.get('流通市值', 0), errors='coerce').values * 1.2
    
    # 过滤
    result = result[result['code'].str.match(r'^[036]')]
    result = result[~result['name'].str.contains('ST|退', na=False)]
    result = result[result['market_cap'] > 0].dropna(subset=['market_cap'])
    
    result = result.reset_index(drop=True)
    print(f"[akshare] 成功获取 {len(result)} 只A股市值数据")
    return result


# ==================== 基金持仓数据 ====================

def get_fund_holdings(fund_code: str, date: str = None) -> pd.DataFrame:
    """
    获取基金持仓数据（季报重仓股）
    
    Args:
        fund_code: 基金代码，如 '021181'
        date: 报告期，如 '2024Q4'，默认取最新
    
    Returns:
        DataFrame with columns: [stock_code, stock_name, pct, holding_value]
        pct: 占净值比例(%)
    """
    for source in ['efinance', 'akshare']:
        try:
            if source == 'efinance':
                return _get_holdings_efinance(fund_code, date)
            elif source == 'akshare':
                return _get_holdings_akshare(fund_code, date)
        except Exception as e:
            print(f"[{source}] 获取基金{fund_code}持仓失败: {e}")
            continue
    raise RuntimeError(f"无法获取基金 {fund_code} 持仓数据")


def _get_holdings_efinance(fund_code: str, date: str = None) -> pd.DataFrame:
    """通过efinance获取基金持仓"""
    import efinance as ef
    
    df = ef.fund.get_invest_position(fund_code=fund_code)
    
    if df is None or df.empty:
        raise RuntimeError("efinance返回空持仓数据")
    
    result = pd.DataFrame()
    result['stock_code'] = df['股票代码'].values
    result['stock_name'] = df['股票名称'].values
    result['pct'] = pd.to_numeric(df['占净值比例'], errors='coerce').values
    result['holding_value'] = pd.to_numeric(df.get('持仓市值', 0), errors='coerce').values
    
    # 去除空行
    result = result.dropna(subset=['stock_code', 'pct'])
    result = result[result['pct'] > 0].reset_index(drop=True)
    
    print(f"[efinance] 基金{fund_code}获取到 {len(result)} 只重仓股")
    return result


def _get_holdings_akshare(fund_code: str, date: str = None) -> pd.DataFrame:
    """通过akshare获取基金持仓"""
    import akshare as ak
    
    df = ak.fund_portfolio_hold_em(symbol=fund_code, date=date or "")
    
    if df is None or df.empty:
        raise RuntimeError("akshare返回空持仓数据")
    
    result = pd.DataFrame()
    result['stock_code'] = df['股票代码'].values if '股票代码' in df.columns else df.get('代码', '').values
    result['stock_name'] = df['股票名称'].values if '股票名称' in df.columns else df.get('名称', '').values
    result['pct'] = pd.to_numeric(df.get('占净值比例', df.get('持仓占比', 0)), errors='coerce').values
    result['holding_value'] = pd.to_numeric(df.get('持仓市值', 0), errors='coerce').values
    
    result = result.dropna(subset=['stock_code', 'pct'])
    result = result[result['pct'] > 0].reset_index(drop=True)
    
    print(f"[akshare] 基金{fund_code}获取到 {len(result)} 只重仓股")
    return result


# ==================== 个股财务指标 ====================

def get_stock_financial_data(stock_codes: List[str]) -> pd.DataFrame:
    """
    获取个股财务指标（用于计算价值/成长因子）
    
    需要的指标:
    - EPS (每股收益)
    - BVPS (每股净资产)
    - 营业收入 / 总股本 (每股营收)
    - 经营现金流 / 总股本 (每股现金流)
    - DPS (每股分红)
    - EPS增长率
    - BVPS增长率
    - 营收增长率
    
    Args:
        stock_codes: 股票代码列表
    
    Returns:
        DataFrame with financial indicators per stock
    """
    # 尝试批量获取
    try:
        return _get_financials_akshare(stock_codes)
    except Exception as e:
        print(f"[akshare] 获取财务数据失败: {e}，使用efinance备用")
    
    try:
        return _get_financials_efinance(stock_codes)
    except Exception as e:
        print(f"[efinance] 获取财务数据也失败: {e}")
        raise RuntimeError("无法获取财务指标数据")


def _get_financials_akshare(stock_codes: List[str]) -> pd.DataFrame:
    """通过akshare获取关键财务指标"""
    import akshare as ak
    
    records = []
    
    # 使用 stock_a_indicator_lg 获取个股指标
    for code in stock_codes[:30]:  # 限制数量避免超频
        try:
            # 获取个股基本面指标
            df = ak.stock_a_indicator_lg(symbol=code)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                records.append({
                    'code': code,
                    'pe_ttm': latest.get('pe_ttm', np.nan),      # 市盈率TTM
                    'pb': latest.get('pb', np.nan),               # 市净率
                    'ps_ttm': latest.get('ps_ttm', np.nan),      # 市销率TTM
                    'dv_ratio': latest.get('dv_ratio', np.nan),  # 股息率(%)
                    'dv_ttm': latest.get('dv_ttm', np.nan),      # 股息TTM
                    'total_mv': latest.get('total_mv', np.nan),  # 总市值(万元)
                })
        except Exception as e:
            records.append({'code': code})
            continue
    
    result = pd.DataFrame(records)
    print(f"[akshare] 获取到 {len(result)} 只股票的财务指标")
    return result


def _get_financials_efinance(stock_codes: List[str]) -> pd.DataFrame:
    """通过efinance获取关键指标（使用实时行情的PE/PB等）"""
    import efinance as ef
    
    records = []
    
    # 获取实时行情（含PE/PB）
    df = ef.stock.get_realtime_quotes(stock_codes)
    
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            records.append({
                'code': row.get('股票代码', ''),
                'pe_ttm': pd.to_numeric(row.get('市盈率-动态', np.nan), errors='coerce'),
                'pb': pd.to_numeric(row.get('市净率', np.nan), errors='coerce'),
                'ps_ttm': np.nan,
                'dv_ratio': pd.to_numeric(row.get('股息率TTM', np.nan), errors='coerce'),
            })
    
    result = pd.DataFrame(records)
    print(f"[efinance] 获取到 {len(result)} 只股票的指标")
    return result


# ==================== 市值门槛值计算 ====================

def calculate_market_cap_thresholds(market_caps: pd.Series) -> Dict[str, float]:
    """
    计算市值门槛值
    
    根据研报定义:
    - LMT(大中盘门限): 累计市值占比达到70%时的股票市值
    - MST(中小盘门限): 累计市值占比达到90%时的股票市值
    
    计算方法:
    1. 将所有股票按市值从大到小排序
    2. 计算累计市值占比
    3. 找到累计占比>=70%的边界股票市值 → LMT
    4. 找到累计占比>=90%的边界股票市值 → MST
    
    Args:
        market_caps: 股票总市值序列（单位：元）
    
    Returns:
        {'LMT': float, 'MST': float}
    """
    sorted_caps = market_caps.dropna().sort_values(ascending=False)
    total_cap = sorted_caps.sum()
    
    if total_cap == 0:
        return {'LMT': 1e10, 'MST': 1e9}  # 默认值
    
    cumsum_ratio = sorted_caps.cumsum() / total_cap
    
    # LMT: 累计市值占比>=70%的最小市值
    lmt_mask = cumsum_ratio >= 0.70
    if lmt_mask.any():
        lmt = sorted_caps[lmt_mask].iloc[-1]
    else:
        lmt = sorted_caps.min()
    
    # MST: 累计市值占比>=90%的最小市值
    mst_mask = cumsum_ratio >= 0.90
    if mst_mask.any():
        mst = sorted_caps[mst_mask].iloc[-1]
    else:
        mst = sorted_caps.min()
    
    return {'LMT': float(lmt), 'MST': float(mst)}


# ==================== 价值-成长因子门限计算 ====================

def calculate_vg_thresholds(stock_data: pd.DataFrame, size_group: str = 'all') -> Dict[str, float]:
    """
    计算价值-成长因子的门限值VT和GT
    
    根据研报:
    - 按规模分类（大盘/中盘/小盘）分别计算VCG的门限
    - VT: VCG分布的33.3%分位数（价值-混合门限）
    - GT: VCG分布的66.7%分位数（混合-成长门限）
    
    Args:
        stock_data: 包含VCG得分的股票数据
        size_group: 规模分组 'large'/'mid'/'small'/'all'
    
    Returns:
        {'VT': float, 'GT': float}
    """
    if 'vcg_score' not in stock_data.columns:
        # 如果没有VCG得分，使用默认门限值
        return {'VT': -0.5, 'GT': 0.5}
    
    vcg = stock_data['vcg_score'].dropna()
    
    if len(vcg) < 10:
        return {'VT': -0.5, 'GT': 0.5}
    
    vt = vcg.quantile(1/3)
    gt = vcg.quantile(2/3)
    
    return {'VT': float(vt), 'GT': float(gt)}


if __name__ == '__main__':
    # 测试
    print("=" * 50)
    print("数据获取模块测试")
    print("=" * 50)
    
    # 测试市值数据
    df = get_all_stock_market_cap()
    print(f"\n全A股票数量: {len(df)}")
    print(df.head())
    
    # 测试门槛计算
    thresholds = calculate_market_cap_thresholds(df['market_cap'])
    print(f"\nLMT(大中盘门限): {thresholds['LMT']/1e8:.2f} 亿")
    print(f"MST(中小盘门限): {thresholds['MST']/1e8:.2f} 亿")
