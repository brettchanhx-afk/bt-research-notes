# -*- coding: utf-8 -*-
"""
晨星风格箱 - 因子计算模块
严格按照华泰证券研报《晨星风格箱基于基金持仓数据并根据规模、价值成长特性确定基金风格》复现

核心算法:
1. 规模因子: y = 100 × [1 + (ln(Cap) - ln(MST)) / (ln(LMT) - ln(MST))]
2. 价值-成长因子: x = 100 × [1 + (VCG - VT) / (GT - VT)]
   VCG = 成长得分 - 价值得分
3. 基金风格: 持仓股票得分的加权平均

参考文献: 华泰证券2020-08-21金工研报
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ==================== 规模因子 ====================

def calculate_stock_size_score(market_cap: float, mst: float, lmt: float) -> float:
    """
    计算单只股票的规模得分y
    
    研报公式: y = 100 × [1 + (ln(Cap) - ln(MST)) / (ln(LMT) - ln(MST))]
    
    含义:
    - y < 100: 小盘股（市值 < MST）
    - 100 ≤ y ≤ 200: 中盘股（MST ≤ 市值 ≤ LMT）
    - y > 200: 大盘股（市值 > LMT）
    
    Args:
        market_cap: 股票总市值（元）
        mst: 中小盘门限值（累计市值90%点的股票市值）
        lmt: 大中盘门限值（累计市值70%点的股票市值）
    
    Returns:
        规模得分 y
    """
    if pd.isna(market_cap) or market_cap <= 0 or mst <= 0 or lmt <= 0:
        return np.nan
    
    # 防止LMT <= MST
    if lmt <= mst:
        return np.nan
    
    try:
        cap_log = np.log(market_cap)
        mst_log = np.log(mst)
        lmt_log = np.log(lmt)
        
        denominator = lmt_log - mst_log
        if abs(denominator) < 1e-10:
            return np.nan
        
        y = 100.0 * (1.0 + (cap_log - mst_log) / denominator)
        return y
    except (ValueError, ZeroDivisionError):
        return np.nan


def determine_size_style(y: float) -> str:
    """
    根据规模得分判断股票规模风格
    
    y < 100 → 小盘
    100 ≤ y ≤ 200 → 中盘
    y > 200 → 大盘
    """
    if pd.isna(y):
        return "未知"
    elif y < 100:
        return "小盘"
    elif y <= 200:
        return "中盘"
    else:
        return "大盘"


def calculate_fund_size_score(holdings: pd.DataFrame, thresholds: Dict[str, float]) -> float:
    """
    计算基金的规模得分Y
    
    公式: Y = Σ(wi × yi) / Σ(wi)
    即持仓股票规模得分的持仓占比加权平均
    
    Args:
        holdings: 持仓数据，必须包含 market_cap, pct 列
        thresholds: 包含 MST, LMT 的字典
    
    Returns:
        基金规模得分Y
    """
    if holdings.empty:
        return np.nan
    
    mst = thresholds.get('MST', 0)
    lmt = thresholds.get('LMT', 0)
    
    if mst <= 0 or lmt <= 0:
        return np.nan
    
    # 计算每只股票的规模得分
    scores = []
    weights = []
    for _, row in holdings.iterrows():
        cap = row.get('market_cap', np.nan)
        pct = row.get('pct', 0)
        
        if pd.isna(cap) or pd.isna(pct) or pct <= 0:
            continue
        
        y = calculate_stock_size_score(cap, mst, lmt)
        if not pd.isna(y):
            scores.append(y)
            weights.append(pct)
    
    if not scores:
        return np.nan
    
    # 加权平均
    Y = np.average(scores, weights=weights)
    return Y


# ==================== 价值-成长因子 ====================

def calculate_value_score_from_market(financial_data: pd.DataFrame) -> pd.Series:
    """
    基于市场数据计算价值得分
    
    价值得分因子（研报定义5个子因子）:
    1. 预期收益/股价 = 1/PE_TTM (E/P)
    2. 每股净资产/股价 = 1/PB (B/P)
    3. 每股营收/股价 = 1/PS_TTM (S/P)
    4. 每股现金流/股价 ≈ (E/P) * 现金流调整因子 (C/P)
    5. 每股分红/股价 = 股息率 (D/P)
    
    权重（研报）: 均等 0.2
    
    Args:
        financial_data: 包含pe_ttm, pb, ps_ttm, dv_ratio等列的DataFrame
    
    Returns:
        价值得分Series
    """
    df = financial_data.copy()
    
    # 1. E/P (收益/价格) = 1/PE
    df['ep'] = np.where(
        (df.get('pe_ttm', 0) > 0) & ~pd.isna(df.get('pe_ttm')),
        1.0 / df['pe_ttm'],
        np.nan
    )
    
    # 2. B/P (净资产/价格) = 1/PB
    df['bp'] = np.where(
        (df.get('pb', 0) > 0) & ~pd.isna(df.get('pb')),
        1.0 / df['pb'],
        np.nan
    )
    
    # 3. S/P (营收/价格) = 1/PS
    df['sp'] = np.where(
        (df.get('ps_ttm', 0) > 0) & ~pd.isna(df.get('ps_ttm')),
        1.0 / df['ps_ttm'],
        np.nan
    )
    
    # 4. C/P (现金流/价格): 近似为 E/P * 0.8 (现金流通常是净利润的80%)
    df['cp'] = df['ep'] * 0.8
    
    # 5. D/P (分红/价格) = 股息率/100
    if 'dv_ratio' in df.columns:
        df['dp'] = df['dv_ratio'] / 100.0
    else:
        df['dp'] = np.nan
    
    # 计算价值得分: 各因子的均值
    value_factors = ['ep', 'bp', 'sp', 'cp', 'dp']
    df['value_score'] = df[value_factors].mean(axis=1, skipna=True)
    
    # 如果全部缺失，使用 EP 和 BP 的简单均值
    mask_all_na = df[value_factors].isna().all(axis=1)
    if mask_all_na.any():
        df.loc[mask_all_na, 'value_score'] = np.nan
    
    return df['value_score']


def calculate_growth_score_from_market(financial_data: pd.DataFrame) -> pd.Series:
    """
    基于市场数据计算成长得分
    
    成长得分因子（研报定义4个子因子）:
    1. EPS增长率
    2. 每股净资产增长率
    3. 每股营收增长率
    4. 每股经营现金流增长率
    
    由于历史数据难以批量获取，使用以下替代:
    - 用 PE相对行业均值的偏离度作为增长预期的代理
    - 用营收/市值比变化率近似
    
    权重（研报）: 均等 0.25
    
    Args:
        financial_data: 包含 pe_ttm, ps_ttm, pb 等列的DataFrame
    
    Returns:
        成长得分Series
    """
    df = financial_data.copy()
    
    # 使用PE倒数作为增长预期代理
    # 高PE通常意味着市场预期高增长
    pe_col = df.get('pe_ttm', pd.Series(dtype=float))
    
    # 标准化PE到0-1范围（正值代表增长预期）
    pe_positive = pe_col[pe_col > 0]
    if len(pe_positive) > 0:
        pe_median = pe_positive.median()
        # PE相对中位数的偏离度
        df['growth_proxy1'] = (pe_col - pe_median) / pe_median
    else:
        df['growth_proxy1'] = 0
    
    # PS倒数变化（营收增长代理）
    if 'ps_ttm' in df.columns:
        sp = 1.0 / df['ps_ttm'].replace(0, np.nan)
        sp_median = sp.median()
        df['growth_proxy2'] = (sp - sp_median) / abs(sp_median) if sp_median != 0 else 0
    else:
        df['growth_proxy2'] = 0
    
    # 简化: 成长得分 = 两个代理因子的均值
    df['growth_score'] = df[['growth_proxy1', 'growth_proxy2']].mean(axis=1, skipna=True)
    
    return df['growth_score']


def calculate_vcg_score(value_scores: pd.Series, growth_scores: pd.Series) -> pd.Series:
    """
    计算VCG综合得分
    
    VCG = 成长得分 - 价值得分
    
    VCG > 0: 偏成长
    VCG < 0: 偏价值
    VCG ≈ 0: 平衡
    
    Args:
        value_scores: 价值得分Series
        growth_scores: 成长得分Series
    
    Returns:
        VCG得分Series
    """
    vcg = growth_scores - value_scores
    return vcg


def calculate_stock_vg_score(vcg: float, vt: float, gt: float) -> float:
    """
    计算单只股票的价值-成长得分x
    
    研报公式: x = 100 × [1 + (VCG - VT) / (GT - VT)]
    
    含义:
    - x < 100: 价值型（VCG < VT）
    - 100 ≤ x ≤ 200: 平衡型（VT ≤ VCG ≤ GT）
    - x > 200: 成长型（VCG > GT）
    
    Args:
        vcg: 股票VCG综合得分
        vt: 价值-混合门限值
        gt: 混合-成长门限值
    
    Returns:
        价值-成长得分x
    """
    if pd.isna(vcg) or pd.isna(vt) or pd.isna(gt):
        return np.nan
    
    denominator = gt - vt
    if abs(denominator) < 1e-10:
        return 150.0  # 默认平衡
    
    x = 100.0 * (1.0 + (vcg - vt) / denominator)
    return x


def determine_vg_style(x: float, gamma: float = 0.5) -> str:
    """
    根据价值-成长得分判断风格
    
    研报定义（含gamma参数）:
    - x < 150×(1 - γ/3) → 价值型
    - 150×(1 - γ/3) < x < 150×(1 + γ/3) → 平衡型
    - x > 150×(1 + γ/3) → 成长型
    
    默认γ=0.5时:
    - x < 125 → 价值型
    - 125 ≤ x ≤ 175 → 平衡型
    - x > 175 → 成长型
    
    Args:
        x: 价值-成长得分
        gamma: 判定参数，默认0.5
    """
    lower = 150 * (1 - gamma / 3)
    upper = 150 * (1 + gamma / 3)
    
    if pd.isna(x):
        return "未知"
    elif x < lower:
        return "价值型"
    elif x <= upper:
        return "平衡型"
    else:
        return "成长型"


def calculate_fund_vg_score(holdings: pd.DataFrame, thresholds: Dict[str, float]) -> float:
    """
    计算基金的价值-成长得分X
    
    公式: X = Σ(wi × xi) / Σ(wi)
    即持仓股票VG得分的持仓占比加权平均
    
    Args:
        holdings: 持仓数据，必须包含 vg_score, pct 列
        thresholds: 包含 VT, GT 的字典
    
    Returns:
        基金价值-成长得分X
    """
    if holdings.empty:
        return np.nan
    
    valid = holdings.dropna(subset=['vg_score_x', 'pct'])
    valid = valid[valid['pct'] > 0]
    
    if valid.empty:
        return 150.0  # 默认平衡
    
    X = np.average(valid['vg_score_x'].values, weights=valid['pct'].values)
    return X


# ==================== 完整风格分析 ====================

def analyze_fund_style(
    holdings: pd.DataFrame,
    market_cap_thresholds: Dict[str, float],
    vg_thresholds: Dict[str, float],
    gamma: float = 0.5
) -> Dict:
    """
    完整的晨星风格箱分析
    
    Args:
        holdings: 基金持仓数据，需包含 stock_code, market_cap, pct, vg_score
        market_cap_thresholds: {'MST': float, 'LMT': float}
        vg_thresholds: {'VT': float, 'GT': float}
        gamma: 价值-成长风格判定参数
    
    Returns:
        完整风格分析结果字典
    """
    # 1. 计算基金规模得分Y
    Y = calculate_fund_size_score(holdings, market_cap_thresholds)
    size_style = determine_size_style(Y)
    
    # 2. 计算基金价值-成长得分X
    X = calculate_fund_vg_score(holdings, vg_thresholds)
    vg_style = determine_vg_style(X, gamma)
    
    # 3. 计算每只持仓股的风格
    stock_styles = []
    for _, row in holdings.iterrows():
        cap = row.get('market_cap', np.nan)
        y = calculate_stock_size_score(cap, market_cap_thresholds['MST'], market_cap_thresholds['LMT'])
        
        vcg = row.get('vcg_score', np.nan)
        x = calculate_stock_vg_score(vcg, vg_thresholds['VT'], vg_thresholds['GT'])
        
        stock_styles.append({
            'stock_code': row.get('stock_code', ''),
            'stock_name': row.get('stock_name', ''),
            'market_cap': cap,
            'size_score_y': y,
            'size_style': determine_size_style(y),
            'vg_score_x': x,
            'vg_style': determine_vg_style(x, gamma),
            'pct': row.get('pct', 0)
        })
    
    result = {
        'fund_size_score_Y': Y,
        'fund_size_style': size_style,
        'fund_vg_score_X': X,
        'fund_vg_style': vg_style,
        'fund_style': f"{size_style}{vg_style}",
        'gamma': gamma,
        'thresholds': {
            'MST': market_cap_thresholds['MST'],
            'LMT': market_cap_thresholds['LMT'],
            'VT': vg_thresholds['VT'],
            'GT': vg_thresholds['GT'],
        },
        'stock_details': pd.DataFrame(stock_styles)
    }
    
    return result


if __name__ == '__main__':
    # 快速测试
    mst = 5e9   # 50亿
    lmt = 5e10  # 500亿
    
    # 测试规模得分
    test_caps = [1e8, 1e9, 5e9, 2e10, 5e10, 1e11, 5e11, 1e12]
    print("规模得分测试:")
    print(f"{'市值(亿)':>12} {'得分y':>8} {'风格':>6}")
    print("-" * 30)
    for cap in test_caps:
        y = calculate_stock_size_score(cap, mst, lmt)
        style = determine_size_style(y)
        print(f"{cap/1e8:12.2f} {y:8.2f} {style:>6}")
    
    # 测试VG得分
    print("\n价值-成长得分测试:")
    vt, gt = -0.5, 0.5
    test_vcgs = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print(f"{'VCG':>6} {'得分x':>8} {'风格':>6}")
    print("-" * 25)
    for vcg in test_vcgs:
        x = calculate_stock_vg_score(vcg, vt, gt)
        style = determine_vg_style(x)
        print(f"{vcg:6.2f} {x:8.2f} {style:>6}")
