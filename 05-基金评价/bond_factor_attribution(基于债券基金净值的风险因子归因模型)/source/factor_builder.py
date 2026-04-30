# -*- coding: utf-8 -*-
"""
风险因子构建模块

功能：
  - 构建利率风险因子（久期、凸性）
  - 构建信用风险因子（信用债-国债利差）
  - 构建可转债风险因子
  - 因子收益率计算
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ============================================================
# 1. 利率风险因子
# ============================================================
def build_duration_factor(
    treasury_index: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """构建久期因子。
    
    久期因子 = -Δy × D
    近似 = -国债指数收益率（久期≈1）
    
    Parameters
    ----------
    treasury_index : pd.DataFrame
        国债指数数据
    window : int
        滚动窗口
    
    Returns
    -------
    pd.DataFrame
        久期因子收益序列
    """
    if len(treasury_index) == 0:
        return pd.DataFrame()
    
    df = treasury_index.copy()
    
    # 久期因子：国债收益率变化（取负）
    df['duration_factor'] = -df['daily_return']
    
    # 滚动标准化
    df['duration_factor_std'] = (
        df['duration_factor'] - df['duration_factor'].rolling(window).mean()
    ) / df['duration_factor'].rolling(window).std()
    
    return df[['duration_factor', 'duration_factor_std']]


def build_convexity_factor(
    treasury_index: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """构建凸性因子。
    
    凸性因子 = 0.5 × C × (Δy)²
    近似 = 国债收益率的平方
    
    Parameters
    ----------
    treasury_index : pd.DataFrame
        国债指数数据
    window : int
        滚动窗口
    
    Returns
    -------
    pd.DataFrame
        凸性因子收益序列
    """
    if len(treasury_index) == 0:
        return pd.DataFrame()
    
    df = treasury_index.copy()
    
    # 凸性因子：收益率变化的平方
    df['convexity_factor'] = 0.5 * (df['daily_return'] ** 2)
    
    # 滚动标准化
    df['convexity_factor_std'] = (
        df['convexity_factor'] - df['convexity_factor'].rolling(window).mean()
    ) / df['convexity_factor'].rolling(window).std()
    
    return df[['convexity_factor', 'convexity_factor_std']]


# ============================================================
# 2. 信用风险因子
# ============================================================
def build_credit_factor(
    corporate_index: pd.DataFrame,
    treasury_index: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """构建信用利差因子。
    
    信用利差因子 = 信用债指数收益 - 国债指数收益
    
    Parameters
    ----------
    corporate_index : pd.DataFrame
        信用债指数数据
    treasury_index : pd.DataFrame
        国债指数数据
    window : int
        滚动窗口
    
    Returns
    -------
    pd.DataFrame
        信用利差因子收益序列
    """
    if len(corporate_index) == 0 or len(treasury_index) == 0:
        return pd.DataFrame()
    
    # 对齐日期
    common_dates = corporate_index.index.intersection(treasury_index.index)
    
    if len(common_dates) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(index=common_dates)
    df['credit_return'] = corporate_index.loc[common_dates, 'daily_return']
    df['treasury_return'] = treasury_index.loc[common_dates, 'daily_return']
    
    # 信用利差因子
    df['credit_factor'] = df['credit_return'] - df['treasury_return']
    
    # 滚动标准化
    df['credit_factor_std'] = (
        df['credit_factor'] - df['credit_factor'].rolling(window).mean()
    ) / df['credit_factor'].rolling(window).std()
    
    return df[['credit_factor', 'credit_factor_std', 'credit_return', 'treasury_return']]


# ============================================================
# 3. 可转债风险因子
# ============================================================
def build_convertible_factor(
    convertible_index: pd.DataFrame,
    treasury_index: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """构建可转债因子。
    
    可转债因子 = 转债指数收益 - 国债指数收益
    
    Parameters
    ----------
    convertible_index : pd.DataFrame
        可转债指数数据
    treasury_index : pd.DataFrame
        国债指数数据
    window : int
        滚动窗口
    
    Returns
    -------
    pd.DataFrame
        可转债因子收益序列
    """
    if len(convertible_index) == 0:
        return pd.DataFrame()
    
    df = convertible_index.copy()
    
    # 如果有国债数据，计算超额收益
    if len(treasury_index) > 0:
        common_dates = df.index.intersection(treasury_index.index)
        if len(common_dates) > 0:
            df = df.loc[common_dates].copy()
            df['treasury_return'] = treasury_index.loc[common_dates, 'daily_return']
            df['convertible_factor'] = df['daily_return'] - df['treasury_return']
        else:
            df['convertible_factor'] = df['daily_return']
    else:
        df['convertible_factor'] = df['daily_return']
    
    # 滚动标准化
    df['convertible_factor_std'] = (
        df['convertible_factor'] - df['convertible_factor'].rolling(window).mean()
    ) / df['convertible_factor'].rolling(window).std()
    
    return df[['convertible_factor', 'convertible_factor_std']]


# ============================================================
# 4. 综合因子构建
# ============================================================
def build_all_factors(
    treasury_index: pd.DataFrame,
    corporate_index: pd.DataFrame = None,
    convertible_index: pd.DataFrame = None,
    window: int = 20
) -> pd.DataFrame:
    """构建全部风险因子。
    
    Parameters
    ----------
    treasury_index : pd.DataFrame
        国债指数
    corporate_index : pd.DataFrame
        信用债指数（可选）
    convertible_index : pd.DataFrame
        可转债指数（可选）
    window : int
        滚动窗口
    
    Returns
    -------
    pd.DataFrame
        全部因子数据
    """
    # 利率因子
    duration_df = build_duration_factor(treasury_index, window)
    convexity_df = build_convexity_factor(treasury_index, window)
    
    # 合并
    factors = pd.concat([duration_df, convexity_df], axis=1)
    
    # 信用因子
    if corporate_index is not None and len(corporate_index) > 0:
        credit_df = build_credit_factor(corporate_index, treasury_index, window)
        if len(credit_df) > 0:
            factors = factors.join(credit_df[['credit_factor', 'credit_factor_std']], how='inner')
    
    # 可转债因子
    if convertible_index is not None and len(convertible_index) > 0:
        conv_df = build_convertible_factor(convertible_index, treasury_index, window)
        if len(conv_df) > 0:
            factors = factors.join(conv_df[['convertible_factor', 'convertible_factor_std']], how='inner')
    
    # 填充缺失
    factors = factors.fillna(0)
    
    print(f'  构建因子: {len(factors)} 条记录, {len(factors.columns)} 个因子')
    
    return factors


# ============================================================
# 5. 因子收益率计算
# ============================================================
def calculate_factor_returns(
    factors: pd.DataFrame,
    factor_names: List[str] = None
) -> pd.DataFrame:
    """计算因子累计收益率。
    
    Parameters
    ----------
    factors : pd.DataFrame
        因子数据
    factor_names : List[str]
        因子名称列表
    
    Returns
    -------
    pd.DataFrame
        因子累计收益
    """
    if factor_names is None:
        factor_names = [col for col in factors.columns if not col.endswith('_std')]
    
    result = pd.DataFrame(index=factors.index)
    
    for name in factor_names:
        if name in factors.columns:
            result[name] = (1 + factors[name]).cumprod() - 1
    
    return result
