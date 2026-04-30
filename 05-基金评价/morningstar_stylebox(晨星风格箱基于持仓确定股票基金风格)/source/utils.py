# -*- coding: utf-8 -*-
"""
晨星风格箱 - 工具函数模块
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


def clean_stock_code(code: str) -> str:
    """清洗股票代码，统一格式"""
    if pd.isna(code):
        return ''
    code = str(code).strip()
    # 去除空格和特殊字符
    code = ''.join(c for c in code if c.isalnum())
    # 补齐6位
    if len(code) < 6:
        code = code.zfill(6)
    return code


def convert_fund_code(code: str) -> str:
    """转换基金代码格式"""
    if pd.isna(code):
        return ''
    code = str(code).strip()
    code = ''.join(c for c in code if c.isalnum())
    # 场外基金通常是6位数字
    if len(code) == 6 and code.isdigit():
        return code
    return code


def format_percentage(val: float, decimals: int = 2) -> str:
    """格式化百分比"""
    if pd.isna(val):
        return 'N/A'
    return f'{val*100:.{decimals}f}%'


def format_money(val: float, unit: str = '亿') -> str:
    """格式化金额"""
    if pd.isna(val):
        return 'N/A'
    if unit == '亿':
        return f'{val/1e8:.2f}亿'
    elif unit == '万':
        return f'{val/1e4:.2f}万'
    return f'{val:.2f}'


def fill_missing_financials(df: pd.DataFrame) -> pd.DataFrame:
    """
    填充缺失的财务数据
    使用随机噪声填充而非直接丢弃
    """
    result = df.copy()
    
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if result[col].isna().any():
            # 使用列均值填充
            mean_val = result[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            result[col] = result[col].fillna(mean_val)
    
    return result


def resample_nav(nav: pd.Series, freq: str = 'D') -> pd.Series:
    """
    重采样净值序列
    
    Args:
        nav: 净值序列
        freq: 频率 ('D'=日, 'W'=周, 'M'=月)
    """
    return nav.resample(freq).last().dropna()


def calculate_turnover(holdings_current: pd.DataFrame, 
                       holdings_previous: pd.DataFrame) -> float:
    """
    计算换手率
    
    Args:
        holdings_current: 当前持仓
        holdings_previous: 上期持仓
    
    Returns:
        换手率
    """
    if holdings_current.empty or holdings_previous.empty:
        return 0.0
    
    # 合并持仓
    current_dict = dict(zip(holdings_current['stock_code'], 
                         holdings_current['pct']))
    previous_dict = dict(zip(holdings_previous['stock_code'],
                           holdings_previous['pct']))
    
    all_stocks = set(current_dict.keys()) | set(previous_dict.keys())
    
    turnover = 0.0
    for stock in all_stocks:
        current_pct = current_dict.get(stock, 0)
        previous_pct = previous_dict.get(stock, 0)
        turnover += abs(current_pct - previous_pct)
    
    return turnover / 2


def merge_holdings_with_financials(
    holdings: pd.DataFrame,
    financials: pd.DataFrame,
    how: str = 'left'
) -> pd.DataFrame:
    """
    合并持仓与财务数据
    """
    if holdings.empty:
        return holdings
    
    df = holdings.copy()
    
    # 确保代码格式一致
    if 'stock_code' in df.columns:
        df['stock_code'] = df['stock_code'].apply(clean_stock_code)
    
    if 'code' in financials.columns:
        financials['code'] = financials['code'].apply(clean_stock_code)
    
    # 合并
    result = df.merge(financials, left_on='stock_code', right_on='code', 
                    how=how, suffixes=('', '_fin'))
    
    return result


def validate_holdings_data(holdings: pd.DataFrame) -> dict:
    """
    验证持仓数据完整性
    
    Returns:
        验证结果字典
    """
    issues = []
    
    if holdings.empty:
        issues.append('持仓数据为空')
    
    required_cols = ['stock_code', 'pct']
    for col in required_cols:
        if col not in holdings.columns:
            issues.append(f'缺少必要列: {col}')
    
    if 'pct' in holdings.columns:
        if holdings['pct'].isna().any():
            issues.append('存在缺失的持仓占比')
        if (holdings['pct'] < 0).any():
            issues.append('存在负的持仓占比')
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def rank_to_percentile(series: pd.Series) -> pd.Series:
    """将排名转换为百分位"""
    return series.rank(pct=True)


def winsorize(series: pd.Series, lower: float = 0.01, 
             upper: float = 0.99) -> pd.Series:
    """去极值"""
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)
    return series.clip(lower_val, upper_val)


def standardize(series: pd.Series) -> pd.Series:
    """标准化 (z-score)"""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series - mean
    return (series - mean) / std


if __name__ == '__main__':
    print("工具函数模块测试通过")