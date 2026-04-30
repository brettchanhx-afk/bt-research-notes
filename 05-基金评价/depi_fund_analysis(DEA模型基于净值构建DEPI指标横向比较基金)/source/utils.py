# -*- coding: utf-8 -*-
"""
工具函数模块
包含：数据清洗、格式转换、收益率计算等通用工具
"""
import os
import pandas as pd
import numpy as np


def setup_chinese_font():
    """配置 matplotlib 中文字体（解决 Windows 方块字问题）。
    关键：plt.style.use() 必须在字体配置之前调用，
    字体配置必须在 style.use() 之后再次应用。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    cache_dir = matplotlib.get_cachedir()
    font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
    if os.path.exists(font_cache):
        try:
            os.remove(font_cache)
        except OSError:
            pass

    # 在 style.use() 之后设置字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120


def standardize(series: pd.Series) -> pd.Series:
    """Z-score 标准化（消除量纲影响，与研报一致）。"""
    mean = series.mean()
    std = series.std()
    if std < 1e-10:
        return series - mean
    return (series - mean) / std


def normalize(series: pd.Series) -> pd.Series:
    """Min-Max 归一化到 [0,1] 区间。"""
    mn = series.min()
    mx = series.max()
    if mx - mn < 1e-10:
        return series - mn
    return (series - mn) / (mx - mn)


def parse_fund_code(code) -> str:
    """解析基金代码，统一返回6位字符串。"""
    code = str(code).strip()
    for suffix in ['.SH', '.SZ', '.OF', '.XSB']:
        if suffix in code.upper():
            code = code.upper().replace(suffix, '')
    return code.zfill(6)


def drop_na_rows(df: pd.DataFrame, pct: float = 0.5) -> pd.DataFrame:
    """删除缺失率超过 pct 的行。"""
    return df.dropna(thresh=int(len(df.columns) * (1 - pct)))


def save_df(df: pd.DataFrame, filename: str, data_dir: str):
    """保存 DataFrame 到 CSV（UTF-8 BOM，兼容 Excel）。"""
    import config
    path = config.DATA_DIR / filename
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  [保存] {path}')


def resample_returns(daily_returns: pd.Series, freq: str = 'Q') -> pd.Series:
    """将日收益率重采样为指定频率（Q=季度，M=月）。
    
    Parameters
    ----------
    daily_returns : pd.Series
        日收益率序列（索引为日期）
    freq : str
        重采样频率，'Q' 季度，'M' 月
    
    Returns
    -------
    pd.Series
        重采样后收益率序列
    """
    if freq == 'Q':
        period_returns = (1 + daily_returns).resample('QE').prod() - 1
    elif freq == 'M':
        period_returns = (1 + daily_returns).resample('ME').prod() - 1
    else:
        period_returns = daily_returns
    return period_returns.dropna()
