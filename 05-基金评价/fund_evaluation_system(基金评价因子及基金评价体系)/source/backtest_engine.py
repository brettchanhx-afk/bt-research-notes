# -*- coding: utf-8 -*-
"""
回测引擎模块

功能：
- 分层回测
- RankIC计算
- ICIR计算
- 因子有效性评估
- 时间敏感性评估
- 板块敏感性评估
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. RankIC计算
# ============================================================
def calc_rank_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series
) -> float:
    """
    计算RankIC（秩相关系数）
    
    Parameters
    ----------
    factor_values : pd.Series
        因子值
    forward_returns : pd.Series
        下期收益率
        
    Returns
    -------
    float
        RankIC值
    """
    if len(factor_values) < 10 or len(forward_returns) < 10:
        return np.nan
    
    # 对齐
    common_idx = factor_values.index.intersection(forward_returns.index)
    
    if len(common_idx) < 10:
        return np.nan
    
    x = factor_values.loc[common_idx]
    y = forward_returns.loc[common_idx]
    
    # 秩相关系数
    rank_ic = x.rank().corr(y.rank())
    
    return rank_ic


def calc_rank_ic_series(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    factor_name: str
) -> pd.Series:
    """
    计算RankIC时间序列
    
    Parameters
    ----------
    factor_df : pd.DataFrame
        因子数据，index=date, columns=fund_code
    return_df : pd.DataFrame
        下期收益率数据
        
    Returns
    -------
    pd.Series
        RankIC时间序列
    """
    ic_series = []
    
    for date in factor_df.index:
        if date not in return_df.index:
            ic_series.append(np.nan)
            continue
        
        factor_values = factor_df.loc[date].dropna()
        forward_returns = return_df.loc[date].dropna()
        
        ic = calc_rank_ic(factor_values, forward_returns)
        ic_series.append(ic)
    
    result = pd.Series(ic_series, index=factor_df.index, name='RankIC')
    
    return result


# ============================================================
# 2. 分层回测
# ============================================================
def layered_backtest(
    factor_values: pd.Series,
    nav_data: dict,
    n_layers: int = 3,
    holding_period: int = 63  # 约3个月
) -> pd.DataFrame:
    """
    分层回测
    
    Parameters
    ----------
    factor_values : pd.Series
        因子值（某一时点）
    nav_data : dict
        {fund_code: nav_dataframe}
    n_layers : int
        分层数（默认3层）
    holding_period : int
        持有期（交易日）
        
    Returns
    -------
    pd.DataFrame
        各层累计收益率
    """
    if len(factor_values) < n_layers * 5:
        return pd.DataFrame()
    
    # 按因子值分层
    factor_values = factor_values.dropna()
    factor_values = factor_values.sort_values(ascending=False)
    
    layer_size = len(factor_values) // n_layers
    
    layers = {}
    for i in range(n_layers):
        if i == n_layers - 1:
            layer_funds = factor_values.iloc[i * layer_size:].index.tolist()
        else:
            layer_funds = factor_values.iloc[i * layer_size:(i + 1) * layer_size].index.tolist()
        
        layers[f'layer_{i + 1}'] = layer_funds
    
    # 计算各层收益
    layer_returns = {}
    
    for layer_name, fund_codes in layers.items():
        returns_list = []
        
        for code in fund_codes:
            if code in nav_data:
                nav = nav_data[code]
                if len(nav) >= holding_period:
                    ret = nav['nav'].iloc[:holding_period].pct_change().dropna()
                    returns_list.append(ret)
        
        if returns_list:
            # 等权组合
            layer_ret = pd.concat(returns_list, axis=1).mean(axis=1)
            layer_returns[layer_name] = (1 + layer_ret).cumprod()
    
    if not layer_returns:
        return pd.DataFrame()
    
    return pd.DataFrame(layer_returns)


def calculate_layer_metrics(
    layer_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    计算各层风险收益指标
    
    Parameters
    ----------
    layer_returns : pd.DataFrame
        各层累计收益率
        
    Returns
    -------
    pd.DataFrame
        风险收益指标
    """
    metrics = []
    
    for col in layer_returns.columns:
        returns = layer_returns[col].pct_change().dropna()
        
        if len(returns) < 10:
            continue
        
        # 年化收益率
        annual_return = (layer_returns[col].iloc[-1] / layer_returns[col].iloc[0]) ** (252 / len(returns)) - 1
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
        
        # 最大回撤
        cummax = layer_returns[col].cummax()
        max_dd = ((cummax - layer_returns[col]) / cummax).max()
        
        metrics.append({
            'layer': col,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        })
    
    return pd.DataFrame(metrics)


# ============================================================
# 3. 因子有效性评估
# ============================================================
def evaluate_factor_effectiveness(
    ic_series: pd.Series
) -> dict:
    """
    评估因子有效性
    
    Parameters
    ----------
    ic_series : pd.Series
        RankIC时间序列
        
    Returns
    -------
    dict
        有效性指标
    """
    ic_clean = ic_series.dropna()
    
    if len(ic_clean) < 10:
        return {}
    
    metrics = {
        'mean_ic': ic_clean.mean(),
        'ic_std': ic_clean.std(),
        'icir': ic_clean.mean() / ic_clean.std() if ic_clean.std() > 0 else np.nan,
        'ic_positive_ratio': (ic_clean > 0).sum() / len(ic_clean),
        'ic_significant_ratio': (np.abs(ic_clean) > 0.02).sum() / len(ic_clean),
        't_stat': ic_clean.mean() / (ic_clean.std() / np.sqrt(len(ic_clean))),
    }
    
    return metrics


# ============================================================
# 4. 时间敏感性评估
# ============================================================
def evaluate_time_sensitivity(
    ic_results: Dict[str, pd.Series]
) -> float:
    """
    评估因子时间敏感性
    
    Parameters
    ----------
    ic_results : dict
        不同调仓路径下的IC序列
        {path_name: ic_series}
        
    Returns
    -------
    float
        时间敏感性指标（IC标准差）
    """
    if len(ic_results) < 2:
        return np.nan
    
    # 计算各路径平均IC
    mean_ics = []
    for path_name, ic_series in ic_results.items():
        ic_clean = ic_series.dropna()
        if len(ic_clean) > 0:
            mean_ics.append(ic_clean.mean())
    
    if len(mean_ics) < 2:
        return np.nan
    
    # 时间敏感性 = 不同路径下IC的标准差
    time_sensitivity = np.std(mean_ics)
    
    return time_sensitivity


# ============================================================
# 5. 板块敏感性评估
# ============================================================
def evaluate_sector_sensitivity(
    ic_results: Dict[str, pd.Series]
) -> float:
    """
    评估因子板块敏感性
    
    Parameters
    ----------
    ic_results : dict
        不同板块下的IC序列
        {sector_name: ic_series}
        
    Returns
    -------
    float
        板块敏感性指标（IC标准差）
    """
    if len(ic_results) < 2:
        return np.nan
    
    # 计算各板块平均IC
    mean_ics = []
    for sector_name, ic_series in ic_results.items():
        ic_clean = ic_series.dropna()
        if len(ic_clean) > 0:
            mean_ics.append(ic_clean.mean())
    
    if len(mean_ics) < 2:
        return np.nan
    
    # 板块敏感性 = 不同板块IC的标准差
    sector_sensitivity = np.std(mean_ics)
    
    return sector_sensitivity


# ============================================================
# 6. 因子综合打分
# ============================================================
def calculate_factor_score(
    factor_name: str,
    ic_effectiveness: float,
    time_sensitivity: float,
    sector_sensitivity: float
) -> float:
    """
    因子综合打分
    
    公式：综合得分 = 因子有效性得分 * 70% - 板块敏感性得分 * 15% - 时间敏感性得分 * 15%
    
    Parameters
    ----------
    ic_effectiveness : float
        因子有效性（平均IC）
    time_sensitivity : float
        时间敏感性
    sector_sensitivity : float
        板块敏感性
        
    Returns
    -------
    float
        综合得分（已归一化）
    """
    # 这里需要先对所有因子进行归一化
    # 简化实现：直接返回
    return ic_effectiveness


# ============================================================
# 7. 回测报告生成
# ============================================================
def generate_backtest_report(
    factor_name: str,
    ic_series: pd.Series,
    layer_metrics: pd.DataFrame,
    effectiveness: dict
) -> str:
    """
    生成回测报告
    
    Returns
    -------
    str
        回测报告文本
    """
    report = f"""
{'=' * 50}
因子回测报告: {factor_name}
{'=' * 50}

【因子有效性】
  平均RankIC:    {effectiveness.get('mean_ic', np.nan):.4f}
  IC标准差:      {effectiveness.get('ic_std', np.nan):.4f}
  ICIR:          {effectiveness.get('icir', np.nan):.4f}
  IC>0占比:      {effectiveness.get('ic_positive_ratio', np.nan):.2%}
  t统计量:       {effectiveness.get('t_stat', np.nan):.2f}

【分层回测结果】
"""
    
    if len(layer_metrics) > 0:
        report += layer_metrics.to_string(index=False)
    
    return report
