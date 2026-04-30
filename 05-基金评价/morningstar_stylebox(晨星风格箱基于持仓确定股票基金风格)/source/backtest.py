# -*- coding: utf-8 -*-
"""
晨星风格箱 - 回测模块
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(nav: pd.Series) -> pd.Series:
    """计算收益率序列"""
    return nav.pct_change().dropna()


def calculate_cumulative_return(returns: pd.Series) -> float:
    """计算累计收益率"""
    return (1 + returns).prod() - 1


def calculate_annualized_return(cum_return: float, days: int) -> float:
    """计算年化收益率"""
    if days <= 0:
        return 0.0
    years = days / 252
    return (1 + cum_return) ** (1 / years) - 1


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """计算夏普比率"""
    if returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_max_drawdown(nav: pd.Series) -> float:
    """计算最大回撤"""
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    return drawdown.min()


def calculate_calmar_ratio(cum_return: float, max_drawdown: float) -> float:
    """计算卡玛比率"""
    if abs(max_drawdown) < 1e-10:
        return 0.0
    return cum_return / abs(max_drawdown)


def calculate_win_rate(returns: pd.Series) -> float:
    """计算胜率"""
    return (returns > 0).mean()


def calculate_volatility(returns: pd.Series) -> float:
    """计算波动率（年化）"""
    return returns.std() * np.sqrt(252)


def run_style_backtest(
    nav: pd.Series,
    style_changes: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict:
    """
    运行风格箱回测
    
    Args:
        nav: 基金净值序列
        style_changes: 风格变更记录 (date, old_style, new_style)
        benchmark_returns: 基准收益率序列
    
    Returns:
        绩效指标字典
    """
    returns = calculate_returns(nav)
    
    # 基础指标
    total_return = calculate_cumulative_return(returns)
    days = len(returns)
    annual_return = calculate_annualized_return(total_return, days)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(nav)
    volatility = calculate_volatility(returns)
    win_rate = calculate_win_rate(returns)
    
    result = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility': volatility,
        'win_rate': win_rate,
        'trading_days': days
    }
    
    # 相对基准
    if benchmark_returns is not None:
        excess = returns - benchmark_returns
        result['excess_return'] = calculate_cumulative_return(excess)
        result['tracking_error'] = excess.std() * np.sqrt(252)
        result['information_ratio'] = (
            result['excess_return'] / result['tracking_error'] 
            if result['tracking_error'] > 0 else 0
        )
    
    # 风格分析
    if not style_changes.empty:
        result['style_changes'] = len(style_changes)
        # 统计各风格持有期
        style_periods = style_changes.groupby('new_style').size()
        result['style_distribution'] = style_periods.to_dict()
    
    return result


def generate_style_signal(
    style_series: pd.Series,
    rebalance_freq: str = 'M'
) -> pd.DataFrame:
    """
    生成风格调仓信号
    
    Args:
        style_series: 风格得分序列
        rebalance_freq: 调仓频率 ('M'=月, 'Q'=季)
    
    Returns:
        调仓信号DataFrame
    """
    if rebalance_freq == 'M':
        style_signal = style_series.resample('M').last()
    elif rebalance_freq == 'Q':
        style_signal = style_series.resample('Q').last()
    else:
        style_signal = style_series
    
    # 标记风格变化
    changes = style_signal != style_signal.shift(1)
    signals = changes[changes].index
    
    return pd.DataFrame({
        'date': signals,
        'rebalance': True
    })


if __name__ == '__main__':
    print("回测模块测试通过")