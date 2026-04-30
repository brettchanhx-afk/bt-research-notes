# -*- coding: utf-8 -*-
"""
回测模块
===================================
基于业绩持续性信号的基金筛选与回测

方法说明：
- 横截面分析法：β显著为正 → 筛选持续性强的基金
- CPR法：CPR > 1 → 筛选赢家组合
- Hurst指数法：H > 0.5 → 筛选具有持续性的基金
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.factor import (
    cross_section_analysis,
    hurst_analysis,
    calculate_cpr,
    comprehensive_persistence_analysis
)
from source.data_loader import calculate_daily_returns, calculate_cumulative_returns
from config import BACKTEST_CONFIG, RISK_FREE_RATE


# ============================================================
# 回测引擎
# ============================================================

def persistence_based_backtest(funds_data, 
                              benchmark_returns=None,
                              method='hurst',
                              rebalance_freq='quarterly',
                              top_n=10,
                              persistence_threshold=0.6,
                              initial_capital=1000000):
    """
    基于业绩持续性的基金筛选与回测
    
    策略逻辑：
    1. 滚动计算各基金的持续性指标
    2. 每到调仓日，选取持续性最强的top_n只基金
    3. 等权持有至下一次调仓
    
    Parameters:
    -----------
    funds_data : dict
        {fund_code: nav_df} 基金净值数据
    benchmark_returns : Series, optional
        基准收益率
    method : str
        持续性评估方法 ('hurst', 'cross_section', 'cpr')
    rebalance_freq : str
        调仓频率 ('monthly', 'quarterly', 'yearly')
    top_n : int
        持仓基金数量
    persistence_threshold : float
        持续性阈值（仅选H > threshold的基金）
    initial_capital : float
        初始资金
        
    Returns:
    --------
    dict : 回测结果，包含持仓记录、收益率序列、绩效指标等
    """
    # 合并所有基金的净值数据
    merged_nav = None
    for fund_code, nav_df in funds_data.items():
        df = nav_df[['date', 'nav']].rename(columns={'nav': f'nav_{fund_code}'})
        if merged_nav is None:
            merged_nav = df
        else:
            merged_nav = pd.merge(merged_nav, df, on='date', how='outer')
    
    if merged_nav is None:
        return None
    
    merged_nav = merged_nav.sort_values('date').reset_index(drop=True)
    merged_nav = merged_nav.set_index('date')
    
    # 计算各基金的收益率
    returns_cols = [col for col in merged_nav.columns if col.startswith('nav_')]
    returns_df = merged_nav[returns_cols].pct_change()
    
    # 确定调仓日期
    if rebalance_freq == 'monthly':
        rebalance_dates = _get_monthly_dates(merged_nav.index)
    elif rebalance_freq == 'quarterly':
        rebalance_dates = _get_quarterly_dates(merged_nav.index)
    elif rebalance_freq == 'yearly':
        rebalance_dates = _get_yearly_dates(merged_nav.index)
    else:
        rebalance_dates = _get_quarterly_dates(merged_nav.index)
    
    # 滚动计算持续性指标
    print(f"\n开始回测...")
    print(f"回测期间: {merged_nav.index[0].strftime('%Y-%m-%d')} 至 {merged_nav.index[-1].strftime('%Y-%m-%d')}")
    print(f"调仓频率: {rebalance_freq}")
    print(f"持仓数量: {top_n}")
    
    # 回测参数
    window = 60  # 计算持续性指标的回看窗口
    step = 20    # 滚动步长
    
    portfolio_returns = []
    holdings_history = []
    persistence_scores = {}
    
    dates = merged_nav.index.tolist()
    
    for i in range(window, len(dates), step):
        window_end = dates[i]
        window_start_idx = max(0, i - window)
        window_start = dates[window_start_idx]
        
        # 计算该窗口内各基金的持续性指标
        scores = {}
        
        for col in returns_cols:
            fund_code = col.replace('nav_', '')
            fund_returns = returns_df[col].iloc[window_start_idx:i].dropna()
            
            if len(fund_returns) < window // 2:
                continue
            
            if method == 'hurst':
                result = hurst_analysis(fund_returns)
                if result:
                    scores[fund_code] = result.get('H', 0.5)
            
            elif method == 'cross_section':
                # 横截面法需要基金池，这里简化为单基金分析
                scores[fund_code] = fund_returns.mean() / fund_returns.std()  # 夏普比近似
            
            elif method == 'cpr':
                # CPR法简化
                result = hurst_analysis(fund_returns)
                if result:
                    H = result.get('H', 0.5)
                    scores[fund_code] = H if H > 0.5 else 1 - H  # 反转发转
        
        persistence_scores[window_end] = scores
        
        # 调仓
        if window_end in rebalance_dates or window_end == dates[-1]:
            # 选取持续性最强的基金
            sorted_funds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_funds = [f for f, s in sorted_funds if s > persistence_threshold][:top_n]
            
            if len(top_funds) > 0:
                holdings_history.append({
                    'date': window_end,
                    'funds': top_funds,
                    'weights': [1.0/len(top_funds)] * len(top_funds),
                    'scores': {f: scores[f] for f in top_funds}
                })
                
                # 计算下一期收益率
                next_start_idx = min(i + step, len(dates) - 1)
                next_end_idx = min(i + step * 2, len(dates))
                
                if next_start_idx < len(dates) and next_end_idx <= len(dates):
                    period_returns = []
                    for fund in top_funds:
                        col = f'nav_{fund}'
                        ret = (merged_nav[col].iloc[next_end_idx-1] / merged_nav[col].iloc[next_start_idx] - 1)
                        period_returns.append(ret)
                    
                    portfolio_ret = np.mean(period_returns)
                    portfolio_returns.append({
                        'start_date': dates[next_start_idx],
                        'end_date': dates[next_end_idx-1],
                        'return': portfolio_ret,
                        'n_holdings': len(top_funds)
                    })
    
    # 计算绩效指标
    if len(portfolio_returns) > 0:
        returns_series = pd.Series([r['return'] for r in portfolio_returns])
        
        # 年化收益
        n_periods_per_year = 12 if rebalance_freq == 'monthly' else 4
        annual_return = returns_series.mean() * n_periods_per_year
        
        # 年化波动率
        annual_vol = returns_series.std() * np.sqrt(n_periods_per_year)
        
        # 夏普比率
        sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns_series > 0).sum() / len(returns_series)
        
        # 基准收益
        if benchmark_returns is not None:
            bm_returns = []
            for r in portfolio_returns:
                bm_start = benchmark_returns.index.get_indexer([r['start_date']], method='pad')[0]
                bm_end = benchmark_returns.index.get_indexer([r['end_date']], method='pad')[0]
                if bm_start >= 0 and bm_end >= 0 and bm_end > bm_start:
                    bm_ret = (benchmark_returns.iloc[bm_end] / benchmark_returns.iloc[bm_start] - 1)
                    bm_returns.append(bm_ret)
            
            if len(bm_returns) > 0:
                benchmark_annual = np.mean(bm_returns) * n_periods_per_year
            else:
                benchmark_annual = 0
        else:
            benchmark_annual = 0
        
        performance = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'benchmark_annual': benchmark_annual,
            'excess_return': annual_return - benchmark_annual,
            'n_trades': len(portfolio_returns),
            'avg_holdings': np.mean([r['n_holdings'] for r in portfolio_returns])
        }
    else:
        performance = None
    
    return {
        'returns': portfolio_returns,
        'holdings': holdings_history,
        'performance': performance,
        'persistence_scores': persistence_scores,
        'config': {
            'method': method,
            'rebalance_freq': rebalance_freq,
            'top_n': top_n,
            'persistence_threshold': persistence_threshold,
            'initial_capital': initial_capital
        }
    }


def _get_monthly_dates(dates):
    """获取每月末日期"""
    df = pd.DataFrame({'date': dates})
    df['yearmonth'] = df['date'].dt.to_period('M')
    return df.groupby('yearmonth')['date'].last().tolist()


def _get_quarterly_dates(dates):
    """获取每季度末日期"""
    df = pd.DataFrame({'date': dates})
    df['yearquarter'] = df['date'].dt.to_period('Q')
    return df.groupby('yearquarter')['date'].last().tolist()


def _get_yearly_dates(dates):
    """获取每年末日期"""
    df = pd.DataFrame({'date': dates})
    df['year'] = df['date'].dt.year
    return df.groupby('year')['date'].last().tolist()


# ============================================================
# 绩效计算
# ============================================================

def calculate_performance_metrics(returns, risk_free_rate=RISK_FREE_RATE):
    """
    计算完整绩效指标
    
    Parameters:
    -----------
    returns : Series or list
        收益率序列
    risk_free_rate : float
        年化无风险利率
        
    Returns:
    --------
    dict : 绩效指标
    """
    returns = pd.Series(returns).dropna()
    
    if len(returns) == 0:
        return None
    
    # 累计收益
    cumulative_return = (1 + returns).prod() - 1
    
    # 年化收益
    n_periods_per_year = 252 / len(returns)  # 假设日频数据
    annual_return = (1 + cumulative_return) ** n_periods_per_year - 1
    
    # 年化波动率
    annual_vol = returns.std() * np.sqrt(n_periods_per_year)
    
    # 夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 索提诺比率（只考虑下行波动）
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(n_periods_per_year) if len(downside_returns) > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # 胜率
    win_rate = (returns > 0).sum() / len(returns)
    
    # 卡玛比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'calmar_ratio': calmar,
        'n_periods': len(returns)
    }


def print_backtest_results(results, benchmark_name='沪深300'):
    """
    打印回测结果
    """
    print(f"\n{'='*60}")
    print("基于业绩持续性的基金筛选回测结果")
    print(f"{'='*60}")
    
    perf = results.get('performance')
    if perf is None:
        print("回测失败，无有效结果")
        return
    
    print(f"\n【绩效指标】")
    print(f"  年化收益率: {perf['annual_return']*100:.2f}%")
    print(f"  年化波动率: {perf['annual_volatility']*100:.2f}%")
    print(f"  夏普比率:   {perf['sharpe_ratio']:.3f}")
    print(f"  最大回撤:   {perf['max_drawdown']*100:.2f}%")
    print(f"  胜率:       {perf['win_rate']*100:.1f}%")
    print(f"  基准年化:   {perf['benchmark_annual']*100:.2f}%")
    print(f"  超额收益:   {perf['excess_return']*100:.2f}%")
    
    print(f"\n【交易统计】")
    print(f"  调仓次数:   {perf['n_trades']}")
    print(f"  平均持仓:   {perf['avg_holdings']:.1f}只")
    
    config = results.get('config', {})
    print(f"\n【策略参数】")
    print(f"  持续性方法: {config.get('method', 'N/A')}")
    print(f"  调仓频率:   {config.get('rebalance_freq', 'N/A')}")
    print(f"  持仓数量:   {config.get('top_n', 'N/A')}")
    print(f"  持续性阈值: {config.get('persistence_threshold', 'N/A')}")


# ============================================================
# 分层回测
# ============================================================

def stratified_persistence_backtest(funds_data, benchmark_returns=None):
    """
    分层持续性回测：比较高持续性组vs低持续性组
    
    Parameters:
    -----------
    funds_data : dict
        基金数据
    benchmark_returns : Series, optional
        基准收益
        
    Returns:
    --------
    dict : 两组的表现对比
    """
    # 合并净值数据
    merged_nav = None
    for fund_code, nav_df in funds_data.items():
        df = nav_df[['date', 'nav']].rename(columns={'nav': f'nav_{fund_code}'})
        if merged_nav is None:
            merged_nav = df
        else:
            merged_nav = pd.merge(merged_nav, df, on='date', how='outer')
    
    merged_nav = merged_nav.sort_values('date').reset_index(drop=True)
    merged_nav = merged_nav.set_index('date')
    
    returns_cols = [col for col in merged_nav.columns if col.startswith('nav_')]
    returns_df = merged_nav[returns_cols].pct_change().dropna()
    
    # 计算所有基金的整体Hurst指数
    fund_hurst = {}
    for col in returns_cols:
        fund_code = col.replace('nav_', '')
        fund_returns = returns_df[col].dropna()
        if len(fund_returns) >= 32:
            result = hurst_analysis(fund_returns)
            if result:
                fund_hurst[fund_code] = result.get('H', 0.5)
    
    # 分组
    sorted_funds = sorted(fund_hurst.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_funds)
    high_group = [f for f, h in sorted_funds[:n//2] if h > 0.5]
    low_group = [f for f, h in sorted_funds[n//2:] if h < 0.5]
    
    # 计算各组收益
    high_returns = []
    low_returns = []
    dates = returns_df.index.tolist()
    
    for i in range(0, len(dates), 20):  # 每20天计算一次
        window_returns = returns_df.iloc[i:i+20]
        
        if len(window_returns) > 0:
            if len(high_group) > 0:
                high_cols = [f'nav_{f}' for f in high_group if f'nav_{f}' in returns_df.columns]
                high_ret = window_returns[high_cols].mean(axis=1).mean()
                high_returns.append(high_ret)
            
            if len(low_group) > 0:
                low_cols = [f'nav_{f}' for f in low_group if f'nav_{f}' in returns_df.columns]
                low_ret = window_returns[low_cols].mean(axis=1).mean()
                low_returns.append(low_ret)
    
    results = {
        'high_persistence_group': {
            'funds': high_group,
            'n_funds': len(high_group),
            'performance': calculate_performance_metrics(high_returns) if high_returns else None
        },
        'low_persistence_group': {
            'funds': low_group,
            'n_funds': len(low_group),
            'performance': calculate_performance_metrics(low_returns) if low_returns else None
        }
    }
    
    return results
