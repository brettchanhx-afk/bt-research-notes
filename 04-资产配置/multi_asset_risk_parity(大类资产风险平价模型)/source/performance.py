"""
性能评估模块

提供策略绩效评估的完整工具集：
- 计算年化收益率、最大回撤、夏普比率、卡玛比率等指标
- 生成回撤序列、滚动收益等分析数据
- 绘制净值曲线、回撤曲线、权重热力图等
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化收益率

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    periods_per_year : int
        每年交易天数/周期数

    Returns:
    --------
    float
        年化收益率
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    if n_years <= 0:
        return 0.0

    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    return annualized_return


def calculate_max_drawdown(returns: pd.Series, periods_per_year: int = 252) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    计算最大回撤及其持续时间

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    periods_per_year : int
        每年交易天数

    Returns:
    --------
    tuple
        (最大回撤比例, 回撤开始日期, 回撤结束日期)
    """
    nav = (1 + returns).cumprod()

    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max

    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()

    running_max_at_min = nav.loc[:max_dd_end].expanding().max().idxmax()
    max_dd_start = running_max_at_min

    return max_dd, max_dd_start, max_dd_end


def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化波动率

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    periods_per_year : int
        每年交易天数

    Returns:
    --------
    float
        年化波动率
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.03,
                           periods_per_year: int = 252) -> float:
    """
    计算夏普比率

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    risk_free_rate : float
        年化无风险利率
    periods_per_year : int
        每年交易天数

    Returns:
    --------
    float
        夏普比率
    """
    ann_return = calculate_annualized_return(returns, periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)

    if ann_vol == 0:
        return 0.0

    sharpe = (ann_return - risk_free_rate) / ann_vol
    return sharpe


def calculate_calmar_ratio(returns: pd.Series,
                            periods_per_year: int = 252) -> float:
    """
    计算卡玛比率（年化收益/最大回撤）

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    periods_per_year : int
        每年交易天数

    Returns:
    --------
    float
        卡玛比率
    """
    ann_return = calculate_annualized_return(returns, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(returns, periods_per_year)

    if max_dd == 0:
        return 0.0

    calmar = ann_return / abs(max_dd)
    return calmar


def calculate_sortino_ratio(returns: pd.Series,
                              risk_free_rate: float = 0.03,
                              periods_per_year: int = 252,
                              target_return: float = 0.0) -> float:
    """
    计算索提诺比率

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    risk_free_rate : float
        年化无风险利率
    periods_per_year : int
        每年交易天数
    target_return : float
        目标收益率

    Returns:
    --------
    float
        索提诺比率
    """
    ann_return = calculate_annualized_return(returns, periods_per_year)

    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.inf

    downside_std = downside_returns.std() * np.sqrt(periods_per_year)

    if downside_std == 0:
        return np.inf

    sortino = (ann_return - risk_free_rate) / downside_std
    return sortino


def calculate_win_rate(returns: pd.Series) -> float:
    """
    计算胜率

    Parameters:
    -----------
    returns : pd.Series
        收益率序列

    Returns:
    --------
    float
        胜率（正收益周期占比）
    """
    return (returns > 0).sum() / len(returns)


def calculate_portfolio_metrics(returns: pd.Series,
                                 risk_free_rate: float = 0.03,
                                 periods_per_year: int = 252) -> Dict:
    """
    计算组合完整绩效指标

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    risk_free_rate : float
        年化无风险利率
    periods_per_year : int
        每年交易天数

    Returns:
    --------
    dict
        包含所有绩效指标的字典
    """
    max_dd, dd_start, dd_end = calculate_max_drawdown(returns, periods_per_year)
    ann_return = calculate_annualized_return(returns, periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(returns, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    win_rate = calculate_win_rate(returns)

    return {
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'max_drawdown': max_dd,
        'max_drawdown_start': dd_start,
        'max_drawdown_end': dd_end,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'win_rate': win_rate,
        'total_return': (1 + returns).prod() - 1,
        'n_periods': len(returns)
    }


def calculate_rolling_metrics(returns: pd.Series,
                               window: int = 252,
                               periods_per_year: int = 252) -> pd.DataFrame:
    """
    计算滚动绩效指标

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    window : int
        滚动窗口大小
    periods_per_year : int
        每年交易天数

    Returns:
    --------
    pd.DataFrame
        滚动绩效指标数据框
    """
    rolling_returns = returns.rolling(window).apply(
        lambda x: (1 + x).prod() - 1, raw=False
    )

    rolling_vol = returns.rolling(window).std() * np.sqrt(periods_per_year)

    rolling_sharpe = (rolling_returns * periods_per_year - 0.03) / (rolling_vol * np.sqrt(window / periods_per_year))

    running_max = returns.cumsum().expanding().max()
    rolling_dd = returns.cumsum() - running_max

    metrics = pd.DataFrame({
        'rolling_return': rolling_returns,
        'rolling_volatility': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_dd
    })

    return metrics


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    计算回撤序列

    Parameters:
    -----------
    returns : pd.Series
        收益率序列

    Returns:
    --------
    pd.Series
        回撤序列
    """
    nav = (1 + returns).cumprod()
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    return drawdown


def calculate_nav(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    计算净值序列

    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    initial_value : float
        初始净值

    Returns:
    --------
    pd.Series
        净值序列
    """
    nav = (1 + returns).cumprod() * initial_value
    return nav


def calculate_yearly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    计算年度收益率

    Parameters:
    -----------
    returns : pd.Series
        收益率序列

    Returns:
    --------
    pd.DataFrame
        年度收益率
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    yearly_returns.index = yearly_returns.index.year

    return yearly_returns


def generate_performance_summary(results_dict: Dict,
                                  risk_free_rate: float = 0.03,
                                  periods_per_year: int = 252) -> pd.DataFrame:
    """
    生成策略绩效对比表

    Parameters:
    -----------
    results_dict : dict
        策略名称到BacktestResult的映射
    risk_free_rate : float
        年化无风险利率
    periods_per_year : int
        每年交易天数

    Returns:
    --------
    pd.DataFrame
        绩效对比表
    """
    summary_data = []

    for name, result in results_dict.items():
        metrics = calculate_portfolio_metrics(
            result.portfolio_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        )

        row = {
            '策略名称': name,
            '年化收益': f"{metrics['annualized_return']:.2%}",
            '最大回撤': f"{metrics['max_drawdown']:.2%}",
            '年化波动': f"{metrics['annualized_volatility']:.2%}",
            '夏普比率': f"{metrics['sharpe_ratio']:.2f}",
            '卡玛比率': f"{metrics['calmar_ratio']:.2f}",
            '索提诺比率': f"{metrics['sortino_ratio']:.2f}",
            '胜率': f"{metrics['win_rate']:.2%}"
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def generate_yearly_performance_table(results_dict: Dict) -> pd.DataFrame:
    """
    生成年度绩效对比表

    Parameters:
    -----------
    results_dict : dict
        策略名称到BacktestResult的映射

    Returns:
    --------
    pd.DataFrame
        年度绩效对比表
    """
    yearly_data = {}

    for name, result in results_dict.items():
        yearly_returns = calculate_yearly_returns(result.portfolio_returns)
        yearly_data[name] = yearly_returns

    yearly_df = pd.DataFrame(yearly_data)
    yearly_df = yearly_df.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')

    return yearly_df


def print_performance_report(name: str, returns: pd.Series,
                               risk_free_rate: float = 0.03,
                               periods_per_year: int = 252):
    """
    打印策略绩效报告

    Parameters:
    -----------
    name : str
        策略名称
    returns : pd.Series
        收益率序列
    risk_free_rate : float
        年化无风险利率
    periods_per_year : int
        每年交易天数
    """
    metrics = calculate_portfolio_metrics(returns, risk_free_rate, periods_per_year)

    print(f"\n{'='*60}")
    print(f"策略绩效报告: {name}")
    print(f"{'='*60}")
    print(f"年化收益率:     {metrics['annualized_return']:.2%}")
    print(f"年化波动率:     {metrics['annualized_volatility']:.2%}")
    print(f"最大回撤:       {metrics['max_drawdown']:.2%}")
    print(f"  回撤开始:     {metrics['max_drawdown_start']}")
    print(f"  回撤结束:     {metrics['max_drawdown_end']}")
    print(f"夏普比率:       {metrics['sharpe_ratio']:.2f}")
    print(f"卡玛比率:       {metrics['calmar_ratio']:.2f}")
    print(f"索提诺比率:     {metrics['sortino_ratio']:.2f}")
    print(f"胜率:           {metrics['win_rate']:.2%}")
    print(f"总收益:         {metrics['total_return']:.2%}")
    print(f"交易周期数:     {metrics['n_periods']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("性能评估模块测试")
    print("=" * 60)

    np.random.seed(42)
    n_periods = 252 * 5

    dates = pd.date_range(start='2018-01-01', periods=n_periods, freq='B')
    returns = pd.Series(np.random.randn(n_periods) * 0.01 + 0.0002, index=dates)

    print(f"\n收益率序列长度: {len(returns)}")
    print(f"时间范围: {returns.index[0]} 至 {returns.index[-1]}")

    nav = calculate_nav(returns)
    print(f"\n净值序列预览: {nav.head()}")
    print(f"最终净值: {nav.iloc[-1]:.4f}")

    drawdown = calculate_drawdown_series(returns)
    print(f"\n最大回撤: {drawdown.min():.2%}")

    yearly_returns = calculate_yearly_returns(returns)
    print(f"\n年度收益率:")
    print(yearly_returns)

    print("\n绩效指标:")
    metrics = calculate_portfolio_metrics(returns)
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['annualized_return', 'max_drawdown', 'annualized_volatility', 'total_return', 'win_rate']:
                print(f"  {key}: {value:.2%}" if abs(value) < 10 else f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print_performance_report("测试策略", returns)