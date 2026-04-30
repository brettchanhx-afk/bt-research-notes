# 策略绩效与风险评价指标计算模块
# 基于empyrical实现完整回测评价体系
import numpy as np
import pandas as pd
import empyrical as ep
from typing import List, Tuple, Union


def adjust_return_series(
    return_series: pd.Series,
    adjust_value: Union[float, pd.Series]
) -> pd.Series:
    """
    对收益序列进行差值调整，用于计算主动收益
    若调整系数为0则直接返回原序列，提升计算效率
    """
    if isinstance(adjust_value, (float, int)) and adjust_value == 0:
        return return_series.copy()
    return return_series - adjust_value


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    计算信息比率，衡量主动收益相对于跟踪误差的性价比
    """
    if len(strategy_returns) < 2:
        return np.nan

    active_returns = adjust_return_series(strategy_returns, benchmark_returns)
    tracking_error = np.std(active_returns, ddof=1)
    
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
        
    return np.mean(active_returns) / tracking_error


def evaluate_strategy_metrics(
    return_data: pd.DataFrame,
    benchmark_label: str = 'benchmark',
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    计算策略完整风险收益指标矩阵
    包含绝对收益指标与相对基准指标
    """
    metric_table = pd.DataFrame()

    # 核心绝对收益指标
    metric_table['年化收益率'] = ep.annual_return(return_data, period=frequency)
    metric_table['累计收益'] = return_data.apply(lambda ser: ep.cum_returns(ser).iloc[-1])
    metric_table['波动率'] = return_data.apply(lambda ser: ep.annual_volatility(ser, period=frequency))
    metric_table['夏普比率'] = return_data.apply(ep.sharpe_ratio, period=frequency)
    metric_table['最大回撤'] = return_data.apply(lambda ser: ep.max_drawdown(ser))
    metric_table['索提诺比率'] = return_data.apply(lambda ser: ep.sortino_ratio(ser, period=frequency))
    metric_table['卡玛比率'] = return_data.apply(lambda ser: ep.calmar_ratio(ser, period=frequency))

    # 相对基准指标（仅当存在基准列时计算）
    if benchmark_label in return_data.columns:
        strategy_cols = [col for col in return_data.columns if col != benchmark_label]
        
        # 信息比率、Alpha、Beta
        metric_table['信息比率'] = return_data[strategy_cols].apply(
            lambda ser: calculate_information_ratio(ser, return_data[benchmark_label])
        )
        metric_table['Alpha'] = return_data[strategy_cols].apply(
            lambda ser: ep.alpha(ser, return_data[benchmark_label], period=frequency)
        )
        metric_table['Beta'] = return_data[strategy_cols].apply(
            lambda ser: ep.beta(ser, return_data[benchmark_label])
        )
        
        # 超额年化收益
        metric_table['超额收益率'] = (
            metric_table['年化收益率'] - metric_table.loc[benchmark_label, '年化收益率']
        )

    return metric_table.T