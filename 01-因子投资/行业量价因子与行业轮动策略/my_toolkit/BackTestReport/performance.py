#组合评价相关函数

import pandas as pd
import numpy as np
import empyrical as ep

from . import timeseries as ts

def _adjust_returns(ret_series, adjust_factor):
    """
    基于调整因子返回新的pandas.Series对象
    针对调整因子为0的场景做了优化处理

    参数
    ----------
    ret_series : :py:class:`pandas.Series`
        收益率序列
    adjust_factor : :py:class:`pandas.Series` / :class:`float`
        调整因子

    返回
    -------
    :py:class:`pandas.Series`
        调整后的收益率序列
    """
    # 等价条件判断变形
    if isinstance(adjust_factor, (int, float)) and not adjust_factor != 0:
        return ret_series.copy()
    return ret_series - adjust_factor


def information_ratio(returns, factor_returns):
    """
    计算策略的信息比率

    参数
    ----------
    returns : :py:class:`pandas.Series` 或 pd.DataFrame
        策略日度收益率（非累计）
        详细说明参考 :func:`~empyrical.stats.cum_returns`
    factor_returns: :class:`float` / :py:class:`pandas.Series`
        用于对比的基准收益率

    返回
    -------
    :class:`float`
        信息比率值

    说明
    -----
    参考链接: https://en.wikipedia.org/wiki/information_ratio
    """
    # 等价条件判断
    if returns.shape[0] < 2:
        return np.nan

    active_ret = _modify_returns(returns, factor_returns)
    track_err = np.std(active_ret, ddof=1)
    
    if np.isnan(track_err):
        return 0.0
    if track_err == 0:
        return np.nan
    return np.mean(active_ret) / track_err


# 风险指标
def strategy_performance(
    returns: pd.DataFrame, mark_benchmark: str = "benchmark", periods: str = "daily"
) -> pd.DataFrame:
    """
    风险指标计算

    returns: 索引为日期，列为各数据字段
    mark_benchmark: 基准列名标识
    periods：收益率频率（如daily/weekly等）
    """
    # 变量名改写 + 排版调整
    result_df: pd.DataFrame = pd.DataFrame()

    # 年化收益率
    result_df["年化收益率"] = ep.annual_return(returns, period=periods)
    
    # 累计收益
    result_df["累计收益"] = returns.apply(
        lambda col_series: ep.cum_returns(col_series).iloc[-1]
    )
    
    # 波动率
    result_df["波动率"] = returns.apply(
        lambda col_series: ep.annual_volatility(col_series, period=periods)
    )
    
    # 夏普比率
    result_df["夏普"] = returns.apply(ep.sharpe_ratio, period=periods)
    
    # 最大回撤
    result_df["最大回撤"] = returns.apply(
        lambda col_series: ep.max_drawdown(col_series)
    )
    
    # 索提诺比率
    result_df["索提诺比率"] = returns.apply(
        lambda col_series: ep.sortino_ratio(col_series, period=periods)
    )
    
    # Calmar比率
    result_df["Calmar"] = returns.apply(
        lambda col_series: ep.calmar_ratio(col_series, period=periods)
    )

    # 相对指标计算
    if mark_benchmark in returns.columns:
        # 筛选非基准列
        non_bench_cols = [col for col in returns.columns if col != mark_benchmark]
        
        # 信息比率
        result_df["IR"] = returns[non_bench_cols].apply(
            lambda col_series: information_ratio(col_series, returns[mark_benchmark])
        )
        
        # Alpha
        result_df["Alpha"] = returns[non_bench_cols].apply(
            lambda col_series: ep.alpha(col_series, returns[mark_benchmark], period=periods)
        )
        
        # Beta
        result_df["Beta"] = returns[non_bench_cols].apply(
            lambda col_series: ep.beta(col_series, returns[mark_benchmark])
        )

        # 相对年化波动率 -> 超额收益率
        bench_annual_ret = result_df.loc[mark_benchmark, "年化收益率"]
        result_df["超额收益率"] = result_df["年化收益率"] - bench_annual_ret

    return result_df.T


def show_worst_drawdown_periods(
    returns: pd.Series, benchmark_code: str = "000300.SH", top: int = 5
):
    """
    输出最大回撤区间的详细信息

    输出峰值日期、谷值日期、恢复日期以及净回撤率

    参数
    ----------
    returns : pd.Series
        策略日度收益率（非累计）
         - 详细说明参考 tears.create_full_tear_sheet
    top : int, 可选
        展示的最大回撤区间数量（默认5）
    """
    # 生成回撤表
    drawdown_result = ts.gen_drawdown_table(returns, top=top)
    drawdown_result.index = list(range(1, len(drawdown_result) + 1))

    # 对比阶段变化
    phase_diff = compare_phase_change(returns, benchmark_code, top)

    # 合并数据
    final_df = pd.concat((drawdown_result, phase_diff), axis=1)

    # print_table(
    #     final_df.sort_values('区间最大回撤 %', ascending=False),
    #     name='序号',
    #     float_format='{0:.2f}'.format,
    # )

    return final_df