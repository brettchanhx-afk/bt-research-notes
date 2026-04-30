## 时间序列最大回撤相关计算
import numpy as np
import datetime as dt
import pandas as pd
from typing import List, Tuple
import empyrical as ep


def get_max_drawdown_underwater(
    underwater: pd.Series,
) -> Tuple[dt.datetime, dt.datetime, dt.datetime]:
    """
    根据已计算滚动回撤的'水下收益率'序列，确定最大回撤的峰值、谷值和恢复日期。
    
    水下收益率序列（underwater）是预先计算好的滚动回撤数据。

    参数
    ----------
    underwater : pd.Series
        策略的水下收益率（滚动回撤）数据。

    返回值
    -------
    peak : datetime
        最大回撤的峰值日期。
    valley : datetime
        最大回撤的谷值日期。
    recovery : datetime
        最大回撤的恢复日期。
    """
    # 确定回撤谷值日期（周期结束点）
    valley_date = underwater.idxmin()
    # 查找峰值日期（谷值之前最后一个零值点）
    peak_date = underwater[:valley_date][underwater[:valley_date] == 0].index[-1]
    # 查找恢复日期（谷值之后第一个零值点）
    try:
        recovery_date = underwater[valley_date:][underwater[valley_date:] == 0].index[0]
    except IndexError:
        recovery_date = np.nan  # 回撤尚未恢复
    return peak_date, valley_date, recovery_date


def get_top_drawdowns(returns: pd.Series, top: int = 10) -> List:
    """
    找出按回撤幅度排序的顶级回撤区间。

    参数
    ----------
    returns : pd.Series
        策略的日度收益率（非累计）。
        - 详细说明参考 tears.create_full_tear_sheet。
    top : int, 可选
        要查找的顶级回撤数量（默认值为10）。

    返回值
    -------
    drawdowns : list
        包含回撤峰值、谷值和恢复日期的列表，参考get_max_drawdown函数。
    """
    returns_copy = returns.copy()
    cumulative_returns = ep.cum_returns(returns_copy, 1.0)
    cumulative_max = np.maximum.accumulate(cumulative_returns)
    underwater_series = cumulative_returns / cumulative_max - 1

    drawdown_list = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater_series)
        # 剔除当前回撤周期的数据
        if not pd.isnull(recovery):
            drop_index = underwater_series[peak:recovery].index[1:-1]
            underwater_series.drop(drop_index, inplace=True)
        else:
            # 回撤尚未结束，仅保留峰值之前的数据
            underwater_series = underwater_series.loc[:peak]

        drawdown_list.append((peak, valley, recovery))
        # 无数据或无回撤时终止循环
        if (len(returns_copy) == 0) or (len(underwater_series) == 0) or (np.min(underwater_series) == 0):
            break

    return drawdown_list


def gen_drawdown_table(returns: pd.Series, top: int = 10) -> pd.DataFrame:
    """
    将顶级回撤区间整理为结构化表格。

    参数
    ----------
    returns : pd.Series
        策略的日度收益率（非累计）。
        - 详细说明参考 tears.create_full_tear_sheet。
    top : int, 可选
        要查找的顶级回撤数量（默认值为10）。

    返回值
    -------
    df_drawdowns : pd.DataFrame
        包含顶级回撤详细信息的DataFrame。
    """
    # 计算累计收益率
    cum_returns_series: pd.Series = ep.cum_returns(returns, 1.0)
    # 获取顶级回撤周期列表
    top_drawdown_periods: List = get_top_drawdowns(returns, top=top)
    
    # 初始化回撤表格
    drawdown_df = pd.DataFrame(
        index=list(range(top)),
        columns=[
            "Net drawdown in %",
            "Peak date",
            "Valley date",
            "Recovery date",
            "Valley Duration",
            "End Duration",
            "Duration",
        ],
    )

    # 填充回撤表格数据
    for idx, (peak_dt, valley_dt, recovery_dt) in enumerate(top_drawdown_periods):
        # 计算峰值到谷值的交易日天数
        drawdown_df.loc[idx, "Valley Duration"] = len(
            pd.date_range(peak_dt, valley_dt, freq="B")
        )
        
        if pd.isnull(recovery_dt):
            # 回撤未恢复时填充空值
            drawdown_df.loc[idx, "End Duration"] = np.nan
            drawdown_df.loc[idx, "Duration"] = np.nan
        else:
            # 计算谷值到恢复日、峰值到恢复日的交易日天数
            drawdown_df.loc[idx, "End Duration"] = len(
                pd.date_range(valley_dt, recovery_dt, freq="B")
            )
            drawdown_df.loc[idx, "Duration"] = len(
                pd.date_range(peak_dt, recovery_dt, freq="B")
            )
        
        # 格式化日期列
        drawdown_df.loc[idx, "Peak date"] = peak_dt.to_pydatetime().strftime("%Y-%m-%d")
        drawdown_df.loc[idx, "Valley date"] = valley_dt.to_pydatetime().strftime("%Y-%m-%d")
        
        if isinstance(recovery_dt, float):
            drawdown_df.loc[idx, "Recovery date"] = recovery_dt
        else:
            drawdown_df.loc[idx, "Recovery date"] = recovery_dt.to_pydatetime().strftime(
                "%Y-%m-%d"
            )
        
        # 计算回撤幅度（百分比）
        drawdown_df.loc[idx, "Net drawdown in %"] = (
            (cum_returns_series.loc[peak_dt] - cum_returns_series.loc[valley_dt]) / cum_returns_series.loc[peak_dt]
        ) * 100

    # 转换日期列为datetime类型
    drawdown_df["Peak date"] = pd.to_datetime(drawdown_df["Peak date"])
    drawdown_df["Valley date"] = pd.to_datetime(drawdown_df["Valley date"])
    drawdown_df["Recovery date"] = pd.to_datetime(drawdown_df["Recovery date"])

    return drawdown_df
