from typing import List, Tuple

import backtrader as bt
import empyrical as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib.ticker import FuncFormatter

from .plotting_utils import (calculate_bin_means, get_strategy_return,
                             trans_minute_to_daily,get_strategy_cumulative_return)


def plot_signal_vs_forward_returns(
    df: pd.DataFrame,
    signal_col: str = "signal",
    forward_returns_col: str = "forward_returns",
    step: float = 0.01,
    threshold: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    ax: plt.Axes = None,
):
    
    bin_means: pd.DataFrame = calculate_bin_means(
        df, signal_col, forward_returns_col, step
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    filter_bins: pd.Series = bin_means.query("count>@threshold")["mean"]
    bin_centers = [interval.mid for interval in filter_bins.index]

    ax.bar(bin_centers, filter_bins, width=0.01, align="center", color="DarkRed")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Signal")
    ax.set_ylabel("Average Forward Returns")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2%}"))
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(bin_centers, filter_bins.cumsum(), color="gold")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2%}"))
    ax2.set_ylabel("Cumulative Sum of Forward Returns")

    return ax


def displot_signal(df:pd.DataFrame)->plt.Axes:
       
    unstack_df: pd.DataFrame = df.unstack().to_frame(name="signal").reset_index()

    grid = sns.FacetGrid(unstack_df, col="code", col_wrap=3, sharex=False, sharey=False)
    grid.map(sns.histplot, "signal")
    ubound: pd.Series = df.mean() + df.std()
    lbound: pd.Series = df.mean() - df.std()
    for ax, uline, lline in zip(grid.axes, ubound.items(), lbound.items()):

        ax.axvline(
            uline[1],
            color="green",
            linewidth=1,
            label=f"+1 Std Dev ({uline[1]:.2f})",
        )
        ax.axvline(
            lline[1],
            color="green",
            linewidth=1,
            label=f"-1 Std Dev ({lline[1]:.2f})",
        )
        ax.legend()

    return grid

def displot_signal_vs_forward_returns(signal_vs_forward_return:pd.DataFrame,step:float=0.01,threshold:int=5)->plt.Axes:
    
    col: int = signal_vs_forward_return.columns.levels[0].size

    fig, axes = plt.subplots(col, 1, figsize=(12, 5 * col))

    for ax, (code, df) in zip(axes, signal_vs_forward_return.groupby(level=0, axis=1)):
        plot_signal_vs_forward_returns(df=df[code],step=step,threshold=threshold, ax=ax)
        ax.set_title(f"{code} 信号与未来收益")

    plt.subplots_adjust(hspace=0.3)

    return axes


def plot_cumulative_return(
    strat: bt.Cerebro,
    benchmark: pd.Series = None,
    ax: plt.Axes = None,
    figure: Tuple[int, int] = (16, 4),
    title: str = "",
) -> plt.Axes:
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figure)

    if title == "":
        title = "Cumulative Return"

       
    get_strategy_cumulative_return(strat,1).plot(ax=ax, label="strategy", color="red")

    if benchmark is not None:
        # 必须是日度数据
        bench:pd.Series = benchmark.resample("D").last().dropna()
        # 基准曲线
        (bench / bench.iloc[0]).plot(color="darkgray", label="benchmark", ax=ax)

    plt.title(title)
    ax.grid()
    ax.axhline(1, color="black", ls="-")
    ax.legend()
    return ax


def plot_annual_returns(
    strat, ax: plt.Axes = None, figsize: Tuple[int, int] = (6, 5), title: str = ""
) -> plt.Axes:
    
    returns: pd.Series = get_strategy_return(strat)

    yearly_returns: pd.Series = ep.aggregate_returns(returns, "yearly")
    # 创建颜色列表，小于0为绿色，大于0为红色
    colors: List[str] = ["green" if val < 0 else "red" for val in yearly_returns]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if title == "":
        title = "Annual Returns"
    ax = yearly_returns.plot.barh(color=colors)
    # 添加垂直线
    ax.axvline(0, color="black", linestyle="-", lw=0.4)
    # 设置 x 轴为百分比显示
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title(title)
    return ax


def plot_intraday_signal(
    signal: pd.DataFrame,
    watch_dt: str,
    ax: plt.Axes = None,
    figsize: Tuple[int, int] = (12, 3),
) -> plt.Axes:

    watch_dt: str = pd.to_datetime(watch_dt).strftime("%Y-%m-%d")

    slice_df: pd.DataFrame = signal.loc[f"{watch_dt} 09:30:00":f"{watch_dt} 15:00:00"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    slice_df["upperbound"].plot(ls="--", color="red", label="upperbound", ax=ax)
    slice_df["signal"].plot(ls="-", color="darkgray", label="close", ax=ax)
    slice_df["lowerbound"].plot(ls="--", color="green", label="lowerbound", ax=ax)
    ax.axvline(x=f"{watch_dt} 10:29:00", color="red", lw=1.1, alpha=0.5)
    ax.axvline(x=f"{watch_dt} 11:29:00", color="red", lw=1.1, alpha=0.5)
    ax.axvline(x=f"{watch_dt} 13:59:00", color="red", lw=1.1, alpha=0.5)
    ax.legend()
    ax.set_title(f"Signal on {watch_dt}")
    return ax


def plot_annual_time_series(strat, minute_data: pd.Series = None) -> so.Plot:
   
    returns: pd.Series = get_strategy_return(strat)

    if minute_data is not None:
        benchmark_rets: pd.Series = trans_minute_to_daily(minute_data).pct_change()

        df: pd.DataFreame = pd.concat(
            (returns, benchmark_rets), axis=1, keys=["strategy", "benchmark"]
        )

    else:

        df: pd.DataFrame = returns.to_frame("strategy")

    group_cum: pd.DataFrame = df.groupby(df.index.year).apply(
        lambda x: ep.cum_returns(x)
    )
    group_cum.index.names = ["year", "date"]
    group_cum.reset_index(inplace=True)

    p = (
        so.Plot(group_cum, x=group_cum["date"].dt.day_of_year, y="strategy")
        .facet(col="year", wrap=3)
        .add(so.Line(color="red"))
    )

 
    if "benchmark" in group_cum.columns:
    
        p = p.add(so.Line(color="darkgray"), y="benchmark")

    return p.layout(size=(16, 4)).label(title="{} year".format)