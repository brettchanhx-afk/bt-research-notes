# 基于backtrader回测结果绘制vectorbt风格图表
from typing import Dict, List, Tuple, Union
from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import empyrical as ep

from ..BackTestReport.timeseries import gen_drawdown_table
from .utils import max_rel_rescale, min_rel_rescale

# 定义图表配色方案
CHART_COLORS: Dict = {
    "gray": "#7f7f7f",
    "red": "#dc3912",
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "blue": "#1f77b4",
    "purple": "#9467bd",
}

# 基础布局配置
BASE_LAYOUT: Dict = dict(
    xaxis_tickformat="%Y-%m-%d",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        traceorder="normal",
    ),
    hovermode="x unified",
)


def make_figure(
    use_widgets: bool = False, *args, **kwargs
) -> Union[go.Figure, go.FigureWidget]:
    """创建绘图对象
    根据配置决定返回普通Figure或交互式FigureWidget
    """
    # 依据use_widgets参数选择绘图对象类型
    figure_obj = go.FigureWidget(*args, **kwargs) if use_widgets else go.Figure(*args, **kwargs)
    return figure_obj


def _plot_orders(trade_log: pd.DataFrame, figure: go.Figure = None) -> go.Figure:
    """绘制订单标记
    参数:
        trade_log: 交易记录数据框，status=2表示已平仓，1表示未平仓
        figure: 待添加标记的绘图对象，默认新建
    返回:
        包含订单标记的绘图对象
    """
    # 筛选已平仓交易
    closed_trades = trade_log.query("status==2")

    # 拆分开仓/平仓数据
    open_trades = closed_trades[["ref", "datein", "pricein", "size"]]
    close_trades = closed_trades[["ref", "dateout", "priceout", "size"]]

    # 添加开仓标记（红色上三角）
    if not open_trades.empty:
        open_marker = go.Scatter(
            x=open_trades["datein"],
            y=open_trades["pricein"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color=CHART_COLORS["red"],
                size=8,
                line=dict(width=1, color=CHART_COLORS["red"]),
            ),
            name="Buy",
            customdata=open_trades.values,
            hovertemplate=(
                f"ref: %{{customdata[0]}}"
                f"<br>Entry Timestamp: %{{x:%Y-%m-%d}}"
                f"<br>Entry Price: %{{y:.2f}}"
                f"<br>Size: %{{customdata[3]:.6f}}"
            ),
        )
        figure.add_trace(open_marker)

    # 添加平仓标记（绿色下三角）
    if not close_trades.empty:
        close_marker = go.Scatter(
            x=close_trades["dateout"],
            y=close_trades["priceout"],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                color=CHART_COLORS["green"],
                size=8,
                line=dict(width=1, color=CHART_COLORS["green"]),
            ),
            name="Sell",
            customdata=close_trades.values,
            hovertemplate=(
                f"ref: %{{customdata[0]}}"
                f"<br>Exit Timestamp: %{{x:%Y-%m-%d}}"
                f"<br>Exit Price: %{{y:.2f}}"
                f"<br>Size: %{{customdata[3]:.6f}}"
            ),
        )
        figure.add_trace(close_marker)

    # 添加未平仓标记（橙色上三角）
    active_trades = trade_log.query("status==1")[["ref", "datein", "pricein", "size"]]
    if not active_trades.empty:
        active_marker = go.Scatter(
            x=active_trades["datein"],
            y=active_trades["pricein"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color=CHART_COLORS["orange"],
                size=8,
                line=dict(width=1, color=CHART_COLORS["orange"]),
            ),
            name="Active",
            customdata=active_trades.values,
            hovertemplate=(
                f"ref: %{{customdata[0]}}"
                f"<br>Entry Timestamp: %{{x:%Y-%m-%d}}"
                f"<br>Entry Price: %{{y:.2f}}"
                f"<br>Size: %{{customdata[3]:.6f}}"
            ),
        )
        figure.add_trace(active_marker)

    return figure


def _plot_position(
    trade_log: pd.DataFrame,
    show_zones: bool = True,
    x_axis_ref: str = "x",
    y_axis_ref: str = "y",
    figure: go.Figure = None,
) -> go.Figure:
    """绘制持仓标记及盈亏区间
    参数:
        trade_log: 交易记录数据框
        show_zones: 是否显示盈亏区间，默认显示
        x_axis_ref: x轴引用，默认"x"
        y_axis_ref: y轴引用，默认"y"
        figure: 绘图对象，默认新建
    返回:
        包含持仓标记的绘图对象
    """
    # 初始化绘图对象
    if figure is None:
        figure = go.Figure()

    # 筛选已平仓交易
    closed_trades = trade_log.query("status==2")

    # 定义开仓标记悬浮提示模板
    entry_hover_template = (
        f"ref: %{{customdata[0]}}"
        f"<br>Size: %{{customdata[1]:.2f}}"
        f"<br>Entry Timestamp: %{{x:%Y-%m-%d}}"
        f"<br>Avg Entry Price: %{{y:.2f}}"
        f"<br>Direction: %{{customdata[4]}}"
    )

    # 添加开仓标记（蓝色正方形）
    entry_marker = go.Scatter(
        x=trade_log["datein"],
        y=trade_log["pricein"],
        mode="markers",
        marker=dict(
            symbol="square",
            color=CHART_COLORS["blue"],
            size=7,
            line=dict(width=1, color=CHART_COLORS["blue"]),
        ),
        name="Entry",
        customdata=trade_log[["ref", "size", "datein", "pricein", "dir"]],
        hovertemplate=entry_hover_template,
    )
    figure.add_trace(entry_marker)

    # 内部函数：绘制平仓标记
    def _draw_exit_markers(filtered_trades: pd.DataFrame, marker_name: str, marker_color: str, **kwargs) -> None:
        exit_custom_data = filtered_trades.values
        exit_hover_template = (
            f"ref: %{{customdata[0]}}"
            f"<br>Size: %{{customdata[1]:.2f}}"
            f"<br>Exit Timestamp: %{{x:%Y-%m-%d}}"
            f"<br>Avg Exit Price: %{{y:.2f}}"
            f"<br>PnL: %{{customdata[4]:.2f}}"
            f"<br>Return: %{{customdata[5]:.2%}}"
            f"<br>Direction: %{{customdata[6]}}"
            f"<br>Duration: %{{customdata[7]}}"
        )

        exit_marker = go.Scatter(
            x=filtered_trades["dateout"],
            y=filtered_trades["priceout"],
            mode="markers",
            marker=dict(
                symbol="square",
                color=marker_color,
                size=7,
                line=dict(width=1, color=CHART_COLORS[marker_color]),
            ),
            name=marker_name,
            customdata=exit_custom_data,
            hovertemplate=exit_hover_template,
        )
        exit_marker.update(** kwargs)
        figure.add_trace(exit_marker)

    # 绘制不同盈亏状态的平仓标记
    _draw_exit_markers(
        closed_trades.query("pnl==0")[["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]],
        "Exit",
        "gray",
    )
    _draw_exit_markers(
        closed_trades.query("pnl>0")[["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]],
        "Exit - Profit",
        "red",
    )
    _draw_exit_markers(
        closed_trades.query("pnl<0")[["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]],
        "Exit - Loss",
        "green",
    )
    _draw_exit_markers(
        trade_log.query("status==1")[["ref", "size", "dateout", "priceout", "pnl", "pnl%", "dir", "nbars"]],
        "Active",
        "orange",
    )

    # 绘制盈亏区间
    if show_zones:
        # 盈利区间（红色半透明）
        profit_trades = trade_log.query("pnl > 0")
        if not profit_trades.empty:
            for _, row_data in profit_trades.iterrows():
                figure.add_shape(
                    type="rect",
                    xref=x_axis_ref,
                    yref=y_axis_ref,
                    x0=row_data["datein"],
                    y0=row_data["pricein"],
                    x1=row_data["dateout"],
                    y1=row_data["priceout"],
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

        # 亏损区间（绿色半透明）
        loss_trades = trade_log.query("pnl < 0")
        if not loss_trades.empty:
            for _, row_data in loss_trades.iterrows():
                figure.add_shape(
                    type="rect",
                    xref=x_axis_ref,
                    yref=y_axis_ref,
                    x0=row_data["datein"],
                    y0=row_data["pricein"],
                    x1=row_data["dateout"],
                    y1=row_data["priceout"],
                    fillcolor="green",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

    return figure


def plot_against(
    main_series: pd.Series,
    compare_series: pd.Series,
    main_trace_params: Dict = None,
    compare_trace_params: Dict = None,
    figure: go.Figure = None,
) -> go.Figure:
    """绘制两条序列对比图，填充差值区域
    参数:
        main_series: 主序列
        compare_series: 对比序列
        main_trace_params: 主序列绘图参数
        compare_trace_params: 对比序列绘图参数
        figure: 绘图对象，默认新建
    返回:
        包含对比序列的绘图对象
    """
    # 初始化参数默认值
    if main_trace_params is None:
        main_trace_params = {}
    if compare_trace_params is None:
        compare_trace_params = {}

    # 对齐两个序列的索引
    aligned_main, aligned_compare = main_series.align(compare_series, axis=0, join="left")

    # 筛选主序列大于对比序列的区域
    positive_diff_mask = aligned_main > aligned_compare

    # 初始化绘图对象
    if figure is None:
        figure = go.Figure()

    # 填充正差值区域（红色半透明）
    if positive_diff_mask.any():
        positive_fill_series = aligned_main.copy()
        positive_fill_series[~positive_diff_mask] = aligned_compare[~positive_diff_mask]

        # 底层透明轨迹（用于填充）
        figure.add_trace(
            go.Scatter(
                x=aligned_compare.index,
                y=aligned_compare.values,
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                opacity=0,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )
        # 填充区域
        figure.add_trace(
            go.Scatter(
                x=positive_fill_series.index,
                y=positive_fill_series.values,
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                opacity=0,
                fill="tonexty",
                connectgaps=False,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )

    # 筛选主序列小于对比序列的区域
    negative_diff_mask = aligned_main < aligned_compare
    # 填充负差值区域（绿色半透明）
    if negative_diff_mask.any():
        negative_fill_series = aligned_main.copy()
        negative_fill_series[~negative_diff_mask] = aligned_compare[~negative_diff_mask]

        # 底层透明轨迹（用于填充）
        figure.add_trace(
            go.Scatter(
                x=aligned_compare.index,
                y=aligned_compare.values,
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                opacity=0,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )
        # 填充区域
        figure.add_trace(
            go.Scatter(
                x=negative_fill_series.index,
                y=negative_fill_series.values,
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                fillcolor="rgba(0, 128, 0, 0.3)",
                opacity=0,
                fill="tonexty",
                connectgaps=False,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        )

    # 添加主序列轨迹
    figure.add_trace(go.Scatter(x=aligned_main.index, y=aligned_main.values, **main_trace_params))
    # 处理对比序列隐藏逻辑
    if compare_trace_params == "hidden":
        compare_trace_params = dict(
            line=dict(color="rgba(0, 0, 0, 0)", width=0),
            opacity=0.0,
            hoverinfo="skip",
            showlegend=False,
            name=None,
        )
    # 添加对比序列轨迹
    figure.add_trace(go.Scatter(x=aligned_compare.index, y=aligned_compare.values, **compare_trace_params))

    return figure


def plot_position(
    price_series: pd.Series,
    trade_log: pd.DataFrame,
    use_widgets: bool,
    **layout_params,
) -> go.Figure:
    """绘制持仓图表（主图+持仓标记）
    参数:
        price_series: 价格时序序列
        trade_log: 交易记录数据框
        use_widgets: 是否使用交互式组件
        layout_params: 布局参数
    返回:
        持仓图表对象
    """
    # 创建基础绘图对象（价格线）
    figure = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Scatter(
                x=price_series.index, y=price_series.values, line=dict(color=CHART_COLORS["blue"]), name="Close"
            )
        ],
    )

    # 更新布局参数
    if layout_params:
        layout_params.update(BASE_LAYOUT)
    else:
        layout_params = deepcopy(BASE_LAYOUT)
        del layout_params["hovermode"]  # 移除悬浮模式配置

    figure.update_layout(** layout_params)
    # 添加持仓标记
    return _plot_position(trade_log, figure=figure)


def plot_orders(
    price_series: pd.Series,
    trade_log: pd.DataFrame,
    use_widgets: bool,
    **layout_params,
) -> go.Figure:
    """绘制订单图表（主图+订单标记）
    参数:
        price_series: 价格时序序列
        trade_log: 交易记录数据框
        use_widgets: 是否使用交互式组件
        layout_params: 布局参数
    返回:
        订单图表对象
    """
    # 更新布局参数
    if layout_params:
        layout_params.update(BASE_LAYOUT)
    else:
        layout_params = deepcopy(BASE_LAYOUT)
        del layout_params["hovermode"]  # 移除悬浮模式配置

    # 创建基础绘图对象（价格线）
    figure = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Scatter(
                x=price_series.index, y=price_series, line=dict(color=CHART_COLORS["blue"]), name="Close"
            )
        ],
    )
    figure.update_layout(** layout_params)
    # 添加订单标记
    return _plot_orders(trade_log, figure=figure)


def plot_cumulative(
    returns_series: pd.Series,
    benchmark_returns: pd.Series = None,
    initial_value: float = 0.0,
    fill_benchmark_gap: bool = False,
    main_line_params: Dict = None,
    benchmark_line_params: Dict = None,
    figure: go.Figure = None,
    use_widgets: bool = False,
    **layout_params,
) -> go.Figure:
    """绘制累计收益曲线
    参数:
        returns_series: 收益序列
        benchmark_returns: 基准收益序列
        initial_value: 初始值
        fill_benchmark_gap: 是否填充与基准的差值区域
        main_line_params: 主曲线参数
        benchmark_line_params: 基准曲线参数
        figure: 绘图对象
        use_widgets: 是否使用交互式组件
        layout_params: 布局参数
    返回:
        累计收益图表对象
    """
    # 初始化绘图对象
    if figure is None:
        figure = make_figure(use_widgets=use_widgets)

    # 更新布局参数
    if layout_params:
        layout_params.update(BASE_LAYOUT)
    else:
        layout_params = BASE_LAYOUT
    figure.update_layout(** layout_params)

    # 绘制基准收益曲线
    if benchmark_returns is not None:
        # 对齐收益序列与基准序列
        aligned_returns, aligned_benchmark = returns_series.align(benchmark_returns, axis=0, join="left")
        # 初始化基准曲线参数
        if benchmark_line_params is None:
            benchmark_line_params = {}
        benchmark_line_params.update(dict(line=dict(color=CHART_COLORS["gray"]), name="Benchmark"))
        # 计算基准累计收益
        benchmark_cum_returns = ep.cum_returns(aligned_benchmark, initial_value)
        # 添加基准曲线
        figure.add_trace(
            go.Scatter(
                x=benchmark_cum_returns.index,
                y=benchmark_cum_returns.values,
                **benchmark_line_params,
            )
        )

    # 绘制主收益曲线
    if main_line_params is None:
        main_line_params = {}
    main_line_params.update(dict(line=dict(color=CHART_COLORS["purple"])))
    # 计算主累计收益
    cum_returns = ep.cum_returns(returns_series, initial_value)

    # 填充与基准的差值区域
    if fill_benchmark_gap:
        figure = plot_against(
            cum_returns,
            benchmark_cum_returns,
            trace_kwargs=main_line_params,
            other_trace_kwargs=benchmark_line_params,
            fig=figure,
        )
    else:
        # 隐藏基准线，填充与初始值的差值
        benchmark_line_params = "hidden"
        # 创建初始值序列
        zero_series = pd.Series(index=cum_returns.index, data=[initial_value] * len(cum_returns))
        figure = plot_against(
            cum_returns,
            zero_series,
            trace_kwargs=main_line_params,
            other_trace_kwargs=benchmark_line_params,
            fig=figure,
        )

    # 添加初始值参考线
    figure.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        y0=initial_value,
        x1=1,
        y1=initial_value,
        line=dict(
            color="gray",
            dash="dash",
        ),
    )

    return figure


def plot_underwater(
    returns_series: pd.Series,
    x_axis_ref: str = "x",
    y_axis_ref: str = "y",
    use_widgets: bool = False,
    **layout_params,
) -> go.Figure:
    """绘制回撤水下图
    参数:
        returns_series: 收益序列
        x_axis_ref: x轴引用
        y_axis_ref: y轴引用
        use_widgets: 是否使用交互式组件
        layout_params: 布局参数
    返回:
        回撤水下图对象
    """
    # 更新布局参数
    if layout_params:
        layout_params.update(BASE_LAYOUT)
    else:
        layout_params = BASE_LAYOUT

    # 计算累计收益（初始值1）和最大回撤
    cum_returns_series = ep.cum_returns(returns_series, 1)
    max_drawdown_series = cum_returns_series / cum_returns_series.cummax() - 1

    # 创建绘图对象（填充回撤区域）
    figure = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Scatter(
                x=max_drawdown_series.index,
                y=max_drawdown_series.values,
                fillcolor="rgba(220,57,18,0.3000)",
                line=dict(color=CHART_COLORS["red"]),
                fill="tozeroy",
                name="UnderWater",
                hovertemplate=(
                    f"<br>Drawdown:%{{y:.2%}}" f"<br>Date:%{{x:%Y-%m-%d}}"),
            )
        ],
    )

    # 更新布局
    figure.update_layout(** layout_params)

    # 添加0轴参考线
    figure.add_shape(
        type="line",
        line=dict(color="gray", dash="dash"),
        xref="paper",
        yref="y",
        x0=0,
        y0=0,
        x1=1,
        y1=0,
    )

    # 设置y轴百分比格式
    y_axis_key = "yaxis" + y_axis_ref[1:]
    figure.layout[y_axis_key]["tickformat"] = ".2%"

    return figure


def plot_pnl(
    trade_log: pd.DataFrame,
    use_percent_scale: bool = True,
    marker_size_limits: Tuple[int, int] = (7, 14),
    opacity_limits: Tuple[float, float] = (0.75, 0.9),
    figure: go.Figure = None,
    x_axis_ref: str = "x",
    y_axis_ref: str = "y",
    use_widgets: bool = False,
    **layout_params,
) -> go.Figure:
    """绘制盈亏散点图
    参数:
        trade_log: 交易记录数据框
        use_percent_scale: 是否使用百分比刻度
        marker_size_limits: 标记大小范围
        opacity_limits: 透明度范围
        figure: 绘图对象
        x_axis_ref: x轴引用
        y_axis_ref: y轴引用
        use_widgets: 是否使用交互式组件
        layout_params: 布局参数
    返回:
        盈亏散点图对象
    """
    # 定义轴参数键名
    x_axis_key = "xaxis" + x_axis_ref[1:]
    y_axis_key = "yaxis" + y_axis_ref[1:]

    # 初始化绘图对象
    if figure is None:
        figure = make_figure(use_widgets=use_widgets)

    # 更新布局参数
    if layout_params:
        layout_params.update(BASE_LAYOUT)
    else:
        layout_params = BASE_LAYOUT
    figure.update_layout(** layout_params)

    # 设置y轴百分比格式
    if use_percent_scale:
        y_axis_config = {}
        y_axis_config[y_axis_key] = dict(tickformat=".2%")
        figure.update_layout(** y_axis_config)

    # 计算标记大小和透明度
    abs_pnl_pct = trade_log["pnl%"].abs()
    marker_sizes = min_rel_rescale(abs_pnl_pct, marker_size_limits)
    marker_opacity = max_rel_rescale(abs_pnl_pct, opacity_limits)

    # 定义交易状态掩码
    open_trade_mask = trade_log["status"] == 1
    closed_trade_mask = trade_log["status"] == 2
    closed_profit_mask = closed_trade_mask & (trade_log["pnl"] > 0)
    closed_loss_mask = closed_trade_mask & (trade_log["pnl"] < 0)

    # 内部函数：绘制盈亏散点
    def _draw_pnl_scatter(filter_mask: pd.Series, scatter_name: str, scatter_color: str) -> None:
        # 定义悬浮提示模板
        hover_template = (
            f"ref: %{{customdata[0]}}"
            f"<br>Exit Timestamp: %{{x:%Y-%m-%d}}"
            f"<br>PnL: %{{customdata[1]:.2f}}"
            f"<br>Return: %{{customdata[2]:.2%}}"
        )
        # 准备自定义数据
        custom_data = trade_log.loc[filter_mask, ["ref", "pnl", "pnl%"]]
        
        # 仅在掩码非空时绘制
        if not filter_mask.empty:
            # 选择y轴数据（百分比/绝对值）
            y_values = trade_log.loc[filter_mask, "pnl%"] if use_percent_scale else trade_log.loc[filter_mask, "pnl"]
            
            pnl_scatter = go.Scatter(
                x=trade_log.loc[filter_mask, "dateout"],
                y=y_values,
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color=scatter_color,
                    size=marker_sizes[filter_mask],
                    opacity=marker_opacity[filter_mask],
                    line=dict(width=1, color=CHART_COLORS[scatter_color]),
                ),
                name=scatter_name,
                customdata=custom_data,
                hovertemplate=hover_template,
            )
            figure.add_trace(pnl_scatter)

    # 绘制不同状态的盈亏散点
    _draw_pnl_scatter(closed_profit_mask, "Closed - Profit", "red")
    _draw_pnl_scatter(closed_loss_mask, "Closed - Loss", "green")
    _draw_pnl_scatter(open_trade_mask, "Open", "orange")

    # 添加0轴参考线
    figure.add_shape(
        type="line",
        xref="paper",
        yref=y_axis_ref,
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        line=dict(
            color="gray",
            dash="dash",
        ),
    )

    return figure


def plot_drawdowns(
    returns_series: pd.Series,
    top_count: int = 5,
    initial_value: float = 0.0,
    show_zones: bool = True,
    x_axis_ref: str = "x",
    y_axis_ref: str = "y",
    time_series_params: Dict = None,
    use_widgets: bool = False,
    figure: go.Figure = None,
    **layout_params,
) -> go.Figure:
    """绘制最大回撤图表
    参数:
        returns_series: 收益序列
        top_count: 显示前N大回撤
        initial_value: 初始值（1=净值，0=累计收益）
        show_zones: 是否显示回撤区间
        x_axis_ref: x轴引用
        y_axis_ref: y轴引用
        time_series_params: 时序曲线参数
        use_widgets: 是否使用交互式组件
        figure: 绘图对象
        layout_params: 布局参数
    返回:
        最大回撤图表对象
    """
    # 名称映射字典
    NAME_MAPPING: Dict = {1.0: "Nav", 0.0: "Cum"}

    # 更新布局参数
    if layout_params:
        layout_params.update(BASE_LAYOUT)
    else:
        layout_params = deepcopy(BASE_LAYOUT)
        del layout_params["hovermode"]  # 移除悬浮模式配置

    # 计算累计收益/净值序列
    value_series = ep.cum_returns(returns_series, initial_value)
    # 生成最大回撤表格
    drawdown_df = gen_drawdown_table(returns_series, top_count)
    drawdown_df["id"] = np.arange(top_count) + 1  # 添加回撤ID
    drawdown_df["peak value"] = drawdown_df["Peak date"].map(value_series)
    drawdown_df["valley value"] = drawdown_df["Valley date"].map(value_series)
    # 处理恢复日期为空的情况
    drawdown_df["recovery value"] = drawdown_df["Recovery date"].map(
        lambda dt: value_series.to_dict().get(dt, value_series.iloc[-1])
    )
    drawdown_df["Net drawdown in %"] /= 100  # 转换为小数

    # 初始化时序曲线参数
    if time_series_params is None:
        time_series_params = {}
    if "name" not in time_series_params:
        time_series_params["name"] = NAME_MAPPING.get(initial_value, "Value")
    time_series_params.update(dict(line=dict(color=CHART_COLORS["blue"])))

    # 初始化绘图对象
    if figure is None:
        figure = make_figure(use_widgets=use_widgets)

    # 添加累计收益/净值曲线
    figure.add_trace(go.Scatter(x=value_series.index, y=value_series.values, **time_series_params))
    figure.update_layout(** layout_params)

    # 绘制峰值标记（蓝色菱形）
    peak_custom_data = drawdown_df[["id", "Peak date"]].values
    peak_scatter = go.Scatter(
        x=drawdown_df["Peak date"],
        y=drawdown_df["peak value"],
        mode="markers",
        marker=dict(
            symbol="diamond",
            color=CHART_COLORS["blue"],
            size=7,
            line=dict(width=1, color=CHART_COLORS["blue"]),
        ),
        name="Peak",
        customdata=peak_custom_data,
        hovertemplate=(
            f"<br>Top Drawdowns id:%{{customdata[0]}}"
            f"<br>Peak Date:%{{x:%Y-%m-%d}}"
            f"<br>Peak Value:%{{y:.2f}}"
        ),
    )
    figure.add_trace(peak_scatter)

    # 筛选已恢复的回撤
    recovered_mask = ~drawdown_df["Recovery date"].isna()

    if recovered_mask.any():
        # 绘制谷值标记（红色菱形）
        valley_custom_data = drawdown_df.loc[recovered_mask, ["id", "Net drawdown in %", "Valley Duration"]].values
        valley_scatter = go.Scatter(
            x=drawdown_df.loc[recovered_mask, "Valley date"],
            y=drawdown_df.loc[recovered_mask, "valley value"],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=CHART_COLORS["red"],
                size=7,
                line=dict(width=1, color=CHART_COLORS["red"]),
            ),
            name="Valley",
            customdata=valley_custom_data,
            hovertemplate=(
                f"<br>Top Drawdowns id: %{{customdata[0]}}"
                f"<br>Valley Date: %{{x:%Y-%m-%d}}"
                f"<br>Valley Value: %{{y:.2f}}"
                f"<br>Drawdown: %{{customdata[1]:.2%}}"
                f"<br>Duration: %{{customdata[2]}}"
            ),
        )
        figure.add_trace(valley_scatter)

        # 绘制回撤区间（红色半透明）
        if show_zones:
            for _, row_data in drawdown_df.loc[recovered_mask].iterrows():
                figure.add_shape(
                    type="rect",
                    xref=x_axis_ref,
                    yref="paper",
                    x0=row_data["Peak date"],
                    y0=0,
                    x1=row_data["Valley date"],
                    y1=1,
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

        # 绘制恢复标记（绿色菱形）
        recovery_custom_data = drawdown_df.loc[recovered_mask, ["id", "Net drawdown in %", "End Duration", "Duration"]].values
        recovery_scatter = go.Scatter(
            x=drawdown_df.loc[recovered_mask, "Recovery date"],
            y=drawdown_df.loc[recovered_mask, "recovery value"],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=CHART_COLORS["green"],
                size=7,
                line=dict(width=1, color=CHART_COLORS["green"]),
            ),
            name="Recovery/Peak",
            customdata=recovery_custom_data,
            hovertemplate=(
                f"<br>Top Drawdowns id: %{{customdata[0]}}"
                f"<br>Recovery/Peak Date: %{{x:%Y-%m-%d}}"
                f"<br>Recovery/Peak Value: %{{y:.2f}}"
                f"<br>Return: %{{customdata[1]:.2%}}"
                f"<br>End Duration: %{{customdata[2]}}"
                f"<br>Duration: %{{customdata[3]}}"
            ),
        )
        figure.add_trace(recovery_scatter)

        # 绘制恢复区间（绿色半透明）
        if show_zones:
            for _, row_data in drawdown_df.loc[recovered_mask].iterrows():
                figure.add_shape(
                    type="rect",
                    xref=x_axis_ref,
                    yref="paper",
                    x0=row_data["Valley date"],
                    y0=0,
                    x1=row_data["Recovery date"],
                    y1=1,
                    fillcolor="green",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

    # 处理未恢复的回撤（活跃回撤）
    active_mask = ~recovered_mask
    if active_mask.any():
        peak_date = drawdown_df.loc[active_mask, "Peak date"].values[0]
        active_date = value_series.index[-1]
        # 计算活跃回撤持续时间
        active_custom_data = drawdown_df.loc[active_mask, ["id", "Net drawdown in %", "Duration"]].copy()
        active_custom_data["Duration"] = len(pd.date_range(peak_date, active_date, freq="B"))
        active_custom_data = active_custom_data.values

        # 绘制活跃回撤标记（橙色菱形）
        active_scatter = go.Scatter(
            x=[active_date],
            y=[drawdown_df.loc[active_mask, "recovery value"].values[0]],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=CHART_COLORS["orange"],
                size=7,
                line=dict(width=1, color=CHART_COLORS["orange"]),
            ),
            name="Active",
            customdata=active_custom_data,
            hovertemplate=(
                f"<br>Top Drawdowns id: %{{customdata[0]}}"
                f"<br>Active Date: %{{x:%Y-%m-%d}}"
                f"<br>Active Value: %{{y:.2}}"
                f"<br>Return: %{{customdata[1]:.2%}}"
                f"<br>Duration: %{{customdata[2]}}"
            ),
        )
        figure.add_trace(active_scatter)

        # 绘制活跃回撤区间（橙色半透明）
        if show_zones:
            figure.add_shape(
                type="rect",
                xref=x_axis_ref,
                yref="paper",
                x0=pd.to_datetime(peak_date),
                y0=0,
                x1=pd.to_datetime(active_date),
                y1=1,
                fillcolor="orange",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

    return figure


def plot_annual_returns(
    returns_data: Union[pd.Series, List], use_widgets: bool = False
) -> go.Figure:
    """绘制年度收益条形图
    参数:
        returns_data: 收益数据（Series或backtrader结果列表）
        use_widgets: 是否使用交互式组件
    返回:
        年度收益图表对象
    """
    # 处理不同输入类型
    if isinstance(returns_data, pd.Series):
        annual_returns = ep.aggregate_returns(returns_data, "yearly")
    elif isinstance(returns_data, List):
        annual_returns = pd.Series(returns_data[0].analyzers._AnnualReturn.get_analysis())
    else:
        raise ValueError("returns类型必须为pd.Series或bt_result.result")

    # 转换索引为字符串
    annual_returns.index = annual_returns.index.map(str)
    # 按收益正负设置颜色
    bar_colors = ["crimson" if val > 0 else "#7a9e9f" for val in annual_returns]

    # 创建条形图
    figure = make_figure(
        use_widgets=use_widgets,
        data=go.Bar(
            x=annual_returns.values,
            y=annual_returns.index,
            orientation="h",
            marker_color=bar_colors,
            name='年度收益情况',
            hovertemplate=(f'<br>Year:%{{y}}' f'<br>Return:%{{x:.2%}}')
        ),
    )

    # 更新布局
    figure.update_layout(
        title={"text": "Annual returns", "x": 0.5, "y": 0.9},
        yaxis_title="Year",
        xaxis_tickformat=".2%",
        yaxis_tickformat='%Y',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    # 添加均值参考线
    figure.add_vline(x=annual_returns.mean(), line_dash="dash")

    return figure


def plot_monthly_heatmap(returns_series: pd.Series, use_widgets: bool = False) -> go.Figure:
    """绘制月度收益热力图
    参数:
        returns_series: 收益序列
        use_widgets: 是否使用交互式组件
    返回:
        月度收益热力图对象
    """
    # 计算月度收益并重构为矩阵
    monthly_returns = ep.aggregate_returns(returns_series, "monthly")
    monthly_returns_matrix = monthly_returns.unstack().round(3)

    # 创建热力图
    figure = make_figure(
        use_widgets=use_widgets,
        data=go.Heatmap(
            z=monthly_returns_matrix.values,
            x=monthly_returns_matrix.columns.map(str),
            y=monthly_returns_matrix.index.map(str),
            text=monthly_returns_matrix.values,
            texttemplate="%{text:.2%}",
            hovertemplate=(f'<br>Year:%{{y}}年' f'<br>Month:%{{x}}月' f'<br>Return:%{{z:.2%}}')
        ),
    )

    # 更新布局
    figure.update_layout(
        title={"text": "Monthly returns (%)", "x": 0.5, "y": 0.9},
        yaxis_title="Year",
        xaxis_title="Month",
    )
    return figure


def plot_monthly_dist(returns_series: pd.Series, use_widgets: bool = False) -> go.Figure:
    """绘制月度收益分布直方图
    参数:
        returns_series: 收益序列
        use_widgets: 是否使用交互式组件
    返回:
        月度收益分布图表对象
    """
    # 计算月度收益
    monthly_returns_df = pd.DataFrame(ep.aggregate_returns(returns_series, "monthly"), columns=["Returns"])
    # 创建直方图
    hist_figure = px.histogram(monthly_returns_df, x="Returns")
    # 计算均值并添加参考线
    mean_return = monthly_returns_df["Returns"].mean()
    hist_figure.add_vline(
        x=mean_return,
        line_dash="dash",
        annotation_text="Mean:{:.2f}".format(mean_return),
    )
    # 更新布局
    hist_figure.update_layout(
        hovermode="x unified",
        title={"text": "Distribution of monthly returns", "x": 0.5, "y": 0.9},
        yaxis_title="Number of months",
        xaxis_tickformat=".2%",
        xaxis_title="Returns",
    )
    # 转换为指定类型的绘图对象
    return make_figure(use_widgets, hist_figure)


def plot_table(
    data_frame: pd.DataFrame, use_widgets: bool = False, index_label: str = ""
) -> go.Figure:
    """绘制数据表格
    参数:
        data_frame: 数据框
        use_widgets: 是否使用交互式组件
        index_label: 索引列名称
    返回:
        表格绘图对象
    """
    # 设置索引名称并重置索引
    data_frame.index.names = [index_label]
    table_data = data_frame.reset_index()

    # 表格样式配置
    header_color = "grey"

    # 创建表格
    figure = make_figure(
        use_widgets=use_widgets,
        data=[
            go.Table(
                header=dict(
                    values=table_data.columns,
                    line_color="darkslategray",
                    fill_color=header_color,
                    align=["left", "center"],
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=table_data.T.values,
                    line_color="darkslategray",
                    align=["left", "center"],
                    font=dict(color="darkslategray", size=11),
                ),
            )
        ],
    )

    return figure