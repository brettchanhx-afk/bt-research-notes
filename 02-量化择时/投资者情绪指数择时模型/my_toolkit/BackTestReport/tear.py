from typing import Dict, List, Union
from collections import namedtuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .utils import get_value_from_traderanalyzerdict
from .performance import strategy_performance
from ..VectorbtStylePlotting import (
    plot_table,
    plot_cumulative,
    plot_drawdowns,
    plot_underwater,
    plot_annual_returns,
    plot_monthly_heatmap,
    plot_monthly_dist,
    plot_orders,
    plot_pnl,
    plot_position,
)


def get_transactions_frame(backtest_result: List) -> pd.DataFrame:
    """将交易记录转换为DataFrame格式

    Args:
        backtest_result (List): 回测结果列表

    Returns:
        pd.DataFrame: 以日期为索引的交易记录DataFrame
    """
    trans_data: Dict = backtest_result[0].analyzers._Transactions.get_analysis()
    df_trans: pd.DataFrame = pd.DataFrame(
        index=list(trans_data.keys()),
        data=np.squeeze(list(trans_data.values())),
        columns=["amount", "price", "sid", "symbol", "value"],
    )
    df_trans.index.names = ["date"]
    df_trans.index = pd.to_datetime(df_trans.index)
    return df_trans


def get_trade_flag(backtest_result: List) -> pd.DataFrame:
    """生成交易买卖标记数据

    Args:
        backtest_result (List): 回测结果列表

    Returns:
        pd.DataFrame: 包含开仓/平仓日期、价格的DataFrame
        注：若priceout/dateout为np.nan表示该交易尚未平仓
    """
    df_transactions = get_transactions_frame(backtest_result)
    df_transactions = df_transactions.astype(
        {"amount": np.int32, "price": np.float32, "value": np.float32}
    )

    trans_count = len(df_transactions)
    list_trade_date = df_transactions.index.tolist()
    list_price = df_transactions["price"].tolist()
    
    # 处理奇数笔交易的情况，补充空值
    if trans_count % 2 != 0:
        list_trade_date.append(np.nan)
        list_price.append(np.nan)
        trans_count += 1

    # 重构交易对数据
    arr_date_flag = np.array([(list_trade_date[i-2:i]) for i in np.arange(2, trans_count + 1, 2)])
    arr_price_flag = np.array([(list_price[i-2:i]) for i in np.arange(2, trans_count + 1, 2)])

    return pd.DataFrame(
        data=np.hstack((arr_date_flag, arr_price_flag)),
        columns=["datein", "dateout", "pricein", "priceout"],
    )


# 回测报告相关函数
def get_backtest_report(price_series: pd.Series, backtest_result: List) -> pd.DataFrame:
    """生成回测业绩报告

    Args:
        price_series (pd.Series): 价格序列
        backtest_result (List): 回测结果列表

    Returns:
        pd.DataFrame: 策略与基准的业绩指标DataFrame
    """
    strategy_returns: pd.Series = pd.Series(backtest_result[0].analyzers._TimeReturn.get_analysis())
    benchmark_returns = price_series.pct_change()

    df_returns: pd.DataFrame = pd.concat((strategy_returns, benchmark_returns), axis=1)
    df_returns.columns = ["策略", "benchmark"]

    return strategy_performance(df_returns)


def create_trade_report_table(trade_analyzer_data: Dict) -> pd.DataFrame:
    """构建交易统计报告表格

    Args:
        trade_analyzer_data (Dict): 交易分析器输出数据

    Returns:
        pd.DataFrame: 交易统计结果表格
    """
    win_count = get_value_from_traderanalyzerdict(trade_analyzer_data, "won", "total")
    profit_amount = get_value_from_traderanalyzerdict(
        trade_analyzer_data, "won", "pnl", "total"
    )
    loss_amount = get_value_from_traderanalyzerdict(
        trade_analyzer_data, "lost", "pnl", "total"
    )
    total_trades = get_value_from_traderanalyzerdict(trade_analyzer_data, "total", "total")
    longest_win_streak = get_value_from_traderanalyzerdict(
        trade_analyzer_data, "streak", "won", "longest"
    )
    longest_loss_streak = get_value_from_traderanalyzerdict(
        trade_analyzer_data, "streak", "lost", "longest"
    )

    dict_result: Dict = {
        "交易总笔数": total_trades,
        "完结的交易笔数": get_value_from_traderanalyzerdict(
            trade_analyzer_data, "total", "closed"
        ),
        "未交易完结笔数": get_value_from_traderanalyzerdict(trade_analyzer_data, "total", "open"),
        "连续获利次数": longest_win_streak if longest_win_streak is not None else np.nan,
        "连续亏损次数": longest_loss_streak if longest_loss_streak is not None else np.nan,
        "胜率(%)": round(win_count / total_trades, 4),
        "盈亏比": round(profit_amount / abs(loss_amount), 4),
        "平均持仓天数": round(
            get_value_from_traderanalyzerdict(trade_analyzer_data, "len", "average"), 2
        ),
        "最大持仓天数": get_value_from_traderanalyzerdict(trade_analyzer_data, "len", "max"),
        "最短持仓天数": get_value_from_traderanalyzerdict(trade_analyzer_data, "len", "min"),
    }

    return pd.DataFrame(dict_result, index=["交易统计"]).T


# 回测分析核心函数
def analysis_rets(
    price_series: pd.Series,
    backtest_result: List,
    benchmark_returns: pd.Series = None,
    use_widgets: bool = False,
) -> namedtuple:
    """分析策略净值表现，生成各类业绩图表和风险表格

    Args:
        price_series (pd.Series): 价格序列（索引为日期）
        backtest_result (List): 回测结果列表
        benchmark_returns (pd.Series, optional): 基准收益序列. Defaults to None.
        use_widgets (bool, optional): 是否使用交互式组件. Defaults to False.

    Returns:
        namedtuple: 包含风险表格和各类业绩图表的命名元组
    """
    ReportTuple = namedtuple(
        "report",
        "risk_table,cumulative_chart,maxdrawdowns_chart,underwater_chart,annual_returns_chart,monthly_heatmap_chart,monthly_dist_chart",
    )
    df_strategy_rets: pd.Series = pd.Series(backtest_result[0].analyzers._TimeReturn.get_analysis())
    
    if benchmark_returns is None:
        benchmark_returns: pd.Series = price_series.pct_change()

    # 对齐策略收益和基准收益的索引
    df_strategy_rets, benchmark_returns = df_strategy_rets.align(benchmark_returns, join="right", axis=0)

    df_combined_returns: pd.DataFrame = pd.concat((df_strategy_rets, benchmark_returns), axis=1)
    df_combined_returns.columns = ["Strategy", "Benchmark"]

    df_report: pd.DataFrame = strategy_performance(df_combined_returns)

    fig_risk_table: go.Figure = plot_table(
        df_report.T.applymap(lambda val: "{:.2%}".format(val)),
        index_name="指标",
        use_widgets=use_widgets,
    )

    fig_cumulative: go.Figure = plot_cumulative(
        df_strategy_rets,
        benchmark_returns,
        main_kwargs=dict(name="Close"),
        yaxis_tickformat=".2%",
        title="Cumulative",
        use_widgets=use_widgets,
    )
    
    fig_drawdowns: go.Figure = plot_drawdowns(
        df_strategy_rets, use_widgets=use_widgets, title="Drawdowns"
    )
    
    fig_underwater: go.Figure = plot_underwater(
        df_strategy_rets, use_widgets=use_widgets, title="Underwater"
    )

    fig_annual_returns: go.Figure = plot_annual_returns(df_strategy_rets, use_widgets=use_widgets)
    fig_monthly_heatmap: go.Figure = plot_monthly_heatmap(
        df_strategy_rets, use_widgets=use_widgets
    )
    fig_monthly_dist: go.Figure = plot_monthly_dist(df_strategy_rets, use_widgets=use_widgets)

    return ReportTuple(
        fig_risk_table,
        fig_cumulative,
        fig_drawdowns,
        fig_underwater,
        fig_annual_returns,
        fig_monthly_heatmap,
        fig_monthly_dist,
    )


def analysis_trade(
    price_data: Union[pd.Series, pd.DataFrame],
    backtest_result: List,
    use_widgets: bool = False
) -> namedtuple:
    """分析交易细节，生成交易报告和相关图表

    Args:
        price_data (Union[pd.Series, pd.DataFrame]): 价格数据（序列或DataFrame）
        backtest_result (List): 回测结果列表
        use_widgets (bool, optional): 是否使用交互式组件. Defaults to False.

    Returns:
        namedtuple: 包含交易报告和各类交易图表的命名元组
    """
    TradeReportTuple = namedtuple(
        "report", "trade_report,pnl_chart,orders_chart,position_chart"
    )
    analyzer_trade: Dict = backtest_result[0].analyzers._TradeAnalyzer.get_analysis()
    df_trade_stats: pd.DataFrame = create_trade_report_table(analyzer_trade)
    
    df_trade_records: pd.DataFrame = pd.DataFrame(
        backtest_result[0].analyzers._TradeRecord.get_analysis()
    )
    df_trade_records = df_trade_records.astype(
        {"datein": np.datetime64, "dateout": np.datetime64}
    )
    
    fig_trade_report: go.Figure = plot_table(df_trade_stats, use_widgets=use_widgets)
    fig_pnl: go.Figure = plot_pnl(df_trade_records, use_widgets=use_widgets, title="PnL")

    if isinstance(price_data, pd.Series):
        fig_orders: go.Figure = plot_orders(
            price_data, df_trade_records, use_widgets=use_widgets, title="Orders"
        )
        fig_position: go.Figure = plot_position(
            price_data, df_trade_records, use_widgets=use_widgets, title="Position"
        )

    elif isinstance(price_data, pd.DataFrame):
        print("TODO:尚未完工")

    return TradeReportTuple(fig_trade_report, fig_pnl, fig_orders, fig_position)


# 以下为原代码中被注释的函数，同步进行同义改写保留
# def analysis_rets(price_series: pd.Series, backtest_result: List) -> List:
#     """净值表现情况

#     Args:
#         price_series (pd.Series): 索引为日期的价格序列
#         backtest_result (List): 回测结果列表
#     """
#     strategy_rets: pd.Series = pd.Series(backtest_result[0].analyzers._TimeReturn.get_analysis())
#     benchmark_rets = price_series.pct_change()
#     benchmark_rets, strategy_rets = benchmark_rets.align(strategy_rets, join="right", axis=0)

#     df_returns: pd.DataFrame = pd.concat((strategy_rets, benchmark_rets), axis=1)
#     df_returns.columns = ["策略", "基准"]

#     df_report: pd.DataFrame = strategy_performance(df_returns)

#     bt_risk_table = plotly_table(
#         df_report.T.applymap(lambda val: "{:.2%}".format(val)), "指标"
#     )

#     cumulative_chart = plot_cumulative(strategy_rets, benchmark_rets)
#     maxdrawdowns_chart = plot_drawdowns(strategy_rets)
#     underwater_chart = plot_underwater(strategy_rets)
#     annual_returns_chart = plot_annual_returns(backtest_result)
#     monthly_return_heatmap_chart = plot_monthly_returns_heatmap(strategy_rets)
#     monthly_return_dist_chart = plot_monthly_returns_dist(strategy_rets)

#     return (
#         bt_risk_table,
#         cumulative_chart,
#         maxdrawdowns_chart,
#         underwater_chart,
#         annual_returns_chart,
#         monthly_return_heatmap_chart,
#         monthly_return_dist_chart,
#     )


# def analysis_trade(price_data: pd.DataFrame, backtest_result: List) -> List:
#     """交易情况分析

#     Args:
#         price_data (pd.DataFrame): 索引为日期的OHLCV数据
#         backtest_result (List): 回测结果列表
#     """

#     df_trade_flags = get_trade_flag(backtest_result)
#     analyzer_trade: Dict = backtest_result[0].analyzers._TradeAnalyzer.get_analysis()
#     df_trade_stats: pd.DataFrame = create_trade_report_table(analyzer_trade)

#     trade_report = plotly_table(df_trade_stats)
#     orders_chart = plotl_order_on_ohlc(price_data, df_trade_flags)

#     df_trade_records: pd.DataFrame = pd.DataFrame(
#         backtest_result[0].analyzers._TradeRecord.get_analysis()
#     )

#     pnl_chart = plot_trade_pnl(df_trade_records)

#     return trade_report, orders_chart, pnl_chart