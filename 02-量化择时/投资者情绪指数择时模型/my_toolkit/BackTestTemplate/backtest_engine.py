# 回测核心模块依赖导入
from collections import namedtuple
from typing import Dict
import datetime

import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.feeds as btfeeds

from .bt_strategy import SignalStrategy


class TradeRecord(bt.Analyzer):
    def __init__(self):
        self.trade_log = []
        self.trade_details = []
        self.accumulated_profit = 0.0

    def notify_trade(self, trade_obj):
        self.current_position = trade_obj
        if not trade_obj.isclosed:
            return
        trade_detail: Dict = self.build_trade_info(trade_obj)
        self.trade_details.append(trade_detail)

    def stop(self):
        """统计最后一笔开仓但尚未平仓的交易信息"""
        current_pos = self.current_position

        if not current_pos.isopen:
            return

        pos_detail: Dict = self.build_trade_info(current_pos)
        self.trade_details.append(pos_detail)

    def build_trade_info(self, trade_obj) -> Dict:
        account_asset = self.strategy.broker.getvalue()
        trade_side = "long" if trade_obj.history[0].event.size > 0 else "short"
        hist_count = len(trade_obj.history)
        trade_bar_num = trade_obj.history[hist_count - 1].status.barlen
        entry_price_val = trade_obj.history[hist_count - 1].status.price
        entry_date_val = bt.num2date(trade_obj.history[0].status.dt)

        # 0=已平仓(偶数条记录) 1=未平仓(奇数条记录)
        is_trade_closed: int = hist_count % 2
        if is_trade_closed:
            # 交易已完成平仓操作
            exit_date_val = bt.num2date(trade_obj.history[hist_count - 1].status.dt)
            exit_price_val = trade_obj.history[hist_count - 1].event.price
            trade_high_price = max(trade_obj.data.high.get(ago=0, size=trade_bar_num + 1))
            trade_low_price = min(trade_obj.data.low.get(ago=0, size=trade_bar_num + 1))
            high_pct = 100 * (trade_high_price - entry_price_val) / entry_price_val
            low_pct = 100 * (trade_low_price - entry_price_val) / entry_price_val

        else:
            # 交易未平仓
            exit_date_val = pd.to_datetime(trade_obj.data.datetime.date(0))
            exit_price_val = trade_obj.data.close[0]
            high_pct = np.nan
            low_pct = np.nan
            trade_bar_num = np.nan

        # 时间维度大于等于日线时，仅保留日期部分
        if trade_obj.data._timeframe >= bt.TimeFrame.Days:
            entry_date_val = entry_date_val.date()
            exit_date_val = exit_date_val.date()

        price_change_percent = 100 * exit_price_val / entry_price_val - 100
        profit_loss_amt = trade_obj.history[hist_count - 1].status.pnlcomm
        profit_loss_percent = 100 * profit_loss_amt / account_asset

        pnl_per_bar_num = profit_loss_amt / trade_bar_num if trade_bar_num else np.nan
        self.accumulated_profit += profit_loss_amt
        position_size = position_value = 0.0

        # 遍历交易历史记录，确定最大持仓规模和对应价值
        for hist_item in trade_obj.history:
            if abs(position_size) < abs(hist_item.status.size):
                position_size = hist_item.status.size
                position_value = hist_item.status.value

        # 区分多空方向计算最大有利/不利波动
        if trade_side == "long":
            max_favourable_excursion = high_pct
            max_adverse_excursion = low_pct
        elif trade_side == "short":
            max_favourable_excursion = -low_pct
            max_adverse_excursion = -high_pct

        return {
            "status": trade_obj.status,  # 1-开仓状态,2-平仓状态
            "ref": trade_obj.ref,
            "ticker": trade_obj.data._name,
            "dir": trade_side,
            "datein": entry_date_val,
            "pricein": entry_price_val,
            "dateout": exit_date_val,
            "priceout": exit_price_val,
            "chng%": round(price_change_percent, 2),
            "pnl": profit_loss_amt,
            "pnl%": round(profit_loss_percent, 2),
            "size": position_size,
            "value": position_value,
            "cumpnl": self.accumulated_profit,
            "nbars": trade_bar_num,
            "pnl/bar": round(pnl_per_bar_num, 2),
            "mfe%": round(max_favourable_excursion, 2),
            "mae%": round(max_adverse_excursion, 2),
        }

    def get_analysis(self):
        return self.trade_details


# 股票交易佣金计算类（含佣金+印花税）
class StockCommission(bt.CommInfoBase):
    params = (
        ("stamp_duty", 0.001),  # 印花税默认0.1%
        ("stocklike", True),     # 标记为股票类资产
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 按百分比收取佣金
        ("percabs", True),       # 佣金为绝对值百分比（非小数形式）
    )

    def _getcommission(self, trade_volume, asset_price, pseudoexec):
        if trade_volume > 0:  # 买入操作，仅收取佣金
            return abs(trade_volume) * asset_price * self.p.commission
        elif trade_volume < 0:  # 卖出操作，佣金+印花税合计收取
            return abs(trade_volume) * asset_price * (self.p.commission + self.p.stamp_duty)
        else:  # 无交易操作，佣金为0
            return 0


class AddSignalData(bt.feeds.PandasData):
    """自定义Pandas数据加载类
    扩展GSISI信号字段，用于回测数据加载
    """
    lines = ("GSISI",)
    params = (("GSISI", -1),)


def get_backtesting(
    data: pd.DataFrame,
    name: str = None,
    strategy: bt.Strategy = SignalStrategy,
    begin_dt: datetime.date = None,
    end_dt: datetime.date = None,
    **kw
) -> namedtuple:
    """回测执行核心函数
    配置说明：
    - 默认百分比滑点：0.0001
    - 信号执行规则：当日信号，次日开盘执行买入
    Args:
        data (pd.DataFrame): 包含信号的OHLC格式数据
        name (str): 标的名称
        strategy (bt.Strategy): 回测使用的策略类

    Returns:
        namedtuple: 包含回测结果(result)和cerebro实例(cerebro)的元组
    """
    BacktestOutput = namedtuple("BacktestOutput", "result,cerebro")

    # 多标的数据加载标识
    is_multi_symbol: bool = kw.get("mulit_add_data", False)
    # 滑点百分比参数
    slip_percent_val: float = kw.get("slippage_perc", 0.0001)
    # 交易佣金参数
    trade_commission: float = kw.get("commission", 0.0002)
    # 印花税参数
    tax_stamp_duty: float = kw.get("stamp_duty", 0.001)
    # 日志显示开关
    log_display: bool = kw.get("show_log", True)

    def load_multi_symbol_data(df: pd.DataFrame) -> None:
        """多标的数据加载函数
        按标的代码分组，加载对应OHLC数据到cerebro
        """
        sorted_index_arr: np.ndarray = df.index.sort_values().unique()
        for stock_symbol, stock_df in df.groupby("code"):
            # 重新索引并排序，保证时间序列连续
            stock_df = stock_df.reindex(sorted_index_arr)
            stock_df.sort_index(inplace=True)
            # 仅保留核心行情字段
            stock_df = stock_df[["open", "high", "low", "close", "volume"]]
            # 成交量空值填充为0，价格字段向前填充
            stock_df.loc[:, "volume"] = stock_df.loc[:, "volume"].fillna(0)
            stock_df.loc[:, ["open", "high", "low", "close"]] = stock_df.loc[
                :, ["open", "high", "low", "close"]
            ].fillna(method="pad")

            # 构建backtrader数据馈送对象
            symbol_feed = btfeeds.PandasData(dataname=stock_df, fromdate=begin_dt, todate=end_dt)
            cerebro.adddata(symbol_feed, name=stock_symbol)

    # 初始化回测引擎
    cerebro = bt.Cerebro()
    # 设置初始资金规模（10亿）
    cerebro.broker.setcash(1000000000.0)
    
    # 处理回测时间范围
    if begin_dt is None or end_dt is None:
        start_date = data.index.min()
        end_date = data.index.max()
    else:
        start_date = pd.to_datetime(begin_dt)
        end_date = pd.to_datetime(end_dt)
    
    # 加载数据（区分单/多标的）
    if is_multi_symbol:
        load_multi_symbol_data(data)
    else:
        signal_data_feed = AddSignalData(dataname=data, fromdate=start_date, todate=end_date)
        cerebro.adddata(signal_data_feed, name=name)

    # 设置滑点（百分比模式）
    if slip_percent_val is not None:
        cerebro.broker.set_slippage_perc(perc=slip_percent_val)

    # 设置交易费用（佣金+印花税）
    if trade_commission is not None and tax_stamp_duty is not None:
        commission_calc = StockCommission(commission=trade_commission, stamp_duty=tax_stamp_duty)
        cerebro.broker.addcommissioninfo(commission_calc)

    # 添加回测策略
    cerebro.addstrategy(strategy, show_log=log_display)
    
    # 注册回测分析器
    # 年度收益率分析器（按252个交易日年化）
    cerebro.addanalyzer(bt.analyzers.Returns, _name="_Returns", tann=252)
    # 交易行为分析器
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="_TradeAnalyzer")
    # 交易记录分析器（含成本）
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="_Transactions")
    # 周期统计分析器
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name="_PeriodStats")
    # 时间序列收益率分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    # SQN（交易质量）分析器
    cerebro.addanalyzer(bt.analyzers.SQN, _name="_SQN")
    # 夏普比率分析器（年化，无风险利率4%，按250个交易日折算）
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="_Sharpe",
        timeframe=bt.TimeFrame.Years,
        riskfreerate=0.04,
        annualize=True,
        factor=250,
    )

    # 自定义交易记录分析器（需开启tradehistory）
    cerebro.addanalyzer(TradeRecord, _name="_TradeRecord")

    # 执行回测（开启交易历史记录）
    backtest_result = cerebro.run(tradehistory=True)

    return BacktestOutput(backtest_result, cerebro)