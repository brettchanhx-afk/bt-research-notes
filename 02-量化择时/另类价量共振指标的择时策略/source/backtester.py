import datetime
from collections import namedtuple
from multiprocessing import cpu_count
from typing import Dict

import backtrader as bt
import backtrader.feeds as btfeeds
import numpy as np
import pandas as pd

from .strategy import VM_Strategy

MAX_CPU: int = int(cpu_count() * 0.8)

class TradeRecord(bt.Analyzer):
    def __init__(self):
        self.history = []
        self.trades = []
        self.cumprofit = 0.0

    def notify_trade(self, trade):

        self.current_trade = trade
        if not trade.isclosed:
            return
        print(1)
        record: Dict = self.get_trade_record(trade)
        self.trades.append(record)

    def stop(self):
       
        trade = self.current_trade

        if not trade.isopen:
            return

        record: Dict = self.get_trade_record(trade)
        self.trades.append(record)

    def get_trade_record(self, trade) -> Dict:

        brokervalue = self.strategy.broker.getvalue()
        dir = "long" if trade.history[0].event.size > 0 else "short"
        size = len(trade.history)
        barlen = trade.history[size - 1].status.barlen
        pricein = trade.history[size - 1].status.price
        datein = bt.num2date(trade.history[0].status.dt)

        if size == 1:
            # 交易没有闭合
            dateout = pd.to_datetime(trade.data.datetime.date(0))
            priceout = trade.data.close[0]
            hp = np.nan
            lp = np.nan
            barlen = np.nan

        elif size == 2:
            # 交易闭合
            dateout = bt.num2date(trade.history[size - 1].status.dt)
            priceout = trade.history[size - 1].event.price
            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein

        if trade.data._timeframe >= bt.TimeFrame.Days:
            datein = datein.date()
            dateout = dateout.date()

        pcntchange = 100 * priceout / pricein - 100
        pnl = trade.history[size - 1].status.pnlcomm
        pnlpcnt = 100 * pnl / brokervalue

        pbar = pnl / barlen if barlen else np.nan
        self.cumprofit += pnl
        size = value = 0.0

        for record in trade.history:

            if abs(size) < abs(record.status.size):
                size = record.status.size
                value = record.status.value

        if dir == "long":
            mfe = hp
            mae = lp
        elif dir == "short":
            mfe = -lp
            mae = -hp

        return {
            "status": trade.status,  # 1-open,2-closed
            "ref": trade.ref,
            "ticker": trade.data._name,
            "dir": dir,
            "datein": datein,
            "pricein": pricein,
            "dateout": dateout,
            "priceout": priceout,
            "chng%": round(pcntchange, 2),
            "pnl": pnl,
            "pnl%": round(pnlpcnt, 2),
            "size": size,
            "value": value,
            "cumpnl": self.cumprofit,
            "nbars": barlen,
            "pnl/bar": round(pbar, 2),
            "mfe%": round(mfe, 2),
            "mae%": round(mae, 2),
        }

    def get_analysis(self):
        return self.trades


# 考虑佣金和印花税的股票百分比费用
class StockCommission(bt.CommInfoBase):
    params = (
        ('stamp_duty', 0.001),
        ('stocklike', True),  # 指定为股票模式
        ('commtype', bt.CommInfoBase.COMM_PERC),  # 使用百分比费用模式
        ('percabs', True),
    )  # commission 不以 % 为单位 # 印花税默认为 0.1%

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入时，只考虑佣金
            return abs(size) * price * self.p.commission
        elif size < 0:  # 卖出时，同时考虑佣金和印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        else:
            return 0


def get_backtesting(data: pd.DataFrame,
                    name: str,
                    strategy: bt.Strategy = VM_Strategy,
                    begin_dt: datetime.date = None,
                    end_dt: datetime.date = None,
                    **kw) -> namedtuple:
    res = namedtuple('Res', 'result,cerebro')

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1e8)
    if (begin_dt is None) or (end_dt is None):
        begin_dt = data.index.min()
        end_dt = data.index.max()
    else:
        begin_dt = pd.to_datetime(begin_dt)
        end_dt = pd.to_datetime(end_dt)
   
    datafeed = bt.feeds.PandasData(dataname=data,
                                   fromdate=begin_dt,
                                   todate=end_dt)
    cerebro.adddata(datafeed, name=name)

    # 设置百分比滑点
    cerebro.broker.set_slippage_perc(perc=0.0001)

    # 设置交易费用
    comminfo = StockCommission(commission=0.0002, stamp_duty=0.001)
    cerebro.broker.addcommissioninfo(comminfo)

    cerebro.addstrategy(strategy)

    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="_AnnualReturn")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="_TradeAnalyzer")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="_Transactions")
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name="_PeriodStats")
    # 返回收益率时序
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(TradeRecord, _name="_TradeRecord")

    result = cerebro.run(tradehistory=True)

    return res(result, cerebro)
