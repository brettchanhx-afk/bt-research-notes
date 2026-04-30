import math
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Callable

import backtrader as bt
import pandas as pd
from backtrader import Strategy
from loguru import logger

__all__ = [
    "NoiseRangeStrategy",
    "NoiseRangeVWAPStrategy",
    "BasePositionStrategy",
    "BasePositionStopLossStrategy",
]


def round_to_hundred(size: int) -> int:
    return math.floor(size / 100) * 100


def calculate_ashare_order_size(money: float, price: float, min_limit: int = 100):
    
    if price <= 0 or money <= 0:
        raise ValueError("股价或资金量需要大于0")

    # 计算能买多少“手”（每手100股）
    number_of_hands = money // (price * min_limit)
    # 转换为股数
    return int(number_of_hands * min_limit)


class NoiseRangeStrategy(bt.Strategy):
    
    params: Dict = dict(
        commission=0.01, hold_num=1, verbose=False
    )  # 每次交易预留1%的交易成本

    def __init__(self) -> None:

        self.order = None
        self.signal: Dict = {d._name: d.signal for d in self.datas}
        self.upper_bound: Dict = {
            d._name: d.upperbound for d in self.datas
        }  # self.data.upperbound
        self.lower_bound: Dict = {
            d._name: d.lowerbound for d in self.datas
        }  # self.data.lowerbound

    def log(self, msg: str, current_dt: pd.Timestamp = None, verbose: bool = True):
        if current_dt is None:
            current_dt: pd.Timestamp = self.datetime.datetime(0)
        if verbose:
            logger.info(f"{current_dt} {msg}")

    def handle_signal(self, symbol: str) -> None:
        """信号处理"""
        size: int = self.getpositionbyname(symbol).size

        if self.signal[symbol][0] > self.upper_bound[symbol][0]:
            if size < 0:
                self._close_and_reverse(symbol, f"{symbol} 空头平仓并开多头", self.buy)
            elif size == 0:
                self._open_position(symbol, f"{symbol} 多头开仓", self.buy)
        elif self.signal[symbol][0] < self.lower_bound[symbol][0]:
            if size > 0:
                self._close_and_reverse(symbol, f"{symbol} 多头平仓并开空头", self.sell)
            elif size == 0:
                self._open_position(symbol, f"{symbol} 空头开仓", self.sell)

    def rebalance(self, symbol: str) -> None:
        """尾盘平仓"""
        if self.getpositionbyname(symbol).size != 0:
            self.order = self.close(data=symbol, exectype=bt.Order.Market)
            self.log(f"{symbol} 收盘平仓", verbose=self.p.verbose)

    def handle_stop_loss(self, symbol) -> None:
        """止盈/止损逻辑"""
        size: int = self.getpositionbyname(symbol).size

        if size > 0 and self.signal[symbol][0] < self.lower_bound[symbol][0]:
            self._close_and_reverse(symbol, f"{symbol} 多头平仓并开空头", self.sell)
        elif size < 0 and self.signal[symbol][0] > self.upper_bound[symbol][0]:
            self._close_and_reverse(symbol, f"{symbol} 空头平仓并开多头", self.buy)

    def _calculate_size(self, symbol: str) -> float:
        # 以下个bar的开盘价买入
        money: float = (
            self.broker.getvalue() * (1 - self.p.commission) / self.p.hold_num
        )

        return calculate_ashare_order_size(money, self.getdatabyname(symbol).close[0])

    def _close_and_reverse(
        self, symbol: str, reason: str, new_action: Callable
    ) -> None:
        self.log(reason, verbose=self.p.verbose)
        self.order = self.close(data=symbol)
        size: int = self._calculate_size(symbol)
        self.order = new_action(data=symbol, size=size, exectype=bt.Order.Market)

    def _open_position(self, symbol: str, reason: str, action: Callable) -> None:
        self.log(reason, verbose=self.p.verbose)
        size: int = self._calculate_size(symbol)
        self.order = action(data=symbol, size=size, exectype=bt.Order.Market)

    def next(self) -> None:

        for data in self.datas:

            if self.datetime.datetime(0) != data.datetime.datetime(0):
                continue

            if self.order:
                self.cancel(self.order)
                self.order = None

            self._run(data._name)

    def prenext(self) -> None:
        self.next()

    def _run(self, symbol: str) -> None:

        current_time: str = bt.num2date(
            self.getdatabyname(symbol).datetime[0]
        ).strftime("%H:%M:%S")

        if current_time in ["10:29:00", "11:29:00", "13:59:00"]:

            self.handle_signal(symbol)

        elif current_time == "15:00:00":

            self.rebalance(symbol)

        else:

            self.handle_stop_loss(symbol)


class NoiseRangeVWAPStrategy(NoiseRangeStrategy):
    
    def __init__(self) -> None:
        super().__init__()

        self.long_stop_loss: Dict = {
            d._name: bt.Max(d.upperbound, d.vwap) for d in self.datas
        }
        self.short_stop_loss: Dict = {
            d._name: bt.Min(d.lowerbound, d.vwap) for d in self.datas
        }

    def handle_stop_loss(self, symbol) -> None:
        """止盈/止损逻辑"""
        size: int = self.getpositionbyname(symbol).size

        if size > 0 and self.signal[symbol][0] <= self.long_stop_loss[symbol][0]:
            self.log("多头止损平仓", verbose=self.p.verbose)
            self.order = self.close(data=symbol)
        elif size < 0 and self.signal[symbol][0] >= self.short_stop_loss[symbol][0]:
            self.log("空头止损平仓", verbose=self.p.verbose)
            self.order = self.close(data=symbol)




@dataclass
class Position:
    _yesterday_amount: int = 0
    _today_amount: int = 0
    _total_amount: int = 0

    @property
    def closeable_amount(self) -> int:
        return self._yesterday_amount

    @property
    def locked_amount(self) -> int:
        return self._today_amount

    @property
    def total_amount(self) -> int:
        return self._total_amount


class Portfolio:

    def __init__(self):

        self.long_positions: Dict[str, Dict] = defaultdict(Position)

    def update_position(self, symbol: str, order: bt.Order):

        size = order.executed.size
        price = order.executed.price

        # 创建初始化仓位
        if symbol not in self.long_positions:

            if size < 0:
                raise ValueError(
                    f"{order.data.datetime.datetime(0)} {symbol} 卖出{size:,}数量大于持仓数量({0:,})!"
                )

            else:
                self.long_positions[symbol]._amount = size
                self.long_positions[symbol]._price = price

        if symbol in self.long_positions:

            position = self.long_positions[symbol]

            if size < 0:
                if abs(size) <= position._yesterday_amount:
                    position._yesterday_amount += size
                else:
                    raise ValueError(
                        f"{order.data.datetime.datetime(0)} {symbol} 卖出{size:,}数量大于持仓数量({position._yesterday_amount:,})!"
                    )

            else:
                position._today_amount = size

            position._total_amount = position._yesterday_amount + position._today_amount

            if position._total_amount < 0:
                raise ValueError(
                    f"{order.data.datetime.datetime(0)} {symbol} 总持仓数量({position._total_amount:,})小于0!"
                )

    def end_of_day(self, symbol: str, strategy: Strategy):

        position = self.long_positions[symbol]

        if position._today_amount >= 0:

            position._yesterday_amount = position._total_amount
            position._today_amount = 0

        else:

            raise ValueError(
                f"{strategy.datetime.datetime(0)} {symbol} 当日卖出数量({position._today_amount:,})小于0"
            )


class BaseStrategy(Strategy):
    
    params = dict(
        commission=0.01, hold_num=1, verbose=False
    )  # 每次交易预留1%的交易成本

    def __init__(self) -> None:
        self.order = None

        self.signal: Dict = {d._name: d.signal for d in self.datas}
        self.upper_bound: Dict = {d._name: d.upperbound for d in self.datas}
        self.lower_bound: Dict = {d._name: d.lowerbound for d in self.datas}
        # 预先计算止损线
        self.long_stop_loss: Dict = {
            d._name: bt.Max(d.upperbound, d.vwap) for d in self.datas
        }
        self.short_stop_loss: Dict = {
            d._name: bt.Min(d.lowerbound, d.vwap) for d in self.datas
        }
        # 记录持仓
        self.portfolio = Portfolio()
        # 建立底仓
        self.is_first_day: Dict = {d._name: True for d in self.datas}
        # 标记当日是否有交易
        self.is_trade_today: Dict = {d._name: False for d in self.datas}
        self.long_trade_today: Dict = {d._name: False for d in self.datas}
        self.short_trade_today: Dict = {d._name: False for d in self.datas}

        self.base_position_pct = 0.5  # 底仓比例

    def log(self, msg: str, current_dt: pd.Timestamp = None, verbose: bool = True):
        if current_dt is None:
            current_dt: pd.Timestamp = self.datetime.datetime(0)
        if verbose:
            logger.info(f"{current_dt} {msg}")

    def create_base_position(self, symbol: str):
        """建立底仓"""

        size: int = self.getpositionbyname(symbol).size
        price: float = self.getpositionbyname(symbol).price
        if self.is_first_day[symbol]:
            
            if size == 0:
                self.log(
                    f"{self.datetime.datetime(0)} {symbol} 创建底仓", verbose=self.p.verbose
                )

                money: float = (
                    self.broker.get_value()
                    / self.p.hold_num
                    * self.base_position_pct
                    * (1 - self.p.commission)
                )

                size: int = calculate_ashare_order_size(
                    money,
                    self.getdatabyname(symbol).close[0],
                )

                self.order = self.buy(data=symbol, size=size)

            # 当日收盘后更新is_first_day
            elif self.datetime.datetime(0).strftime("%H:%M:%S") == "15:00:00":
                self.log(
                    f"{self.datetime.datetime(0)} {symbol} 首个交易日结束,底仓({price*size:,.2f}|{size:,}|{price:.2f})",
                    verbose=self.p.verbose,
                )
                self.is_first_day[symbol] = False

            return True
        return False

    @abstractmethod
    def handle_signal(self, symbol: str) -> None:
        """处理交易信号"""
        pass

    def rebalance(self, symbol: str) -> None:
        """平衡底仓"""
        share_value: float = (
            self.getpositionbyname(symbol).size * self.getpositionbyname(symbol).price
        )
        base_position_value: float = (
            self.broker.get_value() / self.p.hold_num * self.base_position_pct
        )
        target_value: float = base_position_value - share_value
        if share_value == 0:
            size: int = calculate_ashare_order_size(
                target_value * (1 - self.p.commission),
                self.getdatabyname(symbol).close[0],
            )
            self.log(
                f"{symbol} 尾盘,持仓偏差大于阈值,加仓({target_value:,.2f}|{size:,})",
                verbose=self.p.verbose,
            )
            self.order = self.buy(data=symbol, size=size, exectype=bt.Order.Market)
        else:
            target_pct: float = base_position_value / share_value - 1

            # 与目标仓位相差大于3%时，进行调仓
            if target_pct < -0.03:

                size: int = calculate_ashare_order_size(
                    abs(target_value) * (1 - self.p.commission),
                    self.getdatabyname(symbol).close[0],
                )
                if size < self.portfolio.long_positions[symbol].closeable_amount:

                    self.log(
                        f"{symbol} 尾盘,持仓偏差大于阈值,减仓({target_value:,.2f}|{size:,}),可用减仓量({self.portfolio.long_positions[symbol].closeable_amount * self.getpositionbyname(symbol).price:,.2f}|{self.portfolio.long_positions[symbol].closeable_amount:,})",
                        verbose=self.p.verbose,
                    )
                    self.order = self.sell(
                        data=symbol, size=size, exectype=bt.Order.Market
                    )
                else:
                    self.log(
                        f"{symbol} 减仓量({target_value:,.2f}|{size:,})大于可交易卖出量({self.portfolio.long_positions[symbol].closeable_amount * self.getpositionbyname(symbol).price:,.2f}|{self.portfolio.long_positions[symbol].closeable_amount:,}),不操作",
                        verbose=self.p.verbose,
                    )

            elif target_pct > 0.03:

                size: int = calculate_ashare_order_size(
                    target_value * (1 - self.p.commission),
                    self.getdatabyname(symbol).close[0],
                )
                self.log(
                    f"{symbol} 尾盘,持仓偏差大于阈值,加仓({target_value:,.2f}|{size:,})",
                    verbose=self.p.verbose,
                )
                self.order = self.buy(data=symbol, size=size, exectype=bt.Order.Market)

    def handle_stop_loss(self, symbol: str) -> None:
        """止损逻辑"""
        if self.long_trade_today[symbol] and (
            self.signal[symbol][0] <= self.long_stop_loss[symbol][0]
        ):
            if self.portfolio.long_positions[symbol].closeable_amount > 0:
                self.log(
                    f"{symbol} 多头止损,卖出({self.portfolio.long_positions[symbol].closeable_amount * self.getpositionbyname(symbol).price:,.2f}|{self.portfolio.long_positions[symbol].closeable_amount:,})",
                    verbose=self.p.verbose,
                )
                self.order = self.sell(
                    data=symbol,
                    size=self.portfolio.long_positions[symbol].closeable_amount,
                    exectype=bt.Order.Market,
                )
                self.long_trade_today[symbol] = False  # 重置当日交易标记

        if self.short_trade_today[symbol] and (
            self.signal[symbol][0] >= self.short_stop_loss[symbol][0]
        ):

            if self.broker.get_cash() > 0:
                share_value: float = (
                    self.getpositionbyname(symbol).size
                    * self.getpositionbyname(symbol).price
                )
                base_position_value: float = (
                    self.broker.get_value() / self.p.hold_num * self.base_position_pct
                )
                target_value: float = base_position_value - share_value
                # 分仓的金额大于目标金额时，买入平空头
                if (self.broker.get_cash() / self.p.hold_num)> abs(target_value):
                    size: int = calculate_ashare_order_size(
                        abs(target_value) * (1 - self.p.commission),
                        self.getdatabyname(symbol).close[0],
                    )

                    self.log(
                        f"{symbol} 空头止损,买入({target_value:,.2f}|{size:,})",
                        verbose=self.p.verbose,
                    )
                    self.order = self.buy(data=symbol, size=size, exectype=bt.Order.Market)

                    self.short_trade_today[symbol] = False
                else:
                    self.log(f"资金不足，无法买入平{symbol}空头仓位", verbose=self.p.verbose)
            else:
                self.log(f"无剩余资金，无法买入平{symbol}空头仓位", verbose=self.p.verbose)

    def prenext(self):

        self.next()

    def notify_order(self, order):

        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已被处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, ref:{order.ref}，Price: {order.executed.price:,.2f}, Size: {order.executed.size:,}, Cost: {order.executed.value:,.2f}, Comm {order.executed.comm:,.2f}, Stock: {order.data._name}",
                    verbose=self.p.verbose,
                )
            else:  # Sell
                self.log(
                    f"SELL EXECUTED, ref:{order.ref}, Price: {order.executed.price:,.2f}, Size: {order.executed.size:,}, Cost: %{order.executed.value:,.2f}, Comm {order.executed.comm:,.2f}, Stock: {order.data._name}",
                    verbose=self.p.verbose,
                )

            self.portfolio.update_position(order.data._name, order)

    def next(self):

        for data in self.datas:

            if self.datetime.datetime(0)!=data.datetime.datetime(0):
                continue

            if self.order:

                self.cancel(self.order)
                self.order = None

            self._run(data._name)
    

    def _run(self, symbol: str) -> None:

    
        if self.create_base_position(symbol):

            return
        
        current_time: str = self.datetime.datetime(0).strftime("%H:%M:%S")

        if current_time in ["10:29:00", "11:29:00", "13:59:00"]:

        
            self.handle_signal(symbol)

        elif current_time == "14:59:00":

            self.rebalance(symbol)

        elif current_time == "15:00:00":

            self.portfolio.end_of_day(symbol, self)
            self.is_trade_today[symbol] = False

        else:

            self.handle_stop_loss(symbol)


class BasePositionStopLossStrategy(BaseStrategy):

    def handle_signal(self, symbol: str) -> None:

        if self.signal[symbol][0] > self.upper_bound[symbol][0]:
            if not self.long_trade_today[symbol]:
                if self.broker.get_cash() > 0:
                    money: float = self.broker.get_cash() / self.p.hold_num * (1 - self.p.commission)
                    size: int = calculate_ashare_order_size(money, self.getdatabyname(symbol).close[0])

                    self.log(
                        f"{symbol} 多头开仓,新买入(买入金额{money:,.2f}|{size:,}),已有({self.portfolio.long_positions[symbol].closeable_amount * self.getpositionbyname(symbol).price:,.2f}|{self.portfolio.long_positions[symbol].closeable_amount:,})",
                        verbose=self.p.verbose,
                    )
                    # 以下个bar的开盘价买入

                    self.order = self.buy(data=symbol,size=size, exectype=bt.Order.Market)
                    self.long_trade_today[symbol] = True

        elif self.signal[symbol][0] < self.lower_bound[symbol][0]:

            if not self.short_trade_today[symbol]:
                if self.portfolio.long_positions[symbol].closeable_amount > 0:
                    self.log(
                        f"{symbol} 空头开仓，卖出底仓,可卖出仓位({self.portfolio.long_positions[symbol].closeable_amount * self.getpositionbyname(symbol).price:,.2f}|{self.portfolio.long_positions[symbol].closeable_amount:,})",
                        verbose=self.p.verbose,
                    )
                    self.order = self.sell(
                        size=self.portfolio.long_positions[symbol].closeable_amount,
                        exectype=bt.Order.Market,
                    )

                    self.short_trade_today[symbol] = True

        else:
            self.log("{symbol} 无信号触发", verbose=self.p.verbose)


class BasePositionStrategt(BaseStrategy):

    def handle_signal(self, symbol: str) -> None:

        if self.signal[symbol][0] > self.upper_bound[symbol][0]:
            if not self.is_trade_today[symbol]:
                if self.broker.get_cash() > 0:
                    money: float = self.broker.get_cash() / self.p.hold_num * (1 - self.p.commission)
                    size: int = calculate_ashare_order_size(money, self.getdatabyname(symbol).close[0])

                    self.log(
                        f"{symbol} 多头开仓,新买入(买入金额{money:,.2f}|{size:,}),已有({self.portfolio.long_positions[symbol].closeable_amount * self.getpositionbyname(symbol).price:,.2f}|{self.portfolio.long_positions[symbol].closeable_amount:,})",
                        verbose=self.p.verbose,
                    )
                    self.order = self.buy(data=symbol,size=size, exectype=bt.Order.Market)
                    self.is_trade_today[symbol] = True
                    self.long_trade_today[symbol] = True

        elif self.signal[symbol][0] < self.lower_bound[symbol][0]:

            if not self.is_trade_today[symbol]:
                if self.portfolio.long_positions[symbol].closeable_amount > 0:
                    self.log(
                        f"{symbol} 空头开仓，卖出底仓,可卖出仓位({self.portfolio.long_positions[symbol].closeable_amount * self.getpositionbyname(symbol).price:,.2f}|{self.portfolio.long_positions[symbol].closeable_amount:,})",
                        verbose=self.p.verbose,
                    )
                    self.order = self.sell(
                        data=symbol,
                        size=self.portfolio.long_positions[symbol].closeable_amount,
                        exectype=bt.Order.Market,
                    )
                    self.is_trade_today[symbol] = True
                    self.short_trade_today[symbol] = True

        else:
            self.log(f"{symbol} 无信号触发", verbose=self.p.verbose)

    def handle_stop_loss(self, symbol: str) -> None:
        pass
