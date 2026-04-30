from typing import Dict

import backtrader as bt
import pandas as pd
from loguru import logger

__all__ = [
    "RSRSStrategy",
]


def calculate_ashare_order_size(money: float, price: float, min_limit: int = 100):
    
    if price <= 0 or money <= 0:
        raise ValueError("股价或资金量需要大于0")

    # 计算能买多少“手”（每手100股）
    number_of_hands = money // (price * min_limit)
    # 转换为股数
    return int(number_of_hands * min_limit)


class RSRSStrategy(bt.Strategy):
    
    # 每次交易预留1%的交易成本
    params: Dict = dict(commission=0.01, hold_num=1, verbose=False)

    def __init__(self) -> None:

        self.order = None

        self.open_signal: Dict = {
            d._name: bt.indicators.CrossUp(d.signal, d.upperbound) for d in self.datas
        }
        self.close_signal: Dict = {
            d._name: bt.indicators.CrossDown(d.signal, d.lowerbound) for d in self.datas
        }

    def log(self, msg: str, current_dt: pd.Timestamp = None, verbose: bool = True):
        if current_dt is None:
            current_dt: pd.Timestamp = self.datetime.datetime(0)
        if verbose:
            logger.info(f"{current_dt} {msg}")

    def _calculate_size(self, symbol: str) -> float:
        
        # 以下个bar的开盘价买入
        money: float = (
            self.broker.getvalue() * (1 - self.p.commission) / self.p.hold_num
        )

        return calculate_ashare_order_size(money, self.getdatabyname(symbol).close[0])

    def handle_signal(self, symbol: str) -> None:
        """信号处理"""
        size: int = self.getpositionbyname(symbol).size

        if self.open_signal[symbol][0]:
            if not size:
                target_size = self._calculate_size(symbol)
                self.order = self.buy(
                    data=symbol, size=target_size, exectype=bt.Order.Market
                )

        if self.close_signal[symbol][0]:
            if size:
                self.order = self.close(data=symbol, exectype=bt.Order.Market)

    def next(self) -> None:

        for data in self.datas:

            if self.datetime.datetime(0) != data.datetime.datetime(0):
                continue

            if self.order:
                self.cancel(self.order)
                self.order = None

            self.handle_signal(data._name)

    def prenext(self) -> None:
        self.next()


class RSRSTwoSidetrategy(bt.Strategy):
    

    # 每次交易预留1%的交易成本
    params: Dict = dict(commission=0.01, hold_num=1, verbose=False)

    def __init__(self) -> None:
        self.order = None

        self.open_signal: Dict = {
            d._name: bt.Or(
                bt.indicators.CrossUp(d.signal, d.upperbound),
                bt.indicators.CrossDown(d.signal, d.lowerbound),
            )
            for d in self.datas
        }
        self.close_signal: Dict = {
            d._name: bt.Or(
                bt.indicators.CrossDown(d.signal, d.upperbound),
                bt.indicators.CrossUp(d.signal, d.lowerbound),
            )
            for d in self.datas
        }

    def log(self, msg: str, current_dt: pd.Timestamp = None, verbose: bool = True):
        if current_dt is None:
            current_dt: pd.Timestamp = self.datetime.datetime(0)
        if verbose:
            logger.info(f"{current_dt} {msg}")

    def _calculate_size(self, symbol: str) -> float:
        
        # 以下个bar的开盘价买入
        money: float = (
            self.broker.getvalue() * (1 - self.p.commission) / self.p.hold_num
        )

        return calculate_ashare_order_size(money, self.getdatabyname(symbol).close[0])

    def handle_signal(self, symbol: str) -> None:
        """信号处理"""
        size: int = self.getpositionbyname(symbol).size

        if self.open_signal[symbol][0]:
            if not size:
                target_size = self._calculate_size(symbol)
                self.order = self.buy(
                    data=symbol, size=target_size, exectype=bt.Order.Market
                )

        if self.close_signal[symbol][0]:
            if size:
                self.order = self.close(data=symbol, exectype=bt.Order.Market)

    def next(self) -> None:

        for data in self.datas:

            if self.datetime.datetime(0) != data.datetime.datetime(0):
                continue

            if self.order:
                self.cancel(self.order)
                self.order = None

            self.handle_signal(data._name)

    def prenext(self) -> None:
        self.next()
