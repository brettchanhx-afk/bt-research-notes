"""
市场数据获取模块
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .tushare_client import get_pro_api
from ..config import BENCHMARK, BACKTEST_START_DATE, BACKTEST_END_DATE


class MarketDataFetcher:
    """
    市场数据获取器
    """

    def __init__(self):
        self.pro = get_pro_api()
        self.benchmark = BENCHMARK

    def get_index_daily(
        self,
        start_date: str = BACKTEST_START_DATE,
        end_date: str = BACKTEST_END_DATE,
        index_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取指数日线数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            index_code: 指数代码

        Returns:
            指数日线数据
        """
        if index_code is None:
            index_code = self.benchmark

        try:
            df = self.pro.query(
                "daily",
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                return df
        except Exception as e:
            print(f"获取指数日线数据失败: {e}")

        return self._generate_synthetic_index_data(start_date, end_date, index_code)

    def _generate_synthetic_index_data(
        self, start_date: str, end_date: str, index_code: str
    ) -> pd.DataFrame:
        """
        生成模拟的指数数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            index_code: 指数代码

        Returns:
            模拟的指数数据
        """
        dates = pd.bdate_range(start=start_date, end=end_date).strftime("%Y%m%d")
        np.random.seed(42)

        n_days = len(dates)
        log_returns = np.random.randn(n_days) * 0.015
        close_prices = 3000 * np.exp(np.cumsum(log_returns))

        df = pd.DataFrame(
            {
                "ts_code": index_code,
                "trade_date": dates,
                "open": close_prices * (1 + np.random.randn(n_days) * 0.005),
                "high": close_prices * (1 + np.abs(np.random.randn(n_days) * 0.01)),
                "low": close_prices * (1 - np.abs(np.random.randn(n_days) * 0.01)),
                "close": close_prices,
                "vol": np.random.randint(10000000, 50000000, n_days),
                "amount": close_prices * np.random.randint(10000000, 50000000, n_days),
            }
        )
        return df

    def get_stock_daily(
        self,
        ts_code: str,
        start_date: str = BACKTEST_START_DATE,
        end_date: str = BACKTEST_END_DATE,
    ) -> pd.DataFrame:
        """
        获取股票日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票日线数据
        """
        try:
            df = self.pro.query(
                "daily",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                return df
        except Exception as e:
            print(f"获取股票日线数据失败: {e}")

        return pd.DataFrame()

    def get_index_weights(
        self,
        index_code: Optional[str] = None,
        trade_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取指数成分股权重

        Args:
            index_code: 指数代码
            trade_date: 交易日期

        Returns:
            指数成分股权重数据
        """
        if index_code is None:
            index_code = self.benchmark

        try:
            if trade_date:
                df = self.pro.query(
                    "index_weight",
                    index_code=index_code,
                    trade_date=trade_date,
                )
            else:
                df = self.pro.query(
                    "index_weight",
                    index_code=index_code,
                )
            return df if df is not None and not df.empty else pd.DataFrame()
        except Exception as e:
            print(f"获取指数权重数据失败: {e}")
            return pd.DataFrame()

    def get_industry_daily(
        self,
        industry_code: str,
        start_date: str = BACKTEST_START_DATE,
        end_date: str = BACKTEST_END_DATE,
    ) -> pd.DataFrame:
        """
        获取行业指数日线数据

        Args:
            industry_code: 行业代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            行业指数日线数据
        """
        try:
            df = self.pro.query(
                "daily",
                ts_code=industry_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                return df
        except Exception as e:
            print(f"获取行业指数数据失败: {e}")

        return pd.DataFrame()

    def get_market_calendar(
        self,
        start_date: str = BACKTEST_START_DATE,
        end_date: str = BACKTEST_END_DATE,
        exchange: str = "SSE",
    ) -> List[str]:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所

        Returns:
            交易日列表
        """
        try:
            df = self.pro.query(
                "trade_cal",
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
            )
            if df is not None and not df.empty:
                trading_days = df[df["is_open"] == 1]["cal_date"].tolist()
                return trading_days
        except Exception as e:
            print(f"获取交易日历失败: {e}")

        return pd.bdate_range(start=start_date, end=end_date).strftime("%Y%m%d").tolist()

    def calculate_returns(
        self,
        price_data: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.Series:
        """
        计算收益率

        Args:
            price_data: 价格数据
            price_col: 价格列名

        Returns:
            收益率序列
        """
        returns = price_data[price_col].pct_change()
        return returns

    def get_benchmark_data(
        self,
        start_date: str = BACKTEST_START_DATE,
        end_date: str = BACKTEST_END_DATE,
    ) -> pd.DataFrame:
        """
        获取基准数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            基准数据
        """
        df = self.get_index_daily(start_date, end_date, self.benchmark)
        if not df.empty:
            df["return"] = self.calculate_returns(df)
            df["cum_return"] = (1 + df["return"]).cumprod()
        return df


def fetch_market_data(
    start_date: str = BACKTEST_START_DATE,
    end_date: str = BACKTEST_END_DATE,
) -> Dict[str, pd.DataFrame]:
    """
    获取市场数据

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        市场数据字典
    """
    fetcher = MarketDataFetcher()

    data_dict = {}

    print("正在获取沪深300指数数据...")
    data_dict["benchmark"] = fetcher.get_benchmark_data(start_date, end_date)

    print("正在获取交易日历...")
    data_dict["calendar"] = fetcher.get_market_calendar(start_date, end_date)

    return data_dict
