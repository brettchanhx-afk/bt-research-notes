"""
北向资金情绪指数模块

情绪指数构建：
1. 构建13项二级事件指标
2. 根据事件分析结果选取6项有效指标
3. 合成北向资金情绪指数
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from ..utils.data_utils import calculate_historical_percentile


class SentimentIndexBuilder:
    """
    北向资金情绪指数构建器
    """

    def __init__(self):
        self.event_indicators = {}
        self.selected_indicators = []
        self.sentiment_index = None

    def calculate_counter_market_flow(
        self,
        flow_data: pd.DataFrame,
        price_data: pd.DataFrame,
        consecutive_days: int = 1,
    ) -> pd.DataFrame:
        """
        逆市流入/流出指标

        当指数连续下跌时北向资金净流入（逆市流入）或
        当指数连续上涨时北向资金净流出（逆市流出）

        Args:
            flow_data: 流向数据
            price_data: 价格数据
            consecutive_days: 连续天数

        Returns:
            事件指标
        """
        df = flow_data.merge(price_data[["trade_date", "close", "return"]], on="trade_date", how="left")

        df = df.sort_values("trade_date")

        df["price_consecutive"] = (
            (df["close"].diff() > 0).rolling(window=consecutive_days).min()
        )

        df["counter_market_signal"] = 0
        mask = (df["price_consecutive"] == 1) & (df["north_flow"] > 0)
        df.loc[mask, "counter_market_signal"] = 1

        mask = (df["price_consecutive"] == 0) & (df["close"].diff(-1) < 0) & (df["north_flow"] < 0)
        df.loc[mask, "counter_market_signal"] = -1

        return df[["trade_date", "counter_market_signal"]]

    def calculate_large_flow(
        self,
        flow_data: pd.DataFrame,
        window: int = 60,
        upper_percentile: float = 90,
        lower_percentile: float = 10,
    ) -> pd.DataFrame:
        """
        大额流入/流出指标

        单日流入/流出额占据历史高位

        Args:
            flow_data: 流向数据
            window: 历史窗口
            upper_percentile: 上百分位
            lower_percentile: 下百分位

        Returns:
            事件指标
        """
        df = flow_data.copy()
        df = df.sort_values("trade_date")

        df["flow_percentile"] = calculate_historical_percentile(
            df["north_flow"], window
        )

        df["large_flow_signal"] = 0
        df.loc[df["flow_percentile"] >= upper_percentile, "large_flow_signal"] = 1
        df.loc[df["flow_percentile"] <= lower_percentile, "large_flow_signal"] = -1

        return df[["trade_date", "large_flow_signal", "flow_percentile"]]

    def calculate_abnormal_flow(
        self,
        flow_data: pd.DataFrame,
        window: int = 10,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        反常态流入/流出指标

        过去N日内大于等于M日净流入后，出现单日大额流出或相反

        Args:
            flow_data: 流向数据
            window: 窗口大小
            threshold: 阈值

        Returns:
            事件指标
        """
        df = flow_data.copy()
        df = df.sort_values("trade_date")

        df["positive_days"] = (
            (df["north_flow"] > 0).rolling(window=window).sum()
        )
        df["positive_ratio"] = df["positive_days"] / window

        df["abnormal_flow_signal"] = 0

        mask_in = (df["positive_ratio"] >= threshold) & (df["north_flow"] < 0)
        mask_out = (df["positive_ratio"] <= (1 - threshold)) & (df["north_flow"] > 0)

        df.loc[mask_in, "abnormal_flow_signal"] = 1
        df.loc[mask_out, "abnormal_flow_signal"] = -1

        return df[["trade_date", "abnormal_flow_signal"]]

    def calculate_divergence_flow(
        self,
        flow_data: pd.DataFrame,
        price_data: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        资金流与指数信息耦合指标 - 背离

        净流入占据历史高位，收盘价占据历史低位，或相反

        Args:
            flow_data: 流向数据
            price_data: 价格数据
            window: 窗口大小

        Returns:
            事件指标
        """
        df = flow_data.merge(price_data[["trade_date", "close", "vol"]], on="trade_date", how="left")
        df = df.sort_values("trade_date")

        df["flow_ma"] = df["north_flow"].rolling(window=window).mean()
        df["flow_std"] = df["north_flow"].rolling(window=window).std()
        df["flow_zscore"] = (df["north_flow"] - df["flow_ma"]) / df["flow_std"]

        df["price_ma"] = df["close"].rolling(window=window).mean()
        df["price_std"] = df["close"].rolling(window=window).std()
        df["price_zscore"] = (df["close"] - df["price_ma"]) / df["price_std"]

        df["divergence_signal"] = 0
        mask_positive = (df["flow_zscore"] > 1.5) & (df["price_zscore"] < -0.5)
        mask_negative = (df["flow_zscore"] < -1.5) & (df["price_zscore"] > 0.5)

        df.loc[mask_positive, "divergence_signal"] = 1
        df.loc[mask_negative, "divergence_signal"] = -1

        return df[["trade_date", "divergence_signal", "flow_zscore", "price_zscore"]]

    def calculate_consistency_flow(
        self,
        institution_flow_data: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        机构一致性指标

        流入流出的子机构数目占机构总数的比例占历史高位

        Args:
            institution_flow_data: 机构流向数据
            window: 窗口大小

        Returns:
            事件指标
        """
        if institution_flow_data.empty:
            return pd.DataFrame()

        df = institution_flow_data.copy()
        df["positive_flag"] = (df["flow"] > 0).astype(int)
        df["negative_flag"] = (df["flow"] < 0).astype(int)

        agg_df = df.groupby("trade_date").agg(
            positive_count=("positive_flag", "sum"),
            total_count=("institution_code", "nunique"),
        ).reset_index()

        agg_df["consistency_ratio"] = agg_df["positive_count"] / agg_df["total_count"]
        agg_df = agg_df.sort_values("trade_date")
        agg_df["consistency_percentile"] = calculate_historical_percentile(
            agg_df["consistency_ratio"], window
        )

        agg_df["consistency_signal"] = 0
        agg_df.loc[agg_df["consistency_percentile"] >= 75, "consistency_signal"] = 1
        agg_df.loc[agg_df["consistency_percentile"] <= 25, "consistency_signal"] = -1

        return agg_df[["trade_date", "consistency_signal", "consistency_ratio", "consistency_percentile"]]

    def build_sentiment_index(
        self,
        event_data: Dict[str, pd.DataFrame],
        selected_events: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        构建情绪指数

        Args:
            event_data: 事件数据字典
            selected_events: 选择的事件列表
            weights: 事件权重

        Returns:
            情绪指数
        """
        if weights is None:
            weights = {k: 1.0 for k in selected_events}

        merged = None
        for event_name in selected_events:
            if event_name not in event_data:
                continue
            event_df = event_data[event_name].copy()
            event_df = event_df.rename(columns={"signal": event_name})

            if merged is None:
                merged = event_df
            else:
                merged = merged.merge(event_df, on="trade_date", how="outer")

        if merged is None:
            return pd.DataFrame()

        merged = merged.fillna(0)

        signal_cols = [col for col in merged.columns if col in selected_events]
        merged["sentiment_index"] = sum(
            merged[col] * weights.get(col, 1.0) for col in signal_cols
        )

        merged["sentiment_index_normalized"] = (
            merged["sentiment_index"] - merged["sentiment_index"].expanding().mean()
        ) / merged["sentiment_index"].expanding().std()

        self.sentiment_index = merged
        return merged

    def event_analysis(
        self,
        event_data: pd.DataFrame,
        forward_returns: pd.Series,
        event_date_col: str = "trade_date",
    ) -> Dict[str, float]:
        """
        事件分析

        Args:
            event_data: 事件数据
            forward_returns: 未来收益
            event_date_col: 事件日期列

        Returns:
            事件分析结果
        """
        if event_data.empty or forward_returns.empty:
            return {}

        df = event_data.merge(
            forward_returns.reset_index(),
            left_on=event_date_col,
            right_on="trade_date",
            how="inner",
        )

        positive_events = df[df["signal"] > 0]
        negative_events = df[df["signal"] < 0]

        results = {
            "positive_count": len(positive_events),
            "negative_count": len(negative_events),
        }

        if len(positive_events) > 0:
            results["positive_win_rate"] = (positive_events["return"] > 0).mean()
            results["positive_avg_return"] = positive_events["return"].mean()
        else:
            results["positive_win_rate"] = np.nan
            results["positive_avg_return"] = np.nan

        if len(negative_events) > 0:
            results["negative_win_rate"] = (negative_events["return"] < 0).mean()
            results["negative_avg_return"] = negative_events["return"].mean()
        else:
            results["negative_win_rate"] = np.nan
            results["negative_avg_return"] = np.nan

        return results


def calculate_sentiment_index(
    flow_data: pd.DataFrame,
    price_data: pd.DataFrame,
    selected_events: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    便捷函数：计算情绪指数

    Args:
        flow_data: 流向数据
        price_data: 价格数据
        selected_events: 选择的事件列表

    Returns:
        情绪指数
    """
    if selected_events is None:
        selected_events = [
            "counter_market_signal",
            "large_flow_signal",
            "abnormal_flow_signal",
            "divergence_signal",
            "consistency_signal",
        ]

    builder = SentimentIndexBuilder()
    event_data = {}

    print("计算逆市流入/流出指标...")
    event_data["counter_market_signal"] = builder.calculate_counter_market_flow(
        flow_data, price_data
    )

    print("计算大额流入/流出指标...")
    event_data["large_flow_signal"] = builder.calculate_large_flow(flow_data)

    print("计算反常态流入/流出指标...")
    event_data["abnormal_flow_signal"] = builder.calculate_abnormal_flow(flow_data)

    print("计算背离指标...")
    event_data["divergence_signal"] = builder.calculate_divergence_flow(
        flow_data, price_data
    )

    print("构建情绪指数...")
    sentiment_index = builder.build_sentiment_index(event_data, selected_events)

    return sentiment_index
