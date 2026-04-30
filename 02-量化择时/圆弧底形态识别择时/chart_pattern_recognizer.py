from typing import Callable, Dict, Tuple, Union
import numpy as np
import pandas as pd
import talib
from loguru import logger
from scipy.signal import argrelmax, argrelmin

from .utils import calc_smooth


def fetch_prev_high_index(low_points: np.ndarray, high_points: np.ndarray) -> int:
    """查找指定低点之前的最近一个高点索引"""
    latest_low_pos = low_points[-1]
    latest_high_pos = high_points[-1]

    if latest_high_pos < latest_low_pos:
        return latest_high_pos
    return high_points[-2]


class ArcBottomDetector:
    def __init__(
        self,
        ticker_code: str,
        price_data: pd.DataFrame,
        /,
        *,
        smooth_bandwidth: Union[str, float, int] = 1.5,
        smoothing_method: Union[Callable, pd.Series] = calc_smooth,
        distance_limit: float = 10,
        ratio_limit: float = 0.4,
        rate_limit: float = 0.03,
    ) -> None:

        self.ticker_code = ticker_code
        self.price_data = price_data
        self.smooth_bandwidth = smooth_bandwidth
        self.smoothing_method = smoothing_method
        self.distance_limit = distance_limit
        self.ratio_limit = ratio_limit
        self.rate_limit = rate_limit

    @property
    def extract_core_features(self) -> Dict:
        """提取识别圆弧底形态所需的核心特征数据"""
        closing_series = self.price_data["close"].fillna(method="ffill")

        # 对收盘价进行平滑处理，用于精准识别高低点
        if isinstance(self.smoothing_method, Callable):
            smoothed_prices = self.smoothing_method(closing_series, self.smooth_bandwidth)
        elif isinstance(self.smoothing_method, pd.Series):
            smoothed_prices = self.smoothing_method
        else:
            raise ValueError("平滑处理方法仅支持函数或Pandas序列类型！")

        total_length = len(self.price_data)
        current_price = closing_series.iloc[-1]

        # 识别序列中的局部高点与低点位置
        low_indices = argrelmin(smoothed_prices.values)[0]
        high_indices = argrelmax(smoothed_prices.values)[0]

        # 获取距离当前时点最近的有效低点
        nearest_low_pos = low_indices[-1]
        # 获取该低点之前的关键高点位置
        prior_high_pos = fetch_prev_high_index(low_indices, high_indices)

        if prior_high_pos > nearest_low_pos:
            raise ValueError("高点位置不能出现在低点位置之后！")

        # 提取关键价位数据
        lowest_price = closing_series.iloc[nearest_low_pos]
        highest_left_price = closing_series.iloc[prior_high_pos:nearest_low_pos].max()
        highest_right_price = closing_series.iloc[nearest_low_pos:total_length].max()

        # 精准定位左侧最高点的索引
        actual_high_pos = self.price_data.index.get_loc(
            closing_series.iloc[prior_high_pos:nearest_low_pos].idxmax()
        )

        # 计算左右两侧形态的时间跨度
        left_time_span = nearest_low_pos - actual_high_pos
        right_time_span = total_length - nearest_low_pos

        # 计算涨跌幅序列并统计涨跌比例
        daily_change = closing_series.pct_change()
        left_changes = daily_change.iloc[actual_high_pos:nearest_low_pos]
        right_changes = daily_change.iloc[nearest_low_pos:total_length]

        left_sample_size = len(left_changes)
        right_sample_size = len(right_changes)

        # 左侧下跌天数占比、右侧上涨天数占比
        left_drop_ratio = len(left_changes[left_changes < 0]) / left_sample_size if left_sample_size else 0
        right_rise_ratio = len(right_changes[right_changes > 0]) / right_sample_size if right_sample_size else 0

        # 计算平均涨跌幅的绝对值
        left_avg_change = np.abs(left_changes.mean())
        right_avg_change = np.abs(right_changes.mean())

        # 计算单位时间内的价格变化率
        left_price_rate = (highest_left_price / lowest_price - 1) / left_time_span
        right_price_rate = (current_price / lowest_price - 1) / right_time_span

        # 存储核心点位信息供后续使用
        self.key_points = {
            "actual_high_pos": actual_high_pos,
            "nearest_low_pos": nearest_low_pos,
            "prior_high_pos": prior_high_pos,
            "total_length": total_length,
        }

        # 返回完整特征字典
        return {
            "code": self.ticker_code,
            "current_price": current_price,
            "highest_right_price": highest_right_price,
            "highest_left_price": highest_left_price,
            "lowest_price": lowest_price,
            "left_time_span": left_time_span,
            "right_time_span": right_time_span,
            "left_drop_ratio": left_drop_ratio,
            "right_rise_ratio": right_rise_ratio,
            "left_avg_change": left_avg_change,
            "right_avg_change": right_avg_change,
            "left_price_rate": left_price_rate,
            "right_price_rate": right_price_rate,
        }

    def validate_pattern(self) -> bool:
        """验证当前数据是否符合圆弧底形态标准"""
        self.features = self.extract_core_features

        # 条件1：当前价格处于右侧最高点与左侧最高点之间
        price_range_check = (
            (self.features["current_price"] >= self.features["highest_right_price"])
            and (self.features["current_price"] <= self.features["highest_left_price"])
        )

        # 条件2：左右时间跨度均满足最小要求
        time_span_check = (
            (self.features["left_time_span"] > self.distance_limit)
            and (self.features["right_time_span"] > self.distance_limit)
        )

        # 条件3：涨跌比例满足阈值要求
        trend_ratio_check = (
            (self.features["left_drop_ratio"] > self.ratio_limit)
            and (self.features["right_rise_ratio"] > self.ratio_limit)
        )

        # 条件4：平均涨跌幅控制在合理范围内
        avg_change_check = (
            (self.features["left_avg_change"] < self.rate_limit)
            and (self.features["right_avg_change"] < self.rate_limit)
        )

        # 条件5：单位时间价格变化率符合标准
        price_rate_check = (
            (self.features["left_price_rate"] < self.rate_limit)
            and (self.features["right_price_rate"] < self.rate_limit)
        )

        # 所有条件同时满足则判定为有效形态
        return (
            price_range_check
            and time_span_check
            and trend_ratio_check
            and avg_change_check
            and price_rate_check
        )


def detect_arc_bottom_pattern(
    price_data: pd.DataFrame,
    period: int = 200,
    /,
    *,
    min_data_length: int = 252,
    ticker_code: str = None,
    smooth_bandwidth: Union[str, float, int] = 1.5,
) -> Tuple[bool, bool]:
    """
    检测K线数据中是否存在圆弧底形态，并判断是否出现有效买点

    参数：
        price_data: 包含OHLC的行情数据框
        period: 均线计算周期
        min_data_length: 最小数据长度要求
        ticker_code: 标的代码
        smooth_bandwidth: 平滑带宽参数

    返回：
        元组(是否识别到圆弧底形态, 是否满足买入条件)
    """
    if len(price_data) <= min_data_length:
        return False, False

    # 形态识别核心参数配置
    MIN_SPAN_LENGTH = 10
    MIN_TREND_RATIO = 0.4
    MAX_AVG_CHANGE = 0.03
    PRICE_RANGE_TOLERANCE = 0.3

    closing_series = price_data["close"].fillna(method="ffill")
    smoothed_series = calc_smooth(closing_series, smooth_bandwidth)
    # 计算长期均线用于趋势判断
    long_term_ma = talib.EMA(closing_series, period)
    current_ma_value = long_term_ma.iloc[-1]

    # 识别高低点索引
    low_indices = argrelmin(smoothed_series.values)[0]
    high_indices = argrelmax(smoothed_series.values)[0]

    total_length = len(price_data)
    current_price = closing_series.iloc[-1]

    # 获取关键位置索引
    nearest_low_pos = low_indices[-1]
    prior_high_pos = fetch_prev_high_index(low_indices, high_indices)

    if prior_high_pos > nearest_low_pos:
        raise ValueError("高点位置不能晚于低点位置！")

    # 提取关键价格
    lowest_price = closing_series.iloc[nearest_low_pos]
    highest_left_price = closing_series.iloc[prior_high_pos:nearest_low_pos].max()
    actual_high_pos = price_data.index.get_loc(
        closing_series.iloc[prior_high_pos:nearest_low_pos].idxmax()
    )
    highest_right_price = closing_series.iloc[nearest_low_pos:total_length].max()

    # 计算时间跨度
    left_time_span = nearest_low_pos - actual_high_pos
    right_time_span = total_length - nearest_low_pos

    # 计算涨跌数据
    daily_change = closing_series.pct_change()
    left_changes = daily_change.iloc[actual_high_pos:nearest_low_pos]
    right_changes = daily_change.iloc[nearest_low_pos:total_length]

    left_size = len(left_changes)
    right_size = len(right_changes)

    left_drop_ratio = len(left_changes[left_changes < 0]) / left_size if left_size else 0
    right_rise_ratio = len(right_changes[right_changes > 0]) / right_size if right_size else 0

    # 计算平均波动与价格速率
    left_avg_change = np.abs(left_changes.mean())
    right_avg_change = np.abs(right_changes.mean())
    left_price_rate = (highest_left_price / lowest_price - 1) / left_time_span
    right_price_rate = (current_price / lowest_price - 1) / right_time_span

    # 形态验证条件
    price_valid = (current_price >= highest_right_price) and (current_price <= highest_left_price)
    span_valid = (left_time_span > MIN_SPAN_LENGTH) and (right_time_span > MIN_SPAN_LENGTH)
    ratio_valid = (left_drop_ratio > MIN_TREND_RATIO) and (right_rise_ratio > MIN_TREND_RATIO)
    change_valid = (left_avg_change < MAX_AVG_CHANGE) and (right_avg_change < MAX_AVG_CHANGE)
    rate_valid = (left_price_rate < MAX_AVG_CHANGE) and (right_price_rate < MAX_AVG_CHANGE)

    # 形态判定结果
    pattern_detected = price_valid and span_valid and ratio_valid and change_valid and rate_valid

    # 买点验证条件
    above_ma = current_price >= current_ma_value
    in_valid_zone = (
        (1 - PRICE_RANGE_TOLERANCE) * (highest_left_price - lowest_price)
        < current_price - lowest_price
        < (1 + PRICE_RANGE_TOLERANCE) * (highest_left_price - lowest_price)
    )

    # 日志输出识别结果
    current_date = price_data.index[-1].strftime("%Y-%m-%d")
    if pattern_detected:
        logger.info(
            f"日期:{current_date} 标的:{ticker_code} 形态识别:{pattern_detected}, 买点条件 - 均线上方:{above_ma} 价格区间合理:{in_valid_zone}"
        )

    buy_signal_triggered = pattern_detected and above_ma and in_valid_zone
    if buy_signal_triggered:
        logger.info(f"日期:{current_date} 标的:{ticker_code}: 已识别圆弧底形态且触发买点信号")

    return pattern_detected, above_ma and in_valid_zone