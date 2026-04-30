# 趋势识别与波段分析核心工具
# 基于MACD与ATR实现趋势划分、极值点标记与效率指标计算
from collections import defaultdict, namedtuple
from typing import List, Tuple, Dict, Union, Any
import pandas as pd
import numpy as np
from talib import MACD, ATR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def compute_trend_signal(
    dif_series: pd.Series,
    dea_series: pd.Series,
    atr_series: pd.Series,
    scale_coef: float,
    mode: str
) -> pd.Series:
    """
    根据选定模式计算趋势方向信号
    """
    if not isinstance(mode, str):
        raise ValueError("参数mode必须为字符串类型")
    
    mode = mode.upper()
    method_mapping = {
        "A": build_mode_a,
        "B": build_mode_b,
        "C": build_mode_c
    }
    return method_mapping[mode](dif_series, dea_series, atr_series, scale_coef)


def build_mode_a(
    dif_series: pd.Series,
    dea_series: pd.Series,
    *args,
    **kwargs
) -> pd.Series:
    """
    模式A：直接通过DIF与DEA的差值符号判断趋势
    """
    diff_series = dif_series - dea_series
    return diff_series.apply(
        lambda val: 1 if val >= 0 else (-1 if val < 0 else 0)
    )


def build_mode_b(
    dif_series: pd.Series,
    dea_series: pd.Series,
    atr_series: pd.Series,
    scale_coef: float
) -> pd.Series:
    """
    模式B：引入ATR动态阈值过滤噪声，保持趋势持续性
    """
    threshold = atr_series * scale_coef
    diff_series = dif_series - dea_series
    trend_series = pd.Series(index=dif_series.index)
    prev_trend = 0

    for idx, (diff_val, th_val) in enumerate(zip(diff_series, threshold)):
        if idx == 0:
            trend_series.iloc[idx] = 0
            prev_trend = 0
            continue

        if diff_val - th_val >= 0:
            current_trend = 1
        elif diff_val + th_val <= 0:
            current_trend = -1
        else:
            current_trend = prev_trend

        trend_series.iloc[idx] = current_trend
        prev_trend = current_trend

    return trend_series


def cumulative_same_sign(
    dif_series: pd.Series,
    dea_series: pd.Series
) -> pd.Series:
    """
    对同符号的差值进行累加，用于趋势强度累积计算
    """
    result_series = pd.Series(index=dif_series.index)
    diff_series = dif_series - dea_series

    for pos, current_val in enumerate(diff_series):
        if pos == 0:
            result_series.iloc[pos] = 0
            prev_sign = np.sign(current_val)
            prev_accum = current_val
            continue

        current_sign = np.sign(current_val)
        if current_sign == prev_sign:
            result_series.iloc[pos] = current_val + prev_accum
        else:
            result_series.iloc[pos] = 0

        prev_sign = current_sign
        prev_accum = current_val

    return result_series


def build_mode_c(
    dif_series: pd.Series,
    dea_series: pd.Series,
    atr_series: pd.Series,
    scale_coef: float
) -> pd.Series:
    """
    模式C：基于累积差值与动态阈值判断趋势
    """
    accum_series = cumulative_same_sign(dif_series, dea_series)
    threshold = atr_series * scale_coef
    trend_series = pd.Series(index=dif_series.index)
    prev_trend = 0

    for idx, (accum_val, th_val) in enumerate(zip(accum_series, threshold)):
        if idx == 0:
            trend_series.iloc[idx] = 0
            prev_trend = 0
            continue

        if accum_val >= th_val or (prev_trend == 1 and accum_val >= -th_val):
            current_trend = 1
        elif accum_val <= -th_val or (prev_trend == -1 and accum_val <= th_val):
            current_trend = -1
        else:
            current_trend = 0

        trend_series.iloc[idx] = current_trend
        prev_trend = current_trend

    return trend_series


class TrendIdentifier(BaseEstimator, TransformerMixin):
    """
    趋势识别核心类：计算MACD、ATR并生成趋势方向
    """
    def __init__(
        self,
        scale_coef: float,
        mode: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        atr_period: int = 100
    ) -> None:
        self.scale_coef = scale_coef
        self.mode = mode.upper()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.atr_period = atr_period

    def fit(self, X, y=None):
        return self

    def transform(self, price_df: pd.DataFrame, y=None) -> pd.DataFrame:
        dif_res, dea_res, hist_res = MACD(
            price_df["close"],
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        atr_res = ATR(
            price_df["high"],
            price_df["low"],
            price_df["close"],
            self.atr_period
        )

        result_df = price_df[["close"]].copy()
        trend_result = compute_trend_signal(
            dif_res, dea_res, atr_res, self.scale_coef, self.mode
        )
        result_df["raw_diff"] = dif_res - dea_res
        result_df["trend"] = trend_result
        result_df["dif"] = dif_res
        result_df["dea"] = dea_res
        result_df["atr"] = atr_res

        max_lookback = max(
            self.fast_period, self.slow_period,
            self.signal_period, self.atr_period
        )
        return result_df.iloc[max_lookback:]


class ExtremePointMarker(BaseEstimator, TransformerMixin):
    """
    根据趋势标记波段极值点（高点PEAK / 低点VALLEY）
    """
    def __init__(self, trend_col: str) -> None:
        self.trend_col = trend_col

    def fit(self, X, y=None):
        return self

    def transform(self, data_df: pd.DataFrame, y=None) -> pd.DataFrame:
        cleaned_df = data_df.dropna(subset=[self.trend_col]).copy()
        remove_cols = [
            "PEAK", "VALLEY", "PEAK_DATE", "VALLEY_DATE",
            "POINT", "POINT_DATE"
        ]
        exist_cols = [col for col in cleaned_df.columns if col in remove_cols]
        cleaned_df.drop(columns=exist_cols, inplace=True, errors="ignore")

        trend_series = cleaned_df[self.trend_col]
        cleaned_df["segment"] = (trend_series != trend_series.shift(1)).cumsum()

        for seg_id, seg_data in cleaned_df.groupby("segment"):
            if seg_data[self.trend_col].iloc[0] == 1:
                high_idx = seg_data["close"].idxmax()
                cleaned_df.loc[high_idx, "PEAK"] = seg_data.loc[high_idx, "close"]
                cleaned_df.loc[high_idx, "PEAK_DATE"] = high_idx
                cleaned_df.loc[high_idx, "POINT"] = cleaned_df.loc[high_idx, "PEAK"]
                cleaned_df.loc[high_idx, "POINT_DATE"] = high_idx

            if seg_data[self.trend_col].iloc[0] == -1:
                low_idx = seg_data["close"].idxmin()
                cleaned_df.loc[low_idx, "VALLEY"] = seg_data.loc[low_idx, "close"]
                cleaned_df.loc[low_idx, "VALLEY_DATE"] = low_idx
                cleaned_df.loc[low_idx, "POINT"] = cleaned_df.loc[low_idx, "VALLEY"]
                cleaned_df.loc[low_idx, "POINT_DATE"] = low_idx

        last_idx = cleaned_df.index[-1]
        cleaned_df.loc[last_idx, remove_cols] = np.nan
        cleaned_df.drop(columns=["segment"], inplace=True)
        return cleaned_df


class TrendCorrector(BaseEstimator, TransformerMixin):
    """
    基于极值点修正原始趋势信号，生成有效趋势状态
    """
    def __init__(self, target_col: str) -> None:
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, data_df: pd.DataFrame, y=None) -> pd.DataFrame:
        target_col = self.target_col
        work_df = data_df.copy()

        def correction_rule_1(row: pd.Series) -> int:
            if (row[target_col] == 1 and row["close"] <= row["VALLEY_FILLED"]) \
               or (row[target_col] == -1 and row["close"] >= row["PEAK_FILLED"]):
                return -1
            return 1

        def correction_rule_2(row: pd.Series) -> int:
            if row[target_col] != row["prev_trend"]:
                return 1
            elif (row[target_col] == 1 and row["close"] >= row["PEAK_FILLED"]) \
                 or (row[target_col] == -1 and row["close"] <= row["VALLEY_FILLED"]):
                return 1
            return -1

        work_df["prev_trend"] = work_df[target_col].shift(1)
        work_df[["PEAK_FILLED", "VALLEY_FILLED"]] = work_df[["PEAK", "VALLEY"]].ffill()
        work_df["correction_group"] = (work_df[target_col] != work_df[target_col].shift(1)).cumsum()
        valid_df = work_df.dropna(subset=[target_col])

        current_correction = 1
        for idx, row in valid_df.iterrows():
            if current_correction == 1:
                res = correction_rule_1(row)
            else:
                res = correction_rule_2(row)
            valid_df.loc[idx, "correction"] = res
            current_correction = res

        work_df["correction"] = valid_df["correction"]
        work_df["status"] = work_df[target_col] * work_df["correction"]
        work_df.drop(
            columns=["prev_trend", "correction_group", "PEAK_FILLED", "VALLEY_FILLED"],
            inplace=True
        )
        return work_df


class CorrectedPointMarker(BaseEstimator, TransformerMixin):
    """
    基于修正后的趋势状态重新标记极值点
    """
    def __init__(self, status_col: str) -> None:
        self.status_col = status_col

    def fit(self, X, y=None):
        return self

    def transform(self, data_df: pd.DataFrame, y=None) -> pd.DataFrame:
        temp_df = data_df.dropna(subset=[self.status_col]).copy()
        remove_cols = [
            "PEAK", "VALLEY", "PEAK_DATE", "VALLEY_DATE", "POINT", "POINT_DATE"
        ]
        exist_cols = [col for col in temp_df.columns if col in remove_cols]
        temp_df.drop(columns=exist_cols, inplace=True, errors="ignore")

        point_records = extract_segment_extremes(temp_df, self.status_col)
        point_df = pd.DataFrame(
            [item._asdict() for item in point_records.status_map.values()]
        )
        point_df.index = point_df["point_date"]
        point_df.columns = [col.upper() for col in point_df.columns]

        return pd.merge(
            temp_df, point_df, left_index=True, right_index=True, how="left"
        )


class SegmentPointRecorder:
    """
    记录波段极值点信息，支持查询与存储
    """
    def __init__(self) -> None:
        self.PointInfo = namedtuple(
            "PointInfo",
            ["peak", "peak_date", "valley", "valley_date", "point", "point_date"]
        )
        self.status_map: Dict = defaultdict(self.PointInfo)

    def add_record(
        self, key: int,
        peak=None, peak_date=None,
        valley=None, valley_date=None
    ):
        point_val = peak if peak is not None else valley
        point_dt = peak_date if peak_date is not None else valley_date
        self.status_map[key] = self.PointInfo(
            peak=peak, peak_date=peak_date,
            valley=valley, valley_date=valley_date,
            point=point_val, point_date=point_dt
        )

    def get_record(self, key: Any):
        if key not in self.status_map:
            raise KeyError(f"关键字 {key} 不存在")
        return self.status_map[key]


def extract_segment_extremes(
    trend_df: pd.DataFrame,
    status_col: str
) -> SegmentPointRecorder:
    """
    从修正后的趋势中提取完整波段的高低点
    """
    temp_df = trend_df.dropna(subset=[status_col]).copy()
    temp_df["segment_id"] = (
        temp_df[status_col] != temp_df[status_col].shift(1)
    ).cumsum()

    recorder = SegmentPointRecorder()
    prev_status = None
    max_price, min_price = None, None
    max_date, min_date = None, None

    for current_dt, row in temp_df.iterrows():
        current_status = row[status_col]
        current_price = row["close"]
        seg_id = row["segment_id"]

        if prev_status is None:
            max_price, min_price = current_price, current_price
            max_date, min_date = current_dt, current_dt
            prev_status = current_status
            continue

        if current_status != prev_status:
            if prev_status == 1:
                recorder.add_record(seg_id - 1, peak=max_price, peak_date=max_date)
            else:
                recorder.add_record(seg_id - 1, valley=min_price, valley_date=min_date)

            if current_status == 1:
                valley_dt = recorder.get_record(seg_id - 1).valley_date
                slice_data = temp_df.loc[valley_dt:current_dt, "close"]
                max_date = slice_data.idxmax()
                max_price = slice_data.max()
            else:
                peak_dt = recorder.get_record(seg_id - 1).peak_date
                slice_data = temp_df.loc[peak_dt:current_dt, "close"]
                min_date = slice_data.idxmin()
                min_price = slice_data.min()
        else:
            if current_price >= max_price:
                max_price = current_price
                max_date = current_dt
            if current_price <= min_price:
                min_price = current_price
                min_date = current_dt

        prev_status = current_status
    return recorder


class EfficiencyCalculator(BaseEstimator, TransformerMixin):
    """
    计算时间效率与价格效率指标
    """
    def __init__(self, status_col: str, drop_null: bool = True) -> None:
        self.drop_null = drop_null
        self.status_col = status_col

    def fit(self, X, y=None):
        return self

    def transform(self, data_df: pd.DataFrame, y=None) -> pd.DataFrame:
        work_df = data_df.copy()
        if self.drop_null:
            work_df = work_df.dropna(subset=[self.status_col])

        fill_cols = ["PEAK", "VALLEY", "PEAK_DATE", "VALLEY_DATE", "POINT"]
        filled_df = work_df.copy()
        filled_df[fill_cols] = filled_df[fill_cols].ffill()

        filled_df["time_efficiency"] = filled_df.apply(calculate_time_efficiency, axis=1)
        filled_df["price_efficiency"] = filled_df.apply(calculate_price_efficiency, axis=1)
        filled_df[fill_cols] = work_df[fill_cols]
        return filled_df


def calculate_time_efficiency(row: pd.Series) -> float:
    """计算当前时点在波段中的时间位置效率"""
    current_dt = row.name
    if is_closer_to_peak(row):
        return (current_dt - row["PEAK_DATE"]) / (row["PEAK_DATE"] - row["VALLEY_DATE"])
    else:
        return (current_dt - row["VALLEY_DATE"]) / (row["VALLEY_DATE"] - row["PEAK_DATE"])


def calculate_price_efficiency(row: pd.Series) -> float:
    """计算当前价格在波段中的相对位置效率"""
    current_price = row["close"]
    if is_closer_to_peak(row):
        return abs(current_price - row["PEAK"]) / abs(row["PEAK"] - row["VALLEY"])
    else:
        return abs(current_price - row["VALLEY"]) / abs(row["VALLEY"] - row["PEAK"])


def is_closer_to_peak(row: pd.Series) -> bool:
    """判断当前时点更靠近波峰还是波谷"""
    current_dt = row.name
    days_to_peak = (current_dt - row["PEAK_DATE"]).days
    days_to_valley = (current_dt - row["VALLEY_DATE"]).days
    return days_to_peak <= days_to_valley