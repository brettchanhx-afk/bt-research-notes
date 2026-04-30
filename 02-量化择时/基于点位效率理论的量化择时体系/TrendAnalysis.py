# 趋势分析与波段划分核心工具库
# 基于技术指标实现市场趋势识别、高低点标记与效率计算
from collections import defaultdict, namedtuple
from typing import List, Tuple, Dict, Union, Callable, Any
import datetime as dt
import numpy as np
import pandas as pd
import talib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# 趋势划分核心计算函数
def calculate_trend_indicator(
    dif_series: pd.Series,
    dea_series: pd.Series,
    atr_series: pd.Series,
    threshold_rate: float,
    calc_method: str,
    rolling_window: int = 12
) -> pd.Series:
    """
    根据三种不同算法计算趋势划分指标
    参数：
        dif_series: MACD DIF序列
        dea_series: MACD DEA序列
        atr_series: 平均真实波幅序列
        threshold_rate: 阈值调整系数
        calc_method: 计算模式(A/B/C)
        rolling_window: 滚动求和窗口(仅方法C使用)
    """
    if not isinstance(calc_method, str):
        raise ValueError("计算模式必须为字符串类型")
    
    calc_method = calc_method.upper()
    
    # 方法A：直接使用MACD差值
    if calc_method == "A":
        return dif_series - dea_series
    
    # 方法B：MACD差值减去ATR调整项
    elif calc_method == "B":
        return dif_series - dea_series - atr_series * threshold_rate
    
    # 方法C：同号累加结合ATR调整
    elif calc_method == "C":
        stack_array = np.vstack([dif_series.values, dea_series.values]).T
        sign_condition = np.apply_along_axis(
            lambda x: check_same_sign(x[0], x[1]), 1, stack_array
        )
        diff_value = dif_series - dea_series
        rolling_sum = (sign_condition * diff_value).rolling(rolling_window).sum()
        rolling_sum = rolling_sum * np.where(diff_value == 0, 0, 1)
        return rolling_sum + atr_series * threshold_rate
    
    else:
        raise ValueError("计算模式仅支持A、B、C三种类型")


# 符号判断工具函数
def check_same_sign(value_a: float, value_b: float) -> bool:
    """判断两个数值是否符号相同"""
    return np.signbit(value_a) == np.signbit(value_b)


# 趋势方向划分转换器
class TrendDivision(BaseEstimator, TransformerMixin):
    """
    结合MACD与ATR指标划分价格趋势方向
    输出趋势方向与中间技术指标
    """
    def __init__(
        self,
        rate_coef: float,
        div_method: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        atr_period: int = 100
    ) -> None:
        self.rate_coef = rate_coef
        self.div_method = div_method.upper()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.atr_period = atr_period

    def fit(self, X, y=None):
        return self

    def transform(self, price_df: pd.DataFrame, y=None) -> pd.DataFrame:
        # 计算MACD指标
        dif_res, dea_res, macd_hist = talib.MACD(
            price_df["close"],
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        # 计算ATR指标
        atr_res = talib.ATR(
            price_df["high"], price_df["low"], price_df["close"],
            timeperiod=self.atr_period
        )
        # 构建结果数据集
        result_df = price_df[["close"]].copy()
        trend_val = calculate_trend_indicator(
            dif_res, dea_res, atr_res, self.rate_coef, self.div_method
        )
        result_df["trend_value"] = trend_val
        result_df["direction"] = np.sign(trend_val)
        result_df["dif"] = dif_res
        result_df["dea"] = dea_res
        result_df["atr"] = atr_res
        
        # 剔除指标计算期数据
        max_lookback = max(
            self.fast_period, self.slow_period,
            self.signal_period, self.atr_period
        )
        return result_df.iloc[max_lookback:]


# 趋势极值点标记工具
class MarkExtremePoints(BaseEstimator, TransformerMixin):
    """
    根据趋势方向标记波段高点(PEAK)与低点(VALLEY)
    """
    def __init__(self, direction_col: str) -> None:
        self.direction_col = direction_col

    def fit(self, X, y=None):
        return self

    def transform(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        clean_df = input_df.dropna(subset=[self.direction_col]).copy()
        drop_columns = [
            "PEAK", "VALLEY", "PEAK_DATE", "VALLEY_DATE",
            "POINT", "POINT_DATE"
        ]
        exist_cols = [col for col in clean_df.columns if col in drop_columns]
        clean_df.drop(columns=exist_cols, inplace=True, errors="ignore")
        
        dir_series = clean_df[self.direction_col]
        clean_df["group_id"] = (dir_series != dir_series.shift(1)).cumsum()
        
        # 分组标记高低点
        for group, group_data in clean_df.groupby("group_id"):
            if group_data[self.direction_col].iloc[0] == 1:
                max_idx = group_data["close"].idxmax()
                clean_df.loc[max_idx, "PEAK"] = group_data.loc[max_idx, "close"]
                clean_df.loc[max_idx, "PEAK_DATE"] = max_idx
                clean_df.loc[max_idx, "POINT"] = clean_df.loc[max_idx, "PEAK"]
                clean_df.loc[max_idx, "POINT_DATE"] = max_idx
            if group_data[self.direction_col].iloc[0] == -1:
                min_idx = group_data["close"].idxmin()
                clean_df.loc[min_idx, "VALLEY"] = group_data.loc[min_idx, "close"]
                clean_df.loc[min_idx, "VALLEY_DATE"] = min_idx
                clean_df.loc[min_idx, "POINT"] = clean_df.loc[min_idx, "VALLEY"]
                clean_df.loc[min_idx, "POINT_DATE"] = min_idx
        
        # 最后一个时点不做标记
        last_idx = clean_df.index[-1]
        clean_df.loc[last_idx, drop_columns] = np.nan
        clean_df.drop(columns=["group_id"], inplace=True)
        return clean_df


# 趋势修正工具
class CorrectTrendDirection(BaseEstimator, TransformerMixin):
    """
    基于极值点对原始趋势方向进行修正，生成有效趋势状态
    """
    def __init__(self, target_col: str) -> None:
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        work_df = input_df.copy()
        target_col = self.target_col
        
        # 修正规则函数
        def correction_rule1(row: pd.Series) -> int:
            if (row[target_col] == 1) and (row["close"] <= row["VALLEY_FILL"]):
                return -1
            elif (row[target_col] == -1) and (row["close"] >= row["PEAK_FILL"]):
                return -1
            return 1

        def correction_rule2(row: pd.Series) -> int:
            if row[target_col] != row["prev_dir"]:
                return 1
            elif (row[target_col] == 1) and (row["close"] >= row["PEAK_FILL"]):
                return 1
            elif (row[target_col] == -1) and (row["close"] <= row["VALLEY_FILL"]):
                return 1
            return -1

        # 数据预处理
        work_df["prev_dir"] = work_df[target_col].shift(1)
        work_df[["PEAK_FILL", "VALLEY_FILL"]] = work_df[["PEAK", "VALLEY"]].ffill()
        work_df["correction_group"] = (work_df[target_col] != work_df[target_col].shift(1)).cumsum()
        valid_df = work_df.dropna(subset=[target_col])
        
        current_correction = 1
        for idx, row_data in valid_df.iterrows():
            if current_correction == 1:
                res = correction_rule1(row_data)
                valid_df.loc[idx, "correction"] = res
            else:
                res = correction_rule2(row_data)
                valid_df.loc[idx, "correction"] = res
            current_correction = res
        
        work_df["correction"] = valid_df["correction"]
        work_df["status"] = work_df[target_col] * work_df["correction"]
        work_df.drop(
            columns=["prev_dir", "correction_group", "PEAK_FILL", "VALLEY_FILL"],
            inplace=True
        )
        return work_df


# 修正后极值点标记
class MarkCorrectedExtremes(BaseEstimator, TransformerMixin):
    """基于修正后的趋势状态重新标记极值点"""
    def __init__(self, status_col: str) -> None:
        self.status_col = status_col

    def fit(self, X, y=None):
        return self

    def transform(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        temp_df = input_df.dropna(subset=[self.status_col]).copy()
        drop_cols = [
            "PEAK", "VALLEY", "PEAK_DATE", "VALLEY_DATE", "POINT", "POINT_DATE"
        ]
        exist_cols = [col for col in temp_df.columns if col in drop_cols]
        temp_df.drop(columns=exist_cols, inplace=True, errors="ignore")
        
        extreme_records = extract_extreme_points(temp_df, self.status_col)
        point_data = pd.DataFrame(
            [item._asdict() for item in extreme_records.status_map.values()]
        )
        point_data.index = point_data["point_date"]
        point_data.columns = [col.upper() for col in point_data.columns]
        
        return pd.merge(
            temp_df, point_data, left_index=True, right_index=True, how="left"
        )


# 极值点记录类
class ExtremePointRecorder:
    """波段极值点数据存储与查询"""
    def __init__(self) -> None:
        self.status_map: Dict = defaultdict(
            namedtuple("PointInfo", [
                "peak", "peak_date", "valley", "valley_date",
                "point", "point_date"
            ])
        )

    def add_record(
        self, key: int, peak=None, peak_date=None, valley=None, valley_date=None
    ):
        point_val = peak if peak else valley
        point_dt = peak_date if peak_date else valley_date
        self.status_map[key] = self.status_map[0].__class__(
            peak=peak, peak_date=peak_date,
            valley=valley, valley_date=valley_date,
            point=point_val, point_date=point_dt
        )

    def get_record(self, key: Any):
        if key not in self.status_map:
            raise KeyError(f"关键字{key}不存在")
        return self.status_map[key]


# 提取修正后趋势极值点
def extract_extreme_points(
    trend_df: pd.DataFrame, status_col: str
) -> ExtremePointRecorder:
    """根据修正趋势状态提取完整波段极值点"""
    temp_df = trend_df.dropna(subset=[status_col]).copy()
    temp_df["segment_id"] = (
        temp_df[status_col] != temp_df[status_col].shift(1)
    ).cumsum()
    
    recorder = ExtremePointRecorder()
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
            # 记录上一区间极值
            if prev_status == 1:
                recorder.add_record(seg_id - 1, peak=max_price, peak_date=max_date)
            else:
                recorder.add_record(seg_id - 1, valley=min_price, valley_date=min_date)
            
            # 重置区间极值
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


# 时间与价格效率计算
class EfficiencyIndicators(BaseEstimator, TransformerMixin):
    """计算波段时间效率与价格效率指标"""
    def __init__(self, status_col: str, drop_null: bool = True) -> None:
        self.drop_null = drop_null
        self.status_col = status_col

    def fit(self, X, y=None):
        return self

    def transform(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        work_df = input_df.copy()
        if self.drop_null:
            work_df = work_df.dropna(subset=[self.status_col])
        
        fill_cols = ["PEAK", "VALLEY", "PEAK_DATE", "VALLEY_DATE", "POINT"]
        filled_df = work_df.copy()
        filled_df[fill_cols] = filled_df[fill_cols].ffill()
        
        # 计算效率指标
        filled_df["time_efficiency"] = filled_df.apply(calculate_time_efficiency, axis=1)
        filled_df["price_efficiency"] = filled_df.apply(calculate_price_efficiency, axis=1)
        
        # 还原原始标记
        filled_df[fill_cols] = work_df[fill_cols]
        return filled_df


def calculate_time_efficiency(row: pd.Series) -> float:
    """时间位置效率计算"""
    current_dt = row.name
    if check_closer_to_peak(row):
        return (current_dt - row["PEAK_DATE"]) / (row["PEAK_DATE"] - row["VALLEY_DATE"])
    else:
        return (current_dt - row["VALLEY_DATE"]) / (row["VALLEY_DATE"] - row["PEAK_DATE"])


def calculate_price_efficiency(row: pd.Series) -> float:
    """价格位置效率计算"""
    current_price = row["close"]
    if check_closer_to_peak(row):
        return abs(current_price - row["PEAK"]) / abs(row["PEAK"] - row["VALLEY"])
    else:
        return abs(current_price - row["VALLEY"]) / abs(row["VALLEY"] - row["PEAK"])


def check_closer_to_peak(row: pd.Series) -> bool:
    """判断当前时点更接近波峰还是波谷"""
    current_dt = row.name
    days_to_peak = (current_dt - row["PEAK_DATE"]).days
    days_to_valley = (current_dt - row["VALLEY_DATE"]).days
    return days_to_peak <= days_to_valley


# 价格趋势标准化
class PriceTrendNormalizer:
    """价格序列标准化处理，用于趋势强度分析"""
    def __init__(self, price_series: pd.Series) -> None:
        if not isinstance(price_series, pd.Series):
            raise ValueError("输入必须为Pandas序列")
        self.price_series = price_series

    def monotony_normalize(self) -> pd.Series:
        """涨跌单调性标准化"""
        sign_series = self.price_series.pct_change().apply(np.sign)
        return sign_series.cumsum().fillna(0)

    def ma_normalize(self, window: int = 5) -> pd.Series:
        """均线偏离标准化"""
        ma_series = self.price_series.rolling(window).mean()
        sign_series = (self.price_series - ma_series).apply(np.sign).iloc[window-2:]
        return sign_series.cumsum().fillna(0)

    def hybrid_normalize(self, window: int = 5) -> pd.Series:
        """混合模式标准化(涨跌+均线)"""
        change_sign = self.price_series.pct_change().apply(np.sign)
        ma_series = self.price_series.rolling(window).mean()
        ma_sign = (self.price_series - ma_series).apply(np.sign)
        hybrid_sign = (change_sign + ma_sign) / 2
        return hybrid_sign.iloc[window-2:].cumsum().fillna(0)


# 趋势评分计算
class TrendScoring:
    """基于标准化序列计算趋势得分"""
    def __init__(self, normalized_series: pd.Series) -> None:
        if not isinstance(normalized_series, pd.Series):
            raise ValueError("输入必须为Pandas序列")
        self.normalized_series = normalized_series
        self.point_frames: Dict[str, pd.Series] = {}
        self.point_masks: Dict[str, List] = {}
        self.scores: Dict = {}
        self.ScoreTuple = namedtuple("Score", ["trend_score", "action_score"])
        self.method_map = {
            "opposite": self._relative_extreme_points,
            "absolute": self._absolute_extreme_points
        }

    def compute_trend_score(self, method: str):
        """计算趋势核心得分"""
        mask = self.method_map[method]()
        trend_score = np.square(self.normalized_series[mask].diff()).sum()
        action_score = self.normalized_series.diff().sum()
        self.scores[method] = self.ScoreTuple(trend_score, action_score)
        self.point_frames[method] = self.normalized_series[mask]
        self.point_masks[method] = mask

    def compute_final_score(self) -> float:
        """计算最终综合趋势得分"""
        self.compute_trend_score("opposite")
        self.compute_trend_score("absolute")
        score_opp = self.scores["opposite"].trend_score
        score_abs = self.scores["absolute"].trend_score
        max_score = max(score_opp, score_abs)
        return max_score / (len(self.normalized_series) * 1.5)

    def _relative_extreme_points(self) -> List[bool]:
        """相对极值点识别"""
        diff_series = self.normalized_series.diff().fillna(method="bfill")
        flag_series = pd.Series(False, index=self.normalized_series.index)
        prev_diff = None
        
        for idx, current_diff in diff_series.items():
            if prev_diff is None:
                prev_diff = current_diff
                flag_series.iloc[0] = True
                continue
            flag_series[idx] = (current_diff != prev_diff)
            prev_diff = current_diff
        
        flag_series.iloc[0] = True
        flag_series.iloc[-1] = True
        return flag_series.tolist()

    def _absolute_extreme_points(self) -> List[bool]:
        """绝对极值点识别"""
        data_array = self.normalized_series.values
        data_len = len(data_array)
        max_val = np.max(data_array)
        min_val = np.min(data_array)
        max_indices = np.argwhere(data_array == max_val).flatten()
        min_indices = np.argwhere(data_array == min_val).flatten()
        key_points = np.unique(np.concatenate([min_indices, max_indices, [0, data_len-1]]))
        return [idx in key_points for idx in range(data_len)]