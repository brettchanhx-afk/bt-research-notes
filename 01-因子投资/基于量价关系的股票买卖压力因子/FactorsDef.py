import pandas as pd
import numpy as np
import time as tm

# 通用滑动窗口计算工具函数
def sliding_window_calc(dataset, compute_func, window_length) -> pd.Series:
    index_array = np.arange(len(dataset))
    arr_shape = (index_array.size - window_length + 1, window_length)
    arr_strides = (index_array.strides[0], index_array.strides[0])
    
    # 构建滚动窗口索引矩阵
    window_indexes = np.lib.stride_tricks.as_strided(
        index_array, shape=arr_shape, strides=arr_strides, writeable=True
    )
    
    # 批量执行窗口计算
    result_list = [compute_func(dataset.iloc[idx]) for idx in window_indexes]
    return pd.concat(result_list, axis=0)

# 成交量加权平均价格计算核心函数
def calculate_vwap(dataset: pd.DataFrame) -> pd.Series:
    current_idx = dataset.index.get_level_values(1)[-1]
    weight_price = np.average(dataset['close'], weights=dataset['volume'])
    return pd.DataFrame({'vwap': weight_price}, index=[current_idx])

# 价格偏离指标计算核心函数
def calculate_apb(dataset: pd.DataFrame, min_trade_days: int) -> pd.Series:
    current_idx = dataset.index.get_level_values(1)[-1]
    valid_trade_count = dataset['paused'].sum()
    
    # 有效交易天数不足阈值时返回空值
    if valid_trade_count < min_trade_days:
        avg_val = np.mean(dataset['vwap'])
        weighted_val = np.average(dataset['vwap'], weights=dataset['volume'])
        return pd.DataFrame({'apb': avg_val / weighted_val}, index=[current_idx])
    else:
        return pd.DataFrame({'apb': np.nan}, index=[current_idx])

# 滚动窗口计算5日成交量加权平均价格
def compute_rolling_5d_vwap(data_frame):
    return data_frame.groupby(level='code').apply(
        lambda group: sliding_window_calc(
            group, lambda window: calculate_vwap(window), 5)
    )

# 滚动窗口计算30日成交量加权平均价格
def compute_rolling_30d_vwap(data_frame):
    return data_frame.groupby(level='code').apply(
        lambda group: sliding_window_calc(
            group, lambda window: calculate_vwap(window), 30)
    )

# 滚动窗口计算5日价格偏离指标
def compute_rolling_5d_apb(data_frame):
    return data_frame.groupby(level='code').apply(
        lambda group: sliding_window_calc(
            group, lambda window: calculate_apb(window, 3), 5)
    )

# 滚动窗口计算30日价格偏离指标
def compute_rolling_30d_apb(data_frame):
    return data_frame.groupby(level='code').apply(
        lambda group: sliding_window_calc(
            group, lambda window: calculate_apb(window, 15), 30)
    )