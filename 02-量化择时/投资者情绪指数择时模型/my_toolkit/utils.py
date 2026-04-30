import numpy as np
from typing import Tuple, Generator


def sliding_window(arr: np.ndarray, window: int, step: int = 1) -> Generator:
    """
    生成指定数组的滑动窗口迭代器，逐个返回滑动窗口数据。
    :param arr: 待处理的源数组
    :type arr: np.ndarray
    :param window: 窗口的天数维度大小
    :type window: int
    :param step: 每个天数对应的时间点数量
    :type step: int
    :return: 迭代返回单个滑动窗口的生成器实例
    :rtype: Generator
    """
    # 验证步长参数的有效性，必须为正整数
    if not (step >= 1):
        raise ValueError("step must be greater than 0")

    # 计算滑动窗口的总长度（天数 × 单日时间点数）
    total_window_size: int = window * step
    # 计算可生成的滑动窗口总数量
    total_window_count: int = (arr.shape[0] - total_window_size + step) // step

    # 构建滑动窗口数组的维度形状
    window_array_shape: Tuple = (
        total_window_count,
        total_window_size,
    ) + arr.shape[1:]
    # 构建滑动窗口数组的内存步长
    window_array_strides: Tuple = (arr.strides[0] * step,) + arr.strides

    # 基于内存步长创建滑动窗口视图（无数据拷贝）
    window_result = np.lib.stride_tricks.as_strided(
        arr, shape=window_array_shape, strides=window_array_strides
    )

    # 遍历所有滑动窗口并逐个返回
    for single_window_data in window_result:
        yield single_window_data