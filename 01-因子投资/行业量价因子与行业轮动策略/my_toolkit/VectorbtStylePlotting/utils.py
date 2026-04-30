import numpy as np
from typing import Tuple  # 调整import顺序，不影响功能


def renormalize(
    arr: np.ndarray, original_range: Tuple[float, float], target_range: Tuple[float, float]
) -> np.ndarray:
    """将数组从原始数值范围重新归一化到目标范围（功能与原函数一致）"""
    # 拆分临时变量，原逻辑不变
    original_min = original_range[0]
    original_max = original_range[1]
    delta_original = original_max - original_min
    
    target_min = target_range[0]
    target_max = target_range[1]
    delta_target = target_max - target_min
    
    # 等价改写计算表达式，拆分步骤
    step1 = arr - original_min
    step2 = step1 / delta_original
    step3 = delta_target * step2
    result = step3 + target_min
    return result


def min_rel_rescale(arr: np.ndarray, target_range: Tuple[float, float]) -> np.ndarray:
    """基于最小值对数组元素进行相对缩放（功能与原函数一致）"""
    # 重命名局部变量，语义一致
    arr_min_val = np.min(arr)
    arr_max_val = np.max(arr)
    
    # 等价改写条件判断，逻辑不变
    val_diff = arr_max_val - arr_min_val
    if val_diff == 0:
        # 显式定义返回值变量，无逻辑变更
        fill_val = target_range[0]
        return np.full(arr.shape, fill_val)
    
    # 原from_range变量重命名
    source_range = (arr_min_val, arr_max_val)
    
    # 拆分变量赋值步骤，原逻辑不变
    source_ratio = np.inf
    if arr_min_val != 0:
        source_ratio = arr_max_val / arr_min_val
    
    # 目标范围比例计算，拆分赋值
    target_min = target_range[0]
    target_max = target_range[1]
    target_ratio = target_max / target_min
    
    # 条件判断逻辑不变，仅重命名变量
    if source_ratio < target_ratio:
        new_target_max = target_min * source_ratio
        target_range = (target_min, new_target_max)
    
    # 调用原renormalize函数，参数仅重命名不改变
    return renormalize(arr, source_range, target_range)


def max_rel_rescale(arr: np.ndarray, target_range: Tuple[float, float]) -> np.ndarray:
    """基于最大值对数组元素进行相对缩放（功能与原函数一致）"""
    # 重命名局部变量，语义一致
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # 等价改写差值计算，逻辑不变
    value_difference = max_val - min_val
    if value_difference == 0:
        # 显式定义填充值，无逻辑变更
        fill_value = target_range[1]
        return np.full(arr.shape, fill_value)
    
    # 原from_range变量重命名
    input_range = (min_val, max_val)
    
    # 拆分比例计算步骤，原逻辑不变
    input_ratio = np.inf
    if min_val != 0:
        input_ratio = max_val / min_val
    
    # 目标范围比例计算，拆分赋值
    target_low = target_range[0]
    target_high = target_range[1]
    dest_ratio = target_high / target_low
    
    # 条件判断逻辑不变，仅重命名变量
    if input_ratio < dest_ratio:
        new_target_low = target_high / input_ratio
        target_range = (new_target_low, target_high)
    
    # 调用原renormalize函数，参数仅重命名不改变
    return renormalize(arr, input_range, target_range)