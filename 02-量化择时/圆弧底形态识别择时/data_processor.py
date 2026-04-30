import numpy as np
import pandas as pd


def generate_smoothed_series(price_series: pd.Series, bandwidth: float, **kwargs) -> pd.Series:
    """
    基于核岭回归算法对价格序列进行平滑处理
    参数：
        price_series: 带日期索引的原始价格序列
        bandwidth: 平滑带宽配置，支持数值类型或自动计算参数
    返回：
        平滑处理后的时间序列，保持原始索引不变
    """
    from statsmodels.nonparametric.kernel_regression import KernelReg

    # 构建用于回归的一维索引数组
    position_array = np.arange(len(price_series))
    
    # 统一带宽参数格式，适配模型输入要求
    if isinstance(bandwidth, (float, int)):
        bandwidth = [bandwidth]

    # 初始化局部线性核回归模型
    regression_model = KernelReg(
        endog=price_series,
        exog=position_array,
        reg_type="ll",
        var_type="c",
        bw=bandwidth
    )
    
    # 执行拟合并生成平滑结果
    smoothed_result, _ = regression_model.fit(position_array)
    
    # 封装结果并返回与原序列索引一致的Series
    return pd.Series(data=smoothed_result, index=price_series.index)