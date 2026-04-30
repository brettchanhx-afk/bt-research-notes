"""
utils.py - 工具函数模块
提供数据清洗、格式转换、数学计算等通用工具函数
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    验证DataFrame是否包含必需的列
    
    Parameters:
        df: 待验证的DataFrame
        required_cols: 必需的列名列表
    
    Returns:
        bool: 验证是否通过
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")
    return True


def calculate_cumulative_return(returns: pd.Series) -> pd.Series:
    """
    计算累计收益率
    
    Parameters:
        returns: 收益率序列（小数形式，如0.05表示5%）
    
    Returns:
        pd.Series: 累计收益率序列
    """
    return (1 + returns).cumprod() - 1


def geometric_link(returns: List[float]) -> float:
    """
    几何链接计算多期累计收益
    
    Parameters:
        returns: 各期收益率列表（小数形式）
    
    Returns:
        float: 累计收益率
    """
    if not returns:
        return 0.0
    cumulative = 1.0
    for r in returns:
        cumulative *= (1 + r)
    return cumulative - 1


def annualize_return(period_return: float, periods_per_year: int) -> float:
    """
    年化收益率
    
    Parameters:
        period_return: 期间收益率（小数形式）
        periods_per_year: 每年期数（如12表示月度，4表示季度）
    
    Returns:
        float: 年化收益率
    """
    return (1 + period_return) ** periods_per_year - 1


def format_percentage(value: float, decimal: int = 2) -> str:
    """
    将小数格式化为百分比字符串
    
    Parameters:
        value: 小数形式的数值（如0.0523）
        decimal: 小数位数
    
    Returns:
        str: 百分比字符串（如"5.23%"）
    """
    return f"{value * 100:.{decimal}f}%"


def align_dates(dfs: List[pd.DataFrame], date_col: str = 'date') -> List[pd.DataFrame]:
    """
    对齐多个DataFrame的日期索引
    
    Parameters:
        dfs: DataFrame列表
        date_col: 日期列名
    
    Returns:
        List[pd.DataFrame]: 对齐后的DataFrame列表
    """
    # 获取所有日期集合
    all_dates = set()
    for df in dfs:
        all_dates.update(df[date_col].unique())
    
    # 按日期排序
    common_dates = sorted(list(all_dates))
    
    # 重新索引
    aligned_dfs = []
    for df in dfs:
        df_aligned = df.set_index(date_col).reindex(common_dates).reset_index()
        df_aligned.rename(columns={'index': date_col}, inplace=True)
        aligned_dfs.append(df_aligned)
    
    return aligned_dfs


def check_weights_sum(weights: pd.Series, tolerance: float = 0.01) -> bool:
    """
    检查权重之和是否接近1（或100%）
    
    Parameters:
        weights: 权重序列
        tolerance: 容差范围
    
    Returns:
        bool: 是否通过检查
    """
    total = weights.sum()
    # 支持小数形式和百分比形式
    if total > 1.5:  # 百分比形式
        target = 100
    else:
        target = 1
    
    return abs(total - target) <= tolerance * target


def fill_missing_returns(returns: pd.Series, method: str = 'zero') -> pd.Series:
    """
    填充缺失的收益率数据
    
    Parameters:
        returns: 收益率序列
        method: 填充方法 ('zero', 'forward', 'backward')
    
    Returns:
        pd.Series: 填充后的序列
    """
    if method == 'zero':
        return returns.fillna(0)
    elif method == 'forward':
        return returns.fillna(method='ffill')
    elif method == 'backward':
        return returns.fillna(method='bfill')
    else:
        raise ValueError(f"不支持的填充方法: {method}")


def calculate_drawdown(nav: pd.Series) -> pd.Series:
    """
    计算回撤序列
    
    Parameters:
        nav: 净值序列
    
    Returns:
        pd.Series: 回撤序列（负值表示回撤）
    """
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    return drawdown


def get_period_dates(start_date: str, end_date: str, freq: str = 'M') -> pd.DatetimeIndex:
    """
    生成期间日期序列
    
    Parameters:
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率 ('D'日, 'W'周, 'M'月, 'Q'季度)
    
    Returns:
        pd.DatetimeIndex: 日期索引
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def pivot_attribution_results(attribution_df: pd.DataFrame) -> pd.DataFrame:
    """
    将归因结果透视展示
    
    Parameters:
        attribution_df: 归因结果DataFrame，包含date, sector, allocation, selection, interaction列
    
    Returns:
        pd.DataFrame: 透视后的结果
    """
    pivot_df = attribution_df.pivot_table(
        index='date',
        columns='sector',
        values=['allocation', 'selection', 'interaction'],
        aggfunc='sum'
    )
    return pivot_df


def aggregate_by_sector(df: pd.DataFrame, sector_col: str = 'sector', 
                        value_cols: List[str] = None) -> pd.DataFrame:
    """
    按行业聚合数据
    
    Parameters:
        df: 输入DataFrame
        sector_col: 行业列名
        value_cols: 需要聚合的数值列
    
    Returns:
        pd.DataFrame: 聚合后的结果
    """
    if value_cols is None:
        value_cols = [col for col in df.columns if col != sector_col]
    
    return df.groupby(sector_col)[value_cols].sum().reset_index()


def print_attribution_summary(attribution_results: Dict[str, float]):
    """
    打印归因结果摘要
    
    Parameters:
        attribution_results: 归因结果字典
    """
    print("=" * 50)
    print("Brinson绩效归因结果")
    print("=" * 50)
    print(f"总超额收益:     {format_percentage(attribution_results.get('total', 0))}")
    print(f"类别配置收益:   {format_percentage(attribution_results.get('allocation', 0))}")
    print(f"个券选择收益:   {format_percentage(attribution_results.get('selection', 0))}")
    print(f"交互作用收益:   {format_percentage(attribution_results.get('interaction', 0))}")
    print("=" * 50)
    
    # 验证分解是否完整
    total = attribution_results.get('total', 0)
    allocation = attribution_results.get('allocation', 0)
    selection = attribution_results.get('selection', 0)
    interaction = attribution_results.get('interaction', 0)
    
    check_sum = allocation + selection + interaction
    diff = abs(total - check_sum)
    
    if diff > 1e-10:
        print(f"警告: 分解求和 ({format_percentage(check_sum)}) 与总超额收益 ({format_percentage(total)}) 不一致")
        print(f"差异: {format_percentage(diff)}")
    else:
        print("✓ 归因分解验证通过")


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试几何链接
    returns = [0.05, -0.02, 0.03]
    cum_return = geometric_link(returns)
    print(f"几何链接测试: {returns} -> {format_percentage(cum_return)}")
    
    # 测试年化
    annual = annualize_return(0.1, 4)
    print(f"年化收益测试: 季度10% -> 年化{format_percentage(annual)}")
    
    # 测试回撤
    nav = pd.Series([1.0, 1.1, 1.05, 1.15, 1.08])
    dd = calculate_drawdown(nav)
    print(f"回撤测试: {dd.tolist()}")
    
    print("工具函数测试完成!")
