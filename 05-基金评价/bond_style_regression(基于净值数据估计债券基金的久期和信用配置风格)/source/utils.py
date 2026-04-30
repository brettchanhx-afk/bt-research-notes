# -*- coding: utf-8 -*-
"""
utils.py - 工具函数模块

功能：
- 数据清洗与预处理
- 结果导出
- 报告生成
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import os


def align_dates(data_dict: Dict[str, pd.DataFrame], date_col: str = "date") -> Dict[str, pd.DataFrame]:
    """
    对齐多个数据框的日期
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        数据字典
    date_col : str
        日期列名
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        对齐后的数据
    """
    # 获取所有日期
    all_dates = None
    for df in data_dict.values():
        dates = set(df[date_col])
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    
    common_dates = sorted(list(all_dates))
    
    # 过滤数据
    aligned_data = {}
    for key, df in data_dict.items():
        aligned_data[key] = df[df[date_col].isin(common_dates)].sort_values(date_col).reset_index(drop=True)
    
    return aligned_data


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    计算收益率
    
    Parameters:
    -----------
    prices : pd.Series
        价格序列
    method : str
        计算方法："simple"简单收益率，"log"对数收益率
        
    Returns:
    --------
    pd.Series
        收益率序列
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"未知的收益率计算方法: {method}")


def annualize_return(daily_return: float, days: int = 252) -> float:
    """年化收益率"""
    return daily_return * days


def annualize_volatility(daily_vol: float, days: int = 252) -> float:
    """年化波动率"""
    return daily_vol * np.sqrt(days)


def save_results_to_json(results: Dict, filepath: str):
    """
    保存结果到JSON文件
    
    Parameters:
    -----------
    results : Dict
        结果字典
    filepath : str
        文件路径
    """
    # 转换numpy类型为Python原生类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        return obj
    
    # 递归转换
    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [recursive_convert(item) for item in d]
        else:
            return convert(d)
    
    converted = recursive_convert(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 结果已保存: {filepath}")


def generate_report(fund_code: str, fund_name: str,
                   style_result: Dict,
                   rolling_results: pd.DataFrame,
                   stability: Dict,
                   output_dir: str) -> str:
    """
    生成分析报告
    
    Parameters:
    -----------
    fund_code : str
        基金代码
    fund_name : str
        基金名称
    style_result : Dict
        风格估计结果
    rolling_results : pd.DataFrame
        滚动回测结果
    stability : Dict
        稳定性指标
    output_dir : str
        输出目录
        
    Returns:
    --------
    str
        报告文件路径
    """
    report_path = os.path.join(output_dir, f"{fund_code}_report_{datetime.now().strftime('%Y%m%d')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"债券基金风格分析报告\n")
        f.write(f"基金: {fund_name} ({fund_code})\n")
        f.write(f"报告日期: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("=" * 60 + "\n\n")
        
        # 风格估计结果
        f.write("【风格估计结果】\n")
        f.write(f"估计久期: {style_result['duration']:.2f} 年\n")
        f.write(f"久期风格: {style_result['duration_label']}\n")
        f.write(f"估计信用评分: {style_result['credit']:.2f}\n")
        f.write(f"信用风格: {style_result['credit_label']}\n")
        f.write(f"风格箱定位: {style_result['style_box']}\n")
        f.write(f"回归R²: {style_result['r2']:.4f}\n\n")
        
        # 稳定性分析
        f.write("【风格稳定性分析】\n")
        f.write(f"久期均值: {stability['duration']['mean']:.2f} ± {stability['duration']['std']:.2f}\n")
        f.write(f"信用评分均值: {stability['credit']['mean']:.2f} ± {stability['credit']['std']:.2f}\n")
        f.write(f"R²均值: {stability['r2']['mean']:.4f}\n")
        f.write(f"久期风格变化次数: {stability['duration']['changes']}\n")
        f.write(f"信用风格变化次数: {stability['credit']['changes']}\n\n")
        
        # 滚动窗口结果
        f.write("【滚动窗口估计结果】\n")
        f.write(rolling_results.to_string(index=False))
        f.write("\n\n")
        
        # 方法说明
        f.write("【方法说明】\n")
        f.write("本报告基于华泰证券研报《基于净值数据对债券基金久期和信用配置风格进行估计的方法》\n")
        f.write("核心方法：\n")
        f.write("1. 收益率回归: R = α + β₁×R₁ + ... + βₙ×Rₙ\n")
        f.write("2. 久期估计: D = α + β₁×D₁ + ... + βₙ×Dₙ\n")
        f.write("3. 信用估计: C = α + β₁×C₁ + ... + βₙ×Cₙ\n")
        f.write("=" * 60 + "\n")
    
    print(f"[OK] 报告已生成: {report_path}")
    return report_path


def export_to_csv(data: pd.DataFrame, filepath: str):
    """导出数据到CSV"""
    data.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"[OK] 数据已导出: {filepath}")


def print_summary(style_result: Dict, stability: Dict):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("债券基金风格分析结果摘要")
    print("=" * 60)
    print(f"风格箱定位: {style_result['style_box']}")
    print(f"估计久期: {style_result['duration']:.2f} 年 ({style_result['duration_label']})")
    print(f"估计信用: {style_result['credit']:.2f} 分 ({style_result['credit_label']})")
    print(f"回归R^2: {style_result['r2']:.4f}")
    print(f"久期稳定性: {stability['duration']['mean']:.2f} ± {stability['duration']['std']:.2f}")
    print(f"信用稳定性: {stability['credit']['mean']:.2f} ± {stability['credit']['std']:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Utils Module - Test")
