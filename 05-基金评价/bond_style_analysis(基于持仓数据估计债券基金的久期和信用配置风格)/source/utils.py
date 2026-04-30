# -*- coding: utf-8 -*-
"""
utils.py - 债券基金风格分析工具函数

功能:
- 数据清洗与格式转换
- JSON/CSV/Excel 结果导出
- 报告生成
- 路径管理

作者: QClaw Agent | 版本: 1.0.0
"""

from __future__ import annotations

import os
import json
import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 路径管理
# =============================================================================

def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)
    return path


def get_output_dir(fund_code: str = None) -> str:
    """
    获取输出目录

    Parameters
    ----------
    fund_code : str, optional
        基金代码，用于创建子目录

    Returns
    -------
    str
        输出目录路径
    """
    root = get_project_root()
    output_dir = os.path.join(root, "output")
    if fund_code:
        output_dir = os.path.join(output_dir, fund_code)
    return ensure_dir(output_dir)


def get_data_dir() -> str:
    """获取数据目录"""
    root = get_project_root()
    data_dir = os.path.join(root, "data")
    return ensure_dir(data_dir)


# =============================================================================
# 数据清洗
# =============================================================================

def clean_nav_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗净值数据

    - 去除重复日期
    - 填充缺失值
    - 排序
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 确保有日期列
    if "date" not in df.columns:
        if df.index.name == "date":
            df = df.reset_index()
        else:
            df["date"] = pd.to_datetime(df.index)

    # 去重
    df = df.drop_duplicates(subset=["date"], keep="last")

    # 排序
    df = df.sort_values("date").reset_index(drop=True)

    # 数值列处理
    num_cols = ["nav", "cumulative_nav", "daily_return"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_holdings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗持仓数据

    - 去除无市值记录
    - 标准化数值格式
    - 排序
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 确保必要列
    required = ["bond_code", "bond_name"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # 市值/占比数值化
    for col in ["market_value", "pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 去除无效行
    df = df.dropna(subset=["bond_code"])
    df = df[df["bond_code"].astype(str).str.strip() != ""]

    # 排序
    if "market_value" in df.columns:
        df = df.sort_values("market_value", ascending=False).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


# =============================================================================
# 结果导出
# =============================================================================

def export_results(
    results: dict,
    fund_code: str,
    output_format: str = "csv",
    output_dir: str = None,
) -> dict:
    """
    导出分析结果

    Parameters
    ----------
    results : dict
        分析结果字典
    fund_code : str
        基金代码
    output_format : str
        输出格式: 'csv' / 'json' / 'excel'
    output_dir : str, optional
        输出目录

    Returns
    -------
    dict
        导出文件路径
    """
    if output_dir is None:
        output_dir = get_output_dir(fund_code)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    # Summary
    summary_key = f"{output_dir}/{fund_code}_style_summary_{timestamp}"
    if output_format == "csv":
        summary_path = f"{summary_key}.csv"
        pd.DataFrame([results]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    elif output_format == "json":
        summary_path = f"{summary_key}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(results), f, ensure_ascii=False, indent=2)
    else:
        summary_path = f"{summary_key}.xlsx"
        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            pd.DataFrame([results]).to_excel(writer, sheet_name="Summary", index=False)

    paths["summary"] = summary_path
    print(f"[OK] 结果已导出: {summary_path}")
    return paths


def export_holdings(holdings: pd.DataFrame, fund_code: str, period: str = "latest") -> str:
    """
    导出持仓数据
    """
    output_dir = get_output_dir(fund_code)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/{fund_code}_holdings_{period}_{timestamp}.csv"
    holdings.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 持仓已导出: {path}")
    return path


# =============================================================================
# 报告生成
# =============================================================================

def generate_analysis_report(
    fund_code: str,
    fund_name: str,
    style_result: dict,
    perf_metrics: dict = None,
    output_dir: str = None,
) -> str:
    """
    生成文本分析报告

    Returns
    -------
    str
        报告文件路径
    """
    if output_dir is None:
        output_dir = get_output_dir(fund_code)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("=" * 60)
    lines.append(f"债券基金风格分析报告")
    lines.append(f"生成时间: {timestamp}")
    lines.append("=" * 60)
    lines.append(f"基金代码: {fund_code}")
    lines.append(f"基金名称: {fund_name}")
    lines.append("")
    lines.append("【一、风格分析结果】")
    dur_style = style_result.get("duration_style", 0.0)
    dur_label = style_result.get("duration_style_label", "unknown")
    cred_style = style_result.get("credit_style", 0.0)
    cred_label = style_result.get("credit_style_label", "unknown")
    style_box = style_result.get("style_box", "unknown_unknown")

    lines.append(f"  久期风格: {dur_style:.2f} 年 ({dur_label})")
    lines.append(f"  信用风格: {cred_style:.2f} 分 ({cred_label})")
    lines.append(f"  风格箱定位: {style_box}")
    lines.append(f"  持仓数量: {style_result.get('n_holdings', 'N/A')} 只")

    if perf_metrics:
        lines.append("")
        lines.append("【二、绩效指标】")
        lines.append(f"  年化收益率: {perf_metrics.get('annual_return', 0)*100:.2f}%")
        lines.append(f"  年化波动率: {perf_metrics.get('annual_vol', 0)*100:.2f}%")
        lines.append(f"  夏普比率: {perf_metrics.get('sharpe', 0):.4f}")
        lines.append(f"  最大回撤: {perf_metrics.get('max_drawdown', 0)*100:.2f}%")
        lines.append(f"  卡尔马比率: {perf_metrics.get('calmar', 0):.4f}")
        lines.append(f"  总收益: {perf_metrics.get('total_return', 0)*100:.2f}%")

    lines.append("")
    lines.append("=" * 60)
    lines.append("报告结束")

    report_text = "\n".join(lines)
    report_path = f"{output_dir}/{fund_code}_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[OK] 分析报告已生成: {report_path}")
    return report_path


# =============================================================================
# JSON 序列化兼容
# =============================================================================

def _make_serializable(obj: Any) -> Any:
    """将 numpy/pandas 类型转换为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    elif isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    elif isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient="records")
    else:
        return obj


# =============================================================================
# 配置常量
# =============================================================================

# 示例债券基金列表 (纯数据展示，无推荐意图)
SAMPLE_BOND_FUNDS = {
    "000012": "华夏债券A",
    "000084": "博时裕祥A",
    "000355": "景顺长城优选",
    "001001": "华夏希望债券A",
    "020003": "国泰金龙债券A",
}

# 默认参数
DEFAULT_PARAMS = {
    "start_date": "20230101",
    "end_date": None,
    "lookback_days": 252,
    "rebalance_freq": 60,
    "risk_free_rate": 0.0,
}
