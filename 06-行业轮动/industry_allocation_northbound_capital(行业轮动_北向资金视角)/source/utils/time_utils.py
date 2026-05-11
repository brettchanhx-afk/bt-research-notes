"""
时间相关工具函数
"""

from typing import Tuple, List
import pandas as pd


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    """
    获取交易日列表

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD

    Returns:
        交易日列表
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    return [d.strftime("%Y%m%d") for d in dates]


def convert_to_date_str(date_var, format: str = "%Y%m%d") -> str:
    """
    转换日期为字符串格式

    Args:
        date_var: 日期变量 (str, datetime, pd.Timestamp)
        format: 输出格式

    Returns:
        日期字符串
    """
    if isinstance(date_var, str):
        return date_var
    elif isinstance(date_var, (pd.Timestamp, pd.DatetimeIndex)):
        return date_var.strftime(format)
    else:
        return pd.Timestamp(date_var).strftime(format)


def get_frequency_dates(
    start_date: str, end_date: str, frequency: str
) -> List[str]:
    """
    根据频率获取调仓日期

    Args:
        start_date: 开始日期
        end_date: 结束日期
        frequency: 频率 (daily, weekly, biweekly, monthly, bimonthly)

    Returns:
        调仓日期列表
    """
    df = pd.DataFrame({"date": pd.bdate_range(start=start_date, end=end_date)})
    df["date"] = df["date"].dt.strftime("%Y%m%d")

    if frequency == "daily":
        return df["date"].tolist()
    elif frequency == "weekly":
        df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week
        df["year"] = pd.to_datetime(df["date"]).dt.year
        return df.groupby(["year", "week"]).first()["date"].tolist()
    elif frequency == "biweekly":
        df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["biweek"] = (df["week"] - 1) // 2
        return df.groupby(["year", "biweek"]).first()["date"].tolist()
    elif frequency == "monthly":
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        return df.groupby("month").first()["date"].tolist()
    elif frequency == "bimonthly":
        df["bimonth"] = (pd.to_datetime(df["date"]).dt.month - 1) // 2
        df["year"] = pd.to_datetime(df["date"]).dt.year
        return df.groupby(["year", "bimonth"]).first()["date"].tolist()
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")


def split_in_out_sample(
    dates: List[str], split_date: str
) -> Tuple[List[str], List[str]]:
    """
    划分样本内和样本外日期

    Args:
        dates: 日期列表
        split_date: 分割日期

    Returns:
        (样本内日期, 样本外日期)
    """
    in_sample = [d for d in dates if d <= split_date]
    out_of_sample = [d for d in dates if d > split_date]
    return in_sample, out_of_sample
