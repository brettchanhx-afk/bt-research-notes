"""
主动权重因子模块

主动权重因子：相比基准指数权重（沪深300），北向资金行业配置权重的偏配
"""

from typing import Optional
import pandas as pd
import numpy as np
from ..utils.data_utils import normalize_factor, calculate_yoy_change, calculate_qoq_change


class ActiveWeightFactor:
    """
    主动权重因子计算器

    计算方式：北向资金行业配置权重 - 基准指数行业权重
    """

    def __init__(self):
        self.factor_name = "active_weight"

    def calculate(
        self,
        northbound_weight: pd.DataFrame,
        benchmark_weight: pd.DataFrame,
        construction: str = "raw",
    ) -> pd.DataFrame:
        """
        计算主动权重因子

        Args:
            northbound_weight: 北向资金行业权重
            benchmark_weight: 基准指数行业权重
            construction: 构造方式 (raw, yoy, qoq)

        Returns:
            主动权重因子
        """
        if northbound_weight.empty:
            return pd.DataFrame()

        factor_df = northbound_weight.copy()

        if not benchmark_weight.empty:
            factor_df = factor_df.merge(
                benchmark_weight,
                on=["trade_date", "industry_code"],
                how="left",
                suffixes=("", "_benchmark")
            )
            factor_df["active_weight"] = (
                factor_df["weight"] - factor_df["weight_benchmark"]
            )
        else:
            factor_df["active_weight"] = factor_df["weight"]

        if construction == "yoy":
            factor_df = self._apply_yoy(factor_df)
        elif construction == "qoq":
            factor_df = self._apply_qoq(factor_df)

        factor_df["factor"] = factor_df.groupby("trade_date")["active_weight"].transform(
            lambda x: normalize_factor(x) if x.std() > 0 else x
        )

        return factor_df[["trade_date", "industry_code", "factor", "active_weight"]]

    def _apply_yoy(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """应用同比变化"""
        factor_df = factor_df.sort_values(["industry_code", "trade_date"])
        yoy_values = []
        for code in factor_df["industry_code"].unique():
            mask = factor_df["industry_code"] == code
            sub_df = factor_df[mask].copy()
            if len(sub_df) >= 52:
                sub_df["active_weight_yoy"] = calculate_yoy_change(sub_df, "active_weight", 52)
            else:
                sub_df["active_weight_yoy"] = sub_df["active_weight"]
            yoy_values.append(sub_df)
        return pd.concat(yoy_values, ignore_index=True)

    def _apply_qoq(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """应用环比变化"""
        factor_df = factor_df.sort_values(["industry_code", "trade_date"])
        qoq_values = []
        for code in factor_df["industry_code"].unique():
            mask = factor_df["industry_code"] == code
            sub_df = factor_df[mask].copy()
            if len(sub_df) >= 4:
                sub_df["active_weight_qoq"] = calculate_qoq_change(sub_df, "active_weight", 4)
            else:
                sub_df["active_weight_qoq"] = sub_df["active_weight"]
            qoq_values.append(sub_df)
        return pd.concat(qoq_values, ignore_index=True)

    def calculate_northbound_weight(
        self,
        holding_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        计算北向资金行业权重

        Args:
            holding_data: 持股数据

        Returns:
            北向资金行业权重
        """
        if holding_data.empty:
            return pd.DataFrame()

        weight_df = holding_data.copy()
        total_value = weight_df.groupby("trade_date")["holding_value"].transform("sum")
        weight_df["weight"] = weight_df["holding_value"] / total_value

        return weight_df[["trade_date", "industry_code", "weight"]]


def calculate_active_weight_factor(
    northbound_weight: pd.DataFrame,
    benchmark_weight: pd.DataFrame,
    construction: str = "raw",
) -> pd.DataFrame:
    """
    便捷函数：计算主动权重因子

    Args:
        northbound_weight: 北向资金行业权重
        benchmark_weight: 基准指数行业权重
        construction: 构造方式

    Returns:
        主动权重因子
    """
    calculator = ActiveWeightFactor()
    return calculator.calculate(northbound_weight, benchmark_weight, construction)
