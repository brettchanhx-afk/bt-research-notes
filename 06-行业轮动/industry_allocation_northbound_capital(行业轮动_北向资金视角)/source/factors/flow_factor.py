"""
资金流向因子模块

资金流向因子：使用成交额对北向资金流（增减持）进行归一化
"""

from typing import Optional, Dict
import pandas as pd
import numpy as np
from ..utils.data_utils import normalize_factor, rank_factor, calculate_yoy_change, calculate_qoq_change


class CapitalFlowFactor:
    """
    资金流向因子计算器

    计算方式：北向资金在特定行业的成交额 / 全部A股在该行业的成交额
    """

    def __init__(self):
        self.factor_name = "capital_flow"

    def calculate(
        self,
        northbound_flow: pd.DataFrame,
        industry_turnover: pd.DataFrame,
        construction: str = "raw",
    ) -> pd.DataFrame:
        """
        计算资金流向因子

        Args:
            northbound_flow: 北向资金流向数据
            industry_turnover: 行业成交额数据
            construction: 构造方式 (raw, yoy, qoq)

        Returns:
            资金流向因子
        """
        if northbound_flow.empty or industry_turnover.empty:
            return pd.DataFrame()

        factor_df = northbound_flow.copy()

        if "industry_code" not in factor_df.columns:
            factor_df["industry_code"] = "all"

        factor_df = factor_df.merge(
            industry_turnover,
            on=["trade_date", "industry_code"],
            how="left",
            suffixes=("", "_total")
        )

        factor_df["flow_ratio"] = factor_df["north_flow"] / factor_df["total_flow"]
        factor_df["flow_ratio"] = factor_df["flow_ratio"].replace([np.inf, -np.inf], np.nan)

        if construction == "yoy":
            factor_df = self._apply_yoy(factor_df)
        elif construction == "qoq":
            factor_df = self._apply_qoq(factor_df)

        factor_df["factor"] = factor_df.groupby("trade_date")["flow_ratio"].transform(
            lambda x: normalize_factor(x) if x.std() > 0 else x
        )

        return factor_df[["trade_date", "industry_code", "factor", "flow_ratio"]]

    def _apply_yoy(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """应用同比变化"""
        factor_df = factor_df.sort_values(["industry_code", "trade_date"])
        yoy_values = []
        for code in factor_df["industry_code"].unique():
            mask = factor_df["industry_code"] == code
            sub_df = factor_df[mask].copy()
            if len(sub_df) >= 52:
                sub_df["flow_ratio_yoy"] = calculate_yoy_change(sub_df, "flow_ratio", 52)
            else:
                sub_df["flow_ratio_yoy"] = sub_df["flow_ratio"]
            yoy_values.append(sub_df)
        return pd.concat(yoyoy_values, ignore_index=True)

    def _apply_qoq(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """应用环比变化"""
        factor_df = factor_df.sort_values(["industry_code", "trade_date"])
        qoq_values = []
        for code in factor_df["industry_code"].unique():
            mask = factor_df["industry_code"] == code
            sub_df = factor_df[mask].copy()
            if len(sub_df) >= 4:
                sub_df["flow_ratio_qoq"] = calculate_qoq_change(sub_df, "flow_ratio", 4)
            else:
                sub_df["flow_ratio_qoq"] = sub_df["flow_ratio"]
            qoq_values.append(sub_df)
        return pd.concat(qoq_values, ignore_index=True)

    def calculate_net_flow(
        self,
        flow_data: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        计算净流入

        Args:
            flow_data: 流向数据
            window: 窗口期

        Returns:
            净流入序列
        """
        return flow_data["north_flow"].rolling(window=window).sum()

    def calculate_net_flow_rate(
        self,
        flow_data: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        计算净流入率

        Args:
            flow_data: 流向数据
            window: 窗口期

        Returns:
            净流入率序列
        """
        net_flow = self.calculate_net_flow(flow_data, window)
        total_flow = flow_data["total_flow"].rolling(window=window).sum()
        return net_flow / total_flow


def calculate_flow_factor(
    northbound_flow: pd.DataFrame,
    industry_turnover: pd.DataFrame,
    construction: str = "raw",
) -> pd.DataFrame:
    """
    便捷函数：计算资金流向因子

    Args:
        northbound_flow: 北向资金流向数据
        industry_turnover: 行业成交额数据
        construction: 构造方式

    Returns:
        资金流向因子
    """
    calculator = CapitalFlowFactor()
    return calculator.calculate(northbound_flow, industry_turnover, construction)
