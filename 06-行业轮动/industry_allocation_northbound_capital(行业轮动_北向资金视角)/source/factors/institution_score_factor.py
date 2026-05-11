"""
机构打分因子模块

机构打分因子：根据行业净流入的机构数目，对行业进行打分
"""

from typing import Optional
import pandas as pd
import numpy as np
from ..utils.data_utils import normalize_factor, calculate_yoy_change, calculate_qoq_change


class InstitutionScoreFactor:
    """
    机构打分因子计算器

    计算方式：根据净流入某行业的机构数目对行业打分
    """

    def __init__(self):
        self.factor_name = "institution_score"

    def calculate(
        self,
        institution_flow_data: pd.DataFrame,
        construction: str = "raw",
    ) -> pd.DataFrame:
        """
        计算机构打分因子

        Args:
            institution_flow_data: 机构流向数据（包含机构信息和行业信息）
            construction: 构造方式 (raw, yoy, qoq)

        Returns:
            机构打分因子
        """
        if institution_flow_data.empty:
            return pd.DataFrame()

        factor_df = institution_flow_data.copy()

        factor_df["net_inflow_flag"] = (factor_df["flow"] > 0).astype(int)

        score_df = factor_df.groupby(["trade_date", "industry_code"]).agg(
            institution_count=("net_inflow_flag", "sum"),
            total_institution=("institution_code", "nunique"),
        ).reset_index()

        score_df["institution_ratio"] = (
            score_df["institution_count"] / score_df["total_institution"]
        )

        score_df["score"] = score_df["institution_ratio"] * 100

        if construction == "yoy":
            score_df = self._apply_yoy(score_df)
        elif construction == "qoq":
            score_df = self._apply_qoq(score_df)

        score_df["factor"] = score_df.groupby("trade_date")["score"].transform(
            lambda x: normalize_factor(x) if x.std() > 0 else x
        )

        return score_df[["trade_date", "industry_code", "factor", "score", "institution_ratio"]]

    def _apply_yoy(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """应用同比变化"""
        factor_df = factor_df.sort_values(["industry_code", "trade_date"])
        yoy_values = []
        for code in factor_df["industry_code"].unique():
            mask = factor_df["industry_code"] == code
            sub_df = factor_df[mask].copy()
            if len(sub_df) >= 52:
                sub_df["score_yoy"] = calculate_yoy_change(sub_df, "score", 52)
            else:
                sub_df["score_yoy"] = sub_df["score"]
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
                sub_df["score_qoq"] = calculate_qoq_change(sub_df, "score", 4)
            else:
                sub_df["score_qoq"] = sub_df["score"]
            qoq_values.append(sub_df)
        return pd.concat(qoq_values, ignore_index=True)

    def calculate_composite_score(
        self,
        flow_data: pd.DataFrame,
        institution_col: str = "institution_code",
    ) -> pd.DataFrame:
        """
        计算复合机构打分

        Args:
            flow_data: 流向数据
            institution_col: 机构列名

        Returns:
            复合打分
        """
        if flow_data.empty:
            return pd.DataFrame()

        flow_data = flow_data.copy()
        flow_data["is_positive"] = (flow_data["flow"] > 0).astype(int)
        flow_data["is_negative"] = (flow_data["flow"] < 0).astype(int)

        score_df = flow_data.groupby(["trade_date", "industry_code"]).agg(
            positive_count=("is_positive", "sum"),
            negative_count=("is_negative", "sum"),
            total_flow=("flow", "sum"),
        ).reset_index()

        score_df["composite_score"] = (
            score_df["positive_count"] - score_df["negative_count"]
        ) / (score_df["positive_count"] + score_df["negative_count"] + 1)

        return score_df


def calculate_institution_score_factor(
    institution_flow_data: pd.DataFrame,
    construction: str = "raw",
) -> pd.DataFrame:
    """
    便捷函数：计算机构打分因子

    Args:
        institution_flow_data: 机构流向数据
        construction: 构造方式

    Returns:
        机构打分因子
    """
    calculator = InstitutionScoreFactor()
    return calculator.calculate(institution_flow_data, construction)
