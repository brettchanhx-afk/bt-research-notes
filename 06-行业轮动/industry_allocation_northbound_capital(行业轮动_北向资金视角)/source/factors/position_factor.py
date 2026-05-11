"""
持仓市值因子模块

持仓市值因子：北向资金持仓在中信/申万一级行业的流通市值，
除以全部A股在该行业的流通市值
"""

from typing import Optional, Dict
import pandas as pd
import numpy as np
from ..utils.data_utils import normalize_factor, rank_factor, calculate_yoy_change, calculate_qoq_change


class PositionMarketValueFactor:
    """
    持仓市值因子计算器

    计算方式：北向资金在特定行业的持仓市值 / 全部A股在该行业的流通市值
    """

    def __init__(self):
        self.factor_name = "position_market_value"

    def calculate(
        self,
        northbound_holding: pd.DataFrame,
        market_cap_data: pd.DataFrame,
        industry_mapping: pd.DataFrame,
        construction: str = "raw",
    ) -> pd.DataFrame:
        """
        计算持仓市值因子

        Args:
            northbound_holding: 北向资金持股数据
            market_cap_data: 流通市值数据
            industry_mapping: 行业映射数据
            construction: 构造方式 (raw, yoy, qoq)

        Returns:
            持仓市值因子
        """
        if northbound_holding.empty or market_cap_data.empty:
            return pd.DataFrame()

        factor_df = self._merge_data(
            northbound_holding, market_cap_data, industry_mapping
        )

        factor_df["ratio"] = factor_df["north_value"] / factor_df["total_value"]
        factor_df["ratio"] = factor_df["ratio"].replace([np.inf, -np.inf], np.nan)

        if construction == "yoy":
            factor_df = self._apply_yoy(factor_df)
        elif construction == "qoq":
            factor_df = self._apply_qoq(factor_df)

        factor_df["factor"] = factor_df.groupby("trade_date")["ratio"].transform(
            lambda x: normalize_factor(x) if x.std() > 0 else x
        )

        return factor_df[["trade_date", "industry_code", "factor"]]

    def _merge_data(
        self,
        northbound_holding: pd.DataFrame,
        market_cap_data: pd.DataFrame,
        industry_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """合并数据"""
        if "industry_code" not in northbound_holding.columns:
            northbound_holding = northbound_holding.merge(
                industry_mapping, left_on="symbol", right_on="stock_code", how="left"
            )

        merged = northbound_holding.merge(
            market_cap_data, on=["trade_date", "industry_code"], how="outer"
        )
        merged = merged.fillna(0)
        return merged

    def _apply_yoy(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """应用同比变化"""
        factor_df = factor_df.sort_values(["industry_code", "trade_date"])
        factor_df["ratio_yoy"] = factor_df.groupby("industry_code")["ratio"].transform(
            lambda x: calculate_yoy_change(factor_df, "ratio", 52) if len(x) >= 52 else x
        )
        return factor_df

    def _apply_qoq(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """应用环比变化"""
        factor_df = factor_df.sort_values(["industry_code", "trade_date"])
        factor_df["ratio_qoq"] = factor_df.groupby("industry_code")["ratio"].transform(
            lambda x: calculate_qoq_change(factor_df, "ratio", 4) if len(x) >= 4 else x
        )
        return factor_df

    def calculate_composite(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        计算复合因子

        Args:
            factors_dict: 因子字典
            weights: 因子权重

        Returns:
            复合因子
        """
        if weights is None:
            weights = {k: 1.0 / len(factors_dict) for k in factors_dict.keys()}

        merged = None
        for name, df in factors_dict.items():
            if merged is None:
                merged = df.copy()
                merged["weighted_factor"] = merged["factor"] * weights[name]
            else:
                merged = merged.merge(
                    df, on=["trade_date", "industry_code"], how="outer"
                )
                merged["weighted_factor"] += merged["factor"] * weights[name]

        return merged[["trade_date", "industry_code", "weighted_factor"]]


def calculate_position_factor(
    northbound_data: pd.DataFrame,
    market_data: pd.DataFrame,
    construction: str = "raw",
) -> pd.DataFrame:
    """
    便捷函数：计算持仓市值因子

    Args:
        northbound_data: 北向资金数据
        market_data: 市场数据
        construction: 构造方式

    Returns:
        持仓市值因子
    """
    calculator = PositionMarketValueFactor()
    industry_mapping = pd.DataFrame(
        {"stock_code": [], "industry_code": [], "industry_name": []}
    )
    return calculator.calculate(northbound_data, market_data, industry_mapping, construction)
