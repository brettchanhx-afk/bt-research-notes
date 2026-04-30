# -*- coding: utf-8 -*-
"""
factors.py — Barra 风格因子构建模块

因子定义（均为日收益率差值）：
  market   = 沪深300 日收益率          （市场/Beta 因子）
  size     = 中证1000 − 沪深300        （规模因子：小盘 − 大盘）
  value    = 沪深300价值 − 沪深300成长  （价值因子）
  momentum = 中证红利 日收益率          （动量代理）
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FactorBuilder:
    """从原始指数收益率构建 Barra 风格因子。"""

    def __init__(self, index_df: pd.DataFrame):
        """
        Parameters
        ----------
        index_df : DataFrame
            列名为因子名称（market / size / value / growth / momentum），
            值为日收益率，由 DataLoader.load_all_indexes() 返回。
        """
        self.raw = index_df.copy()

    def build(self) -> pd.DataFrame:
        """
        构建并返回因子收益率宽表。
        缺失的因子用零均值小噪声填充，并记录警告。
        """
        factors = pd.DataFrame(index=self.raw.index)

        # 1. 市场因子
        factors["market"] = self._get_col("market_return")

        # 2. 规模因子 = 小盘(中证1000) − 大盘(沪深300)
        size_raw   = self._get_col("size_return",   fallback_zero=True)
        market_raw = self._get_col("market_return", fallback_zero=True)
        factors["size"] = size_raw - market_raw

        # 3. 价值因子 = 价值指数 − 成长指数
        if "value_return" in self.raw.columns and "growth_return" in self.raw.columns:
            factors["value"] = self.raw["value_return"] - self.raw["growth_return"]
        else:
            logger.warning("缺少 value/growth 指数，价值因子用零噪声填充")
            factors["value"] = np.random.normal(0, 0.003, len(factors))

        # 4. 动量因子
        factors["momentum"] = self._get_col("momentum_return")

        factors = factors.dropna()
        logger.info(f"因子构建完成: {factors.shape[0]} 个交易日，{list(factors.columns)}")
        return factors

    def _get_col(self, name: str, fallback_zero: bool = False) -> pd.Series:
        # 尝试直接匹配
        if name in self.raw.columns:
            return self.raw[name]
        # 尝试模糊匹配
        for col in self.raw.columns:
            if name in col or col in name:
                return self.raw[col]
        logger.warning(f"缺少列 '{name}'，{'用零填充' if fallback_zero else '用噪声填充'}")
        if fallback_zero:
            return pd.Series(0.0, index=self.raw.index)
        return pd.Series(np.random.normal(0, 0.005, len(self.raw)), index=self.raw.index)

    @staticmethod
    def describe(factors: pd.DataFrame) -> pd.DataFrame:
        """返回因子描述性统计（均值、标准差、偏度、峰度、年化）。"""
        stats = factors.describe().T
        stats["annualized_mean"] = factors.mean() * 252
        stats["annualized_std"]  = factors.std() * (252 ** 0.5)
        stats["skew"]            = factors.skew()
        stats["kurt"]            = factors.kurt()
        return stats[["mean", "std", "annualized_mean", "annualized_std", "skew", "kurt"]]
