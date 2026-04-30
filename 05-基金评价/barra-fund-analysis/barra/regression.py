# -*- coding: utf-8 -*-
"""
regression.py — Barra 回归分析模块
  - OLS 截面回归（全样本）
  - 滚动窗口回归（风格漂移检测）
"""

import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class BarraRegression:
    """基金收益对 Barra 因子的回归分析。"""

    def __init__(self, fund_return: pd.Series, factors: pd.DataFrame):
        """
        Parameters
        ----------
        fund_return : Series (index=date)
            基金日收益率
        factors     : DataFrame (index=date, columns=[market, size, value, momentum])
            风格因子日收益率
        """
        # 对齐日期
        common_idx = fund_return.index.intersection(factors.index)
        self.y = fund_return.loc[common_idx]
        self.X = factors.loc[common_idx]
        self.results = None
        logger.info(f"回归数据对齐: {len(common_idx)} 个交易日")

    # ── 全样本 OLS ─────────────────────────────────────────
    def fit(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """执行 OLS 回归，返回 statsmodels 结果对象。"""
        X_const = sm.add_constant(self.X)
        model   = sm.OLS(self.y, X_const)
        self.results = model.fit()
        logger.info(f"OLS 回归完成: R²={self.results.rsquared:.4f}, F={self.results.fvalue:.2f}")
        return self.results

    def summary(self) -> str:
        """返回回归结果的文字摘要。"""
        if self.results is None:
            raise RuntimeError("请先调用 fit()")
        return str(self.results.summary())

    def exposures(self) -> pd.Series:
        """返回各因子暴露度（回归系数）。"""
        if self.results is None:
            raise RuntimeError("请先调用 fit()")
        return self.results.params.drop("const", errors="ignore")

    # ── 滚动窗口 ────────────────────────────────────────────
    def rolling_fit(self, window: int = 60) -> pd.DataFrame:
        """
        滚动窗口回归，检测风格漂移。
        返回 DataFrame(index=date, columns=[market,size,value,momentum,r2])
        """
        records = []
        dates   = self.y.index

        for i in range(window, len(dates)):
            idx = dates[i]
            y_win = self.y.iloc[i - window : i]
            X_win = sm.add_constant(self.X.iloc[i - window : i])

            try:
                model = sm.OLS(y_win, X_win).fit()
                row = model.params.drop("const", errors="ignore").to_dict()
                row["r2"] = model.rsquared
                row["date"] = idx
                records.append(row)
            except Exception as e:
                logger.debug(f"滚动窗口 {idx} 回归失败: {e}")
                continue

        df = pd.DataFrame(records).set_index("date")
        logger.info(f"滚动回归完成: {len(df)} 个窗口 (window={window})")
        return df

    # ── 业绩指标 ───────────────────────────────────────────
    @staticmethod
    def performance_metrics(returns: pd.Series) -> dict:
        """
        计算基金业绩指标。
        返回 dict: {cum_return, annual_return, annual_vol, sharpe, max_dd}
        """
        cum_nav = (1 + returns).cumprod()
        cum_ret = cum_nav.iloc[-1] - 1
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * (252 ** 0.5)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

        # 最大回撤
        rolling_max = cum_nav.cummax()
        drawdown    = (cum_nav - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        return {
            "cum_return":    cum_ret,
            "annual_return": ann_ret,
            "annual_vol":    ann_vol,
            "sharpe":        sharpe,
            "max_drawdown":  max_dd,
        }
