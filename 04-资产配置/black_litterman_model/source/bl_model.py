"""
bl_model.py - Black-Litterman 核心模型

实现四步法:
  Step 1: CAPM 先验 (均衡收益)  Π = λ·Σ·w_mkt
  Step 2: 主观观点 (P, Q, Ω)
  Step 3: 后验分布贝叶斯融合
  Step 4: 均值-方差优化

参考文献:
  - Black-Litterman (1992) Global Portfolio Optimization, FAJ
  - Idzorek (2007) A step-by-step guide to Black-Litterman
  - Meucci (2010) The Black-Litterman approach
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal

from source.optimizer import PortfolioOptimizer


# ============================================================================
# 核心: BlackLittermanModel
# ============================================================================

class BlackLittermanModel:
    """
    Black-Litterman 资产配置模型

    Parameters
    ----------
    returns      : pd.DataFrame (T×n) 日频收益率
    market_weights : str or array
        - 'benchmark' → 股10%+债80%+商品10%
        - 'ew'        → 等权
        - 传入 array  → 自定义 (n,)
    risk_aversion : float λ 市场风险厌恶系数 (通常 2.5~10)
    tau           : float τ 观点置信度参数 (1/lookback_months)
    omega_method  : str
        'default'  → Ω = diag(diag(P Σ P^T)) * τ
        'idzorek'  → Ω = τ·Σ·(1-w_mkt)  投资者视角
        'reverse'  → Ω = τ·Σ  无参
    rf_rate       : float 无风险利率 (%)
    view_lookback  : int 观点计算回望天数 (默认1=上期收益作为观点)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_weights: str | np.ndarray = 'benchmark',
        risk_aversion: float = 10.0,
        tau: float = 1 / 60,
        omega_method: Literal['default', 'idzorek', 'reverse'] = 'default',
        rf_rate: float = 2.0,
        view_lookback: int = 1,
    ):
        self.returns = returns.dropna()
        self.n_assets = self.returns.shape[1]
        self.asset_names = list(self.returns.columns)
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.omega_method = omega_method
        self.rf_rate = rf_rate
        self.view_lookback = view_lookback

        # ── 计算 Sigma (协方差矩阵) ──────────────────────────────────────
        self.Sigma_ = self.returns.cov() * 252  # 年化协方差
        # 年化收益率 (用于观点计算)
        self.annual_ret_ = self.returns.mean() * 252

        # ── 市场均衡权重 ────────────────────────────────────────────────
        if isinstance(market_weights, str):
            if market_weights == 'benchmark':
                # 股10% + 债80% + 商品10%
                self.w_mkt_ = self._default_benchmark_weights()
            elif market_weights == 'ew':
                self.w_mkt_ = np.ones(self.n_assets) / self.n_assets
            else:
                raise ValueError(f"Unknown market_weights: {market_weights}")
        else:
            self.w_mkt_ = np.array(market_weights).reshape(-1)

        # ── Step 1: 先验均衡收益 ────────────────────────────────────────
        #   Π = λ · Σ · w_mkt
        self.pi_prior_ = self.risk_aversion * (self.Sigma_ @ self.w_mkt_)
        self.pi_prior_ = pd.Series(self.pi_prior_, index=self.asset_names)

        # ── Step 2: 构建观点矩阵 ────────────────────────────────────────
        self._build_views()

        # ── Step 3: 后验收益 ───────────────────────────────────────────
        self._compute_posterior()

        # ── Step 4: 后验协方差 ─────────────────────────────────────────
        self._compute_posterior_cov()

    # ── 默认基准权重 ───────────────────────────────────────────────────────
    def _default_benchmark_weights(self) -> np.ndarray:
        """股10% + 债80% + 商品10% (按资产名判断)"""
        w = np.zeros(self.n_assets)
        for i, name in enumerate(self.asset_names):
            n = name.upper()
            if any(x in n for x in ['CSI300', 'SP500', 'HSI', '沪深', '标普', '恒生']):
                w[i] = 0.10        # 股票
            elif any(x in n for x in ['GOV', 'CORP', '511', '国债', '企业债']):
                w[i] = 0.80        # 债券
            elif any(x in n for x in ['NHCI', 'GOLD', '1599', '5188', '商品', '黄金']):
                w[i] = 0.10        # 商品
            else:
                w[i] = 1.0 / self.n_assets
        w = w / w.sum()  # 归一化
        return w

    # ── Step 2: 构建观点矩阵 P, Q, Ω ────────────────────────────────────
    def _build_views(self):
        """
        用历史收益率作为主观观点:
          Q_i = r_{t-1} (上月收益率, 年化)
        P = I_k (每个观点对应一个资产)
        Ω = diag(σ_i² · τ) (观点不确定性)
        """
        n = self.n_assets
        lookback = self.view_lookback

        # 观点 = 过去 lookback 期收益率均值 (年化)
        past = self.returns.iloc[-lookback:].mean() * 252
        self.Q_ = past.values.astype(float)   # (n,) 向量

        # P = I_n (每个观点覆盖一个资产)
        self.P_ = np.eye(n)                    # (n, n)

        # Ω = diag( σ_i² · τ ) — 每个观点方差
        var_vector = np.diag(self.Sigma_.values)  # 年化方差 (np.diag on ndarray)
        self.Omega_ = np.diag(var_vector * self.tau)  # (n, n)

    # ── Step 3: 后验收益 ─────────────────────────────────────────────────
    def _compute_posterior(self):
        """
        BL后验收益公式:
          mu_BL = Π + τ·Σ·P^T · (τ·Σ·P + Ω)^{-1} · (Q - P·Π)
        """
        Sigma = self.Sigma_.values
        P = self.P_
        Q = self.Q_.reshape(-1, 1)
        Pi = self.pi_prior_.values.reshape(-1, 1)
        tau = self.tau
        Omega = self.Omega_

        # (τ·Σ·P + Ω)
        M = tau * P @ Sigma + Omega
        M_inv = np.linalg.inv(M)

        # BL收益
        delta_mu = tau * Sigma @ P.T @ M_inv @ (Q - P @ Pi)
        mu_bl = Pi + delta_mu

        self.posterior_mu_ = pd.Series(
            mu_bl.flatten(), index=self.asset_names
        )

    # ── Step 3续: 后验协方差 ──────────────────────────────────────────────
    def _compute_posterior_cov(self):
        """
        后验协方差:
          Σ_BL = Σ + (τ·Σ·P^T) · (τ·Σ·P + Ω)^{-1} · (τ·Σ·P^T)^T
        """
        Sigma = self.Sigma_.values
        P = self.P_
        tau = self.tau
        Omega = self.Omega_

        M = tau * P @ Sigma + Omega
        M_inv = np.linalg.inv(M)
        term = tau * Sigma @ P.T @ M_inv @ (tau * Sigma @ P.T).T
        self.posterior_cov_ = Sigma + term

    # ── 最优权重 ─────────────────────────────────────────────────────────
    def get_weights_bl(self) -> np.ndarray:
        """求解 BL 后验均值-方差最优权重"""
        opt = PortfolioOptimizer()
        return opt.mean_variance(
            mu=self.posterior_mu_.values,
            Sigma=self.posterior_cov_,
            rf=self.rf_rate,
            market_cap=self.w_mkt_,
            risk_aversion=self.risk_aversion,
        )

    def get_weights_mvo(self) -> np.ndarray:
        """纯 MVO 权重 (使用先验收益)"""
        opt = PortfolioOptimizer()
        return opt.mean_variance(
            mu=self.pi_prior_.values,
            Sigma=self.Sigma_.values,
            rf=self.rf_rate,
            market_cap=self.w_mkt_,
            risk_aversion=self.risk_aversion,
        )

    def get_weights_fixed(
        self,
        w_stock: float = 0.10,
        w_bond: float = 0.80,
        w_commodity: float = 0.10,
    ) -> np.ndarray:
        """固定权重 (股10%+债80%+商品10%)"""
        w = np.zeros(self.n_assets)
        for i, name in enumerate(self.asset_names):
            n = name.upper()
            if any(x in n for x in ['CSI300', 'SP500', 'HSI']):
                w[i] = w_stock
            elif any(x in n for x in ['GOV', 'CORP', '511', '国债', '企业债']):
                w[i] = w_bond
            elif any(x in n for x in ['NHCI', 'GOLD', '1599', '5188', '商品', '黄金']):
                w[i] = w_commodity
        w = w / w.sum()
        return w

    # ── 摘要表 ─────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """先验 vs 后验收益对比"""
        return pd.DataFrame({
            '先验Π (%)':    self.pi_prior_,
            '观点Q (%)':    pd.Series(self.Q_, index=self.asset_names),
            '后验μ_BL (%)': self.posterior_mu_,
        })


# ============================================================================
# 便捷入口
# ============================================================================

def run_bl_model(
    returns: pd.DataFrame,
    strategy: Literal['BL_S1', 'BL_S2', 'MVO', 'FIXED'] = 'BL_S1',
    **kwargs,
) -> np.ndarray:
    """
    一键运行 BL/MVO/FIXED 策略，返回权重向量

    Parameters
    ----------
    returns  : pd.DataFrame (T×n) 日频收益率
    strategy : 'BL_S1' | 'BL_S2' | 'MVO' | 'FIXED'
    kwargs   : 透传给 BlackLittermanModel
    """
    bl = BlackLittermanModel(returns=returns, **kwargs)

    if strategy == 'BL_S1':
        return bl.get_weights_bl()
    elif strategy == 'BL_S2':
        bl.risk_aversion = 5.0   # 策略2用更低的 λ
        return bl.get_weights_bl()
    elif strategy == 'MVO':
        return bl.get_weights_mvo()
    elif strategy == 'FIXED':
        return bl.get_weights_fixed()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
