"""
optimizer.py - 带约束均值-方差组合优化器

支持:
  1. cvxopt (推荐，精确求解)
  2. scipy.optimize (无 cvxopt 时的备选)

约束:
  - 权重和 = 1
  - 单资产下限/上限
  - 换手率限制 (|w_new - w_old| ≤ turnover_limit)
"""

import numpy as np
from typing import Optional


# ============================================================================
# PortfolioOptimizer
# ============================================================================

class PortfolioOptimizer:
    """
    均值-方差组合优化器 (带约束)

    目标函数:
      max_w  λ·w^T·μ - 0.5·w^T·Σ·w
      (等价于 min 0.5·w^T·Σ·w - λ·w^T·μ)

    约束:
      w^T·1 = 1
      w_i ≥ 0  (若 long_only=True)
      w_i ≤ cap_i
      |w_i - w_prev_i| ≤ turnover_limit
    """

    def __init__(
        self,
        long_only: bool = True,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        turnover_limit: float = 1.0,   # 单期换手上限
        prev_weights: Optional[np.ndarray] = None,
    ):
        self.long_only = long_only
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.turnover_limit = turnover_limit
        self.prev_weights = prev_weights

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def mean_variance(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        rf: float = 2.0,
        market_cap: Optional[np.ndarray] = None,
        risk_aversion: float = 10.0,
    ) -> np.ndarray:
        """
        求解均值-方差最优权重

        Parameters
        ----------
        mu           : (n,) 预期收益向量 (年化, %)
        Sigma        : (n, n) 协方差矩阵 (年化)
        rf           : float 无风险利率 (%/年)
        market_cap   : (n,) 市值权重 (用于 BL 先验)
        risk_aversion: float 风险厌恶系数

        Returns
        -------
        w : (n,) 最优权重
        """
        n = len(mu)
        mu_excess = mu - rf  # 超额收益

        # ── 尝试 cvxopt ──────────────────────────────────────────────
        try:
            return self._solve_cvxopt(mu_excess, Sigma, risk_aversion)
        except Exception:
            pass

        # ── Fallback: scipy.optimize ───────────────────────────────
        try:
            return self._solve_scipy(mu_excess, Sigma, risk_aversion)
        except Exception:
            pass

        # ── 最后 Fallback: 解析解 (无约束) ─────────────────────────
        return self._solve_analytical(mu_excess, Sigma)

    # ------------------------------------------------------------------
    # cvxopt 求解器
    # ------------------------------------------------------------------
    def _solve_cvxopt(
        self,
        mu_excess: np.ndarray,
        Sigma: np.ndarray,
        risk_aversion: float,
    ) -> np.ndarray:
        import cvxopt

        n = len(mu_excess)
        P = cvxopt.matrix(Sigma * risk_aversion)  # 目标二阶项
        q = cvxopt.matrix(-mu_excess * risk_aversion)  # 目标线性项

        # 等式约束: w^T·1 = 1
        A = cvxopt.matrix(1.0, (1, n))
        b = cvxopt.matrix(1.0)

        # 不等式约束
        #   long_only:  w ≥ min_weight
        #   cap:        w ≤ max_weight
        #   turnover:   w - w_prev ≤ turnover_limit
        #               -(w - w_prev) ≤ turnover_limit
        G_list = []
        h_list = []

        if self.long_only:
            G_list.append(-np.eye(n))
            h_list.append(np.full(n, -self.min_weight))
            G_list.append(np.eye(n))
            h_list.append(np.full(n, self.max_weight))

        if self.prev_weights is not None and self.turnover_limit < 1.0:
            G_turn = np.vstack([np.eye(n), -np.eye(n)])
            h_turn = np.full(2 * n, self.turnover_limit)
            # w - w_prev ≤ turnover_limit
            # -(w - w_prev) ≤ turnover_limit  →  -w + w_prev ≤ turnover_limit
            h_turn[:n] = self.turnover_limit - self.prev_weights
            h_turn[n:] = self.turnover_limit + self.prev_weights
            G_list.append(G_turn)
            h_list.append(h_turn)

        if G_list:
            G = cvxopt.matrix(np.vstack(G_list))
            h = cvxopt.matrix(np.concatenate(h_list))
        else:
            G = None
            h = None

        # 求解
        sol = cvxopt.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
        w = np.array(sol['x']).flatten()

        # 数值容差
        w = np.clip(w, 0.0, 1.0)
        w = w / (w.sum() + 1e-10)  # 重新归一化
        return w

    # ------------------------------------------------------------------
    # scipy fallback
    # ------------------------------------------------------------------
    def _solve_scipy(
        self,
        mu_excess: np.ndarray,
        Sigma: np.ndarray,
        risk_aversion: float,
    ) -> np.ndarray:
        from scipy.optimize import minimize

        n = len(mu_excess)
        risk_av = risk_aversion

        def objective(w):
            port_ret = w @ mu_excess * risk_aversion
            port_var = w @ Sigma @ w * risk_aversion
            return port_var - port_ret  # min (var - ret)

        # 约束
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # 换手约束
        if self.prev_weights is not None and self.turnover_limit < 1.0:
            for i in range(n):
                lo = self.prev_weights[i] - self.turnover_limit
                hi = self.prev_weights[i] + self.turnover_limit
                cons.append({'type': 'ineq', 'fun': lambda w, idx=i: w[idx] - lo})
                cons.append({'type': 'ineq', 'fun': lambda w, idx=i: hi - w[idx]})

        bounds = [(0.0 if self.long_only else -1.0, self.max_weight) for _ in range(n)]
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        w = result.x
        w = np.clip(w, 0.0, 1.0)
        return w / w.sum()

    # ------------------------------------------------------------------
    # 解析解 (无约束/等权)
    # ------------------------------------------------------------------
    def _solve_analytical(
        self,
        mu_excess: np.ndarray,
        Sigma: np.ndarray,
    ) -> np.ndarray:
        """无约束最大收益组合 (仅用于完全失败时兜底)"""
        n = len(mu_excess)
        # 按超额收益比例分配
        pos_mu = np.maximum(mu_excess, 0)
        if pos_mu.sum() > 0:
            return pos_mu / pos_mu.sum()
        return np.ones(n) / n
