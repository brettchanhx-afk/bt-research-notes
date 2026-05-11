"""
Optimization module for macro risk parity and risk minimization strategies.
Implements scipy.optimize based solvers for:
1. Asset Risk Parity (baseline)
2. Macro Risk Parity
3. Macro Risk Minimization
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

from source.risk_attribution import (
    compute_portfolio_factor_risk_contribution,
    build_theta_matrix,
)


def asset_risk_parity(returns, cov_matrix=None):
    """
    Compute asset risk parity weights.
    Objective: minimize sum_i (MRC_i - sigma/N)^2
    where MRC_i is marginal risk contribution of asset i.
    """
    if cov_matrix is None:
        cov_matrix = returns.cov().values

    N = cov_matrix.shape[0]

    def objective(w):
        w = np.abs(w)
        w = w / w.sum()
        port_var = w @ cov_matrix @ w
        port_vol = np.sqrt(port_var + 1e-10)
        mrc = cov_matrix @ w / port_vol
        target_mrc = port_vol / N
        return np.sum((mrc - target_mrc) ** 2)

    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]

    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints={"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0},
                      options={"maxiter": 1000, "ftol": 1e-10})

    weights = np.abs(result.x)
    weights = weights / weights.sum()
    return weights


def macro_risk_parity(returns, B, factor_cov, idio_var=0.01, asset_class_caps=None):
    """
    Compute macro risk parity weights.
    Objective: minimize sum_k (FRC_k - 1/K)^2
    where FRC_k is the factor risk contribution of factor k.

    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (for residual variance estimation)
    B : np.ndarray
        Factor loading matrix (N, K)
    factor_cov : np.ndarray
        Factor covariance matrix (K, K)
    idio_var : float or np.array
        Idiosyncratic variance
    asset_class_caps : dict or None
        e.g. {"国内权益": 0.10, "海外权益": 0.10, "商品": 0.20}

    Returns:
    --------
    weights : np.ndarray
        Optimal asset weights (N,)
    """
    N = B.shape[0]
    K = B.shape[1]

    if np.isscalar(idio_var):
        idio_var_arr = idio_var * np.ones(N)
    else:
        idio_var_arr = np.array(idio_var)

    def objective(w):
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-10)

        frc, idio_frc = compute_portfolio_factor_risk_contribution(
            w, B, factor_cov, idio_var_arr
        )

        total_macro_risk = frc.sum()
        if total_macro_risk < 1e-10:
            return 1e10

        frc_normalized = frc / total_macro_risk
        target = 1.0 / K
        loss = np.sum((frc_normalized - target) ** 2)
        return loss

    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0}]

    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-12})

    weights = np.clip(result.x, 0, 1)
    weights = weights / (weights.sum() + 1e-10)
    return weights


def macro_risk_minimization(returns, B, factor_cov, idio_var=0.01,
                             asset_class_caps=None):
    """
    Compute macro risk minimization weights.
    Objective: minimize sum_k (FRC_k)^2 (i.e., minimize total macro risk)

    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (for residual variance estimation)
    B : np.ndarray
        Factor loading matrix (N, K)
    factor_cov : np.ndarray
        Factor covariance matrix (K, K)
    idio_var : float or np.array
        Idiosyncratic variance
    asset_class_caps : dict or None
        Asset class weight caps, e.g. {"国内权益": 0.10, "海外权益": 0.10, "商品": 0.20}

    Returns:
    --------
    weights : np.ndarray
        Optimal asset weights (N,)
    """
    N = B.shape[0]
    K = B.shape[1]

    if np.isscalar(idio_var):
        idio_var_arr = idio_var * np.ones(N)
    else:
        idio_var_arr = np.array(idio_var)

    def objective(w):
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-10)

        frc, idio_frc = compute_portfolio_factor_risk_contribution(
            w, B, factor_cov, idio_var_arr
        )
        return np.sum(frc ** 2)

    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0}]

    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-12})

    weights = np.clip(result.x, 0, 1)
    weights = weights / (weights.sum() + 1e-10)
    return weights


def macro_risk_budget(returns, B, factor_cov, idio_var=0.01, target_frc=None):
    """
    Compute macro risk budget weights (generalization of parity and minimization).
    Objective: minimize sum_k (FRC_k - target_frc_k)^2

    Parameters:
    -----------
    target_frc : np.array
        Target factor risk contributions (K,)
    """
    N = B.shape[0]
    K = B.shape[1]

    if target_frc is None:
        target_frc = np.ones(K) / K

    if np.isscalar(idio_var):
        idio_var_arr = idio_var * np.ones(N)
    else:
        idio_var_arr = np.array(idio_var)

    def objective(w):
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-10)

        frc, _ = compute_portfolio_factor_risk_contribution(
            w, B, factor_cov, idio_var_arr
        )

        total = frc.sum()
        if total < 1e-10:
            return 1e10

        frc_norm = frc / total
        return np.sum((frc_norm - target_frc) ** 2)

    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]

    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=[{"type": "eq",
                                   "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0}],
                      options={"maxiter": 2000, "ftol": 1e-12})

    weights = np.clip(result.x, 0, 1)
    weights = weights / (weights.sum() + 1e-10)
    return weights


def compute_rolling_cov_matrix(returns, window=36):
    """
    Compute rolling covariance matrix for factor returns.
    """
    covs = []
    for i in range(len(returns)):
        if i < window:
            window_data = returns.iloc[:i+1]
        else:
            window_data = returns.iloc[i-window+1:i+1]
        covs.append(window_data.cov().values)
    return np.array(covs)


def estimate_residual_variance(returns, B, factor_returns, window=36):
    """
    Estimate residual variance for each asset using rolling regression residuals.
    """
    N = returns.shape[1]
    K = factor_returns.shape[1]

    resid_vars = []

    for i in range(len(returns)):
        if i < window:
            Y = returns.iloc[:i+1].values
            X = factor_returns.iloc[:i+1].values
        else:
            Y = returns.iloc[i-window+1:i+1].values
            X = factor_returns.iloc[i-window+1:i+1].values

        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            coef, _, _, _ = np.linalg.lstsq(X_with_const, Y, rcond=None)
            residuals = Y - X_with_const @ coef
            resid_var = np.var(residuals, axis=0) + 1e-6
        except Exception:
            resid_var = 0.01 * np.ones(N)

        resid_vars.append(resid_var)

    return np.array(resid_vars)


if __name__ == "__main__":
    print("Optimization module loaded.")
