"""
Risk attribution module.
Implements Boudt & Benedict (2013) factor risk contribution analysis.
"""
import numpy as np
import pandas as pd


def compute_mrc(gamma, theta):
    """
    Compute Marginal Risk Contributions (MRC) for each factor.

    Parameters:
    -----------
    gamma : np.array
        Combined factor weights (K + N assets), first K are factor exposures,
        last N are asset-specific exposures.
    theta : np.ndarray
        Joint covariance matrix of factor returns and idiosyncratic returns.
        Size should be (K+N) x (K+N).

    Returns:
    --------
    mrc : np.array
        Marginal risk contributions for each factor.
    """
    port_vol = np.sqrt(gamma @ theta @ gamma)
    if port_vol < 1e-10:
        return np.zeros(len(gamma))
    mrc = theta @ gamma / port_vol
    return mrc


def compute_frc(gamma, theta):
    """
    Compute Factor Risk Contributions (FRC) as a fraction of total risk.
    FRC_i = gamma_i * MRC_i / total_portfolio_vol

    Parameters:
    -----------
    gamma : np.array
        Factor weight vector.
    theta : np.ndarray
        Covariance matrix.
    """
    port_vol = np.sqrt(gamma @ theta @ gamma)
    if port_vol < 1e-10:
        return np.zeros(len(gamma))
    mrc = compute_mrc(gamma, theta)
    frc = gamma * mrc / port_vol
    return frc


def compute_portfolio_factor_risk_contribution(w, B, factor_cov, asset_idio_var):
    """
    Compute the factor risk contribution for a portfolio.

    Following Boudt & Benedict (2013):
    Portfolio return: w'R = w'a + w'BF + w'De = alpha + gamma'*(F;e)
    where gamma = [beta_G; delta] with beta_G = B'*w (portfolio factor loadings)
    and delta = w (portfolio idiosyncratic exposures for each asset)

    Parameters:
    -----------
    w : np.array
        Asset weights (N,)
    B : np.ndarray
        Factor loading matrix (N, K) for the current period
    factor_cov : np.ndarray
        Factor covariance matrix (K, K)
    asset_idio_var : np.array or float
        Idiosyncratic variance for each asset (N,) or scalar if same for all

    Returns:
    --------
    frc : np.array
        Factor risk contributions (K,) summing to total macro risk fraction
    idio_frc : float
        Idiosyncratic risk contribution
    """
    K = B.shape[1]
    N = len(w)

    beta_p = B.T @ w
    delta_p = w

    gamma_full = np.concatenate([beta_p, delta_p])

    if np.isscalar(asset_idio_var):
        idio_cov_diag = asset_idio_var * np.ones(N)
    else:
        idio_cov_diag = np.array(asset_idio_var)

    S = factor_cov
    I_diag = idio_cov_diag

    theta_block = np.zeros((K + N, K + N))
    theta_block[:K, :K] = S
    theta_block[K:, K:] = np.diag(I_diag)

    port_vol = np.sqrt(gamma_full @ theta_block @ gamma_full)
    if port_vol < 1e-10:
        return np.zeros(K), 0.0

    mrc = theta_block @ gamma_full / port_vol

    frc_macro = gamma_full[:K] * mrc[:K] / port_vol
    frc_idio = gamma_full[K:] * mrc[K:] / port_vol

    return frc_macro, np.sum(frc_idio)


def compute_all_period_factor_risk(w_history, B_history, factor_cov_history,
                                    residual_var=None):
    """
    Compute factor risk contributions across all periods.

    Parameters:
    -----------
    w_history : np.ndarray
        Asset weights over time (T, N)
    B_history : np.ndarray
        Factor loadings over time (T, N, K)
    factor_cov_history : np.ndarray
        Factor covariances over time (T, K, K)
    residual_var : np.array or None
        Idiosyncratic variance per asset (N,)

    Returns:
    --------
    frc_series : pd.DataFrame
        Time series of factor risk contributions (T, K)
    idio_series : np.array
        Time series of idiosyncratic risk contributions (T,)
    """
    T = len(w_history)
    K = B_history.shape[2] if len(B_history.shape) > 2 else 1

    frc_list = []
    idio_list = []

    if residual_var is None:
        residual_var = 0.01 * np.ones(w_history.shape[1])

    for t in range(T):
        w_t = w_history[t]
        B_t = B_history[t]
        cov_t = factor_cov_history[t]

        frc, idio_frc = compute_portfolio_factor_risk_contribution(
            w_t, B_t, cov_t, residual_var
        )
        frc_list.append(frc)
        idio_list.append(idio_frc)

    return np.array(frc_list), np.array(idio_list)


def build_theta_matrix(factor_cov, idio_var):
    """
    Build the full (K+N) x (K+N) covariance matrix for Boudt & Benedict formula.
    """
    K = factor_cov.shape[0]
    N = len(idio_var)
    theta = np.zeros((K + N, K + N))
    theta[:K, :K] = factor_cov
    theta[K:, K:] = np.diag(idio_var)
    return theta


def compute_mrc_frc_series(w_series, B_series, factor_returns, idio_var=None):
    """
    Compute time series of MRC and FRC for a portfolio.

    Parameters:
    -----------
    w_series : pd.DataFrame
        Portfolio weights over time (T x N)
    B_series : dict or np.ndarray
        Factor loadings per period
    factor_returns : pd.DataFrame
        Factor returns (T x K)
    idio_var : float or np.array
        Idiosyncratic variance

    Returns:
    --------
    mrc_df : pd.DataFrame (T x K)
    frc_df : pd.DataFrame (T x K)
    """
    dates = w_series.index.tolist()
    asset_names = w_series.columns.tolist()
    factor_names = factor_returns.columns.tolist()

    if isinstance(B_series, dict):
        has_betas = True
    else:
        has_betas = False

    if idio_var is None:
        idio_var = 0.01

    factor_cov = factor_returns.iloc[:36].cov().values
    cov_history = []
    for i in range(len(factor_returns)):
        if i < 36:
            window = factor_returns.iloc[:i+1]
        else:
            window = factor_returns.iloc[i-35:i+1]
        cov_history.append(window.cov().values)
    cov_history = np.array(cov_history)

    mrc_records = []
    frc_records = []

    for t, date in enumerate(dates):
        w = w_series.iloc[t].values
        factor_cov_t = cov_history[t] if t < len(cov_history) else cov_history[-1]

        if has_betas:
            B = np.zeros((len(asset_names), len(factor_names)))
            for i, aname in enumerate(asset_names):
                for j, fname in enumerate(factor_names):
                    key = (aname, fname)
                    if key in B_series:
                        B[i, j] = B_series[key][t]
                    else:
                        B[i, j] = 0.0
        else:
            B = B_series[t]

        frc, idio_frc = compute_portfolio_factor_risk_contribution(
            w, B, factor_cov_t, idio_var
        )
        frc_records.append(frc)

        mrc = np.zeros(len(factor_names))
        if frc.sum() > 0:
            gamma = np.concatenate([B.T @ w, w])
            theta = build_theta_matrix(factor_cov_t, idio_var * np.ones(len(w)))
            port_vol = np.sqrt(gamma @ theta @ gamma)
            if port_vol > 1e-10:
                mrc = theta @ gamma / port_vol
                mrc = mrc[:len(factor_names)]
        mrc_records.append(mrc)

    mrc_df = pd.DataFrame(mrc_records, index=dates, columns=factor_names)
    frc_df = pd.DataFrame(frc_records, index=dates, columns=factor_names)

    return mrc_df, frc_df


if __name__ == "__main__":
    print("Risk attribution module loaded.")
