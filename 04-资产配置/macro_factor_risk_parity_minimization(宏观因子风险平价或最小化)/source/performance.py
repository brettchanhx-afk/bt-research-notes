"""
Performance metrics module for strategy evaluation.
Computes annualized return, volatility, max drawdown, Sharpe ratio, etc.
"""
import numpy as np
import pandas as pd


def compute_cumulative_returns(returns):
    """
    Compute cumulative returns from period returns.
    """
    return (1 + returns).cumprod()


def compute_annualized_return(returns, periods_per_year=12):
    """
    Compute annualized return.
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    return ann_return


def compute_annualized_volatility(returns, periods_per_year=12):
    """
    Compute annualized volatility.
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def compute_max_drawdown(cumulative_returns):
    """
    Compute maximum drawdown from cumulative returns.
    """
    if isinstance(cumulative_returns, pd.Series):
        cumulative_returns = cumulative_returns.values
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=12):
    """
    Compute Sharpe ratio.
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = excess_returns.mean() * periods_per_year
    ann_vol = excess_returns.std() * np.sqrt(periods_per_year)
    if ann_vol < 1e-10:
        return 0.0
    return ann_return / ann_vol


def compute_calmar_ratio(returns, periods_per_year=12):
    """
    Compute Calmar ratio (annualized return / max drawdown).
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    if isinstance(returns, pd.Series):
        cum_ret = compute_cumulative_returns(returns)
    else:
        cum_ret = compute_cumulative_returns(pd.Series(returns))
    max_dd = abs(compute_max_drawdown(cum_ret))
    if max_dd < 1e-10:
        return 0.0
    return ann_return / max_dd


def compute_performance_metrics(returns, strategy_name="Strategy",
                                 risk_free_rate=0.0, periods_per_year=12):
    """
    Compute comprehensive performance metrics for a strategy.

    Parameters:
    -----------
    returns : pd.Series
        Periodic returns
    strategy_name : str
        Name of the strategy
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year (12 for monthly)

    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    returns = returns.dropna()
    cumulative = compute_cumulative_returns(returns)
    max_dd = compute_max_drawdown(cumulative)

    ann_return = compute_annualized_return(returns, periods_per_year)
    ann_vol = compute_annualized_volatility(returns, periods_per_year)
    sharpe = compute_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    calmar = compute_calmar_ratio(returns, periods_per_year)

    annual_return_pct = ann_return * 100
    annual_vol_pct = ann_vol * 100
    max_dd_pct = max_dd * 100

    metrics = {
        "strategy": strategy_name,
        "annualized_return": annual_return_pct,
        "annualized_volatility": annual_vol_pct,
        "max_drawdown": max_dd_pct,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "total_return": (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0,
        "n_periods": len(returns),
    }
    return metrics


def compute_turnover(weights_history):
    """
    Compute average monthly turnover from weight history.
    """
    if len(weights_history) < 2:
        return 0.0
    turnovers = []
    for t in range(1, len(weights_history)):
        w_prev = weights_history.iloc[t-1].values
        w_curr = weights_history.iloc[t].values
        turnover = np.sum(np.abs(w_curr - w_prev))
        turnovers.append(turnover)
    return np.mean(turnovers)


def print_performance_summary(metrics_dict):
    """
    Print a formatted performance summary.
    """
    print("\n" + "=" * 60)
    print(f"  Performance Summary: {metrics_dict['strategy']}")
    print("=" * 60)
    print(f"  {'Annualized Return:':<25} {metrics_dict['annualized_return']:.2f}%")
    print(f"  {'Annualized Volatility:':<25} {metrics_dict['annualized_volatility']:.2f}%")
    print(f"  {'Maximum Drawdown:':<25} {metrics_dict['max_drawdown']:.2f}%")
    print(f"  {'Sharpe Ratio:':<25} {metrics_dict['sharpe_ratio']:.4f}")
    print(f"  {'Calmar Ratio:':<25} {metrics_dict['calmar_ratio']:.4f}")
    print(f"  {'Total Return:':<25} {metrics_dict['total_return']:.2f}%")
    print(f"  {'N Periods:':<25} {metrics_dict['n_periods']}")
    print("=" * 60)


def compare_strategies(metrics_list):
    """
    Print a comparison table for multiple strategies.
    """
    print("\n" + "=" * 80)
    print(f"  Strategy Comparison")
    print("=" * 80)
    header = f"  {'Strategy':<25} {'Ann.Ret%':>10} {'Ann.Vol%':>10} {'MaxDD%':>10} {'Sharpe':>10}"
    print(header)
    print("  " + "-" * 75)
    for m in metrics_list:
        line = f"  {m['strategy']:<25} {m['annualized_return']:>10.2f} {m['annualized_volatility']:>10.2f} {m['max_drawdown']:>10.2f} {m['sharpe_ratio']:>10.4f}"
        print(line)
    print("=" * 80)


if __name__ == "__main__":
    print("Performance module loaded.")
