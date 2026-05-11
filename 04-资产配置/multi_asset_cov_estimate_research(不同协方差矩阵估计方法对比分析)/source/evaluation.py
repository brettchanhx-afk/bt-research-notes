import numpy as np
import pandas as pd
from scipy import optimize


class PortfolioEvaluator:
    def __init__(self, risk_free_rate=0.0):
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, weights, asset_returns):
        portfolio_returns = asset_returns @ weights
        return portfolio_returns

    def annualized_return(self, returns, periods_per_year=252):
        if len(returns) == 0:
            return 0.0
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / periods_per_year
        if n_years <= 0:
            return 0.0
        annualized = (1 + total_return) ** (1 / n_years) - 1
        return annualized

    def annualized_volatility(self, returns, periods_per_year=252):
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(periods_per_year)

    def sharpe_ratio(self, returns, periods_per_year=252):
        if len(returns) == 0:
            return 0.0
        ann_return = self.annualized_return(returns, periods_per_year)
        ann_vol = self.annualized_volatility(returns, periods_per_year)
        if ann_vol == 0:
            return 0.0
        return (ann_return - self.risk_free_rate) / ann_vol

    def max_drawdown(self, returns):
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def calmar_ratio(self, returns, periods_per_year=252):
        if len(returns) == 0:
            return 0.0
        ann_return = self.annualized_return(returns, periods_per_year)
        max_dd = abs(self.max_drawdown(returns))
        if max_dd == 0:
            return 0.0
        return ann_return / max_dd

    def sortino_ratio(self, returns, periods_per_year=252):
        if len(returns) == 0:
            return 0.0
        ann_return = self.annualized_return(returns, periods_per_year)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)
        if downside_std == 0:
            return 0.0
        return (ann_return - self.risk_free_rate) / downside_std

    def win_rate(self, returns):
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)

    def calmar_ratio_simple(self, returns, periods_per_year=252):
        ann_return = self.annualized_return(returns, periods_per_year)
        max_dd = abs(self.max_drawdown(returns))
        if max_dd == 0:
            return 0.0
        return ann_return / max_dd

    def evaluate_portfolio(self, weights, asset_returns, periods_per_year=252):
        portfolio_returns = self.calculate_returns(weights, asset_returns)

        metrics = {
            'total_return': (1 + portfolio_returns).prod() - 1,
            'annualized_return': self.annualized_return(portfolio_returns, periods_per_year),
            'annualized_volatility': self.annualized_volatility(portfolio_returns, periods_per_year),
            'sharpe_ratio': self.sharpe_ratio(portfolio_returns, periods_per_year),
            'max_drawdown': self.max_drawdown(portfolio_returns),
            'calmar_ratio': self.calmar_ratio_simple(portfolio_returns, periods_per_year),
            'sortino_ratio': self.sortino_ratio(portfolio_returns, periods_per_year),
            'win_rate': self.win_rate(portfolio_returns),
            'portfolio_returns': portfolio_returns
        }
        return metrics

    def calculate_covariance_rmse(self, estimated_cov, actual_returns, lookback=None):
        if lookback is not None:
            actual_cov = actual_returns.tail(lookback).cov().values
        else:
            actual_cov = actual_returns.cov().values

        n = estimated_cov.shape[0]
        if actual_cov.shape[0] != n:
            return np.nan

        diff = estimated_cov - actual_cov
        rmse = np.sqrt(np.sum(diff ** 2) / (n * n))
        return rmse

    def calculate_tracking_error(self, portfolio_returns, benchmark_returns):
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have the same length")

        diff = portfolio_returns - benchmark_returns
        tracking_error = np.std(diff) * np.sqrt(252)
        return tracking_error

    def calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have the same length")

        diff = portfolio_returns - benchmark_returns
        excess_return = np.mean(diff) * 252
        tracking_error = self.calculate_tracking_error(portfolio_returns, benchmark_returns)

        if tracking_error == 0:
            return 0.0
        return excess_return / tracking_error

    def calculate_var(self, returns, confidence_level=0.95):
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_cvar(self, returns, confidence_level=0.95):
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])