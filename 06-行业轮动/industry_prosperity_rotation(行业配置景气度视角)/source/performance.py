import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    def __init__(self, risk_free_rate=0.03):
        self.risk_free_rate = risk_free_rate
        self.metrics = {}

    def calculate_all_metrics(self, portfolio_values, returns, benchmark_returns=None):
        if portfolio_values is None or len(portfolio_values) == 0:
            return {}

        if isinstance(portfolio_values, list):
            portfolio_values = pd.Series(portfolio_values)

        returns_series = returns if isinstance(returns, pd.Series) else pd.Series(returns) if returns is not None else pd.Series()

        metrics = {}

        metrics['total_return'] = self.calculate_total_return(portfolio_values)
        metrics['annual_return'] = self.calculate_annual_return(portfolio_values, len(returns_series) if len(returns_series) > 0 else 12)
        metrics['annual_volatility'] = self.calculate_annual_volatility(returns_series)
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns_series)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(portfolio_values, returns_series)
        metrics['max_drawdown'] = self.calculate_max_drawdown(portfolio_values)
        metrics['max_drawdown_duration'] = self.calculate_max_drawdown_duration(portfolio_values)
        metrics['win_rate'] = self.calculate_win_rate(returns_series)
        metrics['profit_loss_ratio'] = self.calculate_profit_loss_ratio(returns_series)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns_series)
        metrics['information_ratio'] = self.calculate_information_ratio(returns_series, benchmark_returns) if benchmark_returns is not None else 0

        if benchmark_returns is not None:
            benchmark_series = benchmark_returns if isinstance(benchmark_returns, pd.Series) else pd.Series(benchmark_returns)
            metrics['beta'] = self.calculate_beta(returns_series, benchmark_series)
            metrics['alpha'] = self.calculate_alpha(returns_series, benchmark_series)

        self.metrics = metrics
        return metrics

    def calculate_total_return(self, portfolio_values):
        if len(portfolio_values) < 2:
            return 0

        start_value = portfolio_values.iloc[0]
        end_value = portfolio_values.iloc[-1]

        if start_value == 0:
            return 0

        return ((end_value / start_value) - 1) * 100

    def calculate_annual_return(self, portfolio_values, n_periods=12):
        if len(portfolio_values) < 2 or n_periods <= 0:
            return 0

        total_return = self.calculate_total_return(portfolio_values) / 100

        n_years = max(len(portfolio_values) / n_periods, 1)

        annual_return = (1 + total_return) ** (1 / n_years) - 1

        return annual_return * 100

    def calculate_annual_volatility(self, returns):
        if len(returns) == 0:
            return 0

        return returns.std() * np.sqrt(12) * 100

    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        if len(returns) == 0:
            return 0

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        annual_return = self.calculate_annual_return(pd.Series([1, 1 + returns.mean() * 12]), 12)
        annual_vol = self.calculate_annual_volatility(returns)

        if annual_vol == 0:
            return 0

        sharpe = (annual_return / 100 - risk_free_rate) / (annual_vol / 100)

        return sharpe

    def calculate_sortino_ratio(self, returns, risk_free_rate=None):
        if len(returns) == 0:
            return 0

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        annual_return = self.calculate_annual_return(pd.Series([1, 1 + returns.mean() * 12]), 12)

        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return 0

        downside_vol = downside_returns.std() * np.sqrt(12)

        if downside_vol == 0:
            return 0

        sortino = (annual_return / 100 - risk_free_rate) / downside_vol

        return sortino

    def calculate_max_drawdown(self, portfolio_values):
        if len(portfolio_values) == 0:
            return 0

        if isinstance(portfolio_values, list):
            portfolio_values = pd.Series(portfolio_values)

        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax

        max_dd = drawdown.min() * 100

        return max_dd

    def calculate_max_drawdown_duration(self, portfolio_values):
        if len(portfolio_values) == 0:
            return 0

        if isinstance(portfolio_values, list):
            portfolio_values = pd.Series(portfolio_values)

        cummax = portfolio_values.cummax()
        drawdown = portfolio_values - cummax

        in_drawdown = False
        max_duration = 0
        current_duration = 0

        for i in range(len(portfolio_values)):
            if drawdown.iloc[i] < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0

        if in_drawdown:
            max_duration = max(max_duration, current_duration)

        return max_duration

    def calculate_calmar_ratio(self, portfolio_values, returns):
        if len(portfolio_values) == 0 or len(returns) == 0:
            return 0

        annual_return = self.calculate_annual_return(portfolio_values, len(returns))
        max_dd = abs(self.calculate_max_drawdown(portfolio_values))

        if max_dd == 0:
            return 0

        calmar = annual_return / max_dd

        return calmar

    def calculate_win_rate(self, returns):
        if len(returns) == 0:
            return 0

        win_rate = (returns > 0).sum() / len(returns) * 100

        return win_rate

    def calculate_profit_loss_ratio(self, returns):
        if len(returns) == 0:
            return 0

        gains = returns[returns > 0].mean()
        losses = abs(returns[returns < 0].mean())

        if losses == 0:
            return 0

        return gains / losses

    def calculate_beta(self, returns, benchmark_returns):
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0

        min_len = min(len(returns), len(benchmark_returns))

        returns_adjusted = returns.iloc[-min_len:]
        benchmark_adjusted = benchmark_returns.iloc[-min_len:]

        covariance = returns_adjusted.cov(benchmark_adjusted)
        benchmark_variance = benchmark_adjusted.var()

        if benchmark_variance == 0:
            return 0

        beta = covariance / benchmark_variance

        return beta

    def calculate_alpha(self, returns, benchmark_returns):
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0

        beta = self.calculate_beta(returns, benchmark_returns)

        portfolio_return = self.calculate_annual_return(pd.Series([1, 1 + returns.mean() * 12]), 12)
        benchmark_return = self.calculate_annual_return(pd.Series([1, 1 + benchmark_returns.mean() * 12]), 12)

        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))

        return alpha

    def calculate_information_ratio(self, returns, benchmark_returns):
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0

        min_len = min(len(returns), len(benchmark_returns))

        returns_adjusted = returns.iloc[-min_len:]
        benchmark_adjusted = benchmark_returns.iloc[-min_len:]

        excess_returns = returns_adjusted - benchmark_adjusted

        tracking_error = excess_returns.std()

        if tracking_error == 0:
            return 0

        ir = excess_returns.mean() / tracking_error

        return ir * np.sqrt(12)

    def generate_performance_report(self, portfolio_values, returns, benchmark_returns=None):
        metrics = self.calculate_all_metrics(portfolio_values, returns, benchmark_returns)

        report = """
        ========================================
        Performance Report
        ========================================

        Total Return:             {:.2f}%
        Annual Return:            {:.2f}%
        Annual Volatility:        {:.2f}%
        Sharpe Ratio:             {:.2f}
        Sortino Ratio:            {:.2f}
        Calmar Ratio:             {:.2f}
        Max Drawdown:             {:.2f}%
        Max Drawdown Duration:    {} periods
        Win Rate:                 {:.2f}%
        Profit/Loss Ratio:        {:.2f}
        """.format(
            metrics.get('total_return', 0),
            metrics.get('annual_return', 0),
            metrics.get('annual_volatility', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('sortino_ratio', 0),
            metrics.get('calmar_ratio', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('max_drawdown_duration', 0),
            metrics.get('win_rate', 0),
            metrics.get('profit_loss_ratio', 0)
        )

        if 'alpha' in metrics and 'beta' in metrics:
            report += """
        Alpha:                    {:.2f}%
        Beta:                     {:.2f}
        Information Ratio:        {:.2f}
            """.format(
                metrics.get('alpha', 0),
                metrics.get('beta', 0),
                metrics.get('information_ratio', 0)
            )

        return report


class ReturnAttribution:
    def __init__(self):
        pass

    def attribute_returns(self, portfolio_returns, factor_returns):
        if portfolio_returns is None or factor_returns is None:
            return {}

        attribution = {
            'total_return': portfolio_returns.sum(),
            'factor_contribution': factor_returns.sum() if len(factor_returns) > 0 else 0,
            'selection_contribution': 0,
            'timing_contribution': 0
        }

        attribution['selection_contribution'] = attribution['total_return'] - attribution['factor_contribution']

        return attribution


if __name__ == "__main__":
    print("Testing PerformanceAnalyzer...")

    analyzer = PerformanceAnalyzer(risk_free_rate=0.03)

    sample_values = pd.Series([1000000, 1050000, 1020000, 1100000, 1080000, 1150000])
    sample_returns = pd.Series([0.05, -0.03, 0.08, -0.02, 0.06])

    metrics = analyzer.calculate_all_metrics(sample_values, sample_returns)

    print("Performance metrics calculated:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nPerformance module test completed!")
