import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.results = {}
        self.positions_history = []

    def run_rolling_backtest(
        self,
        returns,
        cov_estimator,
        portfolio_builder,
        method='sample_cov',
        lookback_period=252,
        rebalance_freq='monthly',
        allow_short=False,
        portfolio_type='min_variance',
        target_volatility=None,
        **cov_kwargs
    ):
        if len(returns) < lookback_period + 30:
            print(f"Insufficient data: {len(returns)} rows, need at least {lookback_period + 30}")
            return None

        if rebalance_freq == 'monthly':
            rebalance_dates = self._get_monthly_dates(returns, lookback_period)
        elif rebalance_freq == 'weekly':
            rebalance_dates = self._get_weekly_dates(returns, lookback_period)
        else:
            rebalance_dates = self._get_monthly_dates(returns, lookback_period)

        portfolio_values = []
        weights_history = []
        dates = []

        current_capital = self.initial_capital

        for i, date in enumerate(rebalance_dates):
            date_idx = returns.index.get_loc(date) if date in returns.index else None
            if date_idx is None:
                continue

            historical_data = returns.iloc[:date_idx]
            if len(historical_data) < lookback_period:
                continue

            lookback_data = historical_data.tail(lookback_period)

            try:
                cov_matrix = cov_estimator.get_covariance(
                    lookback_data,
                    method=method,
                    lookback=None,
                    **cov_kwargs
                )

                cov_matrix = cov_estimator.ensure_positive_definite(cov_matrix)

                if portfolio_type == 'min_variance':
                    weights = portfolio_builder.minimum_variance_portfolio(
                        cov_matrix,
                        allow_short=allow_short
                    )
                elif portfolio_type == 'target_volatility' and target_volatility is not None:
                    weights = portfolio_builder.target_volatility_portfolio(
                        cov_matrix,
                        target_volatility=target_volatility,
                        allow_short=allow_short
                    )
                elif portfolio_type == 'risk_parity':
                    weights = portfolio_builder.risk_parity_portfolio(
                        cov_matrix,
                        allow_short=allow_short
                    )
                else:
                    weights = portfolio_builder.minimum_variance_portfolio(
                        cov_matrix,
                        allow_short=allow_short
                    )

            except Exception as e:
                print(f"Error computing weights at {date}: {e}")
                weights = np.ones(returns.shape[1]) / returns.shape[1]

            weights_history.append(weights)
            dates.append(date)

            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                if next_date in returns.index:
                    next_idx = returns.index.get_loc(next_date)
                    period_returns = returns.iloc[date_idx:next_idx]
                else:
                    period_returns = returns.iloc[date_idx:date_idx + 22]
            else:
                period_returns = returns.iloc[date_idx:date_idx + 22]

            if len(period_returns) > 0:
                daily_portfolio_return = period_returns @ weights
                period_return = (1 + daily_portfolio_return).prod() - 1
                current_capital = current_capital * (1 + period_return)
                portfolio_values.append({
                    'date': date,
                    'portfolio_value': current_capital,
                    'weights': weights.copy(),
                    'period_return': period_return,
                    'daily_returns': daily_portfolio_return
                })

        if len(portfolio_values) == 0:
            return None

        result_df = pd.DataFrame(portfolio_values)
        result_df.set_index('date', inplace=True)

        self.results[method] = {
            'portfolio_values': result_df,
            'weights_history': weights_history,
            'dates': dates
        }

        return result_df

    def _get_monthly_dates(self, returns, lookback_period):
        monthly_idx = returns.resample('M').indices
        dates = sorted(list(monthly_idx.keys()))
        # 将交易日数量转换为大致的月份数
        start_idx = int(lookback_period / 22)
        if start_idx < 12:
            start_idx = 12  # 至少需要12个月的数据
        if start_idx >= len(dates):
            start_idx = len(dates) - 1
        return dates[start_idx:]

    def _get_weekly_dates(self, returns, lookback_period):
        weekly_idx = returns.resample('W').indices
        dates = sorted(list(weekly_idx.keys()))
        # 将交易日数量转换为大致的周数
        start_idx = int(lookback_period / 5)
        if start_idx < 52:
            start_idx = 52  # 至少需要52周的数据
        if start_idx >= len(dates):
            start_idx = len(dates) - 1
        return dates[start_idx:]

    def run_multi_method_backtest(
        self,
        returns,
        cov_estimator,
        portfolio_builder,
        methods,
        lookback_period=252,
        rebalance_freq='monthly',
        allow_short=False,
        portfolio_type='min_variance',
        target_volatility=None,
        **cov_kwargs
    ):
        results = {}

        for method in methods:
            print(f"Running backtest for method: {method}")
            result = self.run_rolling_backtest(
                returns=returns,
                cov_estimator=cov_estimator,
                portfolio_builder=portfolio_builder,
                method=method,
                lookback_period=lookback_period,
                rebalance_freq=rebalance_freq,
                allow_short=allow_short,
                portfolio_type=portfolio_type,
                target_volatility=target_volatility,** cov_kwargs
            )
            results[method] = result

        return results

    def get_performance_summary(self, method):
        if method not in self.results or self.results[method] is None:
            return None

        result_df = self.results[method]

        if len(result_df) == 0:
            return None

        daily_returns = []
        for pf in result_df['daily_returns']:
            if isinstance(pf, pd.Series):
                daily_returns.extend(pf.values.flatten().tolist())
            else:
                daily_returns.extend(np.array(pf).flatten().tolist())

        daily_returns = np.array(daily_returns)
        daily_returns = daily_returns[~np.isnan(daily_returns)]

        if len(daily_returns) == 0:
            return None

        total_return = (result_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) if len(result_df) > 0 else 0

        n_years = len(daily_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return) / annualized_volatility if annualized_volatility > 0 else 0

        cumulative = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        summary = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'portfolio_values': result_df['portfolio_value']
        }

        return summary

    def compare_methods(self):
        comparison = {}

        for method, result in self.results.items():
            if result is None:
                continue
            summary = self.get_performance_summary(method)
            if summary:
                comparison[method] = {
                    'annualized_return': summary['annualized_return'],
                    'annualized_volatility': summary['annualized_volatility'],
                    'sharpe_ratio': summary['sharpe_ratio'],
                    'max_drawdown': summary['max_drawdown'],
                    'calmar_ratio': summary['calmar_ratio']
                }

        return pd.DataFrame(comparison).T

    def get_results(self):
        return self.results