"""
回测引擎模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    def __init__(self, initial_capital=1000000, transaction_cost=0.002,
                 risk_free_rate=0.04, target_volatility=None):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

        self.portfolio_value = []
        self.returns = []
        self.positions = []
        self.weights_history = []
        self.trades = []

    def run_asset_backtest(self, strategy, asset_returns_df, factor_dict,
                          asset_mapping, start_date=None, end_date=None,
                          rebalance_freq='M'):
        """
        大类资产配置回测
        """
        if asset_returns_df.empty:
            return {}

        if start_date:
            asset_returns_df = asset_returns_df[asset_returns_df.index >= start_date]
        if end_date:
            asset_returns_df = asset_returns_df[asset_returns_df.index <= end_date]

        if len(asset_returns_df) == 0:
            return {}

        dates = asset_returns_df.index.tolist()
        current_weights = None
        portfolio_values = [self.initial_capital]
        portfolio_returns = []
        weights_history = []

        for i, date in enumerate(dates):
            if i == 0:
                current_weights = {col: 1.0 / len(asset_returns_df.columns)
                                 for col in asset_returns_df.columns}
                weights_history.append((date, current_weights.copy()))
                continue

            returns_today = asset_returns_df.iloc[i]
            factor_today = {k: v.iloc[i] if len(v) > i else 0
                          for k, v in factor_dict.items()}

            factor_views = {}
            for factor_name, factor_series in factor_dict.items():
                if len(factor_series) > i:
                    if factor_series.iloc[i] > factor_series.iloc[i-1]:
                        factor_views[factor_name] = 1
                    else:
                        factor_views[factor_name] = -1

            should_rebalance = False
            if rebalance_freq == 'M':
                if i == len(dates) - 1 or dates[i].month != dates[i-1].month:
                    should_rebalance = True
            elif rebalance_freq == 'Q':
                if i == len(dates) - 1 or (dates[i].quarter != dates[i-1].quarter):
                    should_rebalance = True

            if should_rebalance:
                lookback_returns = asset_returns_df.iloc[max(0, i-120):i]
                if len(lookback_returns) >= 12:
                    new_weights = strategy.build_portfolio(
                        lookback_returns, factor_views, asset_mapping
                    )

                    if current_weights is not None:
                        turnover = sum(abs(new_weights.get(k, 0) - current_weights.get(k, 0))
                                     for k in set(new_weights.keys()) | set(current_weights.keys()))

                        if turnover > 0.01:
                            current_weights = new_weights

            weights_history.append((date, current_weights.copy()))

            portfolio_return = sum(
                current_weights.get(asset, 0) * returns_today.get(asset, 0)
                for asset in current_weights.keys()
            )

            portfolio_returns.append(portfolio_return)
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

        results = self.calculate_performance(portfolio_returns, portfolio_values, weights_history)

        return results

    def run_industry_backtest(self, strategy, industry_returns_df, factor_dict,
                             industry_mapping, benchmark_returns=None,
                             start_date=None, end_date=None,
                             rebalance_freq='M'):
        """
        行业轮动回测
        """
        if industry_returns_df.empty:
            return {}

        if start_date:
            industry_returns_df = industry_returns_df[industry_returns_df.index >= start_date]
            if benchmark_returns is not None:
                benchmark_returns = benchmark_returns[benchmark_returns.index >= start_date]
        if end_date:
            industry_returns_df = industry_returns_df[industry_returns_df.index <= end_date]
            if benchmark_returns is not None:
                benchmark_returns = benchmark_returns[benchmark_returns.index <= end_date]

        if len(industry_returns_df) == 0:
            return {}

        dates = industry_returns_df.index.tolist()
        current_weights = None
        portfolio_values = [self.initial_capital]
        portfolio_returns = []
        weights_history = []
        selected_history = []

        for i, date in enumerate(dates):
            if i == 0:
                all_industries = industry_returns_df.columns.tolist()
                current_weights = {ind: 1.0 / len(all_industries)
                                 for ind in all_industries}
                weights_history.append((date, current_weights.copy()))
                continue

            returns_today = industry_returns_df.iloc[i]

            factor_views = {}
            for factor_name, factor_series in factor_dict.items():
                if len(factor_series) > i:
                    if factor_series.iloc[i] > factor_series.iloc[i-1]:
                        factor_views[factor_name] = 1
                    else:
                        factor_views[factor_name] = -1

            should_rebalance = False
            if rebalance_freq == 'M':
                if i == len(dates) - 1 or dates[i].month != dates[i-1].month:
                    should_rebalance = True
            elif rebalance_freq == 'Q':
                if i == len(dates) - 1 or (dates[i].quarter != dates[i-1].quarter):
                    should_rebalance = True

            if should_rebalance:
                lookback_returns = industry_returns_df.iloc[max(0, i-60):i]
                if len(lookback_returns) >= 12:
                    selected = strategy.select_industries(
                        lookback_returns, factor_views, industry_mapping,
                        use_momentum=True
                    )

                    if selected:
                        current_weights = strategy.calculate_equal_weights(selected)
                        selected_history.append((date, selected))

            weights_history.append((date, current_weights.copy() if current_weights else {}))

            portfolio_return = sum(
                current_weights.get(ind, 0) * returns_today.get(ind, 0)
                for ind in current_weights.keys()
            ) if current_weights else 0

            portfolio_returns.append(portfolio_return)
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

        results = self.calculate_performance(portfolio_returns, portfolio_values, weights_history)

        if benchmark_returns is not None:
            benchmark_common = benchmark_returns.loc[benchmark_returns.index.isin(dates)]
            if len(benchmark_common) > 0:
                common_len = min(len(portfolio_returns), len(benchmark_common))
                strategy_returns = pd.Series(portfolio_returns[:common_len],
                                            index=benchmark_common.index[:common_len])
                tracking_error = (strategy_returns - benchmark_common.iloc[:common_len]).std() * np.sqrt(12)
                results['tracking_error'] = tracking_error
                results['information_ratio'] = (
                    (strategy_returns.mean() - benchmark_common.iloc[:common_len].mean()) * 12 / tracking_error
                    if tracking_error > 0 else 0
                )

        return results

    def calculate_performance(self, returns, portfolio_values, weights_history):
        """
        计算绩效指标
        """
        if not returns:
            return {}

        returns_series = pd.Series(returns)
        values_series = pd.Series(portfolio_values)

        cumulative_return = (1 + returns_series).prod() - 1
        annual_return = returns_series.mean() * 12
        annual_volatility = returns_series.std() * np.sqrt(12)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.cummax()
        drawdown = cum_returns / running_max - 1
        max_drawdown = drawdown.min()

        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        winning_periods = (returns_series > 0).sum()
        total_periods = len(returns_series)
        win_rate = winning_periods / total_periods if total_periods > 0 else 0

        avg_turnover = 0
        if len(weights_history) > 1:
            turnovers = []
            for i in range(1, len(weights_history)):
                w1 = weights_history[i-1][1]
                w2 = weights_history[i][1]
                turnover = sum(abs(w2.get(k, 0) - w1.get(k, 0))
                             for k in set(w1.keys()) | set(w2.keys()))
                turnovers.append(turnover / 2)
            avg_turnover = np.mean(turnovers) if turnovers else 0

        return {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'returns_series': returns_series,
            'portfolio_values': values_series
        }

    def plot_results(self, results, title="Strategy Performance", save_path=None):
        """
        绘制回测结果图
        """
        if 'returns_series' not in results or 'portfolio_values' not in results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        values = results['portfolio_values']
        axes[0, 0].plot(values.index if hasattr(values, 'index') else range(len(values)), values)
        axes[0, 0].set_title('Portfolio Value')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)

        returns = results['returns_series']
        cum_returns = (1 + returns).cumprod() - 1
        axes[0, 1].plot(cum_returns.index if hasattr(cum_returns, 'index') else range(len(cum_returns)), cum_returns)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].grid(True)

        cum_returns_vals = (1 + returns).cumprod()
        running_max = cum_returns_vals.cummax()
        drawdown = cum_returns_vals / running_max - 1
        axes[1, 0].plot(drawdown.index if hasattr(drawdown, 'index') else range(len(drawdown)), drawdown)
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True)

        monthly_returns = returns.groupby(returns.index.to_period('M') if hasattr(returns.index, 'to_period') else 'M').mean()
        axes[1, 1].bar(range(len(monthly_returns)), monthly_returns.values)
        axes[1, 1].set_title('Monthly Returns')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Return')
        axes[1, 1].grid(True)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def get_yearly_statistics(self, results):
        """
        获取年度统计
        """
        if 'returns_series' not in results:
            return pd.DataFrame()

        returns = results['returns_series']

        if hasattr(returns.index, 'year'):
            yearly = returns.groupby(returns.index.year).agg(['sum', 'std'])
            yearly.columns = ['annual_return', 'annual_vol']
            yearly['sharpe'] = (yearly['annual_return'] - self.risk_free_rate) / (yearly['annual_vol'] * np.sqrt(12))
        else:
            n_years = len(returns) // 12
            yearly_data = []
            for i in range(n_years):
                year_returns = returns.iloc[i*12:(i+1)*12]
                yearly_data.append({
                    'year': i + 1,
                    'annual_return': year_returns.sum(),
                    'annual_vol': year_returns.std() * np.sqrt(12)
                })
            yearly = pd.DataFrame(yearly_data).set_index('year')
            yearly['sharpe'] = (yearly['annual_return'] - self.risk_free_rate) / yearly['annual_vol']

        return yearly

    def compare_strategies(self, results_list, names):
        """
        比较多个策略
        """
        comparison = []

        for result, name in zip(results_list, names):
            comparison.append({
                'Strategy': name,
                'Cumulative Return': f"{result.get('cumulative_return', 0):.2%}",
                'Annual Return': f"{result.get('annual_return', 0):.2%}",
                'Annual Volatility': f"{result.get('annual_volatility', 0):.2%}",
                'Sharpe Ratio': f"{result.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown': f"{result.get('max_drawdown', 0):.2%}",
                'Win Rate': f"{result.get('win_rate', 0):.2%}"
            })

        return pd.DataFrame(comparison)
