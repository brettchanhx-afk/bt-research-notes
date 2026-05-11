import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    def __init__(self, initial_capital=10000000, commission_rate=0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.results = {}

    def run_event_driven_backtest(self, signals_df, price_df, lookback_days=60, holding_days=100,
                                   threshold=50, exclude_st=False):
        if signals_df is None or price_df is None:
            return None
        signals_df = signals_df.copy()
        price_df = price_df.copy()
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        price_df['return'] = price_df.groupby('ts_code')['close'].pct_change()
        signals_df['trade_date'] = pd.to_datetime(signals_df['trade_date'])
        signals_df = signals_df.sort_values('trade_date')
        trading_dates = signals_df['trade_date'].unique()
        trading_dates = sorted(trading_dates)
        portfolio_value = [self.initial_capital]
        portfolio_returns = []
        current_positions = {}
        holdings_history = []
        daily_returns = []
        benchmark_returns = []
        for i, date in enumerate(trading_dates[:-holding_days]):
            if date not in price_df['trade_date'].values:
                continue
            day_signals = signals_df[
                (signals_df['trade_date'] == date) &
                (signals_df['institutions_count'] >= threshold)
            ]
            stocks_to_buy = day_signals['ts_code'].tolist()
            if len(stocks_to_buy) > 0:
                stocks_to_buy = [s for s in stocks_to_buy if s not in current_positions.keys()]
            for ts_code in list(current_positions.keys()):
                close_prices = price_df[
                    (price_df['ts_code'] == ts_code) &
                    (price_df['trade_date'] == date)
                ]['close']
                if len(close_prices) == 0:
                    hold_end_date = trading_dates[trading_dates.index(date) + holding_days] if i + holding_days < len(trading_dates) else trading_dates[-1]
                    if date >= hold_end_date:
                        if ts_code in current_positions:
                            del current_positions[ts_code]
                else:
                    if date >= trading_dates[min(i + holding_days, len(trading_dates) - 1)]:
                        if ts_code in current_positions:
                            del current_positions[ts_code]
            if len(stocks_to_buy) > 0:
                allocation_per_stock = self.initial_capital * 0.02
                for ts_code in stocks_to_buy:
                    if ts_code not in current_positions:
                        current_positions[ts_code] = {
                            'buy_date': date,
                            'shares': allocation_per_stock,
                            'entry_price': price_df[
                                (price_df['ts_code'] == ts_code) &
                                (price_df['trade_date'] == date)
                            ]['close'].values[0] if len(price_df[
                                (price_df['ts_code'] == ts_code) &
                                (price_df['trade_date'] == date)
                            ]) > 0 else 0
                        }
            holdings_history.append({
                'date': date,
                'positions': len(current_positions),
                'stocks': list(current_positions.keys())
            })
            day_return = 0
            for ts_code, position in current_positions.items():
                stock_returns = price_df[
                    (price_df['ts_code'] == ts_code) &
                    (price_df['trade_date'] == date)
                ]['return'].values
                if len(stock_returns) > 0:
                    day_return += stock_returns[0] / len(current_positions) if len(current_positions) > 0 else 0
            portfolio_returns.append(day_return)
            portfolio_value.append(portfolio_value[-1] * (1 + day_return))
            next_date_idx = min(i + 1, len(trading_dates) - 1)
            next_date = trading_dates[next_date_idx]
            benchmark_data = price_df[
                (price_df['trade_date'] == next_date)
            ]['return'].values
            benchmark_returns.append(benchmark_data[0] if len(benchmark_data) > 0 else 0)
        results = self._calculate_performance_metrics(portfolio_value, portfolio_returns,
                                                       benchmark_returns, trading_dates)
        results['portfolio_value'] = portfolio_value
        results['holdings_history'] = holdings_history
        return results

    def run_regular_stock_backtest(self, survey_df, price_df, lookback_days=120,
                                    rebalance_freq='monthly', num_stocks=20):
        if survey_df is None or price_df is None:
            return None
        survey_df = survey_df.copy()
        price_df = price_df.copy()
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        price_df['return'] = price_df.groupby('ts_code')['close'].pct_change()
        survey_df['trade_date'] = pd.to_datetime(survey_df['trade_date'])
        survey_df = survey_df.sort_values(['ts_code', 'trade_date'])
        if f'survey_count_{lookback_days}d' not in survey_df.columns:
            survey_df[f'survey_count_{lookback_days}d'] = survey_df.groupby('ts_code')['institutions_count'].transform(
                lambda x: x.rolling(window=lookback_days, min_periods=1).sum()
            )
        trading_dates = sorted(survey_df['trade_date'].unique())
        portfolio_value = [self.initial_capital]
        portfolio_returns = []
        current_holdings = []
        holdings_history = []
        benchmark_returns = []
        if rebalance_freq == 'monthly':
            rebalance_dates = [d for d in trading_dates if d.day <= 5]
        elif rebalance_freq == 'quarterly':
            rebalance_dates = [d for d in trading_dates if d.month in [1, 4, 7, 10] and d.day <= 5]
        elif rebalance_freq == 'weekly':
            rebalance_dates = trading_dates[::5]
        else:
            rebalance_dates = trading_dates
        for i, date in enumerate(trading_dates):
            if date not in price_df['trade_date'].values:
                continue
            if date in rebalance_dates:
                day_data = survey_df[survey_df['trade_date'] == date].copy()
                if len(day_data) > 0:
                    day_data = day_data.sort_values(f'survey_count_{lookback_days}d', ascending=False)
                    top_stocks = day_data['ts_code'].head(num_stocks).tolist()
                    current_holdings = top_stocks
            holdings_history.append({
                'date': date,
                'positions': len(current_holdings),
                'stocks': current_holdings.copy()
            })
            day_return = 0
            if len(current_holdings) > 0:
                for ts_code in current_holdings:
                    stock_return = price_df[
                        (price_df['ts_code'] == ts_code) &
                        (price_df['trade_date'] == date)
                    ]['return'].values
                    if len(stock_return) > 0:
                        day_return += stock_return[0] / len(current_holdings)
            portfolio_returns.append(day_return)
            portfolio_value.append(portfolio_value[-1] * (1 + day_return))
            if i + 1 < len(trading_dates):
                next_date = trading_dates[i + 1]
                benchmark_data = price_df[
                    (price_df['trade_date'] == next_date)
                ]['return'].values
                benchmark_returns.append(benchmark_data[0] if len(benchmark_data) > 0 else 0)
        results = self._calculate_performance_metrics(portfolio_value, portfolio_returns,
                                                       benchmark_returns, trading_dates)
        results['portfolio_value'] = portfolio_value
        results['holdings_history'] = holdings_history
        return results

    def run_industry_rotation_backtest(self, industry_survey_df, industry_price_df,
                                        smooth_window=250, num_industries=5,
                                        rebalance_freq='monthly'):
        if industry_survey_df is None or industry_price_df is None:
            return None
        industry_survey_df = industry_survey_df.copy()
        industry_price_df = industry_price_df.copy()
        industry_price_df['trade_date'] = pd.to_datetime(industry_price_df['trade_date'])
        industry_price_df = industry_price_df.sort_values(['industry', 'trade_date'])
        industry_price_df['return'] = industry_price_df.groupby('industry')['close'].pct_change()
        trading_dates = sorted(industry_price_df['trade_date'].unique())
        portfolio_value = [self.initial_capital]
        portfolio_returns = []
        current_industries = []
        holdings_history = []
        benchmark_returns = []
        if rebalance_freq == 'monthly':
            rebalance_dates = [d for d in trading_dates if d.day <= 5]
        elif rebalance_freq == 'quarterly':
            rebalance_dates = [d for d in trading_dates if d.month in [1, 4, 7, 10] and d.day <= 5]
        else:
            rebalance_dates = trading_dates
        for i, date in enumerate(trading_dates):
            if date not in industry_survey_df['survey_date'].values:
                continue
            if date in rebalance_dates:
                day_data = industry_survey_df[industry_survey_df['survey_date'] == date].copy()
                if len(day_data) > 0:
                    day_data = day_data.sort_values('z_score', ascending=False)
                    top_industries = day_data.head(num_industries)['industry'].tolist()
                    current_industries = top_industries
            holdings_history.append({
                'date': date,
                'positions': len(current_industries),
                'industries': current_industries.copy()
            })
            day_return = 0
            if len(current_industries) > 0:
                for industry in current_industries:
                    ind_return = industry_price_df[
                        (industry_price_df['industry'] == industry) &
                        (industry_price_df['trade_date'] == date)
                    ]['return'].values
                    if len(ind_return) > 0:
                        day_return += ind_return[0] / len(current_industries)
            portfolio_returns.append(day_return)
            portfolio_value.append(portfolio_value[-1] * (1 + day_return))
            if i + 1 < len(trading_dates):
                next_date = trading_dates[i + 1]
                all_return = industry_price_df[
                    (industry_price_df['trade_date'] == next_date)
                ]['return'].mean()
                benchmark_returns.append(all_return if not np.isnan(all_return) else 0)
        results = self._calculate_performance_metrics(portfolio_value, portfolio_returns,
                                                       benchmark_returns, trading_dates)
        results['portfolio_value'] = portfolio_value
        results['holdings_history'] = holdings_history
        return results

    def _calculate_performance_metrics(self, portfolio_value, portfolio_returns,
                                       benchmark_returns, trading_dates):
        if len(portfolio_value) < 2:
            return {}
        portfolio_returns = np.array(portfolio_returns)
        benchmark_returns = np.array(benchmark_returns)
        cumulative_returns = (np.array(portfolio_value) / portfolio_value[0]) - 1
        excess_returns = portfolio_returns - benchmark_returns
        annual_return = (portfolio_value[-1] / portfolio_value[0]) ** (252 / len(portfolio_returns)) - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        win_rate = (excess_returns > 0).mean()
        avg_win = excess_returns[excess_returns > 0].mean() if len(excess_returns[excess_returns > 0]) > 0 else 0
        avg_loss = abs(excess_returns[excess_returns < 0].mean()) if len(excess_returns[excess_returns < 0]) > 0 else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        annualized_excess_return = excess_returns.mean() * 252
        excess_volatility = excess_returns.std() * np.sqrt(252)
        information_ratio = annualized_excess_return / excess_volatility if excess_volatility > 0 else 0
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annualized_excess_return': annualized_excess_return,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_return': cumulative_returns[-1],
            'num_trading_days': len(portfolio_returns),
            'turnover': self._estimate_turnover(portfolio_returns)
        }

    def _calculate_max_drawdown(self, cumulative_returns):
        running_max = np.maximum.accumulate(cumulative_returns + 1)
        drawdown = (cumulative_returns + 1) / running_max - 1
        return drawdown.min()

    def _estimate_turnover(self, returns):
        if len(returns) < 2:
            return 0
        position_changes = np.abs(np.diff(returns))
        turnover = position_changes.mean() * 2
        return turnover * 252

    def plot_backtest_results(self, results, benchmark_returns=None, save_path=None):
        if results is None or 'portfolio_value' not in results:
            return
        portfolio_value = results['portfolio_value']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(portfolio_value)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Trading Days')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0, 0].legend()
        cumulative_returns = np.array(portfolio_value) / portfolio_value[0]
        axes[0, 1].plot(cumulative_returns - 1)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_xlabel('Trading Days')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        metrics_text = f"Annual Return: {results.get('annual_return', 0):.2%}\n"
        metrics_text += f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
        metrics_text += f"Max Drawdown: {results.get('max_drawdown', 0):.2%}\n"
        metrics_text += f"Information Ratio: {results.get('information_ratio', 0):.2f}\n"
        metrics_text += f"Win Rate: {results.get('win_rate', 0):.2%}\n"
        metrics_text += f"Profit/Loss Ratio: {results.get('profit_loss_ratio', 0):.2f}"
        axes[1, 0].text(0.1, 0.5, metrics_text, transform=axes[1, 0].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Performance Metrics')
        if 'holdings_history' in results and len(results['holdings_history']) > 0:
            holdings_df = pd.DataFrame(results['holdings_history'])
            axes[1, 1].plot(holdings_df['date'], holdings_df['positions'])
            axes[1, 1].set_title('Number of Holdings Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Number of Positions')
        else:
            axes[1, 1].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def compare_strategies(self, strategy_results_list, names_list, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, (results, name) in enumerate(zip(strategy_results_list, names_list)):
            if results is not None and 'portfolio_value' in results:
                portfolio_value = np.array(results['portfolio_value'])
                cumulative_returns = portfolio_value / portfolio_value[0]
                axes[0].plot(cumulative_returns, label=name)
        axes[0].set_title('Strategy Comparison - Cumulative Returns')
        axes[0].set_xlabel('Trading Days')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].axhline(y=1, color='r', linestyle='--')
        metrics_data = []
        for results, name in zip(strategy_results_list, names_list):
            if results is not None:
                metrics_data.append({
                    'Strategy': name,
                    'Annual Return': results.get('annual_return', 0),
                    'Sharpe Ratio': results.get('sharpe_ratio', 0),
                    'Max Drawdown': results.get('max_drawdown', 0),
                    'Information Ratio': results.get('information_ratio', 0)
                })
        if len(metrics_data) > 0:
            metrics_df = pd.DataFrame(metrics_data)
            axes[1].axis('off')
            table = axes[1].table(
                cellText=metrics_df.round(4).values,
                colLabels=metrics_df.columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_results(self, results, path):
        if results is None:
            return
        save_results = results.copy()
        if 'portfolio_value' in save_results:
            del save_results['portfolio_value']
        if 'holdings_history' in save_results:
            del save_results['holdings_history']
        pd.DataFrame([save_results]).to_csv(path, index=False)