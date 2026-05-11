import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..backtest import BacktestEngine
import warnings
warnings.filterwarnings('ignore')


class RegularStockStrategy:
    def __init__(self, initial_capital=10000000, commission_rate=0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.backtest_engine = BacktestEngine(initial_capital, commission_rate)
        self.results = {}

    def optimize_parameters(self, survey_df, price_df,
                            lookback_days_list=[10, 20, 40, 60, 120],
                            rebalance_freq_list=['weekly', 'monthly', 'quarterly', 'semi-annual'],
                            num_stocks_list=[20, 50, 80, 100, 200]):
        if survey_df is None or price_df is None:
            return None
        survey_df = survey_df.copy()
        price_df = price_df.copy()
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        survey_df['trade_date'] = pd.to_datetime(survey_df['trade_date'])
        survey_df = survey_df.sort_values(['ts_code', 'trade_date'])
        results = []
        for lookback in lookback_days_list:
            survey_df[f'survey_count_{lookback}d'] = survey_df.groupby('ts_code')['institutions_count'].transform(
                lambda x: x.rolling(window=lookback, min_periods=1).sum()
            )
            for freq in rebalance_freq_list:
                for num in num_stocks_list:
                    backtest_results = self.backtest_engine.run_regular_stock_backtest(
                        survey_df, price_df,
                        lookback_days=lookback,
                        rebalance_freq=freq,
                        num_stocks=num
                    )
                    if backtest_results is not None:
                        results.append({
                            'lookback_days': lookback,
                            'rebalance_freq': freq,
                            'num_stocks': num,
                            'annual_return': backtest_results.get('annual_return', 0),
                            'annual_excess_return': backtest_results.get('annualized_excess_return', 0),
                            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                            'information_ratio': backtest_results.get('information_ratio', 0),
                            'max_drawdown': backtest_results.get('max_drawdown', 0),
                            'win_rate': backtest_results.get('win_rate', 0)
                        })
        return pd.DataFrame(results)

    def run_typical_strategy(self, survey_df, price_df, num_stocks=20):
        return self.backtest_engine.run_regular_stock_backtest(
            survey_df, price_df,
            lookback_days=120,
            rebalance_freq='weekly',
            num_stocks=num_stocks
        )

    def run_strategy_variants(self, survey_df, price_df, num_stocks_list=[20, 50, 80, 100]):
        results = {}
        for num in num_stocks_list:
            results[f'stocks_{num}'] = self.backtest_engine.run_regular_stock_backtest(
                survey_df, price_df,
                lookback_days=120,
                rebalance_freq='weekly',
                num_stocks=num
            )
        return results

    def plot_num_stocks_comparison(self, results_dict, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, results in results_dict.items():
            if results is not None and 'portfolio_value' in results:
                portfolio_value = np.array(results['portfolio_value'])
                cumulative_returns = portfolio_value / portfolio_value[0]
                axes[0].plot(cumulative_returns, label=name)
        axes[0].set_title('Cumulative Returns by Number of Stocks')
        axes[0].set_xlabel('Trading Days')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        metrics_data = []
        for name, results in results_dict.items():
            if results is not None:
                metrics_data.append({
                    'Strategy': name,
                    'Annual Return': results.get('annual_return', 0),
                    'Sharpe Ratio': results.get('sharpe_ratio', 0),
                    'Information Ratio': results.get('information_ratio', 0),
                    'Max Drawdown': results.get('max_drawdown', 0)
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

    def generate_summary(self, strategy_results):
        if strategy_results is None:
            return None
        summary = {
            'strategy_type': 'Regular Stock Selection',
            'annual_return': strategy_results.get('annual_return', 0),
            'annual_volatility': strategy_results.get('annual_volatility', 0),
            'sharpe_ratio': strategy_results.get('sharpe_ratio', 0),
            'max_drawdown': strategy_results.get('max_drawdown', 0),
            'annual_excess_return': strategy_results.get('annualized_excess_return', 0),
            'information_ratio': strategy_results.get('information_ratio', 0),
            'win_rate': strategy_results.get('win_rate', 0),
            'profit_loss_ratio': strategy_results.get('profit_loss_ratio', 0)
        }
        return summary