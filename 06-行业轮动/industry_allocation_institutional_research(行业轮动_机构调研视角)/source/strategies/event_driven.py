import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..backtest import BacktestEngine
import warnings
warnings.filterwarnings('ignore')


class EventDrivenStrategy:
    def __init__(self, initial_capital=10000000, commission_rate=0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.backtest_engine = BacktestEngine(initial_capital, commission_rate)
        self.results = {}

    def optimize_parameters(self, survey_df, price_df,
                            lookback_days_list=[1, 5, 10, 20, 40, 60],
                            holding_days_list=[20, 40, 60, 100, 200],
                            threshold_list=[20, 50, 80, 100, 200, 300]):
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
            for holding in holding_days_list:
                for threshold in threshold_list:
                    survey_df['signal'] = (survey_df[f'survey_count_{lookback}d'] >= threshold).astype(int)
                    backtest_results = self.backtest_engine.run_event_driven_backtest(
                        survey_df, price_df,
                        lookback_days=lookback,
                        holding_days=holding,
                        threshold=threshold
                    )
                    if backtest_results is not None:
                        results.append({
                            'lookback_days': lookback,
                            'holding_days': holding,
                            'threshold': threshold,
                            'annual_return': backtest_results.get('annual_return', 0),
                            'annual_excess_return': backtest_results.get('annualized_excess_return', 0),
                            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                            'information_ratio': backtest_results.get('information_ratio', 0),
                            'max_drawdown': backtest_results.get('max_drawdown', 0),
                            'win_rate': backtest_results.get('win_rate', 0)
                        })
        return pd.DataFrame(results)

    def run_strategy1(self, survey_df, price_df):
        return self.backtest_engine.run_event_driven_backtest(
            survey_df, price_df,
            lookback_days=1,
            holding_days=200,
            threshold=50
        )

    def run_strategy2(self, survey_df, price_df):
        return self.backtest_engine.run_event_driven_backtest(
            survey_df, price_df,
            lookback_days=60,
            holding_days=100,
            threshold=50
        )

    def run_with_filter(self, survey_df, price_df, return_filter=True, capital_filter=True):
        survey_df = survey_df.copy()
        price_df = price_df.copy()
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        price_df['return_5d'] = price_df.groupby('ts_code')['close'].pct_change(5)
        survey_df['trade_date'] = pd.to_datetime(survey_df['trade_date'])
        survey_df = survey_df.sort_values(['ts_code', 'trade_date'])
        survey_df = survey_df.merge(
            price_df[['ts_code', 'trade_date', 'return_5d']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        if return_filter:
            survey_df['signal'] = ((survey_df['institutions_count'] >= 50) &
                                    (survey_df['return_5d'] > 0)).astype(int)
        else:
            survey_df['signal'] = (survey_df['institutions_count'] >= 50).astype(int)
        return self.backtest_engine.run_event_driven_backtest(
            survey_df[survey_df['signal'] == 1], price_df,
            lookback_days=60,
            holding_days=100,
            threshold=50
        )

    def plot_parameter_heatmap(self, param_results, metric='information_ratio', save_path=None):
        if param_results is None or len(param_results) == 0:
            return
        pivot_data = param_results.pivot_table(
            values=metric,
            index='lookback_days',
            columns='holding_days',
            aggfunc='mean'
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels(pivot_data.index)
        ax.set_xlabel('Holding Days')
        ax.set_ylabel('Lookback Days')
        ax.set_title(f'{metric} by Parameters')
        plt.colorbar(im, ax=ax)
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = ax.text(j, i, f'{pivot_data.values[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_summary(self, strategy_results):
        if strategy_results is None:
            return None
        summary = {
            'strategy_type': 'Event-Driven',
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