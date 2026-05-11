import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..backtest import BacktestEngine
import warnings
warnings.filterwarnings('ignore')


class IndustryRotationStrategy:
    def __init__(self, initial_capital=10000000, commission_rate=0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.backtest_engine = BacktestEngine(initial_capital, commission_rate)
        self.results = {}

    def optimize_parameters(self, industry_survey_df, industry_price_df,
                            smooth_window_list=[120, 180, 250],
                            num_industries_list=[3, 5, 10],
                            rebalance_freq_list=['monthly', 'quarterly']):
        if industry_survey_df is None or industry_price_df is None:
            return None
        industry_survey_df = industry_survey_df.copy()
        industry_price_df = industry_price_df.copy()
        industry_price_df['trade_date'] = pd.to_datetime(industry_price_df['trade_date'])
        industry_survey_df['survey_date'] = pd.to_datetime(industry_survey_df['survey_date'])
        results = []
        for window in smooth_window_list:
            industry_survey_df['smoothed_survey'] = industry_survey_df.groupby('industry')['institutions_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            industry_survey_df['z_score'] = industry_survey_df.groupby('industry')['smoothed_survey'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            for num_ind in num_industries_list:
                for freq in rebalance_freq_list:
                    backtest_results = self.backtest_engine.run_industry_rotation_backtest(
                        industry_survey_df, industry_price_df,
                        smooth_window=window,
                        num_industries=num_ind,
                        rebalance_freq=freq
                    )
                    if backtest_results is not None:
                        results.append({
                            'smooth_window': window,
                            'num_industries': num_ind,
                            'rebalance_freq': freq,
                            'annual_return': backtest_results.get('annual_return', 0),
                            'annual_excess_return': backtest_results.get('annualized_excess_return', 0),
                            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                            'information_ratio': backtest_results.get('information_ratio', 0),
                            'max_drawdown': backtest_results.get('max_drawdown', 0),
                            'win_rate': backtest_results.get('win_rate', 0)
                        })
        return pd.DataFrame(results)

    def run_typical_strategy(self, industry_survey_df, industry_price_df, num_industries=5):
        return self.backtest_engine.run_industry_rotation_backtest(
            industry_survey_df, industry_price_df,
            smooth_window=250,
            num_industries=num_industries,
            rebalance_freq='monthly'
        )

    def run_timing_strategy(self, industry_survey_df, industry_price_df, z_threshold_high=1.0, z_threshold_low=0.0):
        industry_survey_df = industry_survey_df.copy()
        industry_price_df = industry_price_df.copy()
        industry_price_df['trade_date'] = pd.to_datetime(industry_price_df['trade_date'])
        industry_price_df = industry_price_df.sort_values(['industry', 'trade_date'])
        industry_price_df['return'] = industry_price_df.groupby('industry')['close'].pct_change()
        industry_survey_df['survey_date'] = pd.to_datetime(industry_survey_df['survey_date'])
        trading_dates = sorted(industry_price_df['trade_date'].unique())
        portfolio_value = [self.initial_capital]
        portfolio_returns = []
        current_position = 0.5
        holdings_history = []
        benchmark_returns = []
        rebalance_dates = [d for d in trading_dates if d.day <= 5]
        for i, date in enumerate(trading_dates):
            if date in rebalance_dates and date in industry_survey_df['survey_date'].values:
                day_data = industry_survey_df[industry_survey_df['survey_date'] == date].copy()
                if len(day_data) > 0:
                    avg_z = day_data['z_score'].mean()
                    if avg_z > z_threshold_high:
                        current_position = 1.0
                    elif avg_z > z_threshold_low:
                        current_position = 0.5
                    else:
                        current_position = 0.0
            holdings_history.append({
                'date': date,
                'position': current_position
            })
            day_return = 0
            for industry in industry_price_df['industry'].unique():
                ind_return = industry_price_df[
                    (industry_price_df['industry'] == industry) &
                    (industry_price_df['trade_date'] == date)
                ]['return'].values
                if len(ind_return) > 0:
                    day_return += ind_return[0] / industry_price_df['industry'].nunique()
            portfolio_returns.append(day_return * current_position)
            portfolio_value.append(portfolio_value[-1] * (1 + day_return * current_position))
            if i + 1 < len(trading_dates):
                next_date = trading_dates[i + 1]
                all_return = industry_price_df[
                    (industry_price_df['trade_date'] == next_date)
                ]['return'].mean()
                benchmark_returns.append(all_return if not np.isnan(all_return) else 0)
        results = self.backtest_engine._calculate_performance_metrics(
            portfolio_value, portfolio_returns, benchmark_returns, trading_dates
        )
        results['portfolio_value'] = portfolio_value
        results['holdings_history'] = holdings_history
        return results

    def plot_industry_zscore(self, industry_survey_df, industry_name, save_path=None):
        industry_data = industry_survey_df[industry_survey_df['industry'] == industry_name].copy()
        if len(industry_data) == 0:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        ax2 = ax.twinx()
        ax.plot(industry_data['survey_date'], industry_data['institutions_count'],
                color='blue', label='Survey Count')
        ax2.plot(industry_data['survey_date'], industry_data['z_score'],
                color='red', label='Z-Score')
        ax.set_xlabel('Date')
        ax.set_ylabel('Survey Count', color='blue')
        ax2.set_ylabel('Z-Score', color='red')
        ax.set_title(f'{industry_name} - Survey Count and Z-Score')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_timing_signals(self, industry_survey_df, industry_price_df, industry_name, save_path=None):
        industry_survey = industry_survey_df[industry_survey_df['industry'] == industry_name].copy()
        industry_price = industry_price_df[industry_price_df['industry'] == industry_name].copy()
        if len(industry_survey) == 0 or len(industry_price) == 0:
            return
        merged = industry_survey.merge(industry_price[['industry', 'trade_date', 'close']],
                                        left_on='survey_date', right_on='trade_date', how='left')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax2 = ax.twinx()
        ax.plot(merged['survey_date'], merged['close'], color='blue', label='Price Index')
        ax2.plot(merged['survey_date'], merged['z_score'], color='red', label='Z-Score', linestyle='--')
        ax.fill_between(merged['survey_date'], 0, merged['z_score'],
                       where=merged['z_score'] > 0, alpha=0.3, color='green', label='High Attention')
        ax.fill_between(merged['survey_date'], 0, merged['z_score'],
                       where=merged['z_score'] <= 0, alpha=0.3, color='red', label='Low Attention')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price Index', color='blue')
        ax2.set_ylabel('Z-Score', color='red')
        ax.set_title(f'{industry_name} - Price Index and Z-Score Timing Signals')
        ax.legend(loc='upper left')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def analyze_all_industries_timing(self, industry_survey_df, industry_price_df):
        results = {}
        industries = industry_survey_df['industry'].unique()
        for industry in industries:
            timing_results = self.run_timing_strategy(
                industry_survey_df[industry_survey_df['industry'] == industry],
                industry_price_df[industry_price_df['industry'] == industry]
            )
            results[industry] = timing_results
        return pd.DataFrame([{
            'industry': ind,
            'annual_return': res.get('annual_return', 0) if res else 0,
            'annual_excess_return': res.get('annualized_excess_return', 0) if res else 0,
            'information_ratio': res.get('information_ratio', 0) if res else 0,
            'max_drawdown': res.get('max_drawdown', 0) if res else 0
        } for ind, res in results.items()])

    def generate_summary(self, strategy_results):
        if strategy_results is None:
            return None
        summary = {
            'strategy_type': 'Industry Rotation',
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