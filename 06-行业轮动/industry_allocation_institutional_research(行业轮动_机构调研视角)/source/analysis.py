import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class Analysis:
    def __init__(self):
        self.results = {}

    def analyze_survey_distribution(self, survey_df, price_df=None):
        if survey_df is None or len(survey_df) == 0:
            return None
        results = {}
        daily_stats = survey_df.groupby('survey_date').agg({
            'institutions_count': ['sum', 'count', 'mean', 'median']
        }).reset_index()
        daily_stats.columns = ['date', 'total_surveys', 'stock_count', 'avg_surveys', 'median_surveys']
        results['daily_stats'] = daily_stats
        if price_df is not None:
            merged = daily_stats.merge(price_df[['trade_date', 'close']], left_on='date', right_on='trade_date', how='left')
            results['correlation'] = merged['total_surveys'].corr(merged['close'])
        return results

    def analyze_survey_by_industry(self, survey_df, industry_df):
        if survey_df is None or industry_df is None:
            return None
        merged = survey_df.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')
        industry_stats = merged.groupby('industry').agg({
            'institutions_count': ['sum', 'mean', 'median', 'count']
        }).reset_index()
        industry_stats.columns = ['industry', 'total_surveys', 'avg_surveys', 'median_surveys', 'survey_count']
        return industry_stats.sort_values('total_surveys', ascending=False)

    def analyze_survey_stock_returns(self, survey_df, price_df, holding_periods=[20, 40, 60, 100, 120]):
        if survey_df is None or price_df is None:
            return None
        results = {}
        survey_df = survey_df.copy()
        price_df = price_df.copy()
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        for period in holding_periods:
            price_df[f'return_{period}d'] = price_df.groupby('ts_code')['close'].pct_change(period)
        survey_merged = survey_df.merge(
            price_df[['ts_code', 'trade_date'] + [f'return_{p}d' for p in holding_periods]],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        threshold_groups = [0, 50, 100, 150, 200, 250, 300]
        for i in range(len(threshold_groups) - 1):
            lower = threshold_groups[i]
            upper = threshold_groups[i + 1]
            group_data = survey_merged[
                (survey_merged['institutions_count'] >= lower) &
                (survey_merged['institutions_count'] < upper)
            ]
            results[f'{lower}_{upper}'] = {}
            for period in holding_periods:
                col = f'return_{period}d'
                if col in group_data.columns:
                    results[f'{lower}_{upper}'][f'avg_return_{period}d'] = group_data[col].mean()
                    results[f'{lower}_{upper}'][f'median_return_{period}d'] = group_data[col].median()
                    results[f'{lower}_{upper}'][f'win_rate_{period}d'] = (group_data[col] > 0).mean()
        return results

    def calculate_win_rate_by_threshold(self, survey_df, price_df, holding_period=120):
        if survey_df is None or price_df is None:
            return None
        survey_df = survey_df.copy()
        price_df = price_df.copy()
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        price_df[f'return_{holding_period}d'] = price_df.groupby('ts_code')['close'].pct_change(holding_period)
        survey_merged = survey_df.merge(
            price_df[['ts_code', 'trade_date', f'return_{holding_period}d']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        thresholds = [20, 50, 80, 100, 200, 300]
        results = []
        for thresh in thresholds:
            group = survey_merged[survey_merged['institutions_count'] >= thresh]
            if len(group) > 0:
                avg_return = group[f'return_{holding_period}d'].mean()
                median_return = group[f'return_{holding_period}d'].median()
                win_rate = (group[f'return_{holding_period}d'] > 0).mean()
                results.append({
                    'threshold': thresh,
                    'avg_return': avg_return,
                    'median_return': median_return,
                    'win_rate': win_rate,
                    'sample_count': len(group)
                })
        return pd.DataFrame(results)

    def analyze_combined_indicators(self, survey_df, price_df, money_flow_df=None):
        if survey_df is None or price_df is None:
            return None
        results = {}
        survey_df = survey_df.copy()
        price_df = price_df.copy()
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        price_df = price_df.sort_values(['ts_code', 'trade_date'])
        price_df['return_5d'] = price_df.groupby('ts_code')['close'].pct_change(5)
        price_df['return_20d'] = price_df.groupby('ts_code')['close'].pct_change(20)
        survey_merged = survey_df.merge(
            price_df[['ts_code', 'trade_date', 'return_5d', 'return_20d']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        if money_flow_df is not None:
            money_flow_df = money_flow_df.copy()
            money_flow_df['trade_date'] = pd.to_datetime(money_flow_df['trade_date'])
            survey_merged = survey_merged.merge(
                money_flow_df[['ts_code', 'trade_date', 'buy_sm_amount', 'net_amount']],
                on=['ts_code', 'trade_date'],
                how='left'
            )
        positive_feedback = survey_merged[
            (survey_merged['return_5d'] > 0) & (survey_merged['institutions_count'] >= 50)
        ]
        results['positive_feedback_count'] = len(positive_feedback)
        if money_flow_df is not None and 'net_amount' in survey_merged.columns:
            capital_inflow = survey_merged[
                (survey_merged['net_amount'] > 0) & (survey_merged['institutions_count'] >= 50)
            ]
            results['capital_inflow_count'] = len(capital_inflow)
        return results

    def analyze_institutional_type_distribution(self, survey_detail_df):
        if survey_detail_df is None or len(survey_detail_df) == 0:
            return None
        if 'institution_type' in survey_detail_df.columns:
            type_dist = survey_detail_df.groupby('institution_type').agg({
                'institutions_count': 'sum'
            }).reset_index()
            type_dist['pct'] = type_dist['institutions_count'] / type_dist['institutions_count'].sum()
            return type_dist.sort_values('pct', ascending=False)
        return None

    def calculate_excess_return_stats(self, strategy_returns, benchmark_returns):
        if strategy_returns is None or benchmark_returns is None:
            return None
        excess_returns = strategy_returns - benchmark_returns
        results = {
            'avg_excess_return': excess_returns.mean(),
            'median_excess_return': excess_returns.median(),
            'std_excess_return': excess_returns.std(),
            'win_rate': (excess_returns > 0).mean(),
            'sharpe_ratio': excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(excess_returns),
            'profit_loss_ratio': self._calculate_profit_loss_ratio(excess_returns)
        }
        return results

    def _calculate_max_drawdown(self, returns):
        if returns is None or len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_profit_loss_ratio(self, returns):
        if returns is None or len(returns) == 0:
            return 0
        profits = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1
        return profits / losses if losses > 0 else 0

    def analyze_timing_signals(self, industry_survey_df, industry_price_df):
        if industry_survey_df is None or industry_price_df is None:
            return None
        results = {}
        industries = industry_survey_df['industry'].unique()
        for industry in industries:
            survey_data = industry_survey_df[industry_survey_df['industry'] == industry]
            price_data = industry_price_df[industry_price_df['industry'] == industry]
            if len(survey_data) > 0 and len(price_data) > 0:
                z_score = survey_data['z_score'].values
                returns = price_data['return'].values
                high_z_mask = z_score > 1
                mid_z_mask = (z_score > 0) & (z_score <= 1)
                low_z_mask = z_score <= 0
                results[industry] = {
                    'high_z_avg_return': returns[high_z_mask].mean() if high_z_mask.sum() > 0 else 0,
                    'mid_z_avg_return': returns[mid_z_mask].mean() if mid_z_mask.sum() > 0 else 0,
                    'low_z_avg_return': returns[low_z_mask].mean() if low_z_mask.sum() > 0 else 0,
                }
        return pd.DataFrame(results).T

    def plot_survey_distribution(self, survey_df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        daily_stats = survey_df.groupby('survey_date')['institutions_count'].sum()
        axes[0, 0].plot(daily_stats.index, daily_stats.values)
        axes[0, 0].set_title('Daily Total Institutional Surveys')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Surveys')
        threshold_groups = [0, 50, 100, 200, 300]
        group_counts = []
        for i in range(len(threshold_groups) - 1):
            lower = threshold_groups[i]
            upper = threshold_groups[i + 1]
            count = len(survey_df[
                (survey_df['institutions_count'] >= lower) &
                (survey_df['institutions_count'] < upper)
            ])
            group_counts.append(count)
        axes[0, 1].bar(range(len(threshold_groups) - 1), group_counts)
        axes[0, 1].set_xticks(range(len(threshold_groups) - 1))
        axes[0, 1].set_xticklabels([f'{threshold_groups[i]}-{threshold_groups[i+1]}' for i in range(len(threshold_groups) - 1)])
        axes[0, 1].set_title('Survey Count Distribution by Threshold')
        axes[0, 1].set_xlabel('Threshold Range')
        axes[0, 1].set_ylabel('Count')
        top_stocks = survey_df.groupby('ts_code')['institutions_count'].sum().nlargest(20)
        axes[1, 0].barh(range(len(top_stocks)), top_stocks.values)
        axes[1, 0].set_yticks(range(len(top_stocks)))
        axes[1, 0].set_yticklabels(top_stocks.index)
        axes[1, 0].set_title('Top 20 Stocks by Total Institutional Surveys')
        axes[1, 0].set_xlabel('Total Survey Count')
        monthly_survey = survey_df.groupby(survey_df['survey_date'].dt.to_period('M'))['institutions_count'].sum()
        axes[1, 1].plot(range(len(monthly_survey)), monthly_survey.values)
        axes[1, 1].set_title('Monthly Institutional Surveys')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Total Surveys')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_return_analysis(self, return_analysis_df, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        thresholds = return_analysis_df['threshold'].values
        median_returns = return_analysis_df['median_return_120d'].values * 100
        win_rates = return_analysis_df['win_rate_120d'].values * 100
        axes[0].bar(range(len(thresholds)), median_returns)
        axes[0].set_xticks(range(len(thresholds)))
        axes[0].set_xticklabels(thresholds)
        axes[0].set_title('Median Return by Survey Threshold (120d Holding)')
        axes[0].set_xlabel('Survey Count Threshold')
        axes[0].set_ylabel('Median Return (%)')
        axes[1].bar(range(len(thresholds)), win_rates)
        axes[1].set_xticks(range(len(thresholds)))
        axes[1].set_xticklabels(thresholds)
        axes[1].set_title('Win Rate by Survey Threshold (120d Holding)')
        axes[1].set_xlabel('Survey Count Threshold')
        axes[1].set_ylabel('Win Rate (%)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self, survey_df, price_df, strategy_results):
        report = {}
        report['total_survey_records'] = len(survey_df) if survey_df is not None else 0
        report['total_institutions'] = survey_df['institutions_count'].sum() if survey_df is not None else 0
        report['unique_stocks_surveyed'] = survey_df['ts_code'].nunique() if survey_df is not None else 0
        report['date_range'] = {
            'start': survey_df['survey_date'].min() if survey_df is not None else None,
            'end': survey_df['survey_date'].max() if survey_df is not None else None
        }
        if strategy_results is not None:
            report['strategy_performance'] = strategy_results
        return report