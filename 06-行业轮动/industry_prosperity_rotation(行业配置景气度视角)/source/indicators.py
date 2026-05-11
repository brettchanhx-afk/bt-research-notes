import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProsperityIndicator:
    def __init__(self):
        self.indicator_mapping = {
            'net_profit_margin': {'name': '销售净利率', 'type': 'profitability', 'direction': 1},
            'gross_profit_margin': {'name': '销售毛利率', 'type': 'profitability', 'direction': 1},
            'roe': {'name': '净资产收益率', 'type': 'profitability', 'direction': 1},
            'roa': {'name': '总资产净利率', 'type': 'profitability', 'direction': 1},
            'nptocostexpense': {'name': '成本费用利润率', 'type': 'profitability', 'direction': 1},
            'oper_rev_yoy': {'name': '营业收入同比增速', 'type': 'growth', 'direction': 1},
            'net_profit_yoy': {'name': '归母净利润增速', 'type': 'growth', 'direction': 1},
            'tot_profit_yoy': {'name': '利润总额增速', 'type': 'growth', 'direction': 1},
            'debt_to_assets': {'name': '资产负债率', 'type': 'capital_structure', 'direction': 1},
            'current_ratio': {'name': '流动比率', 'type': 'solvency', 'direction': -1},
            'quick_ratio': {'name': '速动比率', 'type': 'solvency', 'direction': -1},
            'inv_turn': {'name': '存货周转率', 'type': 'operation', 'direction': 1},
            'assets_turn': {'name': '总资产周转率', 'type': 'operation', 'direction': 1},
            'ar_turn': {'name': '应收账款周转率', 'type': 'operation', 'direction': 1},
            'op_ex_rev_yoy': {'name': '营业收入增速变化', 'type': 'growth', 'direction': 1},
        }

    def calculate_ttm(self, df, col_name):
        if col_name not in df.columns:
            return df[[]].assign(**{col_name: np.nan})

        result = df[['industry_code', 'trade_date', col_name]].copy()
        result = result.sort_values(['industry_code', 'trade_date'])
        result[col_name] = result.groupby('industry_code')[col_name].transform(
            lambda x: x.rolling(window=4, min_periods=4).sum()
        )
        return result

    def calculate_qoq(self, df, col_name):
        if col_name not in df.columns:
            return df[[]].assign(**{f'{col_name}_qoq': np.nan})

        result = df[['industry_code', 'trade_date', col_name]].copy()
        result = result.sort_values(['industry_code', 'trade_date'])
        result[f'{col_name}_qoq'] = result.groupby('industry_code')[col_name].diff()
        return result[[f'{col_name}_qoq']]

    def calculate_yoy(self, df, col_name):
        if col_name not in df.columns:
            return df[[]].assign(**{f'{col_name}_yoy': np.nan})

        result = df[['industry_code', 'trade_date', col_name]].copy()
        result = result.sort_values(['industry_code', 'trade_date'])
        result[f'{col_name}_yoy'] = result.groupby('industry_code')[col_name].pct_change(periods=4)
        return result[[f'{col_name}_yoy']]

    def calculate_growth_acceleration(self, df, col_name):
        if col_name not in df.columns:
            return df[[]].assign(**{f'{col_name}_acc': np.nan})

        result = df[['industry_code', 'trade_date', col_name]].copy()
        result = result.sort_values(['industry_code', 'trade_date'])

        result[f'{col_name}_yoy'] = result.groupby('industry_code')[col_name].pct_change(periods=4)
        result[f'{col_name}_acc'] = result.groupby('industry_code')[f'{col_name}_yoy'].diff()

        return result[[f'{col_name}_acc']]

    def build_industry_indicators(self, industry_financial_data):
        if industry_financial_data is None or len(industry_financial_data) == 0:
            return pd.DataFrame()

        df = industry_financial_data.copy()

        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values(['industry_code', 'trade_date'])

        indicator_cols = ['net_profit_margin', 'gross_profit_margin', 'roe', 'roa',
                         'debt_to_assets', 'current_ratio', 'quick_ratio',
                         'inv_turn', 'assets_turn', 'ar_turn',
                         'op_ex_rev_yoy', 'net_profit_yoy']

        available_cols = [col for col in indicator_cols if col in df.columns]

        signals_list = []

        for col in available_cols:
            if col in self.indicator_mapping:
                direction = self.indicator_mapping[col]['direction']
            else:
                direction = 1

            temp_df = df[['industry_code', 'trade_date', col]].copy()

            temp_df[f'{col}_yoy'] = temp_df.groupby('industry_code')[col].pct_change(periods=4)
            temp_df[f'{col}_qoq'] = temp_df.groupby('industry_code')[col].diff()

            temp_df[f'{col}_signal'] = temp_df[f'{col}_qoq'].apply(
                lambda x: direction if x > 0 else (-direction if x < 0 else 0)
            )

            temp_df = temp_df[['industry_code', 'trade_date', f'{col}_signal']].copy()
            temp_df = temp_df.rename(columns={f'{col}_signal': col})

            signals_list.append(temp_df)

        if not signals_list:
            return pd.DataFrame()

        result = signals_list[0]
        for temp_df in signals_list[1:]:
            result = result.merge(temp_df, on=['industry_code', 'trade_date'], how='outer')

        return result

    def generate_prosperity_signals(self, indicator_data, date):
        if indicator_data is None or len(indicator_data) == 0:
            return pd.DataFrame()

        date_data = indicator_data[indicator_data['trade_date'] == date].copy()

        if len(date_data) == 0:
            return pd.DataFrame()

        signal_cols = [col for col in date_data.columns if col not in ['industry_code', 'trade_date']]

        date_data['signal_count'] = 0
        for col in signal_cols:
            date_data['signal_count'] += date_data[col]

        date_data['signal_count_normalized'] = (date_data['signal_count'] - date_data['signal_count'].mean()) / (date_data['signal_count'].std() + 1e-6)

        date_data = date_data.sort_values('signal_count_normalized', ascending=False)

        return date_data

    def rank_industries(self, indicator_data, date, top_n=5):
        date_data = self.generate_prosperity_signals(indicator_data, date)

        if len(date_data) == 0:
            return pd.DataFrame()

        date_data['rank'] = range(1, len(date_data) + 1)

        return date_data.head(top_n)


class IndicatorValidator:
    def __init__(self):
        self.results = {}

    def validate_single_indicator(self, indicator_data, returns_data, indicator_name):
        if indicator_data is None or returns_data is None:
            return None

        merged = indicator_data.merge(
            returns_data,
            on=['industry_code', 'trade_date'],
            how='inner'
        )

        if len(merged) < 10:
            return None

        signal_col = indicator_name if indicator_name in merged.columns else f'{indicator_name}_signal'

        if signal_col not in merged.columns:
            return None

        long_returns = merged[merged[signal_col] > 0].groupby('trade_date')['return'].mean()
        benchmark_returns = merged.groupby('trade_date')['return'].mean()

        if len(long_returns) == 0:
            return None

        long_cumret = (1 + long_returns / 100).cumprod().iloc[-1] if len(long_returns) > 0 else 1
        benchmark_cumret = (1 + benchmark_returns / 100).cumprod().iloc[-1] if len(benchmark_returns) > 0 else 1

        excess_return = (long_cumret - benchmark_cumret) * 100
        win_rate = (long_returns > benchmark_returns).sum() / len(long_returns) * 100 if len(long_returns) > 0 else 0

        annual_excess = ((long_cumret / benchmark_cumret) ** (1 / (len(long_returns) / 12)) - 1) * 100 if benchmark_cumret != 0 and len(long_returns) > 0 else 0

        self.results[indicator_name] = {
            'annual_excess_return': annual_excess,
            'win_rate': win_rate,
            'long_cum_return': (long_cumret - 1) * 100,
            'benchmark_cum_return': (benchmark_cumret - 1) * 100,
            'excess_return': excess_return,
            'n_signals': len(merged[merged[signal_col] != 0])
        }

        return self.results[indicator_name]

    def validate_all_indicators(self, indicator_data, returns_data, indicator_list):
        results = {}
        for ind_name in indicator_list:
            result = self.validate_single_indicator(indicator_data, returns_data, ind_name)
            if result is not None:
                results[ind_name] = result
        return results

    def get_top_indicators(self, n=20):
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['annual_excess_return'],
            reverse=True
        )
        return sorted_results[:n]

    def calculate_correlation_matrix(self, indicator_data, indicator_list):
        signal_cols = [col for col in indicator_data.columns if col in indicator_list]

        if len(signal_cols) < 2:
            return pd.DataFrame()

        return indicator_data[signal_cols].corr()


class ConsensusIndicator:
    def __init__(self):
        pass

    def build_consensus_indicators(self, consensus_data):
        if consensus_data is None or len(consensus_data) == 0:
            return pd.DataFrame()

        df = consensus_data.copy()

        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values(['industry_code', 'trade_date'])

        df['eps_signal'] = df.groupby('industry_code')['eps_forecast_yoy'].diff()
        df['roe_signal'] = df.groupby('industry_code')['roe_forecast_yoy'].diff()

        df['up_ratio'] = df['up_count'] / (df['up_count'] + df['down_count'] + df['neutral_count'])
        df['up_ratio_signal'] = df.groupby('industry_code')['up_ratio'].diff()

        df['consensus_score'] = (
            df['eps_signal'].fillna(0) * 0.4 +
            df['roe_signal'].fillna(0) * 0.4 +
            df['up_ratio_signal'].fillna(0) * 0.2
        )

        indicator_cols = ['eps_forecast_yoy', 'roe_forecast_yoy', 'up_ratio']
        available_cols = [col for col in indicator_cols if col in df.columns]

        for col in available_cols:
            df[f'{col}_qoq'] = df.groupby('industry_code')[col].diff()

        return df

    def generate_consensus_signals(self, consensus_indicators, date):
        if consensus_indicators is None or len(consensus_indicators) == 0:
            return pd.DataFrame()

        date_data = consensus_indicators[consensus_indicators['trade_date'] == date].copy()

        if len(date_data) == 0:
            return pd.DataFrame()

        signal_cols = ['eps_forecast_yoy_qoq', 'roe_forecast_yoy_qoq', 'up_ratio_qoq']

        for col in signal_cols:
            if col in date_data.columns:
                date_data[col] = date_data[col].fillna(0)

        date_data['consensus_score'] = 0
        for col in signal_cols:
            if col in date_data.columns:
                date_data['consensus_score'] += date_data[col]

        date_data = date_data.sort_values('consensus_score', ascending=False)

        return date_data


if __name__ == "__main__":
    print("Testing ProsperityIndicator...")

    indicator = ProsperityIndicator()
    print(f"Loaded {len(indicator.indicator_mapping)} indicator mappings")

    validator = IndicatorValidator()
    print("IndicatorValidator initialized")

    consensus_indicator = ConsensusIndicator()
    print("ConsensusIndicator initialized")

    print("Indicator module test completed!")
