import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self):
        self.processed_data = {}

    def clean_survey_data(self, survey_df):
        if survey_df is None or len(survey_df) == 0:
            return None
        df = survey_df.copy()
        df = df.dropna(subset=['ts_code', 'survey_date', 'institutions_count'])
        df = df[df['institutions_count'] > 0]
        df['survey_date'] = pd.to_datetime(df['survey_date'])
        df = df.sort_values(['ts_code', 'survey_date'])
        df = df.drop_duplicates(subset=['ts_code', 'survey_date'], keep='first')
        return df

    def filter_valid_stocks(self, survey_df, stock_basic_df):
        if survey_df is None or stock_basic_df is None:
            return survey_df
        valid_codes = set(stock_basic_df['ts_code'].tolist())
        survey_df = survey_df[survey_df['ts_code'].isin(valid_codes)]
        return survey_df

    def add_business_days(self, date, days):
        current_date = date
        added_days = 0
        while added_days < days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                added_days += 1
        return current_date

    def align_survey_with_trading_days(self, survey_df, trading_days):
        if survey_df is None or len(survey_df) == 0:
            return survey_df
        df = survey_df.copy()
        df['announce_date'] = df['survey_date']
        df['trade_date'] = df['survey_date'].apply(
            lambda x: trading_days[trading_days >= x].min() if len(trading_days[trading_days >= x]) > 0 else x
        )
        return df

    def calculate_survey_features(self, survey_df, lookback_days_list=[1, 5, 10, 20, 40, 60]):
        if survey_df is None or len(survey_df) == 0:
            return None
        df = survey_df.copy()
        result_dfs = []
        for lookback in lookback_days_list:
            feature_df = df.groupby('ts_code').apply(
                lambda x: self._calculate_rolling_features(x, lookback)
            ).reset_index(drop=True)
            feature_df = feature_df.rename(columns={'institutions_count': f'survey_count_{lookback}d'})
            result_dfs.append(feature_df)
        merged_df = df.copy()
        for rdf in result_dfs:
            merged_df = merged_df.merge(rdf, on=['ts_code', 'survey_date'], how='left')
        return merged_df

    def _calculate_rolling_features(self, group_df, lookback):
        group_df = group_df.sort_values('survey_date')
        result = []
        for i in range(len(group_df)):
            current_date = group_df.iloc[i]['survey_date']
            start_date = current_date - timedelta(days=lookback)
            mask = (group_df['survey_date'] >= start_date) & (group_df['survey_date'] <= current_date)
            window_data = group_df[mask]
            total_count = window_data['institutions_count'].sum()
            result.append({
                'survey_date': current_date,
                'ts_code': group_df.iloc[i]['ts_code'],
                'institutions_count': total_count
            })
        return pd.DataFrame(result)

    def filter_by_listing_date(self, survey_df, listing_date_threshold=180):
        if survey_df is None:
            return None
        cutoff_date = datetime.now() - timedelta(days=listing_date_threshold)
        survey_df = survey_df[survey_df['survey_date'] >= cutoff_date]
        return survey_df

    def remove_st_stock(self, stock_list, st_stock_list):
        return [s for s in stock_list if s not in st_stock_list]

    def calculate_z_score(self, series):
        if series is None or len(series) == 0:
            return series
        return (series - series.mean()) / series.std()

    def smooth_series(self, series, window=250):
        if series is None or len(series) == 0:
            return series
        return series.rolling(window=window, min_periods=1).mean()

    def merge_with_price_data(self, survey_df, price_df):
        if survey_df is None or price_df is None:
            return None
        survey_df = survey_df.copy()
        price_df = price_df.copy()
        if 'trade_date' not in price_df.columns and 'date' in price_df.columns:
            price_df['trade_date'] = pd.to_datetime(price_df['date'])
        if 'trade_date' not in survey_df.columns:
            survey_df['trade_date'] = survey_df['survey_date']
        merged_df = survey_df.merge(
            price_df[['ts_code', 'trade_date', 'pct_chg', 'amount']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        return merged_df

    def add_industry_info(self, survey_df, industry_df):
        if survey_df is None or industry_df is None:
            return survey_df
        df = survey_df.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')
        return df

    def calculate_future_returns(self, price_df, periods=[20, 40, 60, 100, 120]):
        if price_df is None or len(price_df) == 0:
            return None
        df = price_df.sort_values(['ts_code', 'trade_date'])
        for period in periods:
            df[f'future_return_{period}d'] = df.groupby('ts_code')['close'].pct_change(period)
        return df

    def add_market_cap_info(self, survey_df, daily_stats_df):
        if survey_df is None or daily_stats_df is None:
            return survey_df
        survey_df = survey_df.copy()
        daily_stats_df = daily_stats_df.copy()
        if 'trade_date' not in daily_stats_df.columns:
            daily_stats_df['trade_date'] = pd.to_datetime(daily_stats_df['date'])
        survey_df = survey_df.merge(
            daily_stats_df[['ts_code', 'trade_date', 'total_mv', 'circ_mv']],
            on=['ts_code', 'trade_date'],
            how='left'
        )
        return survey_df

    def aggregate_industry_survey(self, survey_df, industry_col='industry'):
        if survey_df is None or industry_col not in survey_df.columns:
            return None
        industry_survey = survey_df.groupby([industry_col, 'survey_date']).agg({
            'institutions_count': 'sum'
        }).reset_index()
        return industry_survey

    def normalize_industry_survey(self, industry_survey_df, stock_count_df):
        if industry_survey_df is None or stock_count_df is None:
            return None
        merged = industry_survey_df.merge(stock_count_df, left_on=industry_survey_df.index.name,
                                          right_on=stock_count_df.index.name, how='left')
        merged['avg_survey_count'] = merged['institutions_count'] / merged['stock_count']
        return merged

    def create_survey_signal(self, survey_df, threshold=50):
        if survey_df is None:
            return None
        df = survey_df.copy()
        df['signal'] = (df['institutions_count'] >= threshold).astype(int)
        return df

    def resample_to_trading_days(self, survey_df, trading_days):
        if survey_df is None or len(survey_df) == 0:
            return None
        df = survey_df.set_index('survey_date')
        resampled = df.reindex(trading_days, method='ffill')
        resampled = resampled.reset_index().rename(columns={'index': 'trade_date'})
        return resampled

    def calculate_cumulative_survey(self, survey_df, window=250):
        if survey_df is None or len(survey_df) == 0:
            return None
        df = survey_df.sort_values(['ts_code', 'survey_date'])
        df['cumulative_survey'] = df.groupby('ts_code')['institutions_count'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
        return df

    def fill_missing_dates(self, survey_df, start_date, end_date):
        if survey_df is None or len(survey_df) == 0:
            return None
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df = survey_df.set_index('survey_date')
        df = df.reindex(date_range, fill_value=0)
        df = df.reset_index().rename(columns={'index': 'trade_date'})
        return df

    def add_calendar_features(self, df):
        if df is None:
            return None
        df = df.copy()
        if 'trade_date' in df.columns:
            date_col = 'trade_date'
        elif 'survey_date' in df.columns:
            date_col = 'survey_date'
        else:
            return df
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_week'] = df[date_col].dt.dayofweek
        return df


def preprocess_for_event_strategy(survey_df, trading_days, lookback_days=60, holding_days=100, threshold=50):
    preprocessor = DataPreprocessor()
    survey_df = preprocessor.clean_survey_data(survey_df)
    survey_df = preprocessor.align_survey_with_trading_days(survey_df, trading_days)
    survey_df = preprocessor.calculate_survey_features(survey_df, lookback_days_list=[lookback_days])
    survey_df['signal'] = (survey_df[f'survey_count_{lookback_days}d'] >= threshold).astype(int)
    return survey_df


def preprocess_for_regular_strategy(survey_df, trading_days, lookback_days=120):
    preprocessor = DataPreprocessor()
    survey_df = preprocessor.clean_survey_data(survey_df)
    survey_df = preprocessor.align_survey_with_trading_days(survey_df, trading_days)
    lookback_list = [10, 20, 40, 60, 120] if lookback_days == 120 else [lookback_days]
    survey_df = preprocessor.calculate_survey_features(survey_df, lookback_days_list=lookback_list)
    return survey_df


def preprocess_for_industry_strategy(survey_df, industry_df, stock_count_df, smooth_window=250):
    preprocessor = DataPreprocessor()
    survey_df = preprocessor.clean_survey_data(survey_df)
    survey_df = preprocessor.add_industry_info(survey_df, industry_df)
    industry_survey = preprocessor.aggregate_industry_survey(survey_df)
    industry_survey['smoothed_survey'] = preprocessor.smooth_series(
        industry_survey.groupby('industry')['institutions_count'].transform(
            lambda x: x.rolling(window=smooth_window, min_periods=1).mean()
        ), window=smooth_window
    )
    industry_survey['z_score'] = industry_survey.groupby('industry')['smoothed_survey'].transform(
        lambda x: preprocessor.calculate_z_score(x)
    )
    return industry_survey