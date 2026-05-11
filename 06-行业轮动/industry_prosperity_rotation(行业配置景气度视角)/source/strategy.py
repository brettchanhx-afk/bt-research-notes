import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProsperityRotationStrategy:
    def __init__(self, data_loader, rebalance_freq='M', top_n=5, commission_rate=0.0003):
        self.data_loader = data_loader
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.commission_rate = commission_rate

        self.industry_list = None
        self.industry_returns = None
        self.financial_data = None
        self.consensus_data = None
        self.prosperity_data = None
        self.rebalance_dates = None

        self.benchmark_returns = None
        self.strategy_returns = None

    def initialize(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

        print("Initializing strategy...")
        self.industry_list = self._get_industry_list()
        self.rebalance_dates = self._generate_rebalance_dates()

        print(f"Strategy initialized: {len(self.industry_list)} industries, {len(self.rebalance_dates)} rebalance dates")

    def _get_industry_list(self):
        sw_list = self.data_loader.get_sw_industry_list(level=1)
        return sw_list

    def _generate_rebalance_dates(self):
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=self.rebalance_freq)
        return dates.tolist()

    def load_all_data(self, reload=False):
        print("Loading industry price data...")
        self.industry_returns = self._get_industry_returns(reload)

        print("Loading financial data...")
        self.financial_data = self._get_financial_data(reload)

        print("Loading consensus data...")
        self.consensus_data = self._get_consensus_data(reload)

        print("All data loaded successfully!")

    def _get_industry_returns(self, reload=False):
        industry_codes = self.industry_list['index_code'].tolist() if self.industry_list is not None else []

        if len(industry_codes) == 0:
            return pd.DataFrame()

        all_data = []
        for code in industry_codes:
            try:
                df = self.data_loader.get_sw_industry_historical(
                    code,
                    self.start_date.replace('-', ''),
                    self.end_date.replace('-', '')
                )
                if df is not None and len(df) > 0:
                    df['industry_code'] = code
                    all_data.append(df)
            except Exception as e:
                pass

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values(['trade_date', 'industry_code'])

            if 'close' in result.columns:
                result['return'] = result.groupby('industry_code')['close'].pct_change() * 100

            return result

        return pd.DataFrame()

    def _get_financial_data(self, reload=False):
        return self.data_loader.get_industry_financial_aggregate(
            self.start_date.replace('-', ''),
            self.end_date.replace('-', ''),
            reload=reload
        )

    def _get_consensus_data(self, reload=False):
        return self.data_loader.get_consensus_data(
            self.start_date.replace('-', ''),
            self.end_date.replace('-', ''),
            reload=reload
        )

    def calculate_prosperity_indicators(self):
        print("Calculating prosperity indicators...")

        from .indicators import ProsperityIndicator, ConsensusIndicator
        from .composite_indicator import IndustryProsperityCalculator

        prosperity_calc = IndustryProsperityCalculator()

        self.prosperity_data = prosperity_calc.calculate_prosperity_index(
            self.financial_data,
            self.consensus_data,
            self.industry_returns
        )

        print(f"Prosperity data calculated: {len(self.prosperity_data)} records")

        return self.prosperity_data

    def generate_trading_signals(self, date):
        if self.prosperity_data is None or len(self.prosperity_data) == 0:
            return pd.DataFrame()

        date_str = pd.to_datetime(date).strftime('%Y%m%d') if isinstance(date, str) else date

        date_data = self.prosperity_data.copy()
        if 'trade_date' in date_data.columns:
            date_data['trade_date'] = pd.to_datetime(date_data['trade_date'])

        if isinstance(date, str):
            target_date = pd.to_datetime(date)
        else:
            target_date = date

        available_dates = date_data['trade_date'].dropna().unique()
        available_dates = sorted(available_dates)

        closest_date = min(available_dates, key=lambda x: abs(x - target_date))

        date_data = date_data[date_data['trade_date'] == closest_date].copy()

        if len(date_data) == 0:
            return pd.DataFrame()

        score_col = 'prosperity_score' if 'prosperity_score' in date_data.columns else 'composite_score'

        date_data = date_data.sort_values(score_col, ascending=False)
        date_data = date_data.reset_index(drop=True)

        date_data['signal'] = 0
        date_data['rank'] = range(1, len(date_data) + 1)

        top_n = min(self.top_n, len(date_data))
        date_data.loc[:top_n-1, 'signal'] = 1

        return date_data

    def run_backtest(self):
        print("Running backtest...")

        if self.prosperity_data is None or len(self.prosperity_data) == 0:
            print("Error: No prosperity data available")
            return None

        all_signals = []
        all_portfolio_values = []

        initial_capital = 1000000
        current_capital = initial_capital
        positions = {}

        if self.industry_returns is not None and len(self.industry_returns) > 0:
            returns_df = self.industry_returns.copy()
            if 'trade_date' in returns_df.columns:
                returns_df['trade_date'] = pd.to_datetime(returns_df['trade_date'])
            unique_trade_dates = sorted(returns_df['trade_date'].unique())
        else:
            print("Error: No industry returns data available")
            return None

        unique_dates = sorted(self.prosperity_data['trade_date'].dropna().unique())

        def get_nearest_trade_date(target_date, trade_dates):
            target = pd.to_datetime(target_date)
            return min(trade_dates, key=lambda x: abs(x - target))

        for i, date in enumerate(unique_dates):
            signals = self.generate_trading_signals(date)

            if len(signals) == 0:
                continue

            long_industries = signals[signals['signal'] == 1]['industry_code'].tolist()

            if i > 0:
                prev_date = unique_dates[i-1]
                curr_trade_date = get_nearest_trade_date(date, unique_trade_dates)
                prev_trade_date = get_nearest_trade_date(prev_date, unique_trade_dates)

                date_returns = returns_df[returns_df['trade_date'] == curr_trade_date]
                prev_returns = returns_df[returns_df['trade_date'] == prev_trade_date]

                position_value = 0
                for code, weight in positions.items():
                    curr_data = date_returns[date_returns['industry_code'] == code]
                    prev_data = prev_returns[prev_returns['industry_code'] == code]

                    if len(curr_data) > 0 and len(prev_data) > 0:
                        curr_price = curr_data['close'].values[0]
                        prev_price = prev_data['close'].values[0]
                        ret = (curr_price - prev_price) / prev_price if prev_price != 0 else 0
                        position_value += weight * (1 + ret)
                    else:
                        position_value += weight

                current_capital = current_capital * position_value

            if i < len(unique_dates) - 1:
                if len(long_industries) > 0:
                    weight = 1.0 / len(long_industries)
                    positions = {code: weight for code in long_industries}
                else:
                    positions = {}

            all_portfolio_values.append({
                'date': date,
                'portfolio_value': current_capital,
                'signal_date': date
            })

            for code in long_industries:
                all_signals.append({
                    'date': date,
                    'industry_code': code,
                    'signal': 1,
                    'rank': signals[signals['industry_code'] == code]['rank'].values[0] if code in signals['industry_code'].values else 0
                })

        portfolio_df = pd.DataFrame(all_portfolio_values)

        if len(portfolio_df) > 0:
            portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df['cum_return'] = (1 + portfolio_df['return']).cumprod() - 1

        signals_df = pd.DataFrame(all_signals)

        self.strategy_returns = portfolio_df

        print(f"Backtest completed! Final portfolio value: {portfolio_df['portfolio_value'].iloc[-1]:,.2f}")

        return {
            'portfolio_values': portfolio_df,
            'signals': signals_df,
            'initial_capital': initial_capital,
            'final_value': portfolio_df['portfolio_value'].iloc[-1] if len(portfolio_df) > 0 else initial_capital
        }

    def get_benchmark_returns(self):
        if self.industry_returns is None or len(self.industry_returns) == 0:
            return pd.DataFrame()

        benchmark = self.industry_returns.groupby('trade_date')['return'].mean().reset_index()
        benchmark.columns = ['date', 'benchmark_return']

        return benchmark

    def calculate_excess_returns(self, strategy_df, benchmark_df):
        if strategy_df is None or len(strategy_df) == 0:
            return pd.DataFrame()

        merged = strategy_df.merge(benchmark_df, on='date', how='left')

        if 'return' in merged.columns and 'benchmark_return' in merged.columns:
            merged['excess_return'] = merged['return'] - merged['benchmark_return']

        return merged


class MomentumStrategy:
    def __init__(self, lookback_period=20, top_n=5):
        self.lookback_period = lookback_period
        self.top_n = top_n

    def calculate_momentum(self, price_data):
        if len(price_data) < self.lookback_period:
            return pd.Series()

        momentum = price_data.pct_change(periods=self.lookback_period)
        return momentum

    def select_top_industries(self, momentum_scores, date, top_n=None):
        if top_n is None:
            top_n = self.top_n

        date_momentum = momentum_scores[momentum_scores.index <= date].tail(self.lookback_period)

        if len(date_momentum) == 0:
            return []

        top_industries = date_momentum.nlargest(top_n).index.tolist()
        return top_industries


class StrategyEvaluator:
    def __init__(self):
        pass

    def calculate_annual_return(self, returns, periods_per_year=12):
        if len(returns) == 0 or returns is None:
            return 0

        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)

        if n_periods < 1:
            return 0

        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        return annual_return * 100

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.03, periods_per_year=12):
        if len(returns) == 0 or returns is None:
            return 0

        annual_return = self.calculate_annual_return(returns, periods_per_year)
        annual_vol = returns.std() * np.sqrt(periods_per_year) * 100

        if annual_vol == 0:
            return 0

        sharpe = (annual_return / 100 - risk_free_rate) / (annual_vol / 100)
        return sharpe

    def calculate_win_rate(self, returns):
        if len(returns) == 0 or returns is None:
            return 0

        win_rate = (returns > 0).sum() / len(returns) * 100
        return win_rate

    def calculate_max_drawdown(self, portfolio_values):
        if len(portfolio_values) == 0 or portfolio_values is None:
            return 0

        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values - cummax) / cummax

        max_dd = drawdown.min() * 100
        return max_dd

    def evaluate_strategy(self, portfolio_values, returns):
        metrics = {
            'annual_return': self.calculate_annual_return(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'win_rate': self.calculate_win_rate(returns),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values),
            'total_return': ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1) * 100 if len(portfolio_values) > 0 else 0
        }

        return metrics


if __name__ == "__main__":
    print("Testing ProsperityRotationStrategy...")

    from data_loader import DataLoader
    loader = DataLoader()

    strategy = ProsperityRotationStrategy(
        data_loader=loader,
        rebalance_freq='M',
        top_n=5
    )

    print("Strategy initialized")

    print("Strategy module test completed!")
