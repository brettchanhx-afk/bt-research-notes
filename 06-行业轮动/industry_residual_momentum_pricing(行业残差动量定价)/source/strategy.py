import pandas as pd
import numpy as np
from datetime import datetime

class ResidualMomentumStrategy:
    def __init__(self, top_n=5, rebalance_freq='M', fee_rate=0.0):
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self.fee_rate = fee_rate
        self.positions = None
        self.signals = None
        self.portfolio_value = None

    def generate_signals(self, residual_momentum, rebalance_dates=None):
        if residual_momentum.empty:
            return pd.DataFrame()

        if rebalance_dates is None:
            if self.rebalance_freq == 'M':
                rebalance_dates = residual_momentum.resample('M').last().index
            else:
                rebalance_dates = residual_momentum.index

        signals = pd.DataFrame(index=rebalance_dates, columns=residual_momentum.columns)
        signals = signals.fillna(0).astype(float)

        for date in rebalance_dates:
            available_dates = residual_momentum.index[residual_momentum.index <= date]
            if len(available_dates) == 0:
                continue

            latest_date = available_dates[-1]
            mom_values = residual_momentum.loc[latest_date].dropna()

            if len(mom_values) >= self.top_n:
                top_assets = mom_values.nlargest(self.top_n).index.tolist()
                signals.loc[date, top_assets] = 1.0 / self.top_n

        self.signals = signals
        return signals

    def backtest(self, prices, signals, initial_capital=1000000.0):
        if prices.empty or signals.empty:
            return None

        common_dates = prices.index.intersection(signals.index)
        prices = prices.loc[common_dates]
        signals = signals.loc[common_dates]

        returns = prices.pct_change().fillna(0)

        portfolio_returns = (signals.shift(1) * returns).sum(axis=1)

        net_value = (1 + portfolio_returns).cumprod() * initial_capital
        self.portfolio_value = net_value

        turnovers = signals.diff().abs().sum(axis=1) / 1

        if self.fee_rate > 0:
            costs = turnovers * self.fee_rate
            net_value_adjusted = net_value * (1 - costs).cumprod()
            self.portfolio_value = net_value_adjusted

        results = {
            'net_value': net_value,
            'returns': portfolio_returns,
            'turnover': turnovers,
            'signals': signals,
            'prices': prices
        }

        return results

    def calculate_performance(self, returns, benchmark_returns=None):
        if returns is None or len(returns) == 0:
            return {}

        excess_returns = returns - benchmark_returns if benchmark_returns is not None else returns

        annual_return = returns.mean() * 12
        annual_vol = returns.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        performance = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_return': cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        }

        return performance

    def combine_with_other_signal(self, signal1, signal2, method='soft'):
        if method == 'hard':
            combined = (signal1 + signal2) / 2
        else:
            combined = (signal1 + signal2) / 2

        return combined

    def add_defense_signals(self, prices, signals, defense_threshold=0.7):
        price_mom = prices.pct_change(12)
        price_vol = prices.pct_change().rolling(20).std()

        defense_signal = pd.DataFrame(index=signals.index, columns=signals.columns)
        defense_signal = defense_signal.fillna(1.0)

        return signals * defense_signal

    def get_positions(self):
        return self.positions

    def get_signals(self):
        return self.signals

    def get_portfolio_value(self):
        return self.portfolio_value