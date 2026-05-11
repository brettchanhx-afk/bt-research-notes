import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self, initial_capital=1000000, commission_rate=0.0003, slippage=0.0001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

        self.trades = []
        self.equity_curve = []
        self.positions = {}

    def reset(self):
        self.trades = []
        self.equity_curve = []
        self.positions = {}

    def run_backtest(self, signals, price_data, rebalance_dates=None):
        self.reset()

        if signals is None or len(signals) == 0:
            return self._create_results()

        if price_data is None or len(price_data) == 0:
            return self._create_results()

        unique_dates = sorted(signals['date'].unique()) if 'date' in signals.columns else []

        if len(unique_dates) == 0:
            return self._create_results()

        current_capital = self.initial_capital
        current_positions = {}

        for i, date in enumerate(unique_dates):
            date_signals = signals[signals['date'] == date]

            if len(date_signals) == 0:
                continue

            long_codes = date_signals[date_signals['signal'] == 1]['industry_code'].tolist()
            short_codes = date_signals[date_signals['signal'] == -1]['industry_code'].tolist()

            for code in list(current_positions.keys()):
                if code not in long_codes and code not in short_codes:
                    self._close_position(code, date, price_data, current_positions)
                    del current_positions[code]

            n_long = len(long_codes)
            n_short = len(short_codes)

            if n_long > 0:
                long_weight = 1.0 / n_long
                for code in long_codes:
                    if code not in current_positions:
                        self._open_position(code, date, 'long', long_weight, price_data, current_capital)
                        current_positions[code] = {'direction': 'long', 'weight': long_weight}

            if n_short > 0:
                short_weight = 1.0 / n_short
                for code in short_codes:
                    if code not in current_positions:
                        self._open_position(code, date, 'short', short_weight, price_data, current_capital)
                        current_positions[code] = {'direction': 'short', 'weight': short_weight}

            portfolio_value = self._calculate_portfolio_value(current_positions, date, price_data, current_capital)

            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': current_capital,
                'position_value': portfolio_value - current_capital
            })

        return self._create_results()

    def _open_position(self, code, date, direction, weight, price_data, capital):
        price = self._get_price(code, date, price_data)

        if price == 0:
            return

        position_value = capital * weight
        shares = position_value / price

        commission = position_value * self.commission_rate
        slippage_cost = position_value * self.slippage

        total_cost = position_value + commission + slippage_cost

        self.trades.append({
            'date': date,
            'code': code,
            'direction': direction,
            'action': 'open',
            'price': price,
            'shares': shares,
            'value': position_value,
            'commission': commission,
            'slippage': slippage_cost
        })

    def _close_position(self, code, date, price_data, positions):
        if code not in positions:
            return

        position = positions[code]
        price = self._get_price(code, date, price_data)

        if price == 0:
            return

        position_value = positions[code]['weight'] * self.initial_capital
        shares = position_value / price

        commission = position_value * self.commission_rate
        slippage_cost = position_value * self.slippage

        total_proceeds = position_value - commission - slippage_cost

        self.trades.append({
            'date': date,
            'code': code,
            'direction': position['direction'],
            'action': 'close',
            'price': price,
            'shares': shares,
            'value': position_value,
            'commission': commission,
            'slippage': slippage_cost,
            'proceeds': total_proceeds
        })

    def _get_price(self, code, date, price_data):
        date_str = pd.to_datetime(date).strftime('%Y%m%d') if isinstance(date, str) else date

        price_df = price_data[price_data['industry_code'] == code]

        if len(price_df) == 0:
            return 0

        date_prices = price_df[price_df['trade_date'] == date]

        if len(date_prices) > 0:
            return date_prices['close'].values[0]

        return 0

    def _calculate_portfolio_value(self, positions, date, price_data, cash):
        total_value = cash

        for code, pos_info in positions.items():
            price = self._get_price(code, date, price_data)

            if price > 0:
                position_value = pos_info['weight'] * self.initial_capital

                if pos_info['direction'] == 'long':
                    total_value += position_value
                else:
                    total_value -= position_value

        return total_value

    def _create_results(self):
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'initial_capital': self.initial_capital,
            'final_value': equity_df['portfolio_value'].iloc[-1] if len(equity_df) > 0 else self.initial_capital
        }


class PortfolioOptimizer:
    def __init__(self):
        self.optimal_weights = {}

    def optimize_weights(self, returns_data, method='equal_weight'):
        if returns_data is None or len(returns_data) == 0:
            return {}

        if method == 'equal_weight':
            codes = returns_data['industry_code'].unique()
            n = len(codes)
            if n == 0:
                return {}
            weight = 1.0 / n
            return {code: weight for code in codes}

        elif method == 'risk_parity':
            return self._risk_parity(returns_data)

        elif method == 'mean_variance':
            return self._mean_variance(returns_data)

        return {}

    def _risk_parity(self, returns_data):
        codes = returns_data['industry_code'].unique()

        volatilities = {}
        for code in codes:
            code_returns = returns_data[returns_data['industry_code'] == code]['return']
            volatilities[code] = code_returns.std() if len(code_returns) > 0 else 1

        total_vol = sum(1.0 / v for v in volatilities.values())

        weights = {}
        for code in codes:
            weights[code] = (1.0 / volatilities[code]) / total_vol if volatilities[code] != 0 else 0

        return weights

    def _mean_variance(self, returns_data):
        codes = returns_data['industry_code'].unique()

        mean_returns = {}
        cov_matrix = returns_data.pivot_table(values='return', index='trade_date', columns='industry_code')

        if len(cov_matrix) == 0:
            return {code: 1.0/len(codes) for code in codes if len(codes) > 0}

        try:
            inv_cov = np.linalg.inv(cov_matrix.cov())
            ones = np.ones(len(codes))

            weights = inv_cov.dot(ones) / (ones.T.dot(inv_cov).dot(ones))

            return {code: weights[i] for i, code in enumerate(codes)}
        except:
            return {code: 1.0/len(codes) for code in codes if len(codes) > 0}


if __name__ == "__main__":
    print("Testing BacktestEngine...")

    engine = BacktestEngine(initial_capital=1000000)
    print(f"BacktestEngine initialized with capital: {engine.initial_capital}")

    optimizer = PortfolioOptimizer()
    print("PortfolioOptimizer initialized")

    print("Backtest module test completed!")
