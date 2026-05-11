import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from source.strategy import ETFRotationStrategy
from source.config import BACKTEST_CONFIG, STRATEGY_CONFIG, BOND_ETF


class Position:
    def __init__(self, code, amount, avg_cost):
        self.code = code
        self.amount = amount
        self.avg_cost = avg_cost


class BacktestEngine:
    def __init__(self, etf_data_dict, trading_dates, strategy, config=None):
        self.etf_data_dict = etf_data_dict
        self.trading_dates = sorted(trading_dates)
        self.strategy = strategy
        self.config = config or BACKTEST_CONFIG.copy()
        
        self.reset()
    
    def reset(self):
        self.initial_cash = self.config['initial_cash']
        self.cash = self.initial_cash
        self.current_positions = {}
        self.total_value = self.initial_cash
        self.equity_curve = []
        self.positions_history = []
        self.trades = []
        self.daily_injection = self.config['daily_injection']
        self.total_injected = 0
        
        self.strategy.context.reset()
        self.strategy.last_rebalance_date = None
    
    def get_price(self, code, date):
        if code not in self.etf_data_dict:
            return np.nan
        df = self.etf_data_dict[code]
        if date not in df.index:
            return np.nan
        return df.loc[date, 'close']
    
    def calculate_commission(self, amount, price):
        value = amount * price
        commission = value * self.config['commission']
        if commission > 0 and commission < self.config['min_commission']:
            commission = self.config['min_commission']
        return commission
    
    def apply_slippage(self, price, is_buy):
        if is_buy:
            return price * (1 + self.config['slippage'])
        else:
            return price * (1 - self.config['slippage'])
    
    def execute_order(self, code, target_value, date):
        current_price = self.get_price(code, date)
        if np.isnan(current_price):
            return 0
        
        current_amount = self.current_positions.get(code, Position(code, 0, 0)).amount
        current_value = current_amount * current_price
        delta_value = target_value - current_value
        
        if abs(delta_value) < STRATEGY_CONFIG['min_amount'] and target_value > 0:
            return 0
        
        is_buy = delta_value > 0
        trade_price = self.apply_slippage(current_price, is_buy)
        
        if target_value == 0 and current_amount > 0:
            amount_to_trade = current_amount
            commission = self.calculate_commission(amount_to_trade, trade_price)
            proceeds = amount_to_trade * trade_price - commission
            self.cash += proceeds
            del self.current_positions[code]
            
            self.trades.append({
                'date': date,
                'code': code,
                'amount': -amount_to_trade,
                'price': trade_price,
                'value': -amount_to_trade * trade_price,
                'commission': commission
            })
            return -amount_to_trade
        
        if target_value > 0:
            new_amount = int(target_value / trade_price)
            amount_to_trade = new_amount - current_amount
            
            if amount_to_trade == 0:
                return 0
            
            trade_value = abs(amount_to_trade) * trade_price
            commission = self.calculate_commission(abs(amount_to_trade), trade_price)
            
            if amount_to_trade > 0:
                if self.cash < trade_value + commission:
                    available = self.cash - commission
                    amount_to_trade = int(available / trade_price)
                    if amount_to_trade <= 0:
                        return 0
                self.cash -= (amount_to_trade * trade_price + commission)
                
                if code in self.current_positions:
                    old_pos = self.current_positions[code]
                    total_amount = old_pos.amount + amount_to_trade
                    total_cost = old_pos.amount * old_pos.avg_cost + amount_to_trade * trade_price
                    avg_cost = total_cost / total_amount
                    self.current_positions[code] = Position(code, total_amount, avg_cost)
                else:
                    self.current_positions[code] = Position(code, amount_to_trade, trade_price)
            else:
                proceeds = abs(amount_to_trade) * trade_price - commission
                self.cash += proceeds
                if code in self.current_positions:
                    new_amount = self.current_positions[code].amount + amount_to_trade
                    if new_amount > 0:
                        self.current_positions[code].amount = new_amount
                    else:
                        del self.current_positions[code]
            
            self.trades.append({
                'date': date,
                'code': code,
                'amount': amount_to_trade,
                'price': trade_price,
                'value': amount_to_trade * trade_price,
                'commission': commission
            })
            return amount_to_trade
        
        return 0
    
    def update_total_value(self, date):
        position_value = 0
        for code, pos in self.current_positions.items():
            price = self.get_price(code, date)
            if not np.isnan(price):
                position_value += pos.amount * price
        self.total_value = position_value + self.cash
        return self.total_value
    
    def rebalance(self, target_weights, date):
        total_value = self.total_value
        
        current_weights = {}
        for code, pos in self.current_positions.items():
            price = self.get_price(code, date)
            if not np.isnan(price):
                current_weights[code] = (pos.amount * price) / total_value
        
        trade_order = []
        for code in current_weights:
            if code not in target_weights:
                trade_order.append((code, 0.0))
            else:
                trade_order.append((code, target_weights[code]))
        
        for code in target_weights:
            if code not in current_weights:
                trade_order.append((code, target_weights[code]))
        
        trade_order.sort(key=lambda x: x[1])
        
        for code, weight in trade_order:
            target_value = weight * total_value
            self.execute_order(code, target_value, date)
        
        self.update_total_value(date)
    
    def run(self):
        self.reset()
        
        for idx, date in enumerate(self.trading_dates):
            self.cash += self.daily_injection
            self.total_injected += self.daily_injection
            
            available_data = {}
            for code, df in self.etf_data_dict.items():
                if date in df.index:
                    df_slice = df.loc[:date].copy()
                    if len(df_slice) >= STRATEGY_CONFIG['reference_cycle']:
                        available_data[code] = df_slice
            
            if len(available_data) == 0:
                self.update_total_value(date)
                self.equity_curve.append({
                    'date': date,
                    'total_value': self.total_value,
                    'cash': self.cash
                })
                continue
            
            current_positions_dict = {code: True for code in self.current_positions}
            
            if self.strategy.should_rebalance(date):
                target_weights = self.strategy.calculate_target_positions(
                    available_data, date, current_positions_dict
                )
                
                bond_weight = max(0, 1 - sum(target_weights.values()))
                if bond_weight > 0 and BOND_ETF in available_data:
                    target_weights[BOND_ETF] = bond_weight
                
                self.rebalance(target_weights, date)
                self.strategy.update_rebalance_date(date)
            else:
                self.update_total_value(date)
            
            pos_snapshot = {
                'date': date,
                'cash': self.cash,
                'total_value': self.total_value,
                'positions': {code: pos.amount for code, pos in self.current_positions.items()}
            }
            self.positions_history.append(pos_snapshot)
            
            self.equity_curve.append({
                'date': date,
                'total_value': self.total_value,
                'cash': self.cash
            })
        
        return self.get_results()
    
    def get_results(self):
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty and 'date' in equity_df.columns:
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df = equity_df.set_index('date')
        
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty and 'date' in trades_df.columns:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        return {
            'equity': equity_df,
            'trades': trades_df,
            'positions_history': self.positions_history,
            'initial_cash': self.initial_cash,
            'total_injected': self.total_injected,
            'final_value': self.total_value
        }
