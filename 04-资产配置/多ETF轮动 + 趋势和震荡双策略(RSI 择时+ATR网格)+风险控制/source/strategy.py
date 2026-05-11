import numpy as np
import pandas as pd
from datetime import datetime
from source.indicators import (
    TechnicalIndicators, calculate_rsi, calculate_ma, calculate_atr, calculate_stddev
)
from source.config import STRATEGY_CONFIG, INDICATOR_CONFIG


class StrategyContext:
    def __init__(self, etf_pool, config):
        self.etf_pool = etf_pool
        self.config = config
        self.reset()
    
    def reset(self):
        self.init_price_dict = {code: 0 for code in self.etf_pool}
        self.position_dict = {code: 0 for code in self.etf_pool}
        self.transaction_record = {code: 0 for code in self.etf_pool}
        self.min_position = 0


class ETFRotationStrategy:
    def __init__(self, etf_pool, config=None):
        self.config = config or STRATEGY_CONFIG.copy()
        self.etf_pool = etf_pool
        self.context = StrategyContext(etf_pool, self.config)
        self.last_rebalance_date = None
    
    def calculate_position_score(self, etf_code, etf_data, indicators):
        gr = indicators.get_growth_rate(n=21)
        es = indicators.get_es(confidence_level=self.config['confidence_level'], 
                           window=self.config['reference_cycle'])
        var = indicators.get_var(confidence_level=self.config['confidence_level'], 
                            window=self.config['reference_cycle'])
        stock_ratio = self.etf_pool.get(etf_code, 1)
        
        es = es if es != 0 else 0.000001
        var = var if var != 0 else 0.000001
        
        position = (1.0 / es) * (1.0 / var) * (1.02 ** gr) * stock_ratio * 10
        return round(position, 3)
    
    def calculate_target_positions(self, etf_data_dict, current_date, current_positions):
        etf_pool = list(etf_data_dict.keys())
        position_dict = {}
        gr_dict = {}
        es_dict = {}
        var_dict = {}
        
        max_es = -1
        max_var = -1
        
        for code in etf_pool:
            df = etf_data_dict[code]
            indicators = TechnicalIndicators(df)
            
            es = indicators.get_es(confidence_level=self.config['confidence_level'], 
                               window=self.config['reference_cycle'])
            var = indicators.get_var(confidence_level=self.config['confidence_level'], 
                                window=self.config['reference_cycle'])
            gr = indicators.get_growth_rate(n=21)
            
            es_dict[code] = es
            var_dict[code] = var
            gr_dict[code] = gr
            
            if es > max_es:
                max_es = es
            if var > max_var:
                max_var = var
        
        for code in etf_pool:
            df = etf_data_dict[code]
            indicators = TechnicalIndicators(df)
            
            es = es_dict[code]
            var = var_dict[code]
            gr = gr_dict[code]
            stock_ratio = self.etf_pool.get(code, 1)
            
            es = es if es != 0 else 0.000001
            var = var if var != 0 else 0.000001
            
            position = (max_es / es) * (max_var / var) * (1.02 ** gr) * stock_ratio * 10
            position = round(position, 3)
            
            if self.context.transaction_record[code] == 0:
                self.context.transaction_record[code] = position
            else:
                position = position * 0.8 + self.context.transaction_record[code] * 0.2
                self.context.transaction_record[code] = position
            
            position_dict[code] = position
        
        position_sorted = sorted(position_dict.items(), key=lambda x: x[1], reverse=True)
        
        stock_in = None
        stock_out = None
        do_change = False
        
        if len(position_sorted) > self.config['stock_count']:
            stock_in = position_sorted[self.config['stock_count'] - 1][0]
            stock_out = position_sorted[self.config['stock_count']][0]
            do_change = (stock_in not in current_positions and stock_out in current_positions)
        
        position_dict = {}
        idx = 1
        
        for code, value in position_sorted:
            gr = gr_dict[code]
            
            if self.context.init_price_dict[code] == 0 and abs(gr) >= self.config['max_range']:
                self.context.init_price_dict[code] = 1
            elif self.context.init_price_dict[code] == 1 and abs(gr) < self.config['consolidation']:
                self.context.init_price_dict[code] = 0
            
            if self.context.init_price_dict[code] == 1:
                position_dict[code] = 0
            else:
                if do_change and (code == stock_in or code == stock_out):
                    if code == stock_in:
                        position_dict[code] = 0
                    elif code == stock_out:
                        position_dict[code] = value
                elif idx <= self.config['stock_count']:
                    position_dict[code] = value
                else:
                    position_dict[code] = 0
            idx += 1
        
        total_position = sum(position_dict.values())
        if total_position == 0:
            total_position = 1
        
        ratio = {}
        for code in etf_pool:
            stock_position = position_dict.get(code, 0)

            df = etf_data_dict[code]
            indicators = TechnicalIndicators(df)
            rsi_array = indicators.get_rsi(period=self.config.get('rsi_period', 14))
            rsi = rsi_array[-1] if len(rsi_array) > 0 else 50

            if rsi > 30 and (stock_position - self.context.min_position) < 0:
                stock_position = 0
            
            if stock_position > 0:
                ratio[code] = round(stock_position / total_position, 3)
            else:
                ratio[code] = 0
        
        for code in etf_pool:
            if ratio[code] < self.config['min_ratio'] and ratio[code] > 0:
                ratio[code] = 0
            elif ratio[code] > self.config['max_ratio']:
                ratio[code] = self.config['max_ratio']
        
        adjustment = round(self.config['stock_count'] * 1.0 / 2 / 100, 2)
        for idx, (code, _) in enumerate(position_sorted):
            if ratio.get(code, 0) > 0:
                ratio[code] = adjustment + ratio[code]
            adjustment -= 0.01
        
        sum_ratio = sum(ratio.values())
        if sum_ratio > 1:
            ratio = {code: r / sum_ratio for code, r in ratio.items()}
        
        return ratio
    
    def check_buy_sell_signal(self, etf_code, etf_data):
        df = etf_data
        if len(df) < 120:
            return ""
        
        indicators = TechnicalIndicators(df)
        close = indicators.close
        high = indicators.high
        low = indicators.low
        
        rsi = indicators.get_rsi(period=INDICATOR_CONFIG['rsi_period'])
        rsi_avg = calculate_ma(rsi, INDICATOR_CONFIG['rsi_avg_period'])
        rsi_std = calculate_stddev(rsi, INDICATOR_CONFIG['rsi_avg_period'])
        atr = indicators.get_atr(period=INDICATOR_CONFIG['atr_period'])
        
        buy_threshold = rsi_avg[-1] - 1.4 * rsi_std[-1]
        buy_threshold = min(buy_threshold, 30)
        sell_threshold = rsi_avg[-1] + 1.4 * rsi_std[-1]
        sell_threshold = max(sell_threshold, 70)
        
        signal = ""
        if self.stop_loss_check(close):
            signal = "禁止操作"
        elif rsi[-1] > sell_threshold:
            signal = "RSI超买卖出"
        elif rsi[-1] < buy_threshold:
            signal = "RSI超卖买入"
        elif close[-1] > close[-2] + 3 * atr[-1]:
            signal = "ATR突破卖出"
        elif close[-1] < close[-2] - 3 * atr[-1]:
            signal = "ATR突破买入"
        
        return signal
    
    @staticmethod
    def stop_loss_check(close_prices, lag=2, loss=2, more=4):
        if len(close_prices) < lag + 1:
            return False
        rate = abs((close_prices[-1] - close_prices[-lag-1]) / close_prices[-lag-1]) * 100
        return loss < rate < more
    
    def should_rebalance(self, current_date):
        if self.last_rebalance_date is None:
            return True
        
        days_since = (current_date - self.last_rebalance_date).days
        if days_since < 30:
            return False
        
        if current_date.weekday() == self.config['weekday']:
            return True
        
        return False
    
    def update_rebalance_date(self, date):
        self.last_rebalance_date = date
