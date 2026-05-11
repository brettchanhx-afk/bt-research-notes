import numpy as np
import pandas as pd


def calculate_ma(close_prices, period):
    ma = pd.Series(close_prices).rolling(window=period, min_periods=1).mean()
    return ma.values


def calculate_ema(close_prices, period):
    ema = pd.Series(close_prices).ewm(span=period, adjust=False).mean()
    return ema.values


def calculate_rsi(close_prices, period=14):
    if len(close_prices) < period + 1:
        return np.array([50.0] * len(close_prices))
    
    deltas = np.diff(close_prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros_like(close_prices)
    avg_loss = np.zeros_like(close_prices)
    
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in range(period + 1, len(close_prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi[:period] = 50.0
    
    return rsi


def calculate_stddev(prices, period):
    stddev = pd.Series(prices).rolling(window=period, min_periods=1).std()
    return stddev.values


def calculate_atr(high_prices, low_prices, close_prices, period=14):
    tr_list = []
    
    for i in range(len(close_prices)):
        if i == 0:
            tr = high_prices[i] - low_prices[i]
        else:
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i-1])
            tr3 = abs(low_prices[i] - close_prices[i-1])
            tr = max(tr1, tr2, tr3)
        tr_list.append(tr)
    
    tr_array = np.array(tr_list)
    atr = calculate_ma(tr_array, period)
    
    return atr


def calculate_daily_returns(close_prices):
    returns = np.diff(close_prices) / close_prices[:-1]
    returns = np.insert(returns, 0, 0.0)
    return returns


def calculate_var(returns, confidence_level=0.05, window=250):
    if len(returns) < window:
        return 0.01
    
    window_returns = returns[-window:] if len(returns) > window else returns
    sorted_returns = np.sort(window_returns)
    var_index = int(len(sorted_returns) * confidence_level)
    var = -sorted_returns[var_index] if var_index > 0 else 0.01
    
    return var


def calculate_es(returns, confidence_level=0.05, window=250):
    if len(returns) < window:
        return 0.01
    
    window_returns = returns[-window:] if len(returns) > window else returns
    sorted_returns = np.sort(window_returns)
    var_index = int(len(sorted_returns) * confidence_level)
    
    if var_index == 0:
        return 0.01
    
    es = -np.mean(sorted_returns[:var_index])
    return es


def calculate_growth_rate(close_prices, n=21):
    if len(close_prices) < n + 1:
        return 0.0
    
    lc = close_prices[-n-1]
    c = close_prices[-1]
    
    if not np.isnan(lc) and not np.isnan(c) and lc != 0:
        return (c - lc) / lc * 100
    return 0.0


def calculate_volatility(close_prices, period=120):
    if len(close_prices) < period:
        return 0.0
    
    returns = calculate_daily_returns(close_prices)
    volatility = np.std(returns[-period:]) * 100 if len(returns) > period else np.std(returns) * 100
    
    return volatility


def calculate_premium_rate(close_prices, net_values):
    if len(close_prices) == 0 or len(net_values) == 0:
        return 0.0
    
    try:
        premium_rate = (close_prices[-1] - net_values[-1]) / net_values[-1] * 100
        return round(premium_rate, 2)
    except:
        return 0.0


class TechnicalIndicators:
    def __init__(self, df):
        self.df = df.copy()
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.volume = df['volume'].values if 'volume' in df.columns else None
        self.money = df['money'].values if 'money' in df.columns else None
        
        self.returns = calculate_daily_returns(self.close)
    
    def get_rsi(self, period=14):
        return calculate_rsi(self.close, period)
    
    def get_ma(self, period):
        return calculate_ma(self.close, period)
    
    def get_ema(self, period):
        return calculate_ema(self.close, period)
    
    def get_atr(self, period=14):
        return calculate_atr(self.high, self.low, self.close, period)
    
    def get_stddev(self, period):
        return calculate_stddev(self.close, period)
    
    def get_var(self, confidence_level=0.05, window=250):
        return calculate_var(self.returns, confidence_level, window)
    
    def get_es(self, confidence_level=0.05, window=250):
        return calculate_es(self.returns, confidence_level, window)
    
    def get_growth_rate(self, n=21):
        return calculate_growth_rate(self.close, n)
    
    def get_volatility(self, period=120):
        return calculate_volatility(self.close, period)
    
    def get_avg_money(self, period=120):
        if self.money is None:
            return 0
        if len(self.money) < period:
            return np.mean(self.money)
        return np.mean(self.money[-period:])
