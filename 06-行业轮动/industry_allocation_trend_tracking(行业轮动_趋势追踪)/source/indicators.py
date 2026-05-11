"""
趋势追踪指标计算模块
实现研报中的41个趋势追踪指标
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List

def MA(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(window=n).mean()

def EMA(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()

def SMA(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(window=n).mean()

def WMA(close: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1)
    return close.rolling(window=n).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)

def DEMA(close: pd.Series, n: int) -> pd.Series:
    ema1 = EMA(close, n)
    ema2 = EMA(ema1, n)
    return 2 * ema1 - ema2

def TMA(close: pd.Series, n: int) -> pd.Series:
    ma1 = MA(close, n)
    return MA(ma1, n)

def REG(close: pd.Series, n: int) -> pd.Series:
    result = pd.Series(index=close.index, dtype=float)
    for i in range(n - 1, len(close)):
        y = close.iloc[i - n + 1:i + 1].values
        x = np.arange(1, n + 1)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            result.iloc[i] = slope * n + intercept
    return result

def ROC(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(periods=n) * 100

def RET(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(periods=n) * 100

def SROC(close: pd.Series, n: int) -> pd.Series:
    roc = ROC(close, n)
    return EMA(roc, n)

def DPO(close: pd.Series, n: int) -> pd.Series:
    ma = MA(close, n)
    shift = n // 2 + 1
    return close - ma.shift(shift)

def VIDYA(close: pd.Series, n: int, fast: int = 10) -> pd.Series:
    result = pd.Series(index=close.index, dtype=float)
    sc = 2 / (fast + 1)
    for i in range(1, len(close)):
        mom = abs(close.iloc[i] - close.iloc[i - 1])
        mom_sum = mom.rolling(window=n).sum().iloc[i] if i >= n else mom.iloc[:i + 1].sum()
        k = sc * mom / mom_sum if mom_sum != 0 else 0
        if i == 1:
            result.iloc[i] = close.iloc[i]
        else:
            result.iloc[i] = k * close.iloc[i] + (1 - k) * result.iloc[i - 1]
    return result

def KST(close: pd.Series, n: int) -> pd.Series:
    roc1 = ROC(close, 10)
    roc2 = ROC(close, 15)
    roc3 = ROC(close, 20)
    roc4 = ROC(close, 30)
    ma1 = MA(roc1, 10)
    ma2 = MA(roc2, 10)
    ma3 = MA(roc3, 10)
    ma4 = MA(roc4, 15)
    return ma1 + ma2 * 2 + ma3 * 3 + ma4 * 4

def BIAS(close: pd.Series, n: int) -> pd.Series:
    ma = MA(close, n)
    return (close - ma) / ma * 100

def BIAS36(close: pd.Series) -> pd.Series:
    ma3 = MA(close, 3)
    ma6 = MA(close, 6)
    return (ma3 - ma6) / ma6 * 100

def BBI(close: pd.Series) -> pd.Series:
    bbi = (MA(close, 3) + MA(close, 6) + MA(close, 12) + MA(close, 24)) / 4
    return bbi

def CMO(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    gain = diff.where(diff > 0, 0).rolling(window=n).sum()
    loss = (-diff.where(diff < 0, 0)).rolling(window=n).sum()
    return 100 * (gain - loss) / (gain + loss)

def PSY(close: pd.Series, n: int) -> pd.Series:
    return (close.diff() > 0).rolling(window=n).sum() / n * 100

def POS(returns: pd.Series, n: int) -> pd.Series:
    return (returns.rolling(window=n).sum() > 0).astype(int) * 100

def TII(close: pd.Series, n: int, threshold: float) -> pd.Series:
    diff = close.diff()
    gain = diff.where(diff > 0, 0).rolling(window=n).sum()
    loss = (-diff.where(diff < 0, 0)).rolling(window=n).sum()
    tii = 100 * gain / (gain + loss)
    return (tii > threshold * 100).astype(int) * 100

def THRES_AVG(close: pd.Series, n: int, threshold: float) -> pd.Series:
    avg_return = close.pct_change().rolling(window=n).mean()
    return (avg_return > threshold).astype(int) * 100

def MASS(close: pd.Series, n: int, threshold: float) -> pd.Series:
    high_low = close.rolling(window=n).max() - close.rolling(window=n).min()
    ema1 = EMA(high_low, n)
    ema2 = EMA(ema1, n)
    mass = ema1 / ema2
    return (mass > threshold).astype(int) * 100

def UP2DOWN(returns: pd.Series, n: int, threshold: float = 0.0) -> pd.Series:
    up = (returns > 0).rolling(window=n).sum()
    down = (returns <= 0).rolling(window=n).sum()
    signal = (up - down) / n * 100
    return (signal > threshold * 100).astype(int) * 100

def OSC(close: pd.Series, n1: int, n2: int) -> pd.Series:
    ma1 = MA(close, n1)
    ma2 = MA(close, n2)
    return ma1 - ma2

def AVG_LINE(close: pd.Series, n: int) -> pd.Series:
    return close - MA(close, n)

def EXPMA(close: pd.Series, n: int) -> pd.Series:
    return EMA(close, n)

def HULLMA(close: pd.Series, n1: int, n2: int) -> pd.Series:
    hull = 2 * MA(close, n1) - MA(close, n2)
    return MA(hull, int(np.sqrt(n2)))

def ZLMACD(close: pd.Series, n1: int, n2: int) -> pd.Series:
    ema1 = EMA(close, n1)
    ema2 = EMA(close, n2)
    macd = ema1 - ema2
    signal = EMA(macd, n1)
    return macd - signal

def RSIH(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    gain = diff.where(diff > 0, 0).rolling(window=n).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=n).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def TSI(close: pd.Series, n1: int, n2: int) -> pd.Series:
    diff = close.diff()
    abs_diff = abs(diff)
    ema1 = EMA(diff, n1)
    ema2 = EMA(EMA(diff, n1), n2)
    ema_abs1 = EMA(abs_diff, n1)
    ema_abs2 = EMA(EMA(abs_diff, n1), n2)
    return 100 * ema2 / ema_abs2

def ROC_CHANGE(close: pd.Series, n1: int, n2: int) -> pd.Series:
    roc1 = ROC(close, n1)
    roc2 = ROC(close, n2)
    return roc1 - roc2

def MOM(close: pd.Series, n1: int, n2: int) -> pd.Series:
    mom1 = MA(close.pct_change(periods=n1) * 100, n2)
    return mom1

def EFFICIENCY(close: pd.Series, n: int) -> pd.Series:
    change = abs(close - close.shift(n))
    volatility = abs(close.diff()).rolling(window=n).sum()
    return (change / volatility).replace([np.inf, -np.inf], 0) * 100

def INVVOL(returns: pd.Series, n1: int, n2: int) -> pd.Series:
    vol = returns.rolling(window=n2).std()
    mom = MA(returns.rolling(window=n1).sum(), n2)
    return mom / vol

def SHARP_MOM(returns: pd.Series, n1: int, n2: int) -> pd.Series:
    mom = returns.rolling(window=n1).sum()
    vol = returns.rolling(window=n2).std()
    return mom / vol

def PMO(close: pd.Series, n1: int, n2: int, n3: int) -> pd.Series:
    pmo = (close.pct_change(periods=n1) * 100).ewm(span=n2, adjust=False).mean()
    signal = EMA(pmo, n3)
    return pmo - signal

def DBCD(close: pd.Series, n1: int, n2: int, n3: int) -> pd.Series:
    diff = close.diff()
    hist = EMA(diff, n1) - EMA(diff, n2)
    return EMA(hist, n3)

def MACD(close: pd.Series, n1: int, n2: int, n3: int) -> pd.Series:
    ema1 = EMA(close, n1)
    ema2 = EMA(close, n2)
    macd = ema1 - ema2
    signal = EMA(macd, n3)
    return macd - signal

def COPP(close: pd.Series, n1: int, n2: int, n3: int) -> pd.Series:
    copp = ((close - close.shift(n1)) / close.shift(n1) * 100 +
            (close.shift(-1) - close.shift(-n2 + 1)) / close.shift(-n2 + 1) * 100) / 2
    return EMA(copp, n3)

def PPO(close: pd.Series, n1: int, n2: int, n3: int) -> pd.Series:
    ema1 = EMA(close, n1)
    ema2 = EMA(close, n2)
    return ((ema1 - ema2) / ema2 * 100).ewm(span=n3, adjust=False).mean()

def DELTA(close: pd.Series, n1: int, n2: int, n3: int, n4: int) -> pd.Series:
    ma1 = MA(close, n1)
    ma2 = MA(close, n2)
    ma3 = MA(close, n3)
    ma4 = MA(close, n4)
    return (ma1 - ma2) + (ma3 - ma4)

class TrendIndicatorCalculator:
    def __init__(self):
        self.indicator_funcs = {
            'MA': {'func': MA, 'params': {'n': [20, 40, 60, 120]}},
            'EMA': {'func': EMA, 'params': {'n': [20, 40, 60, 120]}},
            'WMA': {'func': WMA, 'params': {'n': [20, 40, 60, 120]}},
            'SMA': {'func': SMA, 'params': {'n': [20, 40, 60, 120]}},
            'DEMA': {'func': DEMA, 'params': {'n': [20, 40, 60, 120]}},
            'TMA': {'func': TMA, 'params': {'n': [20, 40, 60, 120]}},
            'REG': {'func': REG, 'params': {'n': [20, 40, 60, 120]}},
            'ROC': {'func': ROC, 'params': {'n': [20, 40, 60, 120]}},
            'RET': {'func': RET, 'params': {'n': [20, 40, 60, 120]}},
            'SROC': {'func': SROC, 'params': {'n': [20, 40, 60, 120]}},
            'DPO': {'func': DPO, 'params': {'n': [20, 40, 60, 120]}},
            'KST': {'func': KST, 'params': {'n': [20, 40, 60, 120]}},
            'BIAS': {'func': BIAS, 'params': {'n': [20, 40, 60, 120]}},
            'BIAS36': {'func': BIAS36, 'params': {}},
            'BBI': {'func': BBI, 'params': {}},
            'CMO': {'func': CMO, 'params': {'n': [20, 40, 60, 120]}},
            'PSY': {'func': PSY, 'params': {'n': [20, 40, 60, 120]}},
            'POS': {'func': POS, 'params': {'n': [20, 40, 60, 120]}},
            'TII': {'func': TII, 'params': {'n': [20, 40, 60], 'threshold': [0.4, 0.5, 0.6]}},
            'THRES_AVG': {'func': THRES_AVG, 'params': {'n': [20, 40, 60], 'threshold': [0.4, 0.5, 0.6]}},
            'MASS': {'func': MASS, 'params': {'n': [20, 40, 60], 'threshold': [0.4, 0.5, 0.6]}},
            'UP2DOWN': {'func': UP2DOWN, 'params': {'n': [20, 40, 60], 'threshold': [0.4, 0.5, 0.6]}},
            'OSC': {'func': OSC, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'AVG_LINE': {'func': AVG_LINE, 'params': {'n': [20, 40, 60, 120]}},
            'EXPMA': {'func': EXPMA, 'params': {'n': [20, 40, 60, 120]}},
            'HULLMA': {'func': HULLMA, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'ZLMACD': {'func': ZLMACD, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'RSIH': {'func': RSIH, 'params': {'n': [20, 40, 60, 120]}},
            'TSI': {'func': TSI, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'ROC_CHANGE': {'func': ROC_CHANGE, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'MOM': {'func': MOM, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'EFFICIENCY': {'func': EFFICIENCY, 'params': {'n': [20, 40, 60, 120]}},
            'INVVOL': {'func': INVVOL, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'SHARP_MOM': {'func': SHARP_MOM, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250]}},
            'PMO': {'func': PMO, 'params': {'n1': [20, 40, 60], 'n2': [10], 'n3': [60, 120, 250]}},
            'DBCD': {'func': DBCD, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250], 'n3': [10]}},
            'MACD': {'func': MACD, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250], 'n3': [60]}},
            'COPP': {'func': COPP, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250], 'n3': [60]}},
            'PPO': {'func': PPO, 'params': {'n1': [20, 40, 60], 'n2': [60, 120, 250], 'n3': [60]}},
            'DELTA': {'func': DELTA, 'params': {'n1': [20], 'n2': [40], 'n3': [120], 'n4': [250]}},
        }

    def generate_signal(self, data: pd.DataFrame, indicator_name: str, params: dict) -> pd.Series:
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        returns = data['returns'] if 'returns' in data.columns else close.pct_change()

        if indicator_name in ['POS', 'UP2DOWN']:
            return self.indicator_funcs[indicator_name]['func'](returns, **params)
        elif indicator_name in ['INVVOL', 'SHARP_MOM']:
            return self.indicator_funcs[indicator_name]['func'](returns, **params)
        else:
            return self.indicator_funcs[indicator_name]['func'](close, **params)

    def generate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = {}
        for name, config in self.indicator_funcs.items():
            params = config['params']
            if not params:
                signals[name] = config['func'](data['close'] if 'close' in data.columns else data.iloc[:, 0])
            else:
                param_names = list(params.keys())
                if len(param_names) == 1:
                    for p1 in params[param_names[0]]:
                        signals[f"{name}_{p1}"] = self.generate_signal(data, name, {param_names[0]: p1})
                elif len(param_names) == 2:
                    for p1 in params[param_names[0]]:
                        for p2 in params[param_names[1]]:
                            signals[f"{name}_{p1}_{p2}"] = self.generate_signal(data, name, {param_names[0]: p1, param_names[1]: p2})
                elif len(param_names) == 3:
                    for p1 in params[param_names[0]]:
                        for p2 in params[param_names[1]]:
                            for p3 in params[param_names[2]]:
                                signals[f"{name}_{p1}_{p2}_{p3}"] = self.generate_signal(data, name, {param_names[0]: p1, param_names[1]: p2, param_names[2]: p3})
                elif len(param_names) == 4:
                    for p1 in params[param_names[0]]:
                        for p2 in params[param_names[1]]:
                            for p3 in params[param_names[2]]:
                                for p4 in params[param_names[3]]:
                                    signals[f"{name}_{p1}_{p2}_{p3}_{p4}"] = self.generate_signal(data, name, {param_names[0]: p1, param_names[1]: p2, param_names[2]: p3, param_names[3]: p4})
        return pd.DataFrame(signals)

if __name__ == "__main__":
    print("测试趋势追踪指标计算...")
    calculator = TrendIndicatorCalculator()
    print(f"共实现 {len(calculator.indicator_funcs)} 个趋势追踪指标")