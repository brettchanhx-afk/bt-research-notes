import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CrowdednessIndicator:
    def __init__(self):
        self.thresholds = {
            'comp_turn_kurtosis_10': 0.95,
            'comp_turn_kurtosis_5': 0.95,
            'turn_20': 0.90,
            'turn_10': 0.90,
            'turn_40': 0.90,
            'corr_amount_close_40': 0.95,
            'corr_amount_close_60': 0.95,
        }

    def calculate_momentum_indicators(self, price_data: pd.DataFrame, windows=[5, 10, 20, 40]) -> pd.DataFrame:
        close = price_data['close']
        results = pd.DataFrame(index=close.index)
        for w in windows:
            ret = close.pct_change(w)
            results[f'momentum_{w}'] = ret
            results[f'sharpe_momentum_{w}'] = ret / ret.rolling(w).std()
            results[f'bias_momentum_{w}'] = (close - close.rolling(w).mean()) / close.rolling(w).std()
        return results

    def calculate_liquidity_indicators(self, price_data: pd.DataFrame, windows=[10, 20, 40, 60]) -> pd.DataFrame:
        volume = price_data['volume']
        results = pd.DataFrame(index=volume.index)
        for w in windows:
            results[f'turn_{w}'] = volume.rolling(w).mean() / volume.iloc[:w].mean() if len(volume) > w else volume.rolling(w).mean()
            results[f'turn_ma{w}'] = volume.rolling(w).mean()
        return results

    def calculate_volume_price_correlation(self, price_data: pd.DataFrame, windows=[20, 40, 60]) -> pd.DataFrame:
        close = price_data['close']
        amount = price_data['amount']
        volume = price_data['volume']
        results = pd.DataFrame(index=close.index)
        for w in windows:
            close_ret = close.pct_change(w)
            amount_ret = amount.pct_change(w)
            volume_ret = volume.pct_change(w)
            results[f'corr_amount_close_{w}'] = close_ret.rolling(w).corr(amount_ret)
            results[f'corr_volume_close_{w}'] = close_ret.rolling(w).corr(volume_ret)
            turn = volume.rolling(w).mean()
            results[f'corr_turn_close_{w}'] = close_ret.rolling(w).corr(turn)
        return results

    def calculate_volatility_indicators(self, price_data: pd.DataFrame, windows=[10, 20, 40, 60]) -> pd.DataFrame:
        close = price_data['close']
        ret = close.pct_change()
        results = pd.DataFrame(index=close.index)
        for w in windows:
            rolling_ret = ret.rolling(w)
            results[f'vol_{w}'] = rolling_ret.std() * np.sqrt(252)
            def neg_std(x):
                return x[x < 0].std() * np.sqrt(252) if len(x) > 0 else np.nan
            results[f'downvol_{w}'] = ret.rolling(w).apply(neg_std, raw=True)
            if len(ret) > w * 2:
                kurt = ret.rolling(w).apply(lambda x: pd.Series(x).kurt() if len(x) > 3 else np.nan, raw=True)
            else:
                kurt = np.nan
            results[f'kurtosis_{w}'] = kurt
        return results

    def calculate_constituent_stock_indicators(self, constituent_returns: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
        results = pd.DataFrame(index=constituent_returns.index)
        for w in windows:
            ret_mean = constituent_returns.rolling(w).mean()
            ret_std = constituent_returns.rolling(w).std()
            def kurt_func(x):
                return pd.Series(x).kurt() if len(x) > 3 else np.nan
            results[f'comp_ret_kurtosis_{w}'] = ret_mean.rolling(w).apply(kurt_func, raw=True)
            results[f'comp_ret_std_{w}'] = ret_std.mean(axis=1)
            turn_mean = constituent_returns.rolling(w).mean()
            results[f'comp_turn_kurtosis_{w}'] = turn_mean.rolling(w).apply(kurt_func, raw=True)
        return results

    def calculate_all_indicators(self, price_data: pd.DataFrame, constituent_returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        all_indicators = pd.DataFrame(index=price_data.index)
        momentum = self.calculate_momentum_indicators(price_data)
        all_indicators = pd.concat([all_indicators, momentum], axis=1)
        liquidity = self.calculate_liquidity_indicators(price_data)
        all_indicators = pd.concat([all_indicators, liquidity], axis=1)
        correlation = self.calculate_volume_price_correlation(price_data)
        all_indicators = pd.concat([all_indicators, correlation], axis=1)
        volatility = self.calculate_volatility_indicators(price_data)
        all_indicators = pd.concat([all_indicators, volatility], axis=1)
        if constituent_returns is not None:
            constituent = self.calculate_constituent_stock_indicators(constituent_returns)
            all_indicators = pd.concat([all_indicators, constituent], axis=1)
        return all_indicators

    def calculate_percentile(self, indicator_series: pd.Series, window=252) -> pd.Series:
        rolling_max = indicator_series.rolling(window, min_periods=60).max()
        rolling_min = indicator_series.rolling(window, min_periods=60).min()
        percentile = (indicator_series - rolling_min) / (rolling_max - rolling_min + 1e-10)
        return percentile.fillna(0.5)

    def calculate_composite_crowdedness(self, indicators: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=indicators.index)
        key_indicators = ['comp_turn_kurtosis_10', 'turn_20', 'corr_amount_close_40']
        available_indicators = [ind for ind in key_indicators if ind in indicators.columns]
        if not available_indicators:
            available_indicators = ['turn_20', 'corr_amount_close_40'] if 'turn_20' in indicators.columns else list(indicators.columns[:3])
        for ind in available_indicators:
            if ind in indicators.columns:
                result[ind] = self.calculate_percentile(indicators[ind])
        result['composite_crowdedness'] = result[available_indicators].max(axis=1)
        return result

    def is_crowded(self, composite_crowdedness: pd.Series, date: pd.Timestamp) -> bool:
        if date not in composite_crowdedness.index:
            return False
        value = composite_crowdedness.loc[date]
        return value > 0.9

class ThresholdRegressionValidator:
    def __init__(self):
        self.passing_indicators = []

    def validate_indicator(self, indicator: pd.Series, forward_returns: pd.Series, thresholds=np.arange(0.5, 1.0, 0.05)) -> Dict:
        results = {
            'coefficient_trend': [],
            'pvalue_below_01': [],
            'win_rate_trend': [],
            'median_return_trend': []
        }
        for thresh in thresholds:
            high_indicator = indicator > thresh
            if high_indicator.sum() < 20:
                continue
            high_returns = forward_returns[high_indicator]
            low_returns = forward_returns[~high_indicator]
            if len(high_returns) < 5 or len(low_returns) < 5:
                continue
            try:
                coef = np.polyfit([0, 1], [low_returns.mean(), high_returns.mean()], 1)[0]
                results['coefficient_trend'].append((thresh, coef))
            except:
                pass
            pvalue = stats.ttest_1samp(high_returns, 0)[1] if len(high_returns) > 0 else 1
            results['pvalue_below_01'].append((thresh, pvalue < 0.1 and high_returns.mean() < 0))
            win_rate = (high_returns < 0).mean()
            results['win_rate_trend'].append((thresh, win_rate))
            median_ret = high_returns.median()
            results['median_return_trend'].append((thresh, median_ret))
        return results

    def check_trend_decreasing(self, values: list) -> bool:
        if len(values) < 3:
            return True
        trends = np.diff(values)
        return sum(trends) < 0

    def check_significant_negative(self, pvalue_list: list) -> float:
        significant_count = sum(1 for _, is_sig in pvalue_list if is_sig)
        return significant_count / len(pvalue_list) if pvalue_list else 0

    def is_valid_indicator(self, validation_results: Dict) -> Tuple[bool, str]:
        if not validation_results['coefficient_trend']:
            return False, "No valid threshold data"
        coeffs = [c for _, c in validation_results['coefficient_trend']]
        if not self.check_trend_decreasing(coeffs):
            return False, "Coefficient trend not decreasing"
        sig_ratio = self.check_significant_negative(validation_results['pvalue_below_01'])
        if sig_ratio < 0.5:
            return False, f"Only {sig_ratio:.1%} significant negative returns"
        return True, "Valid"

def calculate_forward_returns(price_data: pd.DataFrame, windows=[20]) -> Dict[int, pd.Series]:
    close = price_data['close']
    forward_returns = {}
    for w in windows:
        forward_returns[w] = close.shift(-w) / close - 1
    return forward_returns

def generate_crowdedness_signals(industry_data: Dict[str, pd.DataFrame], forward_window=20) -> pd.DataFrame:
    all_signals = {}
    for code, data in industry_data.items():
        try:
            indicator = CrowdednessIndicator()
            all_indicators = pd.DataFrame(index=data.index)

            close = data['close']
            amount = data['amount']
            volume = data['volume']

            all_indicators['turn_20'] = volume.rolling(20).mean() / volume.iloc[:20].mean() if len(volume) > 20 else volume.rolling(20).mean()

            close_ret = close.pct_change(40)
            amount_ret = amount.pct_change(40)
            all_indicators['corr_amount_close_40'] = close_ret.rolling(40).corr(amount_ret)

            composite = indicator.calculate_composite_crowdedness(all_indicators)
            signals = composite > 0.9
            all_signals[code] = signals['composite_crowdedness']
        except Exception as e:
            print(f"  {code} 拥挤度计算失败: {e}")
            continue
    signals_df = pd.DataFrame(all_signals)
    return signals_df

if __name__ == "__main__":
    print("拥挤度指标模块测试...")
    indicator = CrowdednessIndicator()
    print(f"预设阈值: {indicator.thresholds}")