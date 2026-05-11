"""
CSCV过拟合检验模块
组合对称交叉验证框架计算策略过拟合概率
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.stats import norm

class CSCVTest:
    def __init__(self, n_splits: int = 100):
        self.n_splits = n_splits

    def split_data(self, returns: pd.Series) -> List[Tuple[pd.Series, pd.Series]]:
        """
        将数据分割为测试集和训练集
        采用组合对称交叉验证方式
        """
        n = len(returns)
        if n < 10:
            return []

        splits = []
        for _ in range(self.n_splits):
            indices = np.random.permutation(n)
            split_point = n // 2
            train_idx = indices[:split_point]
            test_idx = indices[split_point:]

            train_returns = returns.iloc[train_idx]
            test_returns = returns.iloc[test_idx]

            splits.append((train_returns, test_returns))

        return splits

    def calculate_oos_sharpe(self, returns: pd.Series) -> float:
        """
        计算样本外夏普比率
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
        return sharpe

    def calculate_is_sharpe(self, returns: pd.Series) -> float:
        """
        计算样本内夏普比率
        """
        return self.calculate_oos_sharpe(returns)

    def compute_overfitting_probability(self, returns: pd.Series) -> float:
        """
        计算过拟合概率
        P(Sharpe_IS > 0 AND Sharpe_OOS < 0) / P(Sharpe_IS > 0)
        """
        splits = self.split_data(returns)

        if len(splits) == 0:
            return 0.5

        positive_is_count = 0
        negative_oos_count = 0

        for train_returns, test_returns in splits:
            is_sharpe = self.calculate_is_sharpe(train_returns)
            oos_sharpe = self.calculate_oos_sharpe(test_returns)

            if is_sharpe > 0:
                positive_is_count += 1
                if oos_sharpe < 0:
                    negative_oos_count += 1

        if positive_is_count == 0:
            return 0.5

        overfitting_prob = negative_oos_count / positive_is_count
        return overfitting_prob

    def run_cscv_analysis(self, signals: pd.Series,
                          prices: pd.Series) -> dict:
        """
        运行完整的CSCV分析
        """
        returns = prices.pct_change().dropna()
        aligned_signals = signals.reindex(returns.index, method='ffill')

        strategy_returns = returns * aligned_signals.shift(1)
        strategy_returns = strategy_returns.dropna()

        splits = self.split_data(strategy_returns)

        is_sharpes = []
        oos_sharpes = []
        overfitting_probs = []

        for train_returns, test_returns in splits:
            is_sharpe = self.calculate_is_sharpe(train_returns)
            oos_sharpe = self.calculate_oos_sharpe(test_returns)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

        overfitting_prob = self.compute_overfitting_probability(strategy_returns)

        return {
            'is_sharpe_mean': np.mean(is_sharpes),
            'is_sharpe_std': np.std(is_sharpes),
            'oos_sharpe_mean': np.mean(oos_sharpes),
            'oos_sharpe_std': np.std(oos_sharpes),
            'overfitting_probability': overfitting_prob,
            'selection_sharpe': self.calculate_oos_sharpe(strategy_returns)
        }

class OverfittingDetector:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def filter_indicators(self, indicator_results: pd.DataFrame,
                          cscv_results: dict) -> pd.DataFrame:
        """
        过滤过拟合概率过高的指标
        """
        filtered = indicator_results.copy()

        for idx in filtered.index:
            if idx in cscv_results:
                if cscv_results[idx]['overfitting_probability'] > self.threshold:
                    filtered.loc[idx, 'filtered'] = True
                else:
                    filtered.loc[idx, 'filtered'] = False
            else:
                filtered.loc[idx, 'filtered'] = True

        return filtered[filtered['filtered'] == False]

    def rank_indicators(self, indicator_results: pd.DataFrame,
                        cscv_results: dict,
                        sort_by: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        根据多重标准对指标排序
        """
        ranked = indicator_results.copy()

        for idx in ranked.index:
            if idx in cscv_results:
                ranked.loc[idx, 'overfit_prob'] = cscv_results[idx]['overfitting_probability']
            else:
                ranked.loc[idx, 'overfit_prob'] = 1.0

        ranked = ranked.sort_values(sort_by, ascending=False)

        return ranked

def calculate_compound_overfitting_prob(prob1: float, prob2: float) -> float:
    """
    计算复合策略的过拟合概率
    当两个指标都看多时才买入
    """
    return prob1 * prob2

class StrategyRobustnessAnalyzer:
    def __init__(self):
        self.cscv_test = CSCVTest(n_splits=100)

    def analyze_strategy_robustness(self, prices: pd.DataFrame,
                                     signals: pd.DataFrame,
                                     strategy_type: str = 'time_series') -> pd.DataFrame:
        """
        分析策略鲁棒性
        """
        results = []

        for col in signals.columns:
            try:
                signal = signals[col]
                returns = prices.pct_change().dropna()
                aligned_signal = signal.reindex(returns.index, method='ffill')

                strategy_returns = returns.mean(axis=1) * aligned_signal.shift(1)
                strategy_returns = strategy_returns.dropna()

                cscv_result = self.cscv_test.run_cscv_analysis(signal, prices.mean(axis=1))

                result = {
                    'indicator': col,
                    'sharpe_ratio': cscv_result['selection_sharpe'],
                    'overfitting_prob': cscv_result['overfitting_probability'],
                    'is_sharpe': cscv_result['is_sharpe_mean'],
                    'oos_sharpe': cscv_result['oos_sharpe_mean']
                }
                results.append(result)

            except Exception as e:
                print(f"分析 {col} 时出错: {e}")

        return pd.DataFrame(results)

if __name__ == "__main__":
    print("测试CSCV过拟合检验...")

    np.random.seed(42)
    n_days = 252
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    prices = pd.Series(100 + np.random.randn(n_days).cumsum(), index=dates)

    signal = pd.Series(np.sign(np.random.randn(n_days)), index=dates)

    cscv = CSCVTest(n_splits=50)
    result = cscv.run_cscv_analysis(signal, prices)

    print(f"样本内夏普比率均值: {result['is_sharpe_mean']:.4f}")
    print(f"样本外夏普比率均值: {result['oos_sharpe_mean']:.4f}")
    print(f"过拟合概率: {result['overfitting_probability']:.4f}")

    print("CSCV过拟合检验测试完成!")