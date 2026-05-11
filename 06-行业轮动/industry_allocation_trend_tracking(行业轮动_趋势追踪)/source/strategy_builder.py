"""
策略构建框架模块
实现研报中的趋势追踪策略构建流程
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .indicators import TrendIndicatorCalculator
from .backtest import BacktestEngine, BacktestResult
from .cscv import CSCVTest, calculate_compound_overfitting_prob

@dataclass
class StrategyConfig:
    name: str
    asset_type: str
    strategy_type: str
    n_select: Optional[int] = None
    target_volatility: Optional[float] = 7.5
    commission_rate: float = 0.001

class TrendFollowingStrategyBuilder:
    def __init__(self, strategy_config: StrategyConfig):
        self.config = strategy_config
        self.indicator_calculator = TrendIndicatorCalculator()
        self.backtest_engine = BacktestEngine(
            rebalance_freq='monthly',
            target_volatility=strategy_config.target_volatility,
            commission_rate=strategy_config.commission_rate
        )
        self.cscv_test = CSCVTest(n_splits=100)

    def build_indicator_pool(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        构建指标池
        """
        all_signals = {}

        for asset_name, df in data.items():
            asset_signals = self.indicator_calculator.generate_all_signals(df)
            for col in asset_signals.columns:
                all_signals[f"{asset_name}_{col}"] = asset_signals[col]

        return pd.DataFrame(all_signals)

    def evaluate_indicators(self, prices: pd.DataFrame,
                           signals: pd.DataFrame) -> pd.DataFrame:
        """
        评估指标表现
        """
        evaluator_results = []

        for col in signals.columns:
            try:
                signal = signals[col]
                result = self.backtest_engine.evaluate_indicator(
                    prices, signal,
                    strategy_type=self.config.strategy_type,
                    n_select=self.config.n_select
                )
                result['indicator'] = col
                evaluator_results.append(result)
            except Exception as e:
                print(f"评估 {col} 时出错: {e}")

        results_df = pd.DataFrame(evaluator_results)
        return results_df.set_index('indicator')

    def filter_by_overfitting(self, prices: pd.DataFrame,
                             signals: pd.DataFrame,
                             results_df: pd.DataFrame,
                             threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
        """
        根据CSCV过拟合概率筛选指标
        """
        cscv_results = {}
        filtered_signals = {}

        for col in signals.columns:
            try:
                cscv_result = self.cscv_test.run_cscv_analysis(signals[col], prices.mean(axis=1))
                cscv_results[col] = cscv_result

                if cscv_result['overfitting_probability'] <= threshold:
                    filtered_signals[col] = signals[col]
            except Exception as e:
                print(f"CSCV检验 {col} 时出错: {e}")

        filtered_results = results_df.loc[list(filtered_signals.keys())]
        return filtered_results, cscv_results

    def create_compound_strategy(self, signals: pd.DataFrame,
                                 results_df: pd.DataFrame,
                                 top_n: int = 10) -> pd.DataFrame:
        """
        创建复合策略
        两两搭配构建复合指标
        """
        top_indicators = results_df.nlargest(top_n, 'sharpe_ratio').index.tolist()
        compound_signals = {}

        for i, ind1 in enumerate(top_indicators):
            for ind2 in top_indicators[i + 1:]:
                try:
                    signal1 = signals[ind1]
                    signal2 = signals[ind2]

                    compound_signal = (signal1 > 0) & (signal2 > 0)
                    compound_signal = compound_signal.astype(int) * 100

                    compound_signals[f"{ind1}_AND_{ind2}"] = compound_signal
                except Exception as e:
                    print(f"创建复合信号 {ind1} AND {ind2} 时出错: {e}")

        return pd.DataFrame(compound_signals)

    def run_full_pipeline(self, prices: pd.DataFrame,
                         data: Dict[str, pd.DataFrame]) -> Dict:
        """
        运行完整策略构建流程
        """
        print(f"开始构建 {self.config.name}...")

        print("1. 生成指标信号...")
        signals = self.build_indicator_pool(data)
        print(f"   共生成 {len(signals.columns)} 个指标信号")

        print("2. 评估指标表现...")
        results_df = self.evaluate_indicators(prices, signals)
        print(f"   表现最好的指标夏普比率: {results_df['sharpe_ratio'].max():.4f}")

        print("3. CSCV过拟合检验...")
        filtered_results, cscv_results = self.filter_by_overfitting(
            prices, signals, results_df
        )
        print(f"   过滤后剩余 {len(filtered_results)} 个指标")

        print("4. 创建复合策略...")
        compound_signals = self.create_compound_strategy(
            signals.loc[:, filtered_results.index],
            filtered_results
        )
        print(f"   创建 {len(compound_signals.columns)} 个复合策略")

        return {
            'signals': signals,
            'results': results_df,
            'filtered_results': filtered_results,
            'cscv_results': cscv_results,
            'compound_signals': compound_signals
        }

class AssetAllocationStrategyBuilder(TrendFollowingStrategyBuilder):
    def __init__(self, strategy_config: StrategyConfig):
        super().__init__(strategy_config)

    def classify_assets(self, assets_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        将资产分类为股票、债券、商品
        """
        classification = {
            'stock': [],
            'bond': [],
            'commodity': []
        }

        stock_keywords = ['沪深300', '上证50', '中证500', '恒生', '标普', '上证']
        bond_keywords = ['国债', '债']
        commodity_keywords = ['黄金', '原油', '铜', '螺纹', '豆粕', 'PTA', '糖', '南华']

        for name in assets_data.keys():
            if any(kw in name for kw in stock_keywords):
                classification['stock'].append(name)
            elif any(kw in name for kw in bond_keywords):
                classification['bond'].append(name)
            elif any(kw in name for kw in commodity_keywords):
                classification['commodity'].append(name)
            else:
                classification['stock'].append(name)

        return classification

    def calculate_risk_parity_weights(self, returns: pd.DataFrame,
                                     lookback: int = 40) -> pd.Series:
        """
        计算风险平价权重
        """
        cov_matrix = returns.iloc[-lookback:].cov()
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vol / inv_vol.sum()
        return weights

    def run(self, assets_data: Dict[str, pd.DataFrame],
            start_date: str, end_date: str) -> Dict:
        """
        运行大类资产配置策略
        """
        print(f"\n{'='*50}")
        print(f"开始构建大类资产配置策略: {self.config.name}")
        print(f"{'='*50}")

        prices_dict = {}
        for name, df in assets_data.items():
            df_indexed = df.set_index('trade_date') if 'trade_date' in df.columns else df
            prices_dict[name] = df_indexed

        prices = pd.DataFrame(prices_dict)
        prices = prices.sort_index()

        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]

        pipeline_results = self.run_full_pipeline(prices, assets_data)

        return {
            'config': self.config,
            'prices': prices,
            'asset_classification': self.classify_assets(assets_data),
            **pipeline_results
        }

class IndustryRotationStrategyBuilder(TrendFollowingStrategyBuilder):
    def __init__(self, strategy_config: StrategyConfig):
        super().__init__(strategy_config)

    def run(self, industry_data: Dict[str, pd.DataFrame],
            start_date: str, end_date: str) -> Dict:
        """
        运行行业轮动策略
        """
        print(f"\n{'='*50}")
        print(f"开始构建行业轮动策略: {self.config.name}")
        print(f"{'='*50}")

        prices_dict = {}
        for name, df in industry_data.items():
            df_indexed = df.set_index('trade_date') if 'trade_date' in df.columns else df
            prices_dict[name] = df_indexed

        prices = pd.DataFrame(prices_dict)
        prices = prices.sort_index()
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]

        pipeline_results = self.run_full_pipeline(prices, industry_data)

        return {
            'config': self.config,
            'prices': prices,
            **pipeline_results
        }

def build_asset_allocation_strategies() -> List[AssetAllocationStrategyBuilder]:
    """
    构建多个大类资产配置策略
    """
    strategies = []

    config1 = StrategyConfig(
        name="大类资产配置策略一",
        asset_type="asset_allocation",
        strategy_type="time_series",
        target_volatility=7.5,
        commission_rate=0.001
    )
    strategies.append(AssetAllocationStrategyBuilder(config1))

    config2 = StrategyConfig(
        name="大类资产配置策略二",
        asset_type="asset_allocation",
        strategy_type="cross_section",
        n_select=3,
        target_volatility=7.5,
        commission_rate=0.001
    )
    strategies.append(AssetAllocationStrategyBuilder(config2))

    return strategies

def build_industry_rotation_strategies() -> List[IndustryRotationStrategyBuilder]:
    """
    构建多个行业轮动策略
    """
    strategies = []

    config1 = StrategyConfig(
        name="行业轮动策略一",
        asset_type="industry",
        strategy_type="cross_section",
        n_select=5,
        commission_rate=0.001
    )
    strategies.append(IndustryRotationStrategyBuilder(config1))

    config2 = StrategyConfig(
        name="行业轮动策略二",
        asset_type="industry",
        strategy_type="cross_section",
        n_select=10,
        commission_rate=0.001
    )
    strategies.append(IndustryRotationStrategyBuilder(config2))

    return strategies

if __name__ == "__main__":
    print("测试策略构建框架...")

    np.random.seed(42)
    n_days = 252
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    test_data = {
        'asset1': pd.DataFrame({
            'close': 100 + np.random.randn(n_days).cumsum(),
            'returns': np.random.randn(n_days) * 0.01
        }, index=dates),
        'asset2': pd.DataFrame({
            'close': 100 + np.random.randn(n_days).cumsum(),
            'returns': np.random.randn(n_days) * 0.01
        }, index=dates)
    }

    config = StrategyConfig(
        name="测试策略",
        asset_type="asset_allocation",
        strategy_type="time_series"
    )

    builder = TrendFollowingStrategyBuilder(config)
    print(f"策略构建器初始化完成: {config.name}")

    print("策略构建框架测试完成!")