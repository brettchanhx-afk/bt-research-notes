# -*- coding: utf-8 -*-
"""
backtest.py - Barra模型回测模块

【功能说明】
1. 滚动窗口Barra归因回测
2. 因子暴露时间序列跟踪
3. 归因稳定性检验
4. 绩效指标汇总

【研报对应】
通过多期时间序列回归，检验基金因子暴露b的时变特征和统计显著性

【版本】
v1.0  2026-04-28
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from source.factor import (
    BarraFactorAttribution,
    FactorAttributionResult,
    FundExposureCalculator
)


# =============================================================================
# 回测配置
# =============================================================================

@dataclass
class BacktestConfig:
    """
    回测配置

    【参数说明】
        rolling_window: 滚动回归窗口（月），默认24
        min_periods: 最少期数要求，默认12
        rebalance_freq: 调仓频率 'M'=月, 'Q'=季
        start_date: 回测开始日期
        end_date: 回测结束日期
        significance_level: 显著性水平
        risk_free_rate: 无风险利率（年化）
    """
    rolling_window: int = 24
    min_periods: int = 12
    rebalance_freq: str = 'M'
    start_date: str = '2022-01-01'
    end_date: str = '2024-12-31'
    significance_level: float = 0.05
    risk_free_rate: float = 0.03


# =============================================================================
# 滚动回归回测
# =============================================================================

class BarraRollingBacktest:
    """
    Barra模型滚动回归回测

    【核心逻辑】
        1. 在每个时间点，使用过去rolling_window个月的数据
        2. 执行时间序列回归，获取当期因子暴露b
        3. 跟踪b的时间序列变化，分析因子暴露的时变特征
        4. 检验b的统计显著性是否稳定
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results: List[FactorAttributionResult] = []

    def run(self,
            fund_returns: pd.Series,
            factor_returns: pd.DataFrame,
            fund_code: str = '') -> Dict[str, pd.DataFrame]:
        """
        执行滚动Barra归因回测

        【参数】
            fund_returns: 基金月度收益率 (索引=日期)
            factor_returns: 因子月度收益率 (索引=日期, 列=因子名)
            fund_code: 基金代码

        【返回】
            Dict包含:
                'rolling_exposure': 滚动因子暴露时间序列
                'rolling_significance': 滚动显著性
                'rolling_r_squared': 'R-sq (rolling)'
                'rolling_alpha': 滚动Alpha
                'period_results': 各期详细结果
        """
        print("=" * 70)
        print(f"Barra滚动归因回测 | 基金: {fund_code}")
        print(f"回测期间: {self.config.start_date} ~ {self.config.end_date}")
        print(f"滚动窗口: {self.config.rolling_window}月")
        print("=" * 70)

        # 对齐日期
        common_dates = fund_returns.index.intersection(factor_returns.index)
        common_dates = common_dates.sort_values()

        if len(common_dates) < self.config.min_periods:
            print(f"错误: 共同日期仅 {len(common_dates)} 期，不足 {self.config.min_periods} 期")
            return {}

        # 结果存储
        rolling_exposure = []
        rolling_significance = []
        rolling_r_squared = []
        rolling_alpha = []
        period_results = []

        factor_names = factor_returns.columns.tolist()

        # 滚动回归
        for i in range(self.config.rolling_window, len(common_dates) + 1):
            window_dates = common_dates[i - self.config.rolling_window:i]
            current_date = common_dates[i - 1]

            # 当期数据
            Y = fund_returns.loc[window_dates].values
            X = factor_returns.loc[window_dates].values

            # 时间序列回归
            result = FundExposureCalculator.time_series_regression(
                Y, X, factor_names=factor_names, add_constant=True
            )
            result.fund_code = fund_code
            result.period = str(current_date.date()) if hasattr(current_date, 'date') else str(current_date)

            # 存储
            exposure_row = {'date': current_date}
            sig_row = {'date': current_date}

            for j, name in enumerate(factor_names):
                if j < len(result.b):
                    exposure_row[name] = result.b[j]
                    sig_row[name] = '*' if (j < len(result.p_values) and
                                            result.p_values[j] < self.config.significance_level) else ''
                else:
                    exposure_row[name] = np.nan
                    sig_row[name] = ''

            rolling_exposure.append(exposure_row)
            rolling_significance.append(sig_row)
            rolling_r_squared.append({
                'date': current_date,
                'r_squared': result.r_squared,
                'adj_r_squared': result.adj_r_squared
            })
            rolling_alpha.append({
                'date': current_date,
                'alpha': result.alpha,
                'alpha_annual': (1 + result.alpha) ** 12 - 1
            })
            period_results.append(result)

        # 转为DataFrame
        exposure_df = pd.DataFrame(rolling_exposure).set_index('date')
        sig_df = pd.DataFrame(rolling_significance).set_index('date')
        rsq_df = pd.DataFrame(rolling_r_squared).set_index('date')
        alpha_df = pd.DataFrame(rolling_alpha).set_index('date')

        self.results = period_results

        # 打印摘要
        print(f"\n滚动回归完成，共 {len(period_results)} 期")
        print(f"\n【因子暴露均值】")
        for name in factor_names:
            mean_val = exposure_df[name].mean()
            std_val = exposure_df[name].std()
            sig_rate = (sig_df[name] == '*').mean()
            print(f"  {name:20s}: 均值={mean_val:>7.4f}, 标准差={std_val:>7.4f}, 显著率={sig_rate*100:>5.1f}%")

        print(f"\n【Alpha统计】")
        print(f"  均值(年化): {alpha_df['alpha_annual'].mean()*100:>8.2f}%")
        print(f"  胜率: {(alpha_df['alpha'] > 0).mean()*100:>8.1f}%")

        print(f"\n【模型拟合度】")
        print(f"  R-sq mean:     {rsq_df['r_squared'].mean():>8.4f}")
        print(f"  Adj R-sq mean: {rsq_df['adj_r_squared'].mean():>8.4f}")

        return {
            'rolling_exposure': exposure_df,
            'rolling_significance': sig_df,
            'rolling_r_squared': rsq_df,
            'rolling_alpha': alpha_df,
            'period_results': period_results
        }


# =============================================================================
# 归因稳定性检验
# =============================================================================

class AttributionStabilityTest:
    """
    归因稳定性检验

    【功能】
    1. 因子暴露时间序列的稳定性（自相关、均值回归）
    2. 显著性的持续性
    3. 模型拟合度的稳定性
    """

    @staticmethod
    def test_exposure_autocorrelation(exposure_df: pd.DataFrame,
                                       lags: int = 3) -> pd.DataFrame:
        """
        检验因子暴露的自相关性

        【参数】
            exposure_df: 滚动因子暴露DataFrame
            lags: 滞后期数

        【返回】
            DataFrame: 各因子各滞后阶的自相关系数
        """
        autocorr_results = {}

        for col in exposure_df.columns:
            acf_values = []
            for lag in range(1, lags + 1):
                acf = exposure_df[col].autocorr(lag=lag)
                acf_values.append(acf)

            autocorr_results[col] = acf_values

        result = pd.DataFrame(autocorr_results,
                              index=[f'lag_{i}' for i in range(1, lags + 1)])
        return result

    @staticmethod
    def test_significance_persistence(sig_df: pd.DataFrame) -> pd.DataFrame:
        """
        检验因子显著性的持续性

        【返回】
            DataFrame: 各因子的显著性统计
        """
        persistence = {}

        for col in sig_df.columns:
            is_sig = (sig_df[col] == '*').astype(int)
            total_periods = len(is_sig)
            sig_periods = is_sig.sum()
            sig_rate = sig_periods / total_periods if total_periods > 0 else 0

            # 最长连续显著期
            max_consecutive = 0
            current_consecutive = 0
            for val in is_sig:
                if val == 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0

            persistence[col] = {
                'sig_periods': sig_periods,
                'total_periods': total_periods,
                'sig_rate': sig_rate,
                'max_consecutive': max_consecutive
            }

        return pd.DataFrame(persistence).T


# =============================================================================
# 绩效指标汇总
# =============================================================================

class PerformanceSummary:
    """
    绩效指标汇总
    """

    @staticmethod
    def calculate(fund_returns: pd.Series,
                  factor_returns: pd.DataFrame,
                  attribution_result: FactorAttributionResult) -> Dict[str, float]:
        """
        计算完整绩效指标

        【返回】
            Dict包含:
                - 年化收益率
                - 年化波动率
                - 夏普比率
                - 最大回撤
                - 信息比率
                - 年化Alpha
                - 因子解释比例
        """
        # 年化收益率
        annual_return = (1 + fund_returns.mean()) ** 12 - 1

        # 年化波动率
        annual_vol = fund_returns.std() * np.sqrt(12)

        # 夏普比率
        sharpe = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cum_nav = (1 + fund_returns).cumprod()
        peak = cum_nav.cummax()
        drawdown = (cum_nav - peak) / peak
        max_dd = drawdown.min()

        # 信息比率（相对因子模型解释部分）
        if attribution_result.time_series is not None and len(attribution_result.time_series) > 0:
            residuals = attribution_result.time_series['residual']
            tracking_error = residuals.std() * np.sqrt(12)
            information_ratio = residuals.mean() * 12 / tracking_error if tracking_error > 0 else 0
        else:
            tracking_error = 0
            information_ratio = 0

        # 因子解释比例
        total_var = fund_returns.var()
        if total_var > 0 and attribution_result.r_squared > 0:
            explained_ratio = attribution_result.r_squared
        else:
            explained_ratio = 0

        # 年化Alpha
        alpha_annual = (1 + attribution_result.alpha) ** 12 - 1

        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'information_ratio': information_ratio,
            'alpha_annual': alpha_annual,
            'factor_explained_ratio': explained_ratio,
            'r_squared': attribution_result.r_squared,
            'adj_r_squared': attribution_result.adj_r_squared,
            'tracking_error_annual': tracking_error
        }

        return metrics

    @staticmethod
    def print_summary(metrics: Dict[str, float], fund_code: str = ''):
        """打印绩效指标摘要"""
        print(f"\n{'='*70}")
        print(f"绩效指标汇总 | 基金: {fund_code}")
        print(f"{'='*70}")
        print(f"  年化收益率:       {metrics['annual_return']*100:>8.2f}%")
        print(f"  年化波动率:       {metrics['annual_volatility']*100:>8.2f}%")
        print(f"  夏普比率:         {metrics['sharpe_ratio']:>8.4f}")
        print(f"  最大回撤:         {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  年化Alpha:        {metrics['alpha_annual']*100:>8.2f}%")
        print(f"  信息比率:         {metrics['information_ratio']:>8.4f}")
        print(f"  跟踪误差(年化):   {metrics['tracking_error_annual']*100:>8.2f}%")
        print(f"  因子解释比例:     {metrics['factor_explained_ratio']*100:>8.2f}%")
        print(f"  R-squared:        {metrics['r_squared']:>8.4f}")
        print(f"  Adj R-squared:    {metrics['adj_r_squared']:>8.4f}")
        print(f"{'='*70}")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Barra回测模块测试\n")

    # 生成测试数据
    np.random.seed(42)
    T = 36  # 3年月度数据
    k = 5
    factor_names = ['SIZE', 'VALUE', 'MOMENTUM', 'VOLATILITY', 'QUALITY']

    dates = pd.date_range('2022-01-01', periods=T, freq='M')
    factor_returns = pd.DataFrame(
        np.random.randn(T, k) * 0.02,
        index=dates, columns=factor_names
    )

    # 模拟基金收益率
    true_b = np.array([1.0, -0.5, 0.8, -0.3, 0.6])
    fund_returns = pd.Series(
        factor_returns.values @ true_b + 0.003 + np.random.randn(T) * 0.005,
        index=dates
    )

    # 滚动回归
    config = BacktestConfig(rolling_window=24, min_periods=12)
    backtest = BarraRollingBacktest(config)
    results = backtest.run(fund_returns, factor_returns, fund_code='TEST_FUND')

    if results:
        # 稳定性检验
        exposure_df = results['rolling_exposure']
        sig_df = results['rolling_significance']

        autocorr = AttributionStabilityTest.test_exposure_autocorrelation(exposure_df)
        print(f"\n因子暴露自相关性:")
        print(autocorr.round(4))

        persistence = AttributionStabilityTest.test_significance_persistence(sig_df)
        print(f"\n显著性持续性:")
        print(persistence)

        # 绩效指标
        final_result = results['period_results'][-1]
        metrics = PerformanceSummary.calculate(fund_returns, factor_returns, final_result)
        PerformanceSummary.print_summary(metrics, fund_code='TEST_FUND')

    print("\n回测模块测试完成!")
