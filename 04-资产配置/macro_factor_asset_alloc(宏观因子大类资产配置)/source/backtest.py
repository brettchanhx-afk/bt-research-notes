"""
回测引擎模块
实现宏观因子配置框架的回测功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from macro_factors import MacroFactorBuilder
from factor_exposure import FactorExposureWithPrior
from portfolio_optimizer import PortfolioOptimizer, MacroFactorAllocator
from risk_analysis import RiskAnalyzer, risk_attribution_analysis


class BacktestEngine:
    """回测引擎"""

    def __init__(self, start_date: str, end_date: str, rebalance_freq: str = 'monthly',
                 factor_deviation: float = 0.05):
        """
        初始化回测引擎

        Parameters:
            start_date: 回测开始日期
            end_date: 回测结束日期
            rebalance_freq: 调仓频率 ('monthly', 'quarterly')
            factor_deviation: 因子偏离值
        """
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_freq = rebalance_freq
        self.factor_deviation = factor_deviation

        self.factor_builder = MacroFactorBuilder(n_factors=6)
        self.exposure_calculator = FactorExposureWithPrior(alpha=0.01)
        self.optimizer = PortfolioOptimizer(lambda_param=0.1)
        self.allocator = MacroFactorAllocator(self.optimizer)
        self.risk_analyzer = RiskAnalyzer()

        self.results = {}
        self.weights_history = []
        self.returns_history = []
        self.factor_history = []

    def prepare_data(self, asset_returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备回测数据

        Parameters:
            asset_returns: 资产收益率

        Returns:
            (月度资产收益率, 月度因子收益率)
        """
        monthly_returns = asset_returns.resample('M').last()

        factor_returns = self.factor_builder.construct_all_factors(monthly_returns)

        return monthly_returns, factor_returns

    def compute_rolling_covariance(self, returns: pd.DataFrame, window: int = 36) -> np.ndarray:
        """
        计算滚动协方差矩阵

        Parameters:
            returns: 收益率数据
            window: 滚动窗口

        Returns:
            协方差矩阵
        """
        if len(returns) < window:
            return returns.cov().values

        recent_returns = returns.iloc[-window:]
        return recent_returns.cov().values

    def compute_rolling_exposure(self, asset_returns: pd.DataFrame,
                                factor_returns: pd.DataFrame,
                                window: int = 36) -> pd.DataFrame:
        """
        计算滚动因子暴露

        Parameters:
            asset_returns: 资产收益率
            factor_returns: 因子收益率
            window: 滚动窗口

        Returns:
            因子暴露矩阵
        """
        if len(asset_returns) < window:
            return self.exposure_calculator.fit(asset_returns, factor_returns)

        start_idx = len(asset_returns) - window
        window_assets = asset_returns.iloc[start_idx:]
        window_factors = factor_returns.iloc[start_idx:]

        return self.exposure_calculator.fit(window_assets, window_factors)

    def compute_residual_returns(self, asset_returns: pd.DataFrame,
                                 factor_returns: pd.DataFrame,
                                 exposure_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        计算残差收益率

        Parameters:
            asset_returns: 资产收益率
            factor_returns: 因子收益率
            exposure_matrix: 因子暴露

        Returns:
            残差收益率
        """
        predicted = self.exposure_calculator.predict_asset_returns(factor_returns)

        common_idx = asset_returns.index.intersection(predicted.index)

        residual = asset_returns.loc[common_idx] - predicted.loc[common_idx]

        return residual

    def run_factor_deviation_backtest(self, asset_returns: pd.DataFrame,
                                     factor_returns: pd.DataFrame,
                                     target_factor: str) -> Dict:
        """
        运行因子偏离回测

        Parameters:
            asset_returns: 资产收益率
            factor_returns: 因子收益率
            target_factor: 目标因子名称

        Returns:
            回测结果字典
        """
        factor_names = factor_returns.columns.tolist()
        asset_names = asset_returns.columns.tolist()

        monthly_returns, monthly_factors = self.prepare_data(asset_returns)

        monthly_factors_filled = monthly_factors.fillna(0)
        monthly_returns_filled = monthly_returns.fillna(0)

        rebalance_dates = self._get_rebalance_dates(monthly_factors_filled)

        base_weights = np.ones(len(asset_names)) / len(asset_names)

        portfolio_values = [1.0]
        benchmark_values = [1.0]
        factor_exposure_history = []
        weights_history = []

        for i, date in enumerate(rebalance_dates):
            if i == 0:
                continue

            train_end = rebalance_dates[i - 1]
            train_returns = monthly_returns_filled.loc[:train_end]
            train_factors = monthly_factors_filled.loc[:train_end]

            if len(train_returns) < 24:
                current_weights = base_weights
            else:
                exposure = self.compute_rolling_exposure(train_returns, train_factors, window=36)

                residual_returns = self.compute_residual_returns(train_returns, train_factors, exposure)
                heterogeneous_var = residual_returns.var().values

                cov_matrix = self.compute_rolling_covariance(train_returns, window=36)

                self.risk_analyzer.compute_factor_covariance(train_factors)
                self.risk_analyzer.compute_heterogeneous_variance(residual_returns)

                factor_deviations = {factor: 0.0 for factor in factor_names}
                if target_factor in factor_deviations:
                    factor_deviations[target_factor] = self.factor_deviation

                self.allocator.set_base_exposures(exposure.values, base_weights)

                current_weights = self.allocator.allocate(
                    base_weights, exposure.values, cov_matrix, heterogeneous_var,
                    factor_deviations
                )

            weights_history.append({
                'date': date,
                'weights': current_weights,
                'target_factor': target_factor
            })

            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
            else:
                next_date = monthly_returns_filled.index[-1]

            period_returns = monthly_returns_filled.loc[date:next_date]

            if not period_returns.empty:
                portfolio_return = (current_weights @ period_returns.T.fillna(0).values)
                portfolio_return = np.nansum(portfolio_return)

                benchmark_return = (base_weights @ period_returns.T.fillna(0).values)
                benchmark_return = np.nansum(benchmark_return)

                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))

                excess_return = portfolio_return - benchmark_return

                self.returns_history.append({
                    'date': next_date,
                    'portfolio_return': portfolio_return,
                    'benchmark_return': benchmark_return,
                    'excess_return': excess_return,
                    'target_factor': target_factor
                })

                current_factor_exposure = exposure.values.T @ current_weights if 'exposure' in dir() else None
                if current_factor_exposure is not None:
                    factor_exposure_history.append({
                        'date': next_date,
                        'factor_exposure': dict(zip(factor_names, current_factor_exposure))
                    })

        results = {
            'portfolio_values': pd.Series(portfolio_values, index=rebalance_dates[:len(portfolio_values)]),
            'benchmark_values': pd.Series(benchmark_values, index=rebalance_dates[:len(benchmark_values)]),
            'weights_history': weights_history,
            'factor_exposure_history': factor_exposure_history
        }

        return results

    def _get_rebalance_dates(self, data: pd.DataFrame) -> list:
        """
        获取调仓日期

        Parameters:
            data: 数据

        Returns:
            调仓日期列表
        """
        if self.rebalance_freq == 'monthly':
            dates = data.resample('M').last().index.tolist()
        elif self.rebalance_freq == 'quarterly':
            dates = data.resample('Q').last().index.tolist()
        else:
            dates = data.index.tolist()

        return dates

    def run_all_factor_backtests(self, asset_returns: pd.DataFrame,
                                 factor_returns: pd.DataFrame) -> Dict:
        """
        运行所有因子的偏离回测

        Parameters:
            asset_returns: 资产收益率
            factor_returns: 因子收益率

        Returns:
            所有因子回测结果
        """
        all_results = {}

        for factor in ['增长', '通胀', '利率', '信用', '汇率', '流动性']:
            print(f"\n正在回测 {factor} 因子偏离策略...")

            try:
                result = self.run_factor_deviation_backtest(
                    asset_returns, factor_returns, target_factor=factor
                )
                all_results[factor] = result
                print(f"  {factor} 因子回测完成")

            except Exception as e:
                print(f"  {factor} 因子回测失败: {e}")

        return all_results

    def compute_performance_metrics(self, returns: pd.Series,
                                   benchmark_returns: pd.Series = None,
                                   risk_free_rate: float = 0.03) -> Dict:
        """
        计算绩效指标

        Parameters:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            risk_free_rate: 无风险利率

        Returns:
            绩效指标字典
        """
        total_return = (1 + returns).prod() - 1

        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (12 / n_periods) - 1 if n_periods > 0 else 0

        annualized_vol = returns.std() * np.sqrt(12)

        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0

        max_drawdown = (returns + 1).cumprod().cummax() - (returns + 1).cumprod()
        max_drawdown = max_drawdown.max()

        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        metrics = {
            '总收益率': total_return,
            '年化收益率': annualized_return,
            '年化波动率': annualized_vol,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '胜率': win_rate
        }

        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(12)
            info_ratio = (excess_returns.mean() * 12) / tracking_error if tracking_error > 0 else 0

            metrics['跟踪误差'] = tracking_error
            metrics['信息比率'] = info_ratio

        return metrics


class BacktestResultAnalyzer:
    """回测结果分析器"""

    def __init__(self, backtest_results: Dict):
        self.results = backtest_results

    def plot_relative_returns(self, factor_name: str, factor_returns: pd.DataFrame,
                             save_path: str = None):
        """
        绘制相对净值与因子走势对比图

        Parameters:
            factor_name: 因子名称
            factor_returns: 因子收益率
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        if factor_name not in self.results:
            print(f"未找到 {factor_name} 的回测结果")
            return

        result = self.results[factor_name]
        portfolio_values = result['portfolio_values']
        benchmark_values = result['benchmark_values']

        relative_values = portfolio_values / benchmark_values

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(relative_values.index, relative_values.values, label='相对净值', linewidth=1.5)
        axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title(f'做多{factor_name}因子 - 相对净值走势', fontsize=14)
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('相对净值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        factor_cumsum = (1 + factor_returns[factor_name].loc[relative_values.index]).cumprod()
        factor_cumsum = factor_cumsum / factor_cumsum.iloc[0]

        axes[1].plot(relative_values.index, relative_values.values, label='相对净值', linewidth=1.5)
        axes[1].plot(factor_cumsum.index, factor_cumsum.values, label=f'{factor_name}因子', linewidth=1.5, alpha=0.7)
        axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title(f'相对净值 vs {factor_name}因子走势', fontsize=14)
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('标准化净值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")

        plt.show()

    def plot_weight_deviation(self, factor_name: str, asset_names: list, save_path: str = None):
        """
        绘制各资产权重偏离图

        Parameters:
            factor_name: 因子名称
            asset_names: 资产名称列表
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        if factor_name not in self.results:
            print(f"未找到 {factor_name} 的回测结果")
            return

        result = self.results[factor_name]
        weights_history = result.get('weights_history', [])

        if not weights_history:
            print(f"未找到 {factor_name} 的权重历史")
            return

        weights_df = pd.DataFrame([w['weights'] for w in weights_history])
        weights_df.index = [w['date'] for w in weights_history]

        mean_weights = weights_df.mean()

        base_weights = np.ones(len(asset_names)) / len(asset_names)
        weight_deviation = mean_weights - base_weights

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['green' if x > 0 else 'red' for x in weight_deviation]
        ax.barh(asset_names, weight_deviation, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('平均权重偏离')
        ax.set_title(f'做多{factor_name}因子 - 各资产权重偏离', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")

        plt.show()

    def generate_summary_report(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        生成回测汇总报告

        Parameters:
            factor_returns: 因子收益率

        Returns:
            汇总报告DataFrame
        """
        summary_data = []

        for factor_name, result in self.results.items():
            portfolio_values = result['portfolio_values']
            benchmark_values = result['benchmark_values']

            portfolio_returns = portfolio_values.pct_change().dropna()
            benchmark_returns = benchmark_values.pct_change().dropna()

            engine = BacktestEngine(self.results[factor_name].get('start_date', '2010-01-01'),
                                  self.results[factor_name].get('end_date', '2023-05-31'))
            metrics = engine.compute_performance_metrics(portfolio_returns, benchmark_returns)

            summary_data.append({
                '因子': factor_name,
                '总收益率': metrics.get('总收益率', 0),
                '年化收益率': metrics.get('年化收益率', 0),
                '年化波动率': metrics.get('年化波动率', 0),
                '夏普比率': metrics.get('夏普比率', 0),
                '最大回撤': metrics.get('最大回撤', 0),
                '信息比率': metrics.get('信息比率', 0)
            })

        summary_df = pd.DataFrame(summary_data)

        return summary_df


if __name__ == "__main__":
    print("测试回测引擎模块...")

    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2023-05-31', freq='M')
    n = len(dates)

    asset_returns = pd.DataFrame({
        '沪深300': np.random.randn(n) * 0.02,
        '中证500': np.random.randn(n) * 0.025,
        '中债国债': np.random.randn(n) * 0.005,
        '中债企业债': np.random.randn(n) * 0.006,
        '中证转债': np.random.randn(n) * 0.015,
        '南华工业品': np.random.randn(n) * 0.025,
        '南华农产品': np.random.randn(n) * 0.02,
        '布伦特原油': np.random.randn(n) * 0.03,
        '沪金': np.random.randn(n) * 0.02,
        '美元兑人民币': np.random.randn(n) * 0.01,
        '恒生指数': np.random.randn(n) * 0.025,
    }, index=dates)

    engine = BacktestEngine('2015-01-01', '2023-05-31', rebalance_freq='monthly', factor_deviation=0.05)

    factor_builder = MacroFactorBuilder(n_factors=6)
    factor_returns = factor_builder.construct_all_factors(asset_returns)

    result = engine.run_factor_deviation_backtest(asset_returns, factor_returns, '增长')

    print(f"\n回测结果:")
    print(f"组合净值序列长度: {len(result['portfolio_values'])}")
    print(f"权重历史记录数: {len(result['weights_history'])}")