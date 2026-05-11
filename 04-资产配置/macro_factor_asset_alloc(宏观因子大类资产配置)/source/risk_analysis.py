"""
风险分析模块
实现宏观风险分解，支持Boudt(2013)风险归因方法
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class RiskAnalyzer:
    """风险分析器"""

    def __init__(self):
        self.factor_cov_matrix = None
        self.asset_cov_matrix = None
        self.heterogeneous_var = None

    def compute_factor_covariance(self, factor_returns: pd.DataFrame) -> np.ndarray:
        """
        计算因子收益率协方差矩阵

        Parameters:
            factor_returns: 因子收益率 (T x K)

        Returns:
            K x K 协方差矩阵
        """
        self.factor_cov_matrix = factor_returns.cov().values
        return self.factor_cov_matrix

    def compute_asset_covariance(self, asset_returns: pd.DataFrame) -> np.ndarray:
        """
        计算资产收益率协方差矩阵

        Parameters:
            asset_returns: 资产收益率 (T x N)

        Returns:
            N x N 协方差矩阵
        """
        self.asset_cov_matrix = asset_returns.cov().values
        return self.asset_cov_matrix

    def compute_heterogeneous_variance(self, residual_returns: pd.DataFrame) -> np.ndarray:
        """
        计算资产异质风险方差

        Parameters:
            residual_returns: 残差收益率 (T x N)

        Returns:
            N 异质风险方差向量
        """
        self.heterogeneous_var = residual_returns.var().values
        return self.heterogeneous_var

    def risk_contribution(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        计算各资产对组合风险的贡献

        Parameters:
            weights: 资产权重 (N,)
            cov_matrix: 资产协方差矩阵 (N x N)

        Returns:
            各资产风险贡献 (N,)
        """
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol

        return risk_contrib

    def factor_risk_contribution(self, weights: np.ndarray, exposure_matrix: np.ndarray,
                                 factor_cov: np.ndarray) -> np.ndarray:
        """
        计算各因子对组合风险的贡献 (Boudt 2013方法)

        Parameters:
            weights: 资产权重 (N,)
            exposure_matrix: 因子暴露矩阵 (N x K)
            factor_cov: 因子协方差矩阵 (K x K)

        Returns:
            各因子风险贡献 (K,)
        """
        factor_exposure = exposure_matrix.T @ weights

        gamma = np.concatenate([factor_exposure, weights])

        theta = np.zeros((len(gamma), len(gamma)))
        theta[:len(factor_exposure), :len(factor_exposure)] = factor_cov
        theta[len(factor_exposure):, len(factor_exposure):] = np.diag(self.heterogeneous_var)

        portfolio_vol = np.sqrt(gamma @ theta @ gamma)

        marginal_gamma = theta @ gamma / portfolio_vol
        factor_risk_contrib = gamma * marginal_gamma / portfolio_vol

        factor_risk = factor_risk_contrib[:len(factor_exposure)]
        heterogeneous_risk = factor_risk_contrib[len(factor_exposure):]

        return factor_risk, heterogeneous_risk

    def decompose_portfolio_risk(self, weights: np.ndarray, exposure_matrix: np.ndarray,
                                 asset_names: list, factor_names: list) -> pd.DataFrame:
        """
        对组合进行宏观风险分解

        Parameters:
            weights: 资产权重
            exposure_matrix: 因子暴露矩阵
            asset_names: 资产名称列表
            factor_names: 因子名称列表

        Returns:
            风险分解DataFrame
        """
        if self.factor_cov_matrix is None or self.heterogeneous_var is None:
            raise ValueError("请先计算因子协方差矩阵和异质风险方差")

        factor_risk, heterogeneous_risk = self.factor_risk_contribution(
            weights, exposure_matrix, self.factor_cov_matrix
        )

        total_factor_risk = np.sum(factor_risk)
        total_hetero_risk = np.sum(heterogeneous_risk)
        total_risk = total_factor_risk + total_hetero_risk

        risk_contrib_pct = factor_risk / total_risk * 100 if total_risk > 0 else np.zeros_like(factor_risk)

        result = pd.DataFrame({
            '因子': factor_names,
            '风险贡献': factor_risk,
            '风险贡献占比(%)': risk_contrib_pct
        })

        hetero_contrib_pct = heterogeneous_risk / total_risk * 100 if total_risk > 0 else np.zeros_like(heterogeneous_risk)
        hetero_df = pd.DataFrame({
            '因子': [f'异质风险_{i}' for i in range(len(heterogeneous_risk))],
            '风险贡献': heterogeneous_risk,
            '风险贡献占比(%)': hetero_contrib_pct
        })

        result = pd.concat([result, hetero_df], ignore_index=True)

        return result

    def decompose_asset_risk(self, weights: np.ndarray, exposure_matrix: np.ndarray,
                            asset_names: list, factor_names: list) -> pd.DataFrame:
        """
        对单个资产进行风险分解

        Parameters:
            weights: 资产权重
            exposure_matrix: 因子暴露矩阵
            asset_names: 资产名称列表
            factor_names: 因子名称列表

        Returns:
            资产风险分解DataFrame
        """
        if self.factor_cov_matrix is None or self.heterogeneous_var is None:
            raise ValueError("请先计算因子协方差矩阵和异质风险方差")

        result_data = []

        for i, asset in enumerate(asset_names):
            asset_weight = weights[i]

            if np.abs(asset_weight) < 1e-6:
                continue

            asset_exposure = exposure_matrix[i, :]

            marginal_factors = self.factor_cov_matrix @ asset_exposure
            factor_contrib = asset_exposure * marginal_factors * asset_weight

            hetero_contrib = self.heterogeneous_var[i] * (asset_weight ** 2)

            total_contrib = np.sum(factor_contrib) + hetero_contrib

            row = {'资产': asset, '权重': asset_weight, '异质风险': hetero_contrib}
            for j, factor in enumerate(factor_names):
                row[f'{factor}风险'] = factor_contrib[j]

            result_data.append(row)

        result = pd.DataFrame(result_data)

        risk_cols = factor_names + ['异质风险']
        if not result.empty:
            result['总风险'] = result[risk_cols].sum(axis=1)

        return result

    def compute_portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        计算组合波动率

        Parameters:
            weights: 资产权重
            cov_matrix: 资产协方差矩阵

        Returns:
            组合波动率
        """
        return np.sqrt(weights @ cov_matrix @ weights)

    def compute_tracking_error(self, portfolio_weights: np.ndarray, benchmark_weights: np.ndarray,
                              cov_matrix: np.ndarray) -> float:
        """
        计算跟踪误差

        Parameters:
            portfolio_weights: 组合权重
            benchmark_weights: 基准权重
            cov_matrix: 资产协方差矩阵

        Returns:
            跟踪误差
        """
        delta_w = portfolio_weights - benchmark_weights
        te = np.sqrt(delta_w @ cov_matrix @ delta_w)
        return te

    def compute_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算Value at Risk

        Parameters:
            returns: 收益率序列
            confidence: 置信水平

        Returns:
            VaR值
        """
        return np.percentile(returns, (1 - confidence) * 100)

    def compute_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算Conditional VaR (Expected Shortfall)

        Parameters:
            returns: 收益率序列
            confidence: 置信水平

        Returns:
            CVaR值
        """
        var = self.compute_var(returns, confidence)
        return returns[returns <= var].mean()


class MacroRiskMonitor:
    """宏观风险监测器"""

    def __init__(self, risk_analyzer: RiskAnalyzer):
        self.risk_analyzer = risk_analyzer
        self.risk_history = []

    def monitor_portfolio_risk(self, date: pd.Timestamp, weights: np.ndarray,
                               exposure_matrix: np.ndarray,
                               asset_names: list, factor_names: list):
        """
        监测组合的宏观风险

        Parameters:
            date: 日期
            weights: 当前权重
            exposure_matrix: 因子暴露矩阵
            asset_names: 资产名称
            factor_names: 因子名称
        """
        risk_decomp = self.risk_analyzer.decompose_portfolio_risk(
            weights, exposure_matrix, asset_names, factor_names
        )

        total_risk = risk_decomp['风险贡献'].sum()

        risk_record = {
            'date': date,
            'total_risk': total_risk,
            'factor_risks': risk_decomp.set_index('因子')['风险贡献'].to_dict()
        }

        self.risk_history.append(risk_record)

    def get_risk_history(self) -> pd.DataFrame:
        """
        获取风险监测历史

        Returns:
            风险历史DataFrame
        """
        if not self.risk_history:
            return pd.DataFrame()

        records = []
        for record in self.risk_history:
            row = {'date': record['date'], 'total_risk': record['total_risk']}
            row.update(record['factor_risks'])
            records.append(row)

        return pd.DataFrame(records).set_index('date')


def risk_attribution_analysis(weights: np.ndarray, exposure_matrix: np.ndarray,
                              factor_returns: pd.DataFrame, asset_returns: pd.DataFrame,
                              asset_names: list, factor_names: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    风险归因分析主函数

    Parameters:
        weights: 资产权重
        exposure_matrix: 因子暴露矩阵
        factor_returns: 因子收益率
        asset_returns: 资产收益率
        asset_names: 资产名称
        factor_names: 因子名称

    Returns:
        (组合风险分解, 资产风险分解)
    """
    analyzer = RiskAnalyzer()

    analyzer.compute_factor_covariance(factor_returns)
    analyzer.compute_heterogeneous_variance(asset_returns)

    portfolio_risk = analyzer.decompose_portfolio_risk(
        weights, exposure_matrix, asset_names, factor_names
    )

    asset_risk = analyzer.decompose_asset_risk(
        weights, exposure_matrix, asset_names, factor_names
    )

    return portfolio_risk, asset_risk


if __name__ == "__main__":
    print("测试风险分析模块...")

    n_assets = 11
    n_factors = 6
    n_periods = 120

    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=n_periods, freq='M')

    factor_returns = pd.DataFrame(
        np.random.randn(n_periods, n_factors) * 0.01,
        index=dates,
        columns=['增长', '通胀', '利率', '信用', '汇率', '流动性']
    )

    asset_returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.02,
        index=dates,
        columns=['沪深300', '中证500', '中债国债', '中债企业债', '中证转债',
                '南华工业品', '南华农产品', '布伦特原油', '沪金', '美元兑人民币', '恒生指数']
    )

    exposure_matrix = np.random.randn(n_assets, n_factors) * 0.3

    weights = np.ones(n_assets) / n_assets

    factor_names = ['增长', '通胀', '利率', '信用', '汇率', '流动性']
    asset_names = asset_returns.columns.tolist()

    portfolio_risk, asset_risk = risk_attribution_analysis(
        weights, exposure_matrix, factor_returns, asset_returns,
        asset_names, factor_names
    )

    print("\n=== 组合风险分解 ===")
    print(portfolio_risk)

    print("\n=== 资产风险分解 ===")
    print(asset_risk.head())