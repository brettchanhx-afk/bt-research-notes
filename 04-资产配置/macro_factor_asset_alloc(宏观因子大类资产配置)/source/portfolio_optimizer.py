"""
资产配置权重优化模块
实现Blyth框架、Greenberg框架及结合两者的最优化框架
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """资产配置权重优化器"""

    def __init__(self, weight_bounds=(0, 1), deviation_bounds=(-1, 1),
                 lambda_param=0.1, risk_aversion=1.0):
        """
        初始化优化器

        Parameters:
            weight_bounds: 资产权重上下限 (lower, upper)
            deviation_bounds: 偏离值上下限
            lambda_param: 异质风险惩罚系数
            risk_aversion: 风险厌恶系数
        """
        self.weight_bounds = weight_bounds
        self.deviation_bounds = deviation_bounds
        self.lambda_param = lambda_param
        self.risk_aversion = risk_aversion
        self.optimal_weights = None

    def compute_tracking_error(self, weights: np.ndarray, base_weights: np.ndarray,
                              cov_matrix: np.ndarray) -> float:
        """
        计算跟踪误差

        Parameters:
            weights: 当前组合权重
            base_weights: 基准组合权重
            cov_matrix: 资产协方差矩阵

        Returns:
            跟踪误差
        """
        delta_w = weights - base_weights
        te = np.sqrt(delta_w @ cov_matrix @ delta_w)
        return te

    def compute_factor_exposure(self, weights: np.ndarray, exposure_matrix: np.ndarray) -> np.ndarray:
        """
        计算组合的因子暴露

        Parameters:
            weights: 资产权重
            exposure_matrix: 因子暴露矩阵 (N x K)

        Returns:
            因子暴露向量 (K,)
        """
        return weights @ exposure_matrix

    def blyth_objective(self, weights: np.ndarray, base_weights: np.ndarray,
                      exposure_matrix: np.ndarray, target_exposure: np.ndarray,
                      cov_matrix: np.ndarray, asset_heterogeneous_var: np.ndarray) -> float:
        """
        Blyth最优化框架目标函数

        Parameters:
            weights: 资产权重
            base_weights: 基准权重
            exposure_matrix: 因子暴露矩阵
            target_exposure: 目标因子暴露
            cov_matrix: 资产协方差矩阵
            asset_heterogeneous_var: 资产异质风险方差

        Returns:
            目标函数值
        """
        delta_w = weights - base_weights

        tracking_error = delta_w @ cov_matrix @ delta_w

        factor_exposure = self.compute_factor_exposure(weights, exposure_matrix)
        exposure_deviation = np.sum((factor_exposure - target_exposure) ** 2)

        heterogeneous_risk = weights @ np.diag(asset_heterogeneous_var) @ weights

        objective = tracking_error + self.lambda_param * exposure_deviation + (1 - self.lambda_param) * heterogeneous_risk

        return objective

    def greenberg_objective(self, weights: np.ndarray, base_weights: np.ndarray,
                           exposure_matrix: np.ndarray, target_exposure: np.ndarray,
                           cov_matrix: np.ndarray, factor_cov_matrix: np.ndarray,
                           asset_heterogeneous_var: np.ndarray) -> float:
        """
        Greenberg最优化框架目标函数

        Parameters:
            weights: 资产权重
            base_weights: 基准权重
            exposure_matrix: 因子暴露矩阵
            target_exposure: 目标因子暴露
            cov_matrix: 资产协方差矩阵
            factor_cov_matrix: 因子协方差矩阵
            asset_heterogeneous_var: 资产异质风险方差

        Returns:
            目标函数值
        """
        factor_exposure = self.compute_factor_exposure(weights, exposure_matrix)
        exposure_deviation = factor_exposure - target_exposure

        active_risk = exposure_deviation @ factor_cov_matrix @ exposure_deviation

        heterogeneous_risk = weights @ np.diag(asset_heterogeneous_var) @ weights

        objective = active_risk + self.lambda_param * heterogeneous_risk

        return objective

    def combined_objective(self, weights: np.ndarray, base_weights: np.ndarray,
                          exposure_matrix: np.ndarray, target_exposure: np.ndarray,
                          cov_matrix: np.ndarray, asset_heterogeneous_var: np.ndarray) -> float:
        """
        结合Blyth和Greenberg的最优化框架目标函数

        参数与上述相同

        Returns:
            目标函数值
        """
        delta_w = weights - base_weights

        tracking_error = delta_w @ cov_matrix @ delta_w

        heterogeneous_risk = weights @ np.diag(asset_heterogeneous_var) @ weights

        factor_exposure = self.compute_factor_exposure(weights, exposure_matrix)
        exposure_deviation_norm = np.linalg.norm(factor_exposure - target_exposure)

        objective = tracking_error + self.lambda_param * heterogeneous_risk + (1 - self.lambda_param) * exposure_deviation_norm ** 2

        return objective

    def optimize_blyth(self, base_weights: np.ndarray, exposure_matrix: np.ndarray,
                      target_exposure: np.ndarray, cov_matrix: np.ndarray,
                      asset_heterogeneous_var: np.ndarray) -> np.ndarray:
        """
        使用Blyth框架优化资产权重

        Parameters:
            base_weights: 基准组合权重
            exposure_matrix: 因子暴露矩阵
            target_exposure: 目标因子暴露
            cov_matrix: 资产协方差矩阵
            asset_heterogeneous_var: 资产异质风险方差

        Returns:
            优化后的资产权重
        """
        n_assets = len(base_weights)
        bounds = [self.weight_bounds] * n_assets

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        delta_bounds = [(self.deviation_bounds[0], self.deviation_bounds[1])] * n_assets
        delta = base_weights

        def objective_with_deviation(weights):
            return self.blyth_objective(
                weights, base_weights, exposure_matrix, target_exposure,
                cov_matrix, asset_heterogeneous_var
            )

        result = minimize(
            objective_with_deviation,
            base_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            self.optimal_weights = result.x
        else:
            print(f"Blyth优化失败: {result.message}")
            self.optimal_weights = base_weights

        return self.optimal_weights

    def optimize_greenberg(self, base_weights: np.ndarray, exposure_matrix: np.ndarray,
                          target_exposure: np.ndarray, cov_matrix: np.ndarray,
                          factor_cov_matrix: np.ndarray,
                          asset_heterogeneous_var: np.ndarray) -> np.ndarray:
        """
        使用Greenberg框架优化资产权重

        Parameters:
            base_weights: 基准组合权重
            exposure_matrix: 因子暴露矩阵
            target_exposure: 目标因子暴露
            cov_matrix: 资产协方差矩阵
            factor_cov_matrix: 因子协方差矩阵
            asset_heterogeneous_var: 资产异质风险方差

        Returns:
            优化后的资产权重
        """
        n_assets = len(base_weights)
        bounds = [self.weight_bounds] * n_assets

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        def objective(weights):
            return self.greenberg_objective(
                weights, base_weights, exposure_matrix, target_exposure,
                cov_matrix, factor_cov_matrix, asset_heterogeneous_var
            )

        result = minimize(
            objective,
            base_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            self.optimal_weights = result.x
        else:
            print(f"Greenberg优化失败: {result.message}")
            self.optimal_weights = base_weights

        return self.optimal_weights

    def optimize_combined(self, base_weights: np.ndarray, exposure_matrix: np.ndarray,
                          target_exposure: np.ndarray, cov_matrix: np.ndarray,
                          asset_heterogeneous_var: np.ndarray) -> np.ndarray:
        """
        使用结合的框架优化资产权重（本文采用的方法）

        Parameters:
            base_weights: 基准组合权重
            exposure_matrix: 因子暴露矩阵
            target_exposure: 目标因子暴露
            cov_matrix: 资产协方差矩阵
            asset_heterogeneous_var: 资产异质风险方差

        Returns:
            优化后的资产权重
        """
        n_assets = len(base_weights)
        bounds = [self.weight_bounds] * n_assets

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        def objective(weights):
            return self.combined_objective(
                weights, base_weights, exposure_matrix, target_exposure,
                cov_matrix, asset_heterogeneous_var
            )

        result = minimize(
            objective,
            base_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            self.optimal_weights = result.x
        else:
            print(f"Combined优化失败: {result.message}")
            self.optimal_weights = base_weights

        return self.optimal_weights

    def risk_parity_weights(self, cov_matrix: np.ndarray, asset_names: list = None) -> np.ndarray:
        """
        计算风险平价权重

        Parameters:
            cov_matrix: 资产协方差矩阵
            asset_names: 资产名称列表

        Returns:
            风险平价权重
        """
        n_assets = cov_matrix.shape[0]

        def risk_contribution(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            target_rc = np.ones(n_assets) * portfolio_vol / n_assets
            return np.sum((risk_contrib - target_rc) ** 2)

        initial_weights = np.ones(n_assets) / n_assets
        bounds = [self.weight_bounds] * n_assets

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(
            risk_contribution,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return result.x
        else:
            return initial_weights

    def equal_weight(self, n_assets: int) -> np.ndarray:
        """
        生成等权重

        Parameters:
            n_assets: 资产数量

        Returns:
            等权重向量
        """
        return np.ones(n_assets) / n_assets

    def get_optimal_weights(self) -> np.ndarray:
        """获取最优权重"""
        return self.optimal_weights


class MacroFactorAllocator:
    """宏观因子资产配置器"""

    def __init__(self, optimizer: PortfolioOptimizer):
        self.optimizer = optimizer
        self.base_exposures = None
        self.target_exposures = None

    def set_base_exposures(self, exposures, weights: np.ndarray) -> np.ndarray:
        """
        设置基准因子暴露

        Parameters:
            exposures: 因子暴露矩阵 (N x K) - DataFrame或numpy数组
            weights: 基准资产权重 (N,)

        Returns:
            基准因子暴露 (K,)
        """
        if isinstance(exposures, pd.DataFrame):
            self.base_exposures = exposures.values.T @ weights
        else:
            self.base_exposures = exposures.T @ weights
        return self.base_exposures

    def compute_target_exposures(self, base_exposures: np.ndarray,
                                factor_deviations: dict) -> np.ndarray:
        """
        计算目标因子暴露

        Parameters:
            base_exposures: 基准因子暴露
            factor_deviations: 因子偏离字典 {因子名: 偏离值}

        Returns:
            目标因子暴露
        """
        target = base_exposures.copy()

        for i, factor_name in enumerate(['增长', '通胀', '利率', '信用', '汇率', '流动性']):
            if factor_name in factor_deviations:
                target[i] += factor_deviations[factor_name]

        self.target_exposures = target
        return target

    def allocate(self, base_weights: np.ndarray, exposure_matrix: np.ndarray,
                cov_matrix: np.ndarray, asset_heterogeneous_var: np.ndarray,
                factor_deviations: dict = None) -> np.ndarray:
        """
        进行宏观因子资产配置

        Parameters:
            base_weights: 基准资产权重
            exposure_matrix: 因子暴露矩阵
            cov_matrix: 资产协方差矩阵
            asset_heterogeneous_var: 资产异质风险方差
            factor_deviations: 因子偏离字典

        Returns:
            优化后的资产权重
        """
        if self.base_exposures is None:
            self.base_exposures = exposure_matrix.T @ base_weights

        if factor_deviations is not None:
            self.target_exposures = self.compute_target_exposures(
                self.base_exposures, factor_deviations
            )
        else:
            self.target_exposures = self.base_exposures

        optimal_weights = self.optimizer.optimize_combined(
            base_weights, exposure_matrix, self.target_exposures,
            cov_matrix, asset_heterogeneous_var
        )

        return optimal_weights


if __name__ == "__main__":
    print("测试资产配置权重优化模块...")

    n_assets = 11
    n_factors = 6

    np.random.seed(42)
    base_weights = np.ones(n_assets) / n_assets

    exposure_matrix = np.random.randn(n_assets, n_factors) * 0.3

    cov_matrix = np.random.randn(n_assets, n_assets) * 0.01
    cov_matrix = cov_matrix @ cov_matrix.T
    cov_matrix = cov_matrix / np.max(np.abs(cov_matrix)) * 0.02

    asset_heterogeneous_var = np.ones(n_assets) * 0.005

    optimizer = PortfolioOptimizer(lambda_param=0.1)

    target_exposure = exposure_matrix.T @ base_weights + np.array([0.05, 0, 0, 0, 0, 0])

    print("测试Blyth优化...")
    weights_blyth = optimizer.optimize_blyth(
        base_weights, exposure_matrix, target_exposure,
        cov_matrix, asset_heterogeneous_var
    )
    print(f"Blyth权重和: {np.sum(weights_blyth):.4f}")

    print("\n测试Combined优化...")
    weights_combined = optimizer.optimize_combined(
        base_weights, exposure_matrix, target_exposure,
        cov_matrix, asset_heterogeneous_var
    )
    print(f"Combined权重和: {np.sum(weights_combined):.4f}")

    print("\n测试风险平价...")
    weights_rp = optimizer.risk_parity_weights(cov_matrix)
    print(f"风险平价权重和: {np.sum(weights_rp):.4f}")