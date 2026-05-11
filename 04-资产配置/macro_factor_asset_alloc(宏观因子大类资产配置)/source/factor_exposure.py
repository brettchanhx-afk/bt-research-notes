"""
因子暴露度计算模块
使用基于先验信息的LASSO模型计算资产对各宏观因子的暴露
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FactorExposureCalculator:
    """因子暴露度计算器"""

    def __init__(self, alpha: float = 0.01, max_iter: int = 5000, random_state: int = 42):
        """
        初始化因子暴露度计算器

        Parameters:
            alpha: LASSO正则化参数
            max_iter: 最大迭代次数
            random_state: 随机种子
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.exposures = None

    def set_prior_signs(self) -> dict:
        """
        设置因子暴露的先验符号（基于经济学意义）

        Returns:
            先验符号字典 {因子名: 符号}
            1 表示正相关，-1 表示负相关，0 表示无先验
        """
        prior_signs = {
            '增长': {
                '沪深300': 1, '中证500': 1, '恒生指数': 1,
                '中债国债': -1, '中债企业债': 0, '中证转债': 0,
                '南华工业品': 1, '南华农产品': 0, '布伦特原油': 1,
                '沪金': 0, '美元兑人民币': -1
            },
            '通胀': {
                '沪深300': -1, '中证500': 0, '恒生指数': -1,
                '中债国债': -1, '中债企业债': -1, '中证转债': -1,
                '南华工业品': 1, '南华农产品': 1, '布伦特原油': 1,
                '沪金': 1, '美元兑人民币': 0
            },
            '利率': {
                '沪深300': -1, '中证500': -1, '恒生指数': -1,
                '中债国债': 1, '中债企业债': 1, '中证转债': 1,
                '南华工业品': 0, '南华农产品': 0, '布伦特原油': 0,
                '沪金': 0, '美元兑人民币': 0
            },
            '信用': {
                '沪深300': 0, '中证500': 0, '恒生指数': 0,
                '中债国债': -1, '中债企业债': 1, '中证转债': 1,
                '南华工业品': 0, '南华农产品': 0, '布伦特原油': 0,
                '沪金': 0, '美元兑人民币': 0
            },
            '汇率': {
                '沪深300': -1, '中证500': -1, '恒生指数': -1,
                '中债国债': 0, '中债企业债': 0, '中证转债': 0,
                '南华工业品': 0, '南华农产品': 0, '布伦特原油': 0,
                '沪金': 1, '美元兑人民币': 1
            },
            '流动性': {
                '沪深300': 1, '中证500': 1, '恒生指数': 1,
                '中债国债': -1, '中债企业债': 0, '中证转债': 1,
                '南华工业品': 0, '南华农产品': 0, '布伦特原油': 0,
                '沪金': 0, '美元兑人民币': -1
            }
        }
        return prior_signs

    def fit_lasso_with_prior(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                            prior_signs: dict = None) -> pd.DataFrame:
        """
        使用带先验信息的LASSO回归计算因子暴露

        Parameters:
            asset_returns: 资产收益率 (T x N)
            factor_returns: 因子收益率 (T x K)
            prior_signs: 先验符号字典

        Returns:
            DataFrame: 因子暴露矩阵 (N x K)
        """
        if asset_returns.empty or factor_returns.empty:
            return pd.DataFrame()

        common_idx = asset_returns.index.intersection(factor_returns.index)
        if len(common_idx) == 0:
            return pd.DataFrame()

        X = factor_returns.loc[common_idx].values
        assets = asset_returns.columns.tolist()
        factors = factor_returns.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)

        exposures = {}

        for i, asset in enumerate(assets):
            y = asset_returns.loc[common_idx, asset].values

            model = Lasso(
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
                warm_start=False
            )

            try:
                model.fit(X_scaled, y)

                coef = model.coef_

                if prior_signs is not None and asset in prior_signs.get('增长', {}):
                    for j, factor in enumerate(factors):
                        if factor in prior_signs and asset in prior_signs[factor]:
                            sign = prior_signs[factor][asset]
                            if sign != 0 and np.sign(coef[j]) != sign and np.abs(coef[j]) > 1e-6:
                                coef[j] = np.abs(coef[j]) * sign

                exposures[asset] = coef

            except Exception as e:
                print(f"LASSO回归失败 {asset}: {e}")
                exposures[asset] = np.zeros(len(factors))

        self.exposures = pd.DataFrame(exposures, index=factors).T

        return self.exposures

    def fit_cv_lasso(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                    alphas: list = None) -> pd.DataFrame:
        """
        使用交叉验证的LASSO回归计算因子暴露

        Parameters:
            asset_returns: 资产收益率
            factor_returns: 因子收益率
            alphas: 正则化参数候选列表

        Returns:
            DataFrame: 因子暴露矩阵
        """
        if alphas is None:
            alphas = np.logspace(-4, 0, 20)

        common_idx = asset_returns.index.intersection(factor_returns.index)
        if len(common_idx) == 0:
            return pd.DataFrame()

        X = factor_returns.loc[common_idx].values
        assets = asset_returns.columns.tolist()
        factors = factor_returns.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)

        exposures = {}

        for asset in assets:
            y = asset_returns.loc[common_idx, asset].values

            model = LassoCV(
                alphas=alphas,
                cv=5,
                max_iter=self.max_iter,
                random_state=self.random_state
            )

            try:
                model.fit(X_scaled, y)
                exposures[asset] = model.coef_
            except Exception as e:
                print(f"CV-LASSO回归失败 {asset}: {e}")
                exposures[asset] = np.zeros(len(factors))

        self.exposures = pd.DataFrame(exposures, index=factors).T

        return self.exposures

    def get_exposures(self) -> pd.DataFrame:
        """获取因子暴露矩阵"""
        return self.exposures

    def predict_asset_returns(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        基于因子暴露预测资产收益

        Parameters:
            factor_returns: 因子收益率

        Returns:
            预测的资产收益率
        """
        if self.exposures is None:
            return pd.DataFrame()

        predicted = pd.DataFrame(
            self.exposures.values @ factor_returns.T.values,
            index=self.exposures.index,
            columns=factor_returns.index
        ).T

        return predicted

    def get_residual_risks(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.Series:
        """
        计算各资产的残差风险（异质风险）

        Parameters:
            asset_returns: 实际资产收益率
            factor_returns: 因子收益率

        Returns:
            各资产的残差风险
        """
        if self.exposures is None:
            return pd.Series()

        common_idx = asset_returns.index.intersection(factor_returns.index)
        if len(common_idx) == 0:
            return pd.Series()

        residuals = {}

        for asset in asset_returns.columns:
            if asset in self.exposures.index:
                exposure = self.exposures.loc[asset].values
                predicted = factor_returns.loc[common_idx].values @ exposure
                actual = asset_returns.loc[common_idx, asset].values
                residual = actual - predicted
                residuals[asset] = np.std(residual)

        return pd.Series(residuals)

    def analyze_exposure_stability(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                                  rolling_window: int = 36) -> pd.DataFrame:
        """
        分析因子暴露的稳定性（滚动窗口）

        Parameters:
            asset_returns: 资产收益率
            factor_returns: 因子收益率
            rolling_window: 滚动窗口大小

        Returns:
            滚动暴露度DataFrame
        """
        common_idx = asset_returns.index.intersection(factor_returns.index)
        if len(common_idx) < rolling_window:
            return pd.DataFrame()

        dates = common_idx[rolling_window:]
        rolling_exposures = {}

        for i, date in enumerate(dates):
            start_idx = i
            end_idx = i + rolling_window

            window_assets = asset_returns.loc[common_idx[start_idx:end_idx]]
            window_factors = factor_returns.loc[common_idx[start_idx:end_idx]]

            window_exposures = self.fit_lasso_with_prior(
                window_assets, window_factors,
                prior_signs=self.set_prior_signs()
            )

            if not window_exposures.empty:
                rolling_exposures[date] = window_exposures.mean()

        if rolling_exposures:
            return pd.DataFrame(rolling_exposures).T
        return pd.DataFrame()


class FactorExposureWithPrior(FactorExposureCalculator):
    """带先验信息的因子暴露计算器"""

    def __init__(self, alpha: float = 0.01, max_iter: int = 5000, random_state: int = 42):
        super().__init__(alpha, max_iter, random_state)
        self.prior_signs = self.set_prior_signs()

    def fit(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        使用带先验信息的LASSO回归拟合模型

        Parameters:
            asset_returns: 资产收益率
            factor_returns: 因子收益率

        Returns:
            因子暴露矩阵
        """
        return self.fit_lasso_with_prior(asset_returns, factor_returns, self.prior_signs)


if __name__ == "__main__":
    print("测试因子暴露度计算模块...")

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

    factor_returns = pd.DataFrame({
        '增长': np.random.randn(n) * 0.015,
        '通胀': np.random.randn(n) * 0.01,
        '利率': np.random.randn(n) * 0.008,
        '信用': np.random.randn(n) * 0.012,
        '汇率': np.random.randn(n) * 0.01,
        '流动性': np.random.randn(n) * 0.015,
    }, index=dates)

    calculator = FactorExposureCalculator(alpha=0.01)
    exposures = calculator.fit_lasso_with_prior(asset_returns, factor_returns)
    print(f"因子暴露矩阵形状: {exposures.shape}")
    print(f"因子暴露矩阵:\n{exposures}")

    residual_risks = calculator.get_residual_risks(asset_returns, factor_returns)
    print(f"\n残差风险:\n{residual_risks}")