"""
宏观因子构建模块
通过PCA降维和资产组合方式构造增长、通胀、利率、信用、汇率、流动性六大宏观因子
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MacroFactorBuilder:
    """宏观因子构建器"""

    def __init__(self, n_factors: int = 6):
        """
        初始化宏观因子构建器

        Parameters:
            n_factors: 因子数量，默认6个（增长、通胀、利率、信用、汇率、流动性）
        """
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors, whiten=False)
        self.scaler = StandardScaler()
        self.factor_names = ['增长', '通胀', '利率', '信用', '汇率', '流动性'][:n_factors]

    def fit_pca_factors(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """
        使用PCA方法从资产收益中提取因子

        Parameters:
            asset_returns: 资产收益率矩阵 (T x N)

        Returns:
            DataFrame: 因子收益率 (T x K)
        """
        returns_clean = asset_returns.dropna()
        if returns_clean.empty:
            return pd.DataFrame()

        returns_scaled = self.scaler.fit_transform(returns_clean)

        factors = self.pca.fit_transform(returns_scaled)

        factor_df = pd.DataFrame(
            factors,
            index=returns_clean.index,
            columns=[f'PCA_Factor_{i+1}' for i in range(self.n_factors)]
        )

        return factor_df

    def construct_growth_factor(self, asset_returns: pd.DataFrame) -> pd.Series:
        """
        构造增长因子
        使用股票指数（沪深300、中证500、恒生指数）和商品指数（南华工业品）的组合

        Parameters:
            asset_returns: 资产收益率

        Returns:
            增长因子收益率序列
        """
        equity_assets = ['沪深300', '中证500', '恒生指数']
        commodity_assets = ['南华工业品']

        growth = pd.Series(0.0, index=asset_returns.index)

        for asset in equity_assets:
            if asset in asset_returns.columns:
                growth = growth + asset_returns[asset].fillna(0)

        for asset in commodity_assets:
            if asset in asset_returns.columns:
                growth = growth + 0.5 * asset_returns[asset].fillna(0)

        equity_weight = sum(1 for a in equity_assets if a in asset_returns.columns)
        growth = growth / equity_weight if equity_weight > 0 else growth

        return growth

    def construct_inflation_factor(self, asset_returns: pd.DataFrame) -> pd.Series:
        """
        构造通胀因子
        使用南华农产品、布伦特原油、沪金等商品类资产

        Parameters:
            asset_returns: 资产收益率

        Returns:
            通胀因子收益率序列
        """
        commodity_assets = ['南华农产品', '布伦特原油', '沪金']

        inflation = pd.Series(0.0, index=asset_returns.index)

        for asset in commodity_assets:
            if asset in asset_returns.columns:
                inflation = inflation + asset_returns[asset].fillna(0)

        n_assets = sum(1 for a in commodity_assets if a in asset_returns.columns)
        inflation = inflation / n_assets if n_assets > 0 else inflation

        return inflation

    def construct_interest_rate_factor(self, asset_returns: pd.DataFrame) -> pd.Series:
        """
        构造利率因子
        使用债券类资产（中债国债、中证转债）

        Parameters:
            asset_returns: 资产收益率

        Returns:
            利率因子收益率序列
        """
        bond_assets = ['中债国债', '中证转债']

        rate = pd.Series(0.0, index=asset_returns.index)

        for asset in bond_assets:
            if asset in asset_returns.columns:
                rate = rate + asset_returns[asset].fillna(0)

        n_assets = sum(1 for a in bond_assets if a in asset_returns.columns)
        rate = rate / n_assets if n_assets > 0 else rate

        return rate

    def construct_credit_factor(self, asset_returns: pd.DataFrame) -> pd.Series:
        """
        构造信用因子
        使用企业债与国债的利差组合

        Parameters:
            asset_returns: 资产收益率

        Returns:
            信用因子收益率序列
        """
        if '中债企业债' in asset_returns.columns and '中债国债' in asset_returns.columns:
            credit = asset_returns['中债企业债'].fillna(0) - asset_returns['中债国债'].fillna(0)
        elif '中债企业债' in asset_returns.columns:
            credit = asset_returns['中债企业债'].fillna(0)
        else:
            credit = pd.Series(0.0, index=asset_returns.index)

        return credit

    def construct_exchange_rate_factor(self, asset_returns: pd.DataFrame) -> pd.Series:
        """
        构造汇率因子
        使用美元兑人民币汇率

        Parameters:
            asset_returns: 资产收益率

        Returns:
            汇率因子收益率序列
        """
        if '美元兑人民币' in asset_returns.columns:
            fx = asset_returns['美元兑人民币'].fillna(0)
        else:
            fx = pd.Series(0.0, index=asset_returns.index)

        return fx

    def construct_liquidity_factor(self, asset_returns: pd.DataFrame) -> pd.Series:
        """
        构造流动性因子
        使用中小盘股票（中证500）相对大盘股票（沪深300）的超额收益

        Parameters:
            asset_returns: 资产收益率

        Returns:
            流动性因子收益率序列
        """
        if '中证500' in asset_returns.columns and '沪深300' in asset_returns.columns:
            liquidity = asset_returns['中证500'].fillna(0) - asset_returns['沪深300'].fillna(0)
        elif '中证500' in asset_returns.columns:
            liquidity = asset_returns['中证500'].fillna(0)
        else:
            liquidity = pd.Series(0.0, index=asset_returns.index)

        return liquidity

    def construct_all_factors(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """
        构造所有六大宏观因子

        Parameters:
            asset_returns: 资产收益率

        Returns:
            DataFrame: 宏观因子收益率 (T x 6)
        """
        factors = pd.DataFrame(index=asset_returns.index)

        factors['增长'] = self.construct_growth_factor(asset_returns)
        factors['通胀'] = self.construct_inflation_factor(asset_returns)
        factors['利率'] = self.construct_interest_rate_factor(asset_returns)
        factors['信用'] = self.construct_credit_factor(asset_returns)
        factors['汇率'] = self.construct_exchange_rate_factor(asset_returns)
        factors['流动性'] = self.construct_liquidity_factor(asset_returns)

        return factors

    def high_frequency_factor(self, raw_factors: pd.DataFrame, window: int = 12) -> pd.DataFrame:
        """
        将低频宏观因子高频化
        使用滚动窗口的资产组合方式

        Parameters:
            raw_factors: 原始低频因子
            window: 滚动窗口（月度数据使用12个月）

        Returns:
            高频化的因子序列
        """
        hf_factors = raw_factors.rolling(window=window, min_periods=window//2).apply(
            lambda x: (1 + x.prod()) ** (12 / len(x)) - 1 if len(x) > 0 else 0
        )

        return hf_factors

    def orthogonalize_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        对因子进行对称正交化（Lowdin正交化）

        Parameters:
            factors: 原始因子矩阵

        Returns:
            正交化后的因子矩阵
        """
        factors_clean = factors.dropna()
        if factors_clean.empty:
            return factors

        try:
            Phi = factors_clean.values
            U, D, Vt = np.linalg.svd(Phi, full_matrices=False)
            orthogonalized = U @ Vt

            result = pd.DataFrame(
                orthogonalized,
                index=factors_clean.index,
                columns=factors.columns
            )

            return result

        except Exception as e:
            print(f"正交化失败: {e}")
            return factors

    def get_factor_covariance(self, factors: pd.DataFrame) -> np.ndarray:
        """
        计算因子收益率的协方差矩阵

        Parameters:
            factors: 因子收益率

        Returns:
            K x K 协方差矩阵
        """
        return factors.cov().values

    def analyze_factor_explained_variance(self, asset_returns: pd.DataFrame) -> dict:
        """
        分析各因子对资产收益的解释程度

        Parameters:
            asset_returns: 资产收益率

        Returns:
            解释方差比例字典
        """
        pca_factors = self.fit_pca_factors(asset_returns)
        explained_var = self.pca.explained_variance_ratio_

        return {
            f'Factor_{i+1}': var
            for i, var in enumerate(explained_var)
        }


def build_raw_macro_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    构建原始宏观因子（用于对比高频因子）

    Parameters:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        原始宏观因子DataFrame
    """
    from data_fetcher import get_all_assets_data, resample_to_monthly

    asset_data = get_all_assets_data(start_date, end_date)

    if not asset_data:
        print("警告: 未能获取资产数据")
        return pd.DataFrame()

    monthly_returns = {}
    for asset, df in asset_data.items():
        if not df.empty:
            monthly = resample_to_monthly(df)
            monthly_returns[asset] = monthly['return']

    if not monthly_returns:
        return pd.DataFrame()

    asset_returns_df = pd.DataFrame(monthly_returns)

    builder = MacroFactorBuilder(n_factors=6)
    raw_factors = builder.construct_all_factors(asset_returns_df)

    return raw_factors


if __name__ == "__main__":
    print("测试宏观因子构建模块...")

    test_data = pd.DataFrame({
        '沪深300': np.random.randn(100) * 0.02,
        '中证500': np.random.randn(100) * 0.02,
        '中债国债': np.random.randn(100) * 0.005,
        '中债企业债': np.random.randn(100) * 0.006,
        '中证转债': np.random.randn(100) * 0.015,
        '南华工业品': np.random.randn(100) * 0.025,
        '南华农产品': np.random.randn(100) * 0.02,
        '布伦特原油': np.random.randn(100) * 0.03,
        '沪金': np.random.randn(100) * 0.02,
        '美元兑人民币': np.random.randn(100) * 0.01,
        '恒生指数': np.random.randn(100) * 0.025,
    }, index=pd.date_range('2020-01-01', periods=100, freq='M'))

    builder = MacroFactorBuilder(n_factors=6)

    pca_factors = builder.fit_pca_factors(test_data)
    print(f"PCA因子形状: {pca_factors.shape}")

    manual_factors = builder.construct_all_factors(test_data)
    print(f"手动因子形状: {manual_factors.shape}")

    orthogonal_factors = builder.orthogonalize_factors(manual_factors)
    print(f"正交化因子形状: {orthogonal_factors.shape}")

    explained = builder.analyze_factor_explained_variance(test_data)
    print(f"解释方差: {explained}")