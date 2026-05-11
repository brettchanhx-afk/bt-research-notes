"""
蒙特卡洛模拟模块
基于AR(1)模型和几何布朗运动生成虚拟价格序列
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from scipy.linalg import cholesky

class MonteCarloSimulator:
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def generate_ar1_series(self, n_days: int, mu: float, sigma: float,
                           rho: float, initial_price: float = 1000) -> Tuple[pd.Series, pd.Series]:
        """
        基于AR(1)模型生成单资产虚拟序列
        b_t = rho * b_{t-1} + e_t, e_t ~ N(0, sigma)
        r_t = b_t - mean(b) + mu
        """
        prices = [initial_price]
        returns_list = []

        b = np.zeros(n_days)
        b[0] = np.random.normal(0, sigma)

        for t in range(1, n_days):
            e_t = np.random.normal(0, sigma)
            b[t] = rho * b[t - 1] + e_t

        b_mean = np.mean(b)
        r = b - b_mean + mu
        returns_list = r

        price = initial_price
        for rt in r[1:]:
            price = price * (1 + rt)
            prices.append(price)

        price_series = pd.Series(prices)
        return_series = pd.Series(returns_list)

        return price_series, return_series

    def generate_multi_asset_gbm(self, n_days: int, n_assets: int,
                                  mu: np.ndarray, sigma: np.ndarray,
                                  correlation_matrix: np.ndarray,
                                  initial_prices: np.ndarray = None) -> np.ndarray:
        """
        基于几何布朗运动生成多资产虚拟序列
        r_t = mu + sigma * L * e_t, where L * L^T = correlation_matrix
        """
        if initial_prices is None:
            initial_prices = np.full(n_assets, 1000.0)
        elif len(initial_prices) != n_assets:
            raise ValueError("initial_prices length must match n_assets")

        L = cholesky(correlation_matrix, lower=True)
        log_returns = np.zeros((n_days, n_assets))

        for t in range(n_days):
            e = np.random.standard_normal(n_assets)
            epsilon = L @ e
            log_returns[t] = mu + sigma * epsilon

        price_paths = np.zeros((n_days + 1, n_assets))
        price_paths[0] = initial_prices

        for t in range(n_days):
            price_paths[t + 1] = price_paths[t] * (1 + log_returns[t])

        return price_paths

    def generate_single_asset_scenarios(self, n_days: int, n_scenarios: int,
                                         mu_range: List[float], sigma_range: List[float],
                                         rho_range: List[float]) -> List[Dict]:
        """
        生成单资产多场景虚拟序列
        """
        scenarios = []
        for mu in mu_range:
            for sigma in sigma_range:
                for rho in rho_range:
                    for scenario_id in range(n_scenarios):
                        prices, returns = self.generate_ar1_series(
                            n_days, mu, sigma, rho
                        )
                        scenarios.append({
                            'scenario_id': scenario_id,
                            'mu': mu,
                            'sigma': sigma,
                            'rho': rho,
                            'prices': prices,
                            'returns': returns
                        })
        return scenarios

    def generate_multi_asset_scenarios(self, n_days: int, n_scenarios: int,
                                       mus: np.ndarray, sigmas: np.ndarray,
                                       correlation_matrix: np.ndarray) -> List[Dict]:
        """
        生成多资产虚拟序列
        """
        scenarios = []
        n_assets = len(mus)

        for scenario_id in range(n_scenarios):
            price_paths = self.generate_multi_asset_gbm(
                n_days, n_assets, mus, sigmas, correlation_matrix
            )
            scenarios.append({
                'scenario_id': scenario_id,
                'price_paths': price_paths
            })
        return scenarios

    def extract_asset_parameters(self, returns_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从实际资产收益率序列提取风险收益特征
        """
        daily_returns = returns_df.mean()
        daily_vol = returns_df.std()
        correlation = returns_df.corr().values

        return daily_returns.values, daily_vol.values, correlation

class VirtualSequenceGenerator:
    def __init__(self, trading_days: int = 252):
        self.trading_days = trading_days

    def map_to_trading_dates(self, price_series: pd.Series,
                              start_date: str = "20050104",
                              end_date: str = "20191231") -> pd.Series:
        """
        将虚拟序列映射到实际交易日
        """
        date_range = pd.bdate_range(start=start_date, end=end_date, freq='B')
        n_available = min(len(price_series), len(date_range))

        mapped_prices = pd.Series(index=date_range[:n_available], dtype=float)
        for i in range(n_available):
            mapped_prices.iloc[i] = price_series.iloc[i]

        return mapped_prices

    def create_virtual_sequences_ar1(self, mu: float, sigma: float,
                                      rho: float, n_sequences: int,
                                      n_days: int) -> List[pd.Series]:
        """
        创建多个AR(1)虚拟序列
        """
        simulator = MonteCarloSimulator()
        sequences = []

        for _ in range(n_sequences):
            prices, _ = simulator.generate_ar1_series(n_days, mu, sigma, rho)
            mapped = self.map_to_trading_dates(prices)
            sequences.append(mapped)

        return sequences

    def create_virtual_sequences_gbm(self, mus: np.ndarray, sigmas: np.ndarray,
                                     correlation_matrix: np.ndarray,
                                     n_sequences: int, n_days: int) -> List[np.ndarray]:
        """
        创建多个几何布朗运动虚拟序列
        """
        simulator = MonteCarloSimulator()
        sequences = []

        for _ in range(n_sequences):
            price_paths = simulator.generate_multi_asset_gbm(
                n_days, len(mus), mus, sigmas, correlation_matrix
            )
            sequences.append(price_paths)

        return sequences

if __name__ == "__main__":
    print("测试蒙特卡洛模拟...")

    simulator = MonteCarloSimulator(random_state=42)

    mu_test = 0.0005
    sigma_test = 0.01
    rho_test = 0.1
    n_days_test = 100

    prices, returns = simulator.generate_ar1_series(n_days_test, mu_test, sigma_test, rho_test)
    print(f"AR(1)序列 - 价格范围: {prices.min():.2f} ~ {prices.max():.2f}")
    print(f"AR(1)序列 - 收益率均值: {returns.mean():.6f}, 标准差: {returns.std():.6f}")

    n_assets = 3
    mus_test = np.array([0.001, 0.0005, 0.0008])
    sigmas_test = np.array([0.015, 0.01, 0.012])
    corr_test = np.array([[1.0, 0.3, 0.2],
                          [0.3, 1.0, 0.25],
                          [0.2, 0.25, 1.0]])

    price_paths = simulator.generate_multi_asset_gbm(n_days_test, n_assets, mus_test, sigmas_test, corr_test)
    print(f"GBM多资产序列 - 形状: {price_paths.shape}")
    print(f"GBM多资产序列 - 资产1最终价格: {price_paths[-1, 0]:.2f}")

    print("蒙特卡洛模拟测试完成!")