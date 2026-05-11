"""
动态因子模型 (Dynamic Factor Model) 模块
实现DFM模型的参数估计，使用EM算法
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

from .utils import standardize_series, fill_missing_with_interpolation


@dataclass
class DFMParameters:
    """
    DFM模型参数类
    """
    factor_loadings: np.ndarray
    factor_transition: np.ndarray
    idiosyncratic_var: np.ndarray
    latent_factors: np.ndarray
    idiosyncratic_factors: np.ndarray


class DynamicFactorModel:
    """
    动态因子模型 (DFM)

    基于研报中的描述：
    - 两个假设：经济变量同源性、经济周期内生性
    - 三个方程：隐含状态方程、隐含因子状态转移方程、特质因子状态转移方程
    """

    def __init__(self, n_factors: int = 3, n_idiosyncratic: int = 5,
                 max_iter: int = 100, tol: float = 1e-6, random_state: int = 42):
        """
        初始化DFM模型

        Parameters:
        -----------
        n_factors : int
            隐含因子数量
        n_idiosyncratic : int
            特质因子数量
        max_iter : int
            EM算法最大迭代次数
        tol : float
            收敛阈值
        random_state : int
            随机种子
        """
        self.n_factors = n_factors
        self.n_idiosyncratic = n_idiosyncratic
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.params: Optional[DFMParameters] = None
        self.is_fitted = False

        np.random.seed(random_state)

    def _initialize_parameters(self, Y: np.ndarray, n_obs: int, n_series: int) -> Dict:
        """
        初始化模型参数

        Parameters:
        -----------
        Y : np.ndarray
            观测数据 (n_series x T)
        n_obs : int
            观测数
        n_series : int
            序列数

        Returns:
        --------
        Dict
            初始参数
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y.T).T

        n_factors_init = min(self.n_factors, n_series, n_obs)
        pca = PCA(n_components=n_factors_init)
        try:
            pca_factors = pca.fit_transform(Y_scaled.T).T
        except:
            pca_factors = np.random.randn(n_factors_init, n_obs) * 0.1

        Lambda = np.zeros((n_series, self.n_factors + self.n_idiosyncratic))
        if n_series >= n_factors_init:
            loadings = pca.components_[:n_factors_init].T
            loadings = loadings / (np.std(loadings, axis=0) + 1e-10)
            Lambda[:, :n_factors_init] = loadings

        Psi = np.ones(n_series) * 0.5

        F = pca_factors[:self.n_factors, :]
        if self.n_factors > F.shape[0]:
            F = np.vstack([F, np.random.randn(self.n_factors - F.shape[0], n_obs) * 0.1])

        V = np.random.randn(self.n_idiosyncratic, n_obs) * 0.1

        A = np.eye(self.n_factors) * 0.5
        for i in range(min(self.n_factors, 3)):
            A[i, i] = 0.3 + 0.2 * np.random.rand()

        Q = np.eye(self.n_factors) * 0.1

        return {
            'Lambda': Lambda,
            'Psi': Psi,
            'F': F,
            'V': V,
            'A': A,
            'Q': Q
        }

    def _e_step(self, params: Dict, Y: np.ndarray, Y_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        EM算法的E步：估计隐含因子

        Parameters:
        -----------
        params : Dict
            当前参数
        Y : np.ndarray
            观测数据
        Y_mask : np.ndarray
            缺失值掩码 (1表示观测，0表示缺失)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, float]
            (隐含因子, 特质因子, 似然值)
        """
        Lambda = params['Lambda']
        Psi = params['Psi']
        F = params['F']
        V = params['V']
        A = params['A']
        Q = params['Q']

        n_factors = self.n_factors
        n_idio = self.n_idiosyncratic
        n_series, T = Y.shape

        Lambda_f = Lambda[:, :n_factors]
        Lambda_v = Lambda[:, n_factors:]

        Psi_inv = 1.0 / (Psi + 1e-10)
        Lambda_f_diag = Lambda_f * Psi_inv[:, np.newaxis]

        H = Lambda_f.T @ np.diag(Psi_inv) @ Lambda_f

        F_pred = np.zeros((n_factors, T))
        P_pred = np.zeros((n_factors, n_factors, T))

        for t in range(1, T):
            if t == 0:
                F_pred[:, t] = np.zeros(n_factors)
                P_pred[:, :, t] = np.eye(n_factors)
            else:
                F_pred[:, t] = A @ F[:, t-1]
                P_pred[:, :, t] = A @ P_pred[:, :, t-1] @ A.T + Q

            obs_mask = Y_mask[:, t] > 0.5
            if obs_mask.sum() > n_factors:
                y_centered = Y[obs_mask, t] - Lambda_v[obs_mask] @ V[:, t]
                S = Lambda_f[obs_mask].T @ np.diag(Psi_inv[obs_mask]) @ Lambda_f[obs_mask]

                try:
                    S_inv = np.linalg.inv(S + H + np.eye(n_factors) * 1e-6)
                    K = P_pred[:, :, t] @ Lambda_f_diag[obs_mask] @ S_inv

                    F[:, t] = F_pred[:, t] + K @ (y_centered - Lambda_f[obs_mask] @ F_pred[:, t])
                    P_pred[:, :, t] = (np.eye(n_factors) - K @ Lambda_f[obs_mask]) @ P_pred[:, :, t]
                except:
                    F[:, t] = F_pred[:, t]

        log_likelihood = self._calculate_log_likelihood(params, Y, Y_mask, F)

        return F, V, log_likelihood

    def _m_step(self, params: Dict, Y: np.ndarray, Y_mask: np.ndarray,
                F: np.ndarray, V: np.ndarray) -> Dict:
        """
        EM算法的M步：更新参数

        Parameters:
        -----------
        params : Dict
            当前参数
        Y : np.ndarray
            观测数据
        Y_mask : np.ndarray
            缺失值掩码
        F : np.ndarray
            隐含因子
        V : np.ndarray
            特质因子

        Returns:
        --------
        Dict
            更新后的参数
        """
        Lambda = params['Lambda'].copy()
        Psi = params['Psi'].copy()
        A = params['A'].copy()
        Q = params['Q'].copy()

        n_series, T = Y.shape
        n_factors = self.n_factors

        Lambda_f = Lambda[:, :n_factors]
        Lambda_v = Lambda[:, n_factors:]

        for i in range(n_series):
            obs_mask = Y_mask[i, :] > 0.5
            if obs_mask.sum() < 5:
                continue

            F_obs = F[:, obs_mask]
            y_obs = Y[i, obs_mask]

            try:
                Lambda_f_i = F_obs @ F_obs.T
                Lambda_f_i_inv = np.linalg.inv(Lambda_f_i + np.eye(n_factors) * 1e-6)

                Lambda_f[i, :] = (y_obs @ F_obs.T) @ Lambda_f_i_inv
                Psi[i] = np.mean((y_obs - Lambda_f[i, :n_factors] @ F_obs) ** 2)
                Psi[i] = max(Psi[i], 0.01)
            except:
                pass

        for i in range(n_series):
            obs_mask = Y_mask[i, :] > 0.5
            if obs_mask.sum() < 5:
                continue
            Lambda_v[i, :] = 0.1 * np.random.randn(self.n_idiosyncratic)

        for j in range(n_factors):
            try:
                F_lag = F[j, 1:]
                F_curr = F[j, :-1]

                valid = np.isfinite(F_lag) & np.isfinite(F_curr)
                if valid.sum() < 5:
                    continue

                A[j, j] = np.corrcoef(F_lag[valid], F_curr[valid])[0, 1]
                A[j, j] = np.clip(A[j, j], -0.99, 0.99)
            except:
                pass

        residuals = F[:, 1:] - A @ F[:, :-1]
        Q = np.diag(np.var(residuals, axis=1)) * 0.5
        Q = Q + np.eye(n_factors) * 0.01

        params['Lambda'] = Lambda
        params['Psi'] = Psi
        params['A'] = A
        params['Q'] = Q

        return params

    def _calculate_log_likelihood(self, params: Dict, Y: np.ndarray,
                                   Y_mask: np.ndarray, F: np.ndarray) -> float:
        """
        计算对数似然值

        Parameters:
        -----------
        params : Dict
            参数
        Y : np.ndarray
            观测数据
        Y_mask : np.ndarray
            缺失值掩码
        F : np.ndarray
            隐含因子

        Returns:
        --------
        float
            对数似然值
        """
        Lambda = params['Lambda']
        Psi = params['Psi']
        n_series, T = Y.shape
        n_factors = self.n_factors

        Lambda_f = Lambda[:, :n_factors]

        ll = 0.0
        for t in range(T):
            obs_mask = Y_mask[:, t] > 0.5
            if obs_mask.sum() < n_factors:
                continue

            y_obs = Y[obs_mask, t]
            Lambda_obs = Lambda_f[obs_mask]
            f = F[:, t]

            residual = y_obs - Lambda_obs @ f
            Psi_obs = Psi[obs_mask]

            try:
                ll_t = -0.5 * np.sum(residual ** 2 / Psi_obs) - 0.5 * np.sum(np.log(Psi_obs))
                ll += ll_t
            except:
                pass

        return ll

    def fit(self, Y: np.ndarray, mask: Optional[np.ndarray] = None) -> 'DynamicFactorModel':
        """
        拟合DFM模型

        Parameters:
        -----------
        Y : np.ndarray
            观测数据 (n_series x T)
        mask : np.ndarray, optional
            缺失值掩码

        Returns:
        --------
        self
        """
        n_series, T = Y.shape

        if mask is None:
            mask = np.ones_like(Y)

        params = self._initialize_parameters(Y, T, n_series)

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            F, V, ll = self._e_step(params, Y, mask)

            params = self._m_step(params, Y, mask, F, V)

            if iteration > 0:
                ll_change = abs(ll - prev_ll)
                if ll_change < self.tol:
                    print(f"EM算法在第{iteration + 1}次迭代后收敛")
                    break

            prev_ll = ll

            if (iteration + 1) % 20 == 0:
                print(f"EM算法迭代 {iteration + 1}, 对数似然: {ll:.4f}")

        self.params = DFMParameters(
            factor_loadings=params['Lambda'],
            factor_transition=params['A'],
            idiosyncratic_var=params['Psi'],
            latent_factors=F,
            idiosyncratic_factors=V
        )

        self.is_fitted = True
        print(f"DFM模型拟合完成")

        return self

    def get_latent_factors(self) -> np.ndarray:
        """
        获取隐含因子序列

        Returns:
        --------
        np.ndarray
            隐含因子 (n_factors x T)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        return self.params.latent_factors.copy()

    def predict(self, n_periods: int = 1) -> np.ndarray:
        """
        预测未来隐含因子

        Parameters:
        -----------
        n_periods : int
            预测期数

        Returns:
        --------
        np.ndarray
            预测的隐含因子
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")

        F = self.params.latent_factors
        A = self.params.factor_transition

        last_factors = F[:, -1:]
        predictions = []

        current_factors = last_factors.copy()
        for _ in range(n_periods):
            current_factors = A @ current_factors
            predictions.append(current_factors.copy())

        return np.hstack(predictions)

    def get_factor_loadings(self) -> np.ndarray:
        """
        获取因子载荷矩阵

        Returns:
        --------
        np.ndarray
            因子载荷 (n_series x (n_factors + n_idiosyncratic))
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        return self.params.factor_loadings.copy()


class DFMSentimentIndex:
    """
    基于DFM的行业景气度指数构建器
    """

    def __init__(self, n_factors: int = 3, n_idiosyncratic: int = 5,
                 max_iter: int = 100, tol: float = 1e-6):
        """
        初始化DFM情感指数

        Parameters:
        -----------
        n_factors : int
            隐含因子数量
        n_idiosyncratic : int
            特质因子数量
        max_iter : int
            EM算法最大迭代次数
        tol : float
            收敛阈值
        """
        self.n_factors = n_factors
        self.n_idiosyncratic = n_idiosyncratic
        self.max_iter = max_iter
        self.tol = tol

        self.model: Optional[DynamicFactorModel] = None
        self.sentiment_index: Optional[pd.Series] = None

    def build_index(self, data: pd.DataFrame, smooth: bool = True,
                     smooth_window: int = 3) -> pd.Series:
        """
        构建行业景气度指数

        Parameters:
        -----------
        data : pd.DataFrame
            输入数据 (T x n_series)
        smooth : bool
            是否平滑
        smooth_window : int
            平滑窗口大小

        Returns:
        --------
        pd.Series
            景气度指数
        """
        Y = data.values.T.astype(np.float64)

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if not np.isfinite(Y[i, j]):
                    Y[i, j] = np.nanmean(Y[i, :]) if np.any(np.isfinite(Y[i, :])) else 0

        mask = np.ones_like(Y)
        mask[np.isnan(Y)] = 0

        Y = np.nan_to_num(Y, nan=0.0)

        self.model = DynamicFactorModel(
            n_factors=self.n_factors,
            n_idiosyncratic=self.n_idiosyncratic,
            max_iter=self.max_iter,
            tol=self.tol
        )

        self.model.fit(Y, mask)

        latent_factors = self.model.get_latent_factors()

        first_factor = latent_factors[0, :]

        sentiment = pd.Series(
            first_factor,
            index=data.index,
            name='sentiment_index'
        )

        if smooth:
            sentiment = sentiment.rolling(window=smooth_window, min_periods=1).mean()

        sentiment = (sentiment - sentiment.mean()) / sentiment.std() if sentiment.std() > 1e-10 else sentiment

        self.sentiment_index = sentiment

        return sentiment

    def get_realtime_index(self, historical_data: pd.DataFrame,
                           new_observations: np.ndarray) -> float:
        """
        获取实时更新的景气度指数（新息）

        Parameters:
        -----------
        historical_data : pd.DataFrame
            历史数据
        new_observations : np.ndarray
            新观测值

        Returns:
        --------
        float
            实时景气度指数
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用build_index")

        F = self.model.get_latent_factors()
        Lambda = self.model.get_factor_loadings()

        Lambda_f = Lambda[:, :self.n_factors]

        last_factors = F[:, -1]

        if len(new_observations) == len(Lambda_f):
            residual = new_observations - Lambda_f @ last_factors

            sensitivity = np.linalg.lstsq(
                Lambda_f.T @ Lambda_f + np.eye(self.n_factors) * 0.1,
                Lambda_f.T,
                rcond=None
            )[0]

            factor_update = sensitivity @ residual

            realtime_index = last_factors[0] + factor_update[0]
        else:
            realtime_index = last_factors[0]

        return realtime_index
