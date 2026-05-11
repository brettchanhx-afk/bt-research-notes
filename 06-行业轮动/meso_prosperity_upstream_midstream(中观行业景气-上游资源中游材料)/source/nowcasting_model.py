"""
Nowcasting模型核心模块
实现简化版Nowcasting模型参数估计和景气度指数构建

参考研报: 华泰证券-中观景气度之上游资源中游材料 (2021-10-14)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class NowcastingModel:
    """
    Nowcasting模型实现

    模型方程:
    1) y_i,t = b_i*f_t + e_i,t (if w_i,t = 1) or 0 (if w_i,t = 0)
    2) f_t = a_1*f_{t-1} + a_2*f_{t-2} + delta_t
    3) e_i,t = h_i1*e_i,t-1 + h_i2*e_i,t-2 + phi_i,t

    简化求解方法:
    1) PCA初始化景气度指数
    2) OLS拟合隐含状态方程
    3) OLS拟合状态转移方程
    4) 预测缺失值
    5) 再次PCA得到最终指数
    """

    def __init__(self, n_components: int = 1, p: int = 2):
        """
        Parameters:
        -----------
        n_components : int
            隐含因子数量，通常设为1
        p : int
            自回归阶数，默认2阶
        """
        self.n_components = n_components
        self.p = p
        self.factors_ = None
        self.loadings_ = None
        self.idiosyncratic_ = None
        self.ar_params_ = None
        self.idio_ar_params_ = None
        self.scaler_ = StandardScaler()

    def _initialize_with_pca(self, X: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用PCA初始化景气度指数

        Parameters:
        -----------
        X : np.ndarray (T, n)
            代理指标矩阵
        mask : np.ndarray (T, n)
            缺失值标记矩阵，1为有效值，0为缺失

        Returns:
        --------
        f_init : np.ndarray (T,)
            初始化的隐含因子
        loadings : np.ndarray (n,)
            因子载荷
        """
        valid_mask = mask.all(axis=0)
        X_valid = X[:, valid_mask]

        if X_valid.shape[1] < self.n_components:
            valid_mask = mask.any(axis=0)
            X_valid = X[:, valid_mask]

        pca = PCA(n_components=self.n_components)
        f_init = pca.fit_transform(X_valid)
        loadings_temp = pca.components_.T

        loadings = np.zeros((X.shape[1], self.n_components))
        loadings[valid_mask] = loadings_temp

        if self.n_components == 1:
            f_init = f_init.flatten()

        return f_init, loadings

    def _estimate_loadings(self, X: np.ndarray, f: np.ndarray,
                          mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用OLS估计因子载荷和特质因子

        Parameters:
        -----------
        X : np.ndarray (T, n)
            代理指标矩阵
        f : np.ndarray (T,) or (T, r)
            隐含因子
        mask : np.ndarray (T, n)
            缺失值标记

        Returns:
        --------
        loadings : np.ndarray (n, r)
            因子载荷
        idiosyncratic : np.ndarray (T, n)
            特质因子
        """
        n = X.shape[1]
        r = f.shape[1] if len(f.shape) > 1 else 1

        if r == 1:
            f = f.reshape(-1, 1)

        loadings = np.zeros((n, r))
        idiosyncratic = np.zeros((X.shape[0], n))

        for i in range(n):
            valid_idx = mask[:, i] == 1
            if valid_idx.sum() < r + 1:
                continue

            X_i = X[valid_idx, i]
            f_valid = f[valid_idx]

            reg = Ridge(alpha=0.1)
            reg.fit(f_valid, X_i)
            loadings[i] = reg.coef_
            idiosyncratic[valid_idx, i] = X_i - reg.predict(f_valid)

        return loadings, idiosyncratic

    def _estimate_ar_params(self, f: np.ndarray) -> np.ndarray:
        """
        估计因子自回归参数

        Parameters:
        -----------
        f : np.ndarray (T,)
            隐含因子序列

        Returns:
        --------
        ar_params : np.ndarray (p,)
            自回归系数
        """
        T = len(f)
        if T < self.p + 1:
            return np.ones(self.p) / self.p

        y = f[self.p:]
        X = np.zeros((T - self.p, self.p))

        for i in range(self.p):
            X[:, i] = f[self.p - i - 1:T - i - 1]

        reg = LinearRegression()
        reg.fit(X, y)

        return reg.coef_

    def _estimate_idio_ar_params(self, e: np.ndarray, mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        估计特质因子自回归参数

        Parameters:
        -----------
        e : np.ndarray (T, n)
            特质因子矩阵
        mask : np.ndarray (T, n)
            缺失值标记

        Returns:
        --------
        idio_ar_params : dict
            每个指标的自回归系数
        """
        idio_ar_params = {}

        for i in range(e.shape[1]):
            valid_idx = mask[:, i] == 1
            e_i = e[valid_idx, i]

            if len(e_i) < self.p + 1:
                idio_ar_params[i] = np.zeros(self.p)
                continue

            y = e_i[self.p:]
            X = np.zeros((len(e_i) - self.p, self.p))

            for j in range(self.p):
                X[:, j] = e_i[self.p - j - 1:len(e_i) - j - 1]

            reg = LinearRegression()
            reg.fit(X, y)

            idio_ar_params[i] = reg.coef_

        return idio_ar_params

    def _predict_factors(self, f: np.ndarray, ar_params: np.ndarray,
                        steps: int = 1) -> np.ndarray:
        """
        预测未来因子值

        Parameters:
        -----------
        f : np.ndarray (T,)
            历史因子序列
        ar_params : np.ndarray (p,)
            自回归系数
        steps : int
            预测步数

        Returns:
        --------
        f_pred : np.ndarray (steps,)
            预测的因子值
        """
        f_pred = np.zeros(steps)

        for t in range(steps):
            f_t = 0
            for j in range(self.p):
                if t - j - 1 >= 0:
                    f_t += ar_params[j] * f_pred[t - j - 1]
                else:
                    idx = len(f) - (j - t) - 1
                    if idx >= 0:
                        f_t += ar_params[j] * f[idx]

            f_pred[t] = f_t

        return f_pred

    def _predict_idiosyncratic(self, e: np.ndarray, idio_ar_params: Dict,
                             indices: np.ndarray, mask: np.ndarray,
                             steps: int = 1) -> np.ndarray:
        """
        预测特质因子

        Parameters:
        -----------
        e : np.ndarray (T, n)
            特质因子历史
        idio_ar_params : dict
            每个指标的自回归系数
        indices : np.ndarray
            需要预测的指标索引
        mask : np.ndarray (T, n)
            缺失值标记
        steps : int
            预测步数

        Returns:
        --------
        e_pred : np.ndarray (steps, len(indices))
            预测的特质因子
        """
        e_pred = np.zeros((steps, len(indices)))

        for t in range(steps):
            for idx, i in enumerate(indices):
                e_t = 0
                ar_params = idio_ar_params.get(i, np.zeros(self.p))
                for j in range(self.p):
                    if t - j - 1 >= 0:
                        e_t += ar_params[j] * e_pred[t - j - 1, idx]
                    else:
                        e_len = e.shape[0]
                        e_idx = e_len - (j - t) - 1
                        if e_idx >= 0:
                            valid_idx = mask[e_idx, i] == 1
                            if valid_idx:
                                e_t += ar_params[j] * e[e_idx, i]

                e_pred[t, idx] = e_t

        return e_pred

    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None,
           max_iter: int = 5) -> 'NowcastingModel':
        """
        拟合Nowcasting模型

        Parameters:
        -----------
        X : np.ndarray (T, n)
            代理指标矩阵
        mask : np.ndarray (T, n), optional
            缺失值标记矩阵，1为有效值，0为缺失
            如果为None，则假设没有缺失值
        max_iter : int
            最大迭代次数

        Returns:
        --------
        self
        """
        if mask is None:
            mask = np.ones_like(X)

        X_centered = X.copy()

        for i in range(X.shape[1]):
            valid_idx = mask[:, i] == 1
            if valid_idx.any():
                mean_val = X[valid_idx, i].mean()
                X_centered[valid_idx, i] = X[valid_idx, i] - mean_val
                X_centered[~valid_idx, i] = mean_val

        f = np.zeros((X.shape[0], self.n_components))
        loadings = np.zeros((X.shape[1], self.n_components))
        e = np.zeros_like(X)
        ar_params = np.zeros(self.n_components * self.p).reshape(self.n_components, self.p)

        for iteration in range(max_iter):
            if iteration == 0:
                f_temp, loadings_temp = self._initialize_with_pca(X_centered, mask)
                f[:, 0] = f_temp if self.n_components == 1 else f_temp[:, 0]
                loadings[:, 0] = loadings_temp.flatten() if self.n_components == 1 else loadings_temp[:, 0]
            else:
                loadings, e = self._estimate_loadings(X_centered, f, mask)

                for r in range(self.n_components):
                    ar_params[r] = self._estimate_ar_params(f[:, r])

                idio_ar_params = self._estimate_idio_ar_params(e, mask)

                for t in range(X.shape[0]):
                    for i in range(X.shape[1]):
                        if mask[t, i] == 0:
                            if t == 0:
                                continue

                            f_pred = self._predict_factors(
                                f[:t, 0], ar_params[0], steps=1
                            ) if self.n_components == 1 else self._predict_factors(
                                f[:t, r], ar_params[r], steps=1
                            )

                            idx = np.where(~mask[t])[0]
                            e_pred = self._predict_idio_ar(
                                e[:t, i], idio_ar_params, i, steps=1
                            )

                            if len(e_pred) == 0:
                                continue

                            if self.n_components == 1:
                                X_centered[t, i] = loadings[i, 0] * f_pred[0] + e_pred[0]
                            else:
                                X_centered[t, i] = sum(loadings[i, r] * f_pred[r]
                                                      for r in range(self.n_components)) + e_pred[0]

                X_filled = X_centered.copy()
                for i in range(X.shape[1]):
                    valid_idx = mask[:, i] == 1
                    if (~valid_idx).any():
                        X_filled[~valid_idx, i] = X_centered[~valid_idx, i]

                pca = PCA(n_components=self.n_components)
                f_new = pca.fit_transform(X_filled)
                loadings_temp = pca.components_.T
                loadings = np.zeros((X.shape[1], self.n_components))
                loadings[mask.all(axis=0)] = loadings_temp

                if self.n_components == 1:
                    f = f_new.reshape(-1, 1)
                else:
                    f = f_new

        self.factors_ = f
        self.loadings_ = loadings
        self.idiosyncratic_ = e
        self.ar_params_ = ar_params
        self.idio_ar_params_ = self._estimate_idio_ar_params(e, mask)

        return self

    def _predict_idio_ar(self, e: np.ndarray, idio_ar_params: Dict,
                         idx: int, steps: int = 1) -> np.ndarray:
        """预测单个特质因子"""
        e_pred = np.zeros(steps)
        ar_params = idio_ar_params.get(idx, np.zeros(self.p))

        for t in range(steps):
            e_t = 0
            for j in range(self.p):
                if t - j - 1 >= 0:
                    e_t += ar_params[j] * e_pred[t - j - 1]
                else:
                    e_idx = len(e) - (j - t) - 1
                    if e_idx >= 0:
                        e_t += ar_params[j] * e[e_idx]
            e_pred[t] = e_t

        return e_pred

    def get_factors(self) -> np.ndarray:
        """获取隐含因子（景气度指数）"""
        return self.factors_

    def get_loadings(self) -> np.ndarray:
        """获取因子载荷"""
        return self.loadings_

    def predict(self, X_future: Optional[np.ndarray] = None,
               steps: int = 1) -> np.ndarray:
        """
        预测未来因子值

        Parameters:
        -----------
        X_future : np.ndarray, optional
            未来的代理指标值
        steps : int
            预测步数

        Returns:
        --------
        f_pred : np.ndarray
            预测的因子值
        """
        if self.factors_ is None:
            raise ValueError("模型尚未拟合")

        if self.n_components == 1:
            return self._predict_factors(self.factors_, self.ar_params_[0], steps)
        else:
            f_pred = np.zeros((steps, self.n_components))
            for r in range(self.n_components):
                f_pred[:, r] = self._predict_factors(
                    self.factors_[:, r], self.ar_params_[r], steps
                )
            return f_pred


class SentimentIndexBuilder:
    """景气度指数构建器"""

    def __init__(self, industry_name: str):
        self.industry_name = industry_name
        self.model = NowcastingModel(n_components=1, p=2)
        self.indicators = {}
        self.realtime_index = None
        self.global_index = None

    def add_indicator(self, name: str, data: pd.DataFrame,
                     date_col: str = 'trade_date', value_col: str = 'value'):
        """
        添加代理指标

        Parameters:
        -----------
        name : str
            指标名称
        data : pd.DataFrame
            指标数据
        date_col : str
            日期列名
        value_col : str
            值列名
        """
        if name not in self.indicators:
            self.indicators[name] = {}

        df = data.copy()
        if date_col in df.columns:
            df = df.set_index(date_col)

        self.indicators[name]['data'] = df[value_col]
        self.indicators[name]['value_col'] = value_col

    def build_index(self, start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> Tuple[pd.Series, pd.Series]:
        """
        构建景气度指数

        Parameters:
        -----------
        start_date : str, optional
            开始日期
        end_date : str, optional
            结束日期

        Returns:
        --------
        realtime_idx : pd.Series
            实时景气度指数
        global_idx : pd.Series
            全局景气度指数
        """
        if len(self.indicators) == 0:
            raise ValueError("请先添加代理指标")

        all_dates = set()
        for name, ind_data in self.indicators.items():
            dates = ind_data['data'].index
            if isinstance(dates, pd.DatetimeIndex):
                all_dates.update(dates)

        if start_date:
            all_dates = {d for d in all_dates if d >= pd.to_datetime(start_date)}
        if end_date:
            all_dates = {d for d in all_dates if d <= pd.to_datetime(end_date)}

        all_dates = sorted(all_dates)

        indicator_matrix = []
        mask_matrix = []
        indicator_names = []

        for name, ind_data in self.indicators.items():
            indicator_names.append(name)
            data = ind_data['data']

            values = []
            masks = []

            for date in all_dates:
                if date in data.index:
                    val = data.loc[date]
                    if pd.notna(val):
                        values.append(val)
                        masks.append(1)
                    else:
                        values.append(0)
                        masks.append(0)
                else:
                    values.append(0)
                    masks.append(0)

            indicator_matrix.append(values)
            mask_matrix.append(masks)

        X = np.array(indicator_matrix).T
        mask = np.array(mask_matrix).T

        self.model.fit(X, mask)

        factors = self.model.get_factors()

        if len(factors) != len(all_dates):
            min_len = min(len(factors), len(all_dates))
            factors = factors[:min_len]
            all_dates = all_dates[:min_len]

        self.global_index = pd.Series(factors, index=all_dates)

        self.realtime_index = self.global_index

        return self.realtime_index, self.global_index

    def get_indicator_loadings(self) -> pd.DataFrame:
        """
        获取各指标的因子载荷

        Returns:
        --------
        pd.DataFrame
        """
        if self.model.loadings_ is None:
            return pd.DataFrame()

        loadings = self.model.get_loadings()
        indicator_names = list(self.indicators.keys())

        df = pd.DataFrame({
            'indicator': indicator_names,
            'loading': loadings.flatten()
        })

        return df.sort_values('loading', ascending=False)


def calculate_roe_reproduction(index: pd.Series, roe: pd.Series) -> float:
    """
    计算ROE复现度（R²）

    Parameters:
    -----------
    index : pd.Series
        景气度指数
    roe : pd.Series
        ROE_TTM

    Returns:
    --------
    r2 : float
        R²值
    """
    common_idx = index.index.intersection(roe.index)

    if len(common_idx) < 10:
        return 0.0

    index_aligned = index.loc[common_idx]
    roe_aligned = roe.loc[common_idx]

    correlation = index_aligned.corr(roe_aligned)

    return correlation ** 2


def calculate_direction_accuracy(index: pd.Series, roe: pd.Series,
                                 window: int = 3) -> Tuple[float, float]:
    """
    计算景气度变化方向预测准确率

    Parameters:
    -----------
    index : pd.Series
        景气度指数
    roe : pd.Series
        ROE_TTM
    window : int
        滚动窗口大小，默认3个月

    Returns:
    --------
    latest_direction_acc : float
        最新一期方向准确率
    prediction_acc : float
        下期预测准确率
    """
    index_rolled = index.rolling(window).mean()
    roe_rolled = roe.rolling(window).mean()

    index_change = index_rolled.diff(12)
    roe_change = roe_rolled.diff(4)

    common_idx = index_change.index.intersection(roe_change.index)
    index_change = index_change.loc[common_idx]
    roe_change = roe_change.loc[common_idx]

    index_direction = (index_change > 0).astype(int)
    roe_direction = (roe_change > 0).astype(int)

    valid_idx = ~(index_direction.isna() | roe_direction.isna())
    index_direction = index_direction[valid_idx]
    roe_direction = roe_direction[valid_idx]

    if len(index_direction) < 5:
        return 0.0, 0.0

    latest_direction_acc = (index_direction.iloc[-1] == roe_direction.iloc[-1])

    index_pred = index.shift(-1)
    index_pred_rolled = index_pred.rolling(window).mean()
    index_pred_change = index_pred_rolled.diff(12)

    index_pred_direction = (index_pred_change > 0)

    common_pred_idx = index_pred_direction.index.intersection(roe_change.index)
    index_pred_dir = index_pred_direction.loc[common_pred_idx]
    roe_dir = roe_change.loc[common_pred_idx]

    valid_pred_idx = ~(index_pred_dir.isna() | roe_dir.isna())
    index_pred_dir = index_pred_dir[valid_pred_idx]
    roe_dir = roe_dir[valid_pred_idx]

    if len(index_pred_dir) < 5:
        return float(latest_direction_acc), 0.0

    prediction_acc = (index_pred_dir.iloc[-1] == roe_dir.iloc[-1])

    return float(latest_direction_acc), float(prediction_acc)


if __name__ == '__main__':
    print("测试Nowcasting模型...")

    np.random.seed(42)
    T = 100
    n_indicators = 10

    f_true = np.cumsum(np.random.randn(T))
    loadings = np.random.rand(n_indicators) * 0.5 + 0.5
    e = np.random.randn(T, n_indicators) * 0.3

    X = f_true.reshape(-1, 1) * loadings.reshape(1, -1) + e

    model = NowcastingModel(n_components=1, p=2)
    model.fit(X)

    factors = model.get_factors()
    print(f"估计的因子长度: {len(factors)}")
    print(f"因子前10个值: {factors[:10]}")

    print("\n测试景气度指数构建器...")
    builder = SentimentIndexBuilder("测试行业")

    dates = pd.date_range('2010-01-01', periods=100, freq='M')
    for i in range(n_indicators):
        data = pd.DataFrame({
            'trade_date': dates,
            'value': X[:, i]
        })
        builder.add_indicator(f'indicator_{i}', data)

    realtime_idx, global_idx = builder.build_index()

    print(f"实时景气度指数长度: {len(realtime_idx)}")
    print(f"全局景气度指数长度: {len(global_idx)}")

    loadings_df = builder.get_indicator_loadings()
    print(f"\n因子载荷:\n{loadings_df.head()}")
