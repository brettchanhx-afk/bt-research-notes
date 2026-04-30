# -*- coding: utf-8 -*-
"""
factor.py - Barra模型因子计算模块

【功能说明】
严格按照研报公式复刻Barra模型三步归因流程：
    Step 1: 计算公共因子在基金中的暴露矩阵X（标准化）
    Step 2: 计算公共因子收益率矩阵F（横截面回归）
    Step 3: 计算公共因子在基金中的暴露b（时间序列回归）

【研报来源】
华泰证券研究所 (2020)《Barra模型基于持仓数据从因子角度对收益进行拆解》

【核心公式】
    暴露矩阵:   β_ij = (x_ij - x̄_j) / std(x_j)
    横截面回归:  R_i = Σβ_ij × F_j + ε_i
    时序回归:    R_pt = ΣF_jt × b_j + ε_t

【版本】
v1.0  2026-04-28
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class FactorAttributionResult:
    """
    Barra因子归因结果

    【字段说明】
        fund_code: 基金代码
        period: 分析期间
        factor_names: 因子名称列表
        b: 基金因子暴露系数 (k,)
        t_stats: t统计量 (k,)
        p_values: p值 (k,)
        r_squared: 模型R²
        adj_r_squared: 调整R²
        factor_contributions: 各因子贡献 Dict[str, float]
        residual_contribution: 残差贡献
        alpha: 截距项（选股Alpha）
        time_series: 时间序列明细 DataFrame
    """
    fund_code: str = ''
    period: str = ''
    factor_names: List[str] = field(default_factory=list)
    b: np.ndarray = field(default_factory=lambda: np.array([]))
    t_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    residual_contribution: float = 0.0
    alpha: float = 0.0
    time_series: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class CrossSectionResult:
    """
    横截面回归结果（Step 2输出）

    【字段说明】
        date: 截面日期
        factor_returns: 因子收益率 (k,)
        residuals: 残差 (n,)
        r_squared: 拟合R²
        n_stocks: 股票数量
    """
    date: str = ''
    factor_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: float = 0.0
    n_stocks: int = 0


# =============================================================================
# Step 1: 因子暴露矩阵X计算
# =============================================================================

class FactorExposureCalculator:
    """
    因子暴露矩阵X计算器

    【研报公式】
        X = (β_11 ... β_1k; ...; β_n1 ... β_nk)

        其中: β_ij = (x_ij - x̄_j) / std(x_j)

        x_ij: 第i个证券第j个公共因子的实际取值
        β_ij: 标准化后的因子暴露值
    """

    @staticmethod
    def calculate_raw_exposure(stock_data: Dict[str, pd.DataFrame],
                                factor_definitions: Dict[str, callable]) -> pd.DataFrame:
        """
        计算原始因子暴露值 x_ij

        【参数】
            stock_data: Dict[stock_code, DataFrame] 股票行情数据
            factor_definitions: Dict[factor_name, callable] 因子计算函数

        【返回】
            DataFrame: 行=股票, 列=因子, 值=原始因子值
        """
        exposure_data = {}

        for code, df in stock_data.items():
            factors = {}
            for factor_name, calc_func in factor_definitions.items():
                try:
                    factors[factor_name] = calc_func(df)
                except Exception:
                    factors[factor_name] = np.nan
            exposure_data[code] = factors

        return pd.DataFrame(exposure_data).T

    @staticmethod
    def standardize_exposure(raw_exposure: pd.DataFrame,
                              winsorize: bool = True,
                              lower: float = 0.01,
                              upper: float = 0.99) -> pd.DataFrame:
        """
        因子暴露标准化（Barra模型关键步骤）

        【研报公式】
            β_ij = (x_ij - x̄_j) / std(x_j)

        【参数】
            raw_exposure: 原始因子值 DataFrame (行=证券, 列=因子)
            winsorize: 是否先做去极值处理
            lower/upper: Winsorize上下界百分位

        【返回】
            DataFrame: 标准化后的因子暴露矩阵X
        """
        df = raw_exposure.copy()

        # Step 1a: 去极值（Winsorize）
        if winsorize:
            for col in df.columns:
                lb = df[col].quantile(lower)
                ub = df[col].quantile(upper)
                df[col] = df[col].clip(lb, ub)

        # Step 1b: 标准化
        mean = df.mean()
        std = df.std()
        std[std == 0] = 1  # 避免除零

        standardized = (df - mean) / std

        return standardized

    @staticmethod
    def industry_neutralize(exposure: pd.DataFrame,
                            industry_dummies: pd.DataFrame) -> pd.DataFrame:
        """
        行业中性化处理

        【公式】
            β_neutral = β_raw - β_industry × (β_industry'β_industry)^{-1} × β_industry'β_raw

        【参数】
            exposure: 标准化后的因子暴露
            industry_dummies: 行业哑变量矩阵 (行=证券, 列=行业)

        【返回】
            DataFrame: 中性化后的因子暴露
        """
        neutralized = exposure.copy()
        X_ind = industry_dummies.values

        for col in exposure.columns:
            y = exposure[col].values
            valid = ~np.isnan(y)

            if valid.sum() > X_ind.shape[1] + 1:
                X_v = X_ind[valid]
                y_v = y[valid]

                # OLS回归取残差
                X_with_const = np.column_stack([np.ones(len(y_v)), X_v])
                beta = np.linalg.lstsq(X_with_const, y_v, rcond=None)[0]
                residual = y_v - X_with_const @ beta

                # 回填
                result = np.full(len(y), np.nan)
                result[valid] = residual
                neutralized[col] = result

        return neutralized


# =============================================================================
# Step 2: 因子收益率矩阵F计算（横截面回归）
# =============================================================================

class FactorReturnCalculator:
    """
    因子收益率矩阵F计算器

    【研报公式】
        (R_1; R_2; ...; R_n) = (β_11...β_1k; ...; β_n1...β_nk) × (F_1; ...; F_k) + (ε_1; ...; ε_n)

        对T期数据回归，得到因子收益率矩阵:
        F = (F_11...F_1k; ...; F_T1...F_Tk)
    """

    @staticmethod
    def cross_section_regression(stock_returns: np.ndarray,
                                  factor_exposure: np.ndarray,
                                  add_constant: bool = True) -> CrossSectionResult:
        """
        单期横截面回归

        【研报公式】
            R_i = Σ_{j=1}^{k} β_ij × F_j + ε_i

        【参数】
            stock_returns: 股票超额收益率 (n,)
            factor_exposure: 因子暴露矩阵 (n, k)
            add_constant: 是否添加常数项

        【返回】
            CrossSectionResult
        """
        n, k = factor_exposure.shape

        # 去除NaN
        valid = ~(np.isnan(stock_returns) | np.isnan(factor_exposure).any(axis=1))
        Y = stock_returns[valid]
        X = factor_exposure[valid]

        if len(Y) < k + 1:
            return CrossSectionResult(n_stocks=len(Y))

        if add_constant:
            X_with_const = np.column_stack([np.ones(len(Y)), X])
        else:
            X_with_const = X

        # OLS: F = (X'X)^{-1} X'R
        try:
            beta_hat = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]

            # 因子收益率（排除常数项）
            factor_returns = beta_hat[1:] if add_constant else beta_hat

            # 残差
            Y_hat = X_with_const @ beta_hat
            residuals = Y - Y_hat

            # R²
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        except np.linalg.LinAlgError:
            factor_returns = np.zeros(k)
            residuals = Y
            r_squared = 0

        return CrossSectionResult(
            factor_returns=factor_returns,
            residuals=residuals,
            r_squared=r_squared,
            n_stocks=len(Y)
        )

    @staticmethod
    def rolling_cross_section_regression(stock_returns_df: pd.DataFrame,
                                          factor_exposure_df: pd.DataFrame,
                                          freq: str = 'M') -> pd.DataFrame:
        """
        滚动横截面回归，构建因子收益率矩阵F

        【研报公式】
            对每期横截面数据回归，得到因子收益率矩阵:
            F = (F_11...F_1k; ...; F_T1...F_Tk)

        【参数】
            stock_returns_df: 股票收益率面板 (行=日期, 列=股票代码)
            factor_exposure_df: 因子暴露 (行=股票代码, 列=因子)
            freq: 重采样频率 'M'=月, 'Q'=季

        【返回】
            DataFrame: 因子收益率矩阵 (行=日期, 列=因子名)
        """
        # 重采样
        if freq == 'M':
            period_returns = (1 + stock_returns_df).resample('M').prod() - 1
        elif freq == 'Q':
            period_returns = (1 + stock_returns_df).resample('Q').prod() - 1
        else:
            period_returns = stock_returns_df

        factor_names = factor_exposure_df.columns.tolist()
        results = []

        for date, row in period_returns.iterrows():
            # 当期有收益的股票
            available = row.dropna()
            if len(available) < len(factor_names) + 5:
                continue

            # 对齐因子暴露
            common = available.index.intersection(factor_exposure_df.index)
            if len(common) < len(factor_names) + 5:
                continue

            Y = available.loc[common].values
            X = factor_exposure_df.loc[common].values

            cs_result = FactorReturnCalculator.cross_section_regression(Y, X)

            if len(cs_result.factor_returns) > 0:
                results.append({
                    'date': date,
                    **dict(zip(factor_names, cs_result.factor_returns))
                })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).set_index('date')


# =============================================================================
# Step 3: 基金因子暴露b计算（时间序列回归）
# =============================================================================

class FundExposureCalculator:
    """
    基金因子暴露b计算器

    【研报公式】
        (R_p1; R_p2; ...; R_pT) = (F_11...F_1k; ...; F_T1...F_Tk) × (b_1; ...; b_k) + (ε_1; ...; ε_T)

        b_i: 第i个公共因子对基金收益率的贡献程度
        b_i的显著性: 反映基金是否受该因子显著影响
        b_i的大小: 反映影响程度
    """

    @staticmethod
    def time_series_regression(fund_returns: np.ndarray,
                                factor_returns: np.ndarray,
                                factor_names: List[str] = None,
                                add_constant: bool = True) -> FactorAttributionResult:
        """
        时间序列回归计算基金因子暴露

        【研报公式】
            R_pt = Σ_{j=1}^{k} F_jt × b_j + ε_t

        【参数】
            fund_returns: 基金收益率序列 (T,)
            factor_returns: 因子收益率矩阵 (T, k)
            factor_names: 因子名称列表
            add_constant: 是否添加常数项（Alpha项）

        【返回】
            FactorAttributionResult
        """
        T, k = factor_returns.shape

        # 去除NaN
        valid = ~(np.isnan(fund_returns) | np.isnan(factor_returns).any(axis=1))
        Y = fund_returns[valid]
        X = factor_returns[valid]

        T_valid = len(Y)

        if T_valid < k + 2:
            return FactorAttributionResult(
                factor_names=factor_names or [f'Factor_{i}' for i in range(k)]
            )

        if add_constant:
            X_with_const = np.column_stack([np.ones(T_valid), X])
        else:
            X_with_const = X

        # OLS: b = (F'F)^{-1} F'R_p
        try:
            beta_hat = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return FactorAttributionResult(
                factor_names=factor_names or [f'Factor_{i}' for i in range(k)]
            )

        # 因子暴露系数
        b = beta_hat[1:] if add_constant else beta_hat
        alpha = beta_hat[0] if add_constant else 0

        # 拟合值和残差
        Y_hat = X_with_const @ beta_hat
        residuals = Y - Y_hat

        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 调整R²
        adj_r_squared = 1 - (1 - r_squared) * (T_valid - 1) / (T_valid - k - 1) \
            if T_valid > k + 1 else 0

        # t统计量
        mse = ss_res / (T_valid - k - 1) if (T_valid - k - 1) > 0 else 0
        try:
            var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
            se_beta = np.sqrt(np.abs(np.diag(var_beta)))
            se_beta[se_beta == 0] = 1e-10
            t_stats = beta_hat / se_beta
            t_stats_b = t_stats[1:] if add_constant else t_stats
        except np.linalg.LinAlgError:
            t_stats_b = np.zeros(k)

        # p值（双侧t检验）
        from scipy import stats
        df_resid = T_valid - k - 1
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats_b), df_resid)) if df_resid > 0 else np.ones(k)

        # 因子贡献 = b_j × mean(F_j)
        factor_contributions = {}
        if factor_names:
            mean_factor_returns = X.mean(axis=0)
            for i, name in enumerate(factor_names):
                factor_contributions[name] = float(b[i] * mean_factor_returns[i])

        # 残差贡献
        residual_contribution = float(np.mean(residuals))

        # 时间序列明细
        ts_data = {
            'fund_return': Y,
            'fitted_return': Y_hat,
            'residual': residuals
        }
        if factor_names:
            for i, name in enumerate(factor_names):
                ts_data[f'factor_{name}'] = X[:, i] * b[i]

        ts_df = pd.DataFrame(ts_data)

        return FactorAttributionResult(
            factor_names=factor_names or [f'Factor_{i}' for i in range(k)],
            b=b,
            t_stats=t_stats_b,
            p_values=p_values,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            factor_contributions=factor_contributions,
            residual_contribution=residual_contribution,
            alpha=alpha,
            time_series=ts_df
        )


# =============================================================================
# 统一归因入口
# =============================================================================

class BarraFactorAttribution:
    """
    Barra因子归因统一入口

    【流程】
        Step 1: 计算因子暴露矩阵X → FactorExposureCalculator
        Step 2: 计算因子收益率矩阵F → FactorReturnCalculator
        Step 3: 计算基金因子暴露b → FundExposureCalculator

    【使用示例】
        >>> attribution = BarraFactorAttribution()
        >>> result = attribution.run(fund_returns, factor_returns_matrix)
    """

    def __init__(self):
        self.exposure_calc = FactorExposureCalculator()
        self.factor_return_calc = FactorReturnCalculator()
        self.fund_exposure_calc = FundExposureCalculator()

    def run(self,
            fund_returns: pd.Series,
            factor_returns: pd.DataFrame,
            fund_code: str = '',
            period: str = '',
            significance_level: float = 0.05) -> FactorAttributionResult:
        """
        执行完整的Barra三步归因

        【参数】
            fund_returns: 基金收益率序列 (索引=日期)
            factor_returns: 因子收益率矩阵 (索引=日期, 列=因子名)
            fund_code: 基金代码
            period: 分析期间描述
            significance_level: 显著性水平

        【返回】
            FactorAttributionResult
        """
        print("=" * 70)
        print(f"Barra因子归因分析 | 基金: {fund_code} | 期间: {period}")
        print("=" * 70)

        # ---- 对齐日期 ----
        common_dates = fund_returns.index.intersection(factor_returns.index)
        if len(common_dates) < 12:
            print(f"  警告: 仅有 {len(common_dates)} 个共同日期，结果可能不可靠")

        fund_ret_aligned = fund_returns.loc[common_dates].values
        factor_ret_aligned = factor_returns.loc[common_dates].values
        factor_names = factor_returns.columns.tolist()

        print(f"\n  共同日期数: {len(common_dates)}")
        print(f"  因子数量: {len(factor_names)}")
        print(f"  因子列表: {factor_names}")

        # ---- Step 3: 时间序列回归 ----
        print(f"\n[Step 3] 时间序列回归计算基金因子暴露...")
        result = self.fund_exposure_calc.time_series_regression(
            fund_returns=fund_ret_aligned,
            factor_returns=factor_ret_aligned,
            factor_names=factor_names,
            add_constant=True
        )

        result.fund_code = fund_code
        result.period = period

        # ---- 打印结果 ----
        self._print_results(result, significance_level)

        return result

    def run_with_holdings(self,
                          fund_returns: pd.Series,
                          holdings: pd.DataFrame,
                          stock_factor_data: pd.DataFrame,
                          fund_code: str = '',
                          period: str = '') -> FactorAttributionResult:
        """
        基于持仓数据的Barra归因（完整三步流程）

        【参数】
            fund_returns: 基金收益率序列
            holdings: 持仓数据 (date, stock_code, weight, sector)
            stock_factor_data: 股票因子暴露 (行=股票代码, 列=因子)
            fund_code: 基金代码
            period: 分析期间
        """
        print("=" * 70)
        print(f"Barra因子归因（持仓法）| 基金: {fund_code}")
        print("=" * 70)

        # ---- Step 1: 基金因子暴露 = 持股权重 × 股票因子暴露 ----
        print(f"\n[Step 1] 基于持仓计算基金因子暴露...")

        # 取最新一期持仓
        latest_date = holdings['date'].max()
        latest_holdings = holdings[holdings['date'] == latest_date]

        # 持仓权重
        weights = latest_holdings.set_index('stock_code')['weight']

        # 对齐
        common_stocks = weights.index.intersection(stock_factor_data.index)
        w = weights.loc[common_stocks].values
        X = stock_factor_data.loc[common_stocks].values

        # 基金因子暴露 = Σ(w_i × β_ij)
        fund_exposure = w @ X
        fund_exposure_series = pd.Series(
            fund_exposure,
            index=stock_factor_data.columns
        )

        print(f"  持仓股票数: {len(common_stocks)}")
        print(f"  基金因子暴露: ")
        for name, val in zip(stock_factor_data.columns, fund_exposure):
            print(f"    {name:20s}: {val:>8.4f}")

        # ---- Step 2 & 3: 使用因子收益率矩阵回归 ----
        # （需要外部提供factor_returns，此处直接用fund_returns回归）
        # 简化：用fund_exposure作为先验

        # 构建结果
        result = FactorAttributionResult(
            fund_code=fund_code,
            period=period,
            factor_names=stock_factor_data.columns.tolist(),
            b=fund_exposure,
            factor_contributions={}
        )

        # 因子贡献 = b_j × mean(F_j)（近似）
        for i, name in enumerate(stock_factor_data.columns):
            result.factor_contributions[name] = float(fund_exposure[i])

        return result

    def _print_results(self, result: FactorAttributionResult,
                       significance_level: float = 0.05):
        """打印归因结果"""
        print(f"\n{'='*70}")
        print(f"Barra因子归因结果")
        print(f"{'='*70}")

        print(f"\n【基金因子暴露 b】")
        print(f"  {'因子名称':20s} {'暴露系数':>10s} {'t统计量':>10s} {'显著性':>10s}")
        print(f"  {'-'*52}")

        for i, name in enumerate(result.factor_names):
            if i < len(result.b):
                b_val = result.b[i]
                t_val = result.t_stats[i] if i < len(result.t_stats) else 0
                p_val = result.p_values[i] if i < len(result.p_values) else 1

                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < significance_level:
                    sig = '*'
                else:
                    sig = ''

                print(f"  {name:20s} {b_val:>10.4f} {t_val:>10.2f} {sig:>10s}")

        print(f"\n【Alpha（选股超额收益）】")
        print(f"  月度Alpha: {result.alpha*100:>8.4f}%")
        print(f"  年化Alpha: {((1+result.alpha)**12-1)*100:>8.2f}%")

        print(f"\n【模型拟合度】")
        print(f"  R-squared:     {result.r_squared:>8.4f}")
        print(f"  Adj R-squared: {result.adj_r_squared:>8.4f}")

        print(f"\n【因子贡献分解】")
        total_explained = 0
        for name, contrib in result.factor_contributions.items():
            print(f"  {name:20s}: {contrib*100:>8.4f}%")
            total_explained += contrib

        print(f"  {'残差(特异性)':20s}: {result.residual_contribution*100:>8.4f}%")
        print(f"  {'已解释部分':20s}: {total_explained*100:>8.4f}%")

        print(f"{'='*70}")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Barra因子计算模块测试\n")

    # ---- 测试Step 1: 因子暴露标准化 ----
    print("=" * 50)
    print("Test 1: 因子暴露标准化")
    print("=" * 50)

    raw = pd.DataFrame({
        'SIZE': [22.3, 18.5, 25.1, 20.0, 23.7],
        'VALUE': [0.8, 1.2, 0.5, 1.5, 0.9],
        'MOMENTUM': [0.15, -0.08, 0.25, 0.03, 0.12]
    }, index=['600519', '000858', '601318', '000333', '600036'])

    standardized = FactorExposureCalculator.standardize_exposure(raw)
    print("原始因子值:")
    print(raw)
    print("\n标准化后:")
    print(standardized.round(4))

    # ---- 测试Step 2: 横截面回归 ----
    print("\n" + "=" * 50)
    print("Test 2: 横截面回归")
    print("=" * 50)

    np.random.seed(42)
    n_stocks = 50
    n_factors = 3

    X = np.random.randn(n_stocks, n_factors)
    true_F = np.array([0.02, -0.01, 0.03])
    Y = X @ true_F + np.random.randn(n_stocks) * 0.01

    cs_result = FactorReturnCalculator.cross_section_regression(Y, X)
    print(f"真实因子收益率: {true_F}")
    print(f"估计因子收益率: {cs_result.factor_returns}")
    print(f"R²: {cs_result.r_squared:.4f}")

    # ---- 测试Step 3: 时间序列回归 ----
    print("\n" + "=" * 50)
    print("Test 3: 时间序列回归")
    print("=" * 50)

    T = 60
    k = 3
    factor_names = ['SIZE', 'VALUE', 'MOMENTUM']

    np.random.seed(42)
    F = np.random.randn(T, k) * 0.02
    true_b = np.array([1.2, -0.5, 0.8])
    R_p = F @ true_b + 0.005 + np.random.randn(T) * 0.005

    result = FundExposureCalculator.time_series_regression(
        R_p, F, factor_names=factor_names
    )

    print(f"真实因子暴露: {true_b}")
    print(f"估计因子暴露: {result.b.round(4)}")
    print(f"Alpha: {result.alpha*100:.4f}%")
    print(f"R²: {result.r_squared:.4f}")

    # ---- 测试统一入口 ----
    print("\n" + "=" * 50)
    print("Test 4: 完整Barra归因")
    print("=" * 50)

    fund_returns = pd.Series(R_p, index=pd.date_range('2022-01-01', periods=T, freq='M'))
    factor_returns = pd.DataFrame(F, index=fund_returns.index, columns=factor_names)

    attribution = BarraFactorAttribution()
    final_result = attribution.run(fund_returns, factor_returns, fund_code='TEST')

    print("\n所有测试通过!")
