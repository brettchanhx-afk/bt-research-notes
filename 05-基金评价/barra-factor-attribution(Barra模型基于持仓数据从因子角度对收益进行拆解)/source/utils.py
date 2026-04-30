# -*- coding: utf-8 -*-
"""
utils.py - Barra模型因子归因工具函数模块

【功能说明】
提供数据清洗、格式转换、数学计算、回归分析等通用工具函数

【依赖库】
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0

【作者】
金融工程量化工程师

【版本】
v1.0  2026-04-28

【更新时间】
2026-04-28: 初始版本
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# 第一部分：数据验证与处理
# =============================================================================

def validate_dataframe(df: pd.DataFrame, required_cols: List[str],
                       raise_error: bool = True) -> Tuple[bool, List[str]]:
    """
    验证DataFrame是否包含必需的列

    【参数】
        df: 待验证的DataFrame
        required_cols: 必需的列名列表
        raise_error: 是否在验证失败时抛出异常

    【返回】
        Tuple[bool, List[str]]: (验证是否通过, 缺失列列表)
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    is_valid = len(missing_cols) == 0

    if not is_valid and raise_error:
        raise ValueError(f"DataFrame缺少必需列: {missing_cols}")

    return is_valid, missing_cols


def validate_weights(weights: Union[pd.Series, np.ndarray],
                    tolerance: float = 0.01) -> bool:
    """
    验证权重之和是否接近1（或100%）

    【参数】
        weights: 权重序列
        tolerance: 容差范围，默认1%

    【返回】
        bool: 验证是否通过
    """
    if isinstance(weights, pd.Series):
        total = weights.sum()
    else:
        total = np.sum(weights)

    # 支持小数形式(0.9-1.1)和百分比形式(90-110)
    if total > 1.5:
        target = 100.0
    else:
        target = 1.0

    return abs(total - target) <= tolerance * target


def handle_missing_data(df: pd.DataFrame,
                       method: str = 'drop',
                       fill_value: Any = None) -> pd.DataFrame:
    """
    处理缺失数据

    【参数】
        df: 输入DataFrame
        method: 处理方法 ('drop', 'ffill', 'bfill', 'zero', 'mean', 'median', 'value')
        fill_value: 当method='value'时使用的填充值

    【返回】
        pd.DataFrame: 处理后的DataFrame
    """
    df_copy = df.copy()

    if method == 'drop':
        return df_copy.dropna()
    elif method == 'ffill':
        return df_copy.fillna(method='ffill')
    elif method == 'bfill':
        return df_copy.fillna(method='bfill')
    elif method == 'zero':
        return df_copy.fillna(0)
    elif method == 'mean':
        return df_copy.fillna(df_copy.mean())
    elif method == 'median':
        return df_copy.fillna(df_copy.median())
    elif method == 'value':
        return df_copy.fillna(fill_value)
    else:
        raise ValueError(f"不支持的缺失值处理方法: {method}")


def remove_outliers(data: pd.Series,
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.Series:
    """
    去除异常值

    【参数】
        data: 输入数据
        method: 'iqr'（四分位距）或 'zscore'（Z分数）
        threshold: 阈值（IQR倍数或Z分数标准差倍数）

    【返回】
        pd.Series: 处理后的数据
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores <= threshold]
    else:
        raise ValueError(f"不支持的异常值检测方法: {method}")


# =============================================================================
# 第二部分：收益率计算
# =============================================================================

def calculate_simple_return(prices: pd.Series) -> pd.Series:
    """
    计算简单收益率

    【公式】
        r_t = (P_t - P_{t-1}) / P_{t-1}

    【参数】
        prices: 价格序列

    【返回】
        pd.Series: 收益率序列
    """
    return prices.pct_change()


def calculate_log_return(prices: pd.Series) -> pd.Series:
    """
    计算对数收益率

    【公式】
        r_t = ln(P_t / P_{t-1})

    【注意】
        对数收益率具有时间可加性，适合用于多期累计收益计算
    """
    return np.log(prices / prices.shift(1))


def calculate_cumulative_return(returns: pd.Series,
                                 method: str = 'simple') -> float:
    """
    计算累计收益率

    【公式】
        - 简单法: R = Π(1 + r_i) - 1
        - 对数法: R = Σln(1 + r_i)

    【参数】
        returns: 收益率序列
        method: 'simple' 或 'log'

    【返回】
        float: 累计收益率
    """
    if method == 'simple':
        return (1 + returns.fillna(0)).prod() - 1
    elif method == 'log':
        return np.log((1 + returns.fillna(0)).prod())
    else:
        raise ValueError(f"不支持的方法: {method}")


def geometric_link(returns: List[float]) -> float:
    """
    几何链接计算多期累计收益【Barra模型核心】

    【公式】
        R_T = (1 + r_1)(1 + r_2)...(1 + r_T) - 1
    """
    if not returns:
        return 0.0

    cumulative = 1.0
    for r in returns:
        cumulative *= (1 + r)

    return cumulative - 1


def annualize_return(period_return: float,
                    periods_per_year: int,
                    method: str = 'compound') -> float:
    """
    年化收益率

    【公式】
        - 复合年化: R_annual = (1 + R_period)^(n/T) - 1
    """
    if method == 'compound':
        return (1 + period_return) ** periods_per_year - 1
    else:
        return period_return * periods_per_year


# =============================================================================
# 第三部分：因子暴露计算
# =============================================================================

def standardize_exposure(factor_values: pd.DataFrame) -> pd.DataFrame:
    """
    因子暴露标准化（Barra模型关键步骤）

    【公式】
        β_ij = (x_ij - μ_j) / σ_j

    【参数】
        factor_values: 因子原始值DataFrame (行=证券, 列=因子)

    【返回】
        pd.DataFrame: 标准化后的因子暴露矩阵

    【说明】
        Barra模型中，将因子暴露进行标准化处理，使得不同因子的暴露具有可比性
    """
    # 计算每列的均值和标准差
    mean = factor_values.mean()
    std = factor_values.std()

    # 标准化处理（避免除以零）
    std[std == 0] = 1
    standardized = (factor_values - mean) / std

    return standardized


def winsorize_factor(factor_values: pd.DataFrame,
                    lower: float = 0.01,
                    upper: float = 0.99) -> pd.DataFrame:
    """
    因子暴露去极值处理（Winsorize）

    【公式】
        x_ij = min(max(x_ij, P_lower), P_upper)

    【参数】
        factor_values: 因子原始值
        lower: 下界百分位
        upper: 上界百分位
    """
    winsorized = factor_values.copy()
    for col in winsorized.columns:
        lower_bound = winsorized[col].quantile(lower)
        upper_bound = winsorized[col].quantile(upper)
        winsorized[col] = winsorized[col].clip(lower_bound, upper_bound)
    return winsorized


def neutralize_factor(factor_values: pd.DataFrame,
                     neutralize_columns: List[str]) -> pd.DataFrame:
    """
    因子中性化处理

    【公式】
        β_neutral = β_raw - β_industry × (β_industry'β_industry)^(-1) × β_industry'β_raw

    【参数】
        factor_values: 待中性化的因子暴露矩阵
        neutralize_columns: 用于中性化的列（如行业哑变量）
    """
    # 简化处理：残差法
    neutralized = factor_values.copy()

    for col in factor_values.columns:
        if col not in neutralize_columns:
            # 对每个因子，分别对中性化变量回归
            X = factor_values[neutralize_columns].values
            y = factor_values[col].values

            if X.shape[1] > 0 and len(y) > X.shape[1]:
                try:
                    # 添加常数项
                    X_with_const = np.column_stack([np.ones(len(y)), X])
                    # OLS回归
                    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    # 计算残差
                    residual = y - X_with_const @ beta
                    neutralized[col] = residual
                except:
                    pass

    return neutralized


# =============================================================================
# 第四部分：回归分析
# =============================================================================

def cross_sectional_regression(Y: np.ndarray,
                               X: np.ndarray,
                               add_constant: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    横截面回归（Barra模型第二步）

    【公式】
        R_i = Σβ_ij × F_j + ε_i

    【参数】
        Y: 股票超额收益率 (n×1)
        X: 因子暴露矩阵 (n×k)
        add_constant: 是否添加常数项

    【返回】
        Tuple[np.ndarray, np.ndarray]: (因子收益率, 残差)
    """
    if add_constant:
        X_with_const = np.column_stack([np.ones(len(Y)), X])
    else:
        X_with_const = X

    # OLS估计: β = (X'X)^(-1) X'y
    try:
        beta, residuals, _, _, _ = np.linalg.lstsq(X_with_const, Y, rcond=None)
        factor_returns = beta[1:] if add_constant else beta  # 排除常数项
        residual = residuals if len(residuals) > 0 else Y - X_with_const @ beta
    except:
        factor_returns = np.zeros(X.shape[1])
        residual = Y

    return factor_returns, residual


def time_series_regression(Y: np.ndarray,
                          X: np.ndarray,
                          add_constant: bool = True) -> Dict[str, float]:
    """
    时间序列回归（Barra模型第三步）

    【公式】
        R_pt = ΣF_jt × b_j + ε_t

    【参数】
        Y: 组合收益率序列 (T×1)
        X: 因子收益率矩阵 (T×k)
        add_constant: 是否添加常数项

    【返回】
        Dict: {
            'b': 因子暴露系数,
            'residual': 残差,
            'r_squared': 'R-squared',
            't_stats': t统计量
        }
    """
    if add_constant:
        X_with_const = np.column_stack([np.ones(len(Y)), X])
    else:
        X_with_const = X

    # OLS估计
    try:
        beta, residuals, rank, s, _ = np.linalg.lstsq(X_with_const, Y, rcond=None)

        # 计算R-squared
        y_pred = X_with_const @ beta
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 计算t统计量
        n = len(Y)
        k = X_with_const.shape[1]
        mse = ss_res / (n - k) if (n - k) > 0 else 0
        var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_beta = np.sqrt(np.diag(var_beta))
        t_stats = beta / se_beta

        result = {
            'b': beta[1:] if add_constant else beta,
            'residual': residuals if len(residuals) > 0 else Y - y_pred,
            'r_squared': r_squared,
            't_stats': t_stats[1:] if add_constant else t_stats,
            'const': beta[0] if add_constant else 0
        }
    except Exception as e:
        result = {
            'b': np.zeros(X.shape[1]),
            'residual': Y,
            'r_squared': 0,
            't_stats': np.zeros(X.shape[1]),
            'const': 0
        }

    return result


def calculate_factor_contribution(factor_exposure: np.ndarray,
                                  factor_returns: np.ndarray) -> np.ndarray:
    """
    计算各因子对组合收益的贡献

    【公式】
        Contribution_j = b_j × F_j

    【参数】
        factor_exposure: 因子暴露 (k×1)
        factor_returns: 因子收益率 (k×1)

    【返回】
        np.ndarray: 各因子贡献 (k×1)
    """
    return factor_exposure * factor_returns


# =============================================================================
# 第五部分：风险指标计算
# =============================================================================

def calculate_tracking_error(fund_returns: pd.Series,
                            benchmark_returns: pd.Series) -> float:
    """
    计算跟踪误差

    【公式】
        TE = std(R_p - R_b)
    """
    excess_returns = fund_returns - benchmark_returns
    return excess_returns.std()


def calculate_information_ratio(fund_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               periods_per_year: int = 252) -> float:
    """
    计算信息比率

    【公式】
        IR = (E[R_p] - E[R_b]) / TE
    """
    excess_returns = fund_returns - benchmark_returns
    mean_excess = excess_returns.mean()
    tracking_error = excess_returns.std()

    if tracking_error == 0:
        return 0

    return (mean_excess * periods_per_year) / (tracking_error * np.sqrt(periods_per_year))


def calculate_max_drawdown(nav: pd.Series) -> Tuple[float, str, str]:
    """
    计算最大回撤

    【公式】
        MDD = min(NAV_t / max(NAV_0:t) - 1)
    """
    peak = nav.cummax()
    drawdown = (nav - peak) / peak

    max_dd_idx = drawdown.idxmin()
    peak_idx = nav[:max_dd_idx].idxmax() if max_dd_idx in nav.index else None

    return drawdown[max_dd_idx], str(peak_idx), str(max_dd_idx)


def calculate_sharpe_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.03,
                           periods_per_year: int = 252) -> float:
    """
    计算夏普比率

    【公式】
        SR = (E[R_p] - R_f) / σ_p
    """
    annual_return = returns.mean() * periods_per_year
    annual_vol = returns.std() * np.sqrt(periods_per_year)

    if annual_vol == 0:
        return 0

    return (annual_return - risk_free_rate) / annual_vol


# =============================================================================
# 第六部分：格式转换与输出
# =============================================================================

def format_percentage(value: float,
                      decimal: int = 2,
                      include_sign: bool = True) -> str:
    """
    将小数格式化为百分比字符串
    """
    if include_sign and value > 0:
        return f"+{value * 100:.{decimal}f}%"
    else:
        return f"{value * 100:.{decimal}f}%"


def format_number(value: float, decimal: int = 2) -> str:
    """
    格式化数字为千分位字符串
    """
    return f"{value:,.{decimal}f}"


def print_attribution_summary(attribution_result: Dict[str, Any]):
    """
    打印Barra归因结果摘要

    【参数】
        attribution_result: Barra归因结果字典
    """
    print("=" * 70)
    print("Barra因子归因结果")
    print("=" * 70)

    print("\n【因子暴露系数】")
    b = attribution_result.get('b', [])
    factor_names = attribution_result.get('factor_names', [])
    t_stats = attribution_result.get('t_stats', [])

    for i, (name, bi, ti) in enumerate(zip(factor_names, b, t_stats)):
        sig = "***" if abs(ti) > 3 else "**" if abs(ti) > 2 else "*" if abs(ti) > 1.5 else ""
        print(f"  {name:20s}: {bi:>8.4f}  (t={ti:>6.2f}) {sig}")

    print(f"\n【模型拟合度】")
    print(f"  R-squared:    {attribution_result.get('r_squared', 0):.4f}")
    print(f"  年化跟踪误差: {attribution_result.get('tracking_error', 0)*100:.2f}%")

    print(f"\n【因子贡献分解】")
    contributions = attribution_result.get('contributions', {})
    for name, contrib in contributions.items():
        print(f"  {name:20s}: {format_percentage(contrib)}")

    print("=" * 70)


def save_attribution_results(result: Dict[str, Any],
                            output_dir: str,
                            prefix: str = "barra"):
    """
    保存归因结果到CSV文件

    【参数】
        result: 归因结果字典
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子暴露结果
    if 'b' in result and len(result['b']) > 0:
        exposure_df = pd.DataFrame({
            'factor': result.get('factor_names', []),
            'exposure': result['b'],
            't_statistic': result.get('t_stats', []),
            'contribution': result.get('contributions', {}).values()
        })
        exposure_file = os.path.join(output_dir, f"{prefix}_factor_exposure.csv")
        exposure_df.to_csv(exposure_file, index=False, encoding='utf-8-sig')
        print(f"因子暴露结果已保存: {exposure_file}")

    # 保存时间序列数据
    if 'time_series' in result:
        ts_file = os.path.join(output_dir, f"{prefix}_time_series.csv")
        result['time_series'].to_csv(ts_file, index=False, encoding='utf-8-sig')
        print(f"时间序列数据已保存: {ts_file}")


# =============================================================================
# 第七部分：数据对齐与合并
# =============================================================================

def align_dates(dfs: List[pd.DataFrame],
              date_col: str = 'date') -> List[pd.DataFrame]:
    """
    对齐多个DataFrame的日期索引
    """
    if not dfs:
        return []

    all_dates = set()
    for df in dfs:
        if date_col in df.columns:
            all_dates.update(df[date_col].unique())

    common_dates = sorted(list(all_dates))

    aligned_dfs = []
    for df in dfs:
        if date_col in df.columns:
            df_aligned = df.set_index(date_col).reindex(common_dates).reset_index()
            df_aligned.columns = [date_col] + list(df_aligned.columns[1:])
            aligned_dfs.append(df_aligned)
        else:
            aligned_dfs.append(df)

    return aligned_dfs


def merge_factor_data(holdings: pd.DataFrame,
                     factor_data: pd.DataFrame,
                     on_stock: bool = True) -> pd.DataFrame:
    """
    合并持仓数据与因子数据
    """
    if on_stock and 'stock_code' in holdings.columns and 'stock_code' in factor_data.columns:
        return pd.merge(holdings, factor_data, on='stock_code', how='left')
    else:
        return pd.merge(holdings, factor_data, left_index=True, right_index=True, how='left')


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("测试工具函数模块...")

    # 测试标准化
    factor_values = pd.DataFrame({
        'SIZE': [1e10, 5e10, 1e11, 5e11],
        'VALUE': [0.5, 1.0, 1.5, 2.0]
    })
    standardized = standardize_exposure(factor_values)
    print("\n原始因子值:")
    print(factor_values)
    print("\n标准化后:")
    print(standardized)

    # 测试横截面回归
    Y = np.array([0.05, 0.03, 0.08, 0.02])
    X = np.array([
        [1.0, 0.5],
        [0.8, 0.3],
        [1.2, 0.7],
        [0.6, 0.2]
    ])
    factor_returns, residual = cross_sectional_regression(Y, X)
    print("\n因子收益率:", factor_returns)

    # 测试时间序列回归
    Y = np.array([0.01, 0.02, -0.01, 0.015, 0.025, 0.01])
    X = np.random.randn(6, 3)
    result = time_series_regression(Y, X)
    print("\n因子暴露系数:", result['b'])
    print("R-squared:", result['r_squared'])

    print("\n工具函数模块测试完成!")
