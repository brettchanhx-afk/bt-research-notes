# -*- coding: utf-8 -*-
"""
工具函数模块 - 基金选股择时能力定量评价模型
包含：数据清洗、格式转换、日志打印等通用工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

# ==================== matplotlib 中文字体设置 ====================
def setup_chinese_font():
    """
    设置 matplotlib 中文字体。
    必须在 plt.style.use() 之后调用，防止字体配置被覆盖。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 120

    # 可选：清除字体缓存
    cache_dir = matplotlib.get_cachedir()
    font_cache = os.path.join(cache_dir, 'fontlist-v330.json')
    if os.path.exists(font_cache):
        try:
            os.remove(font_cache)
        except OSError:
            pass


import os
setup_chinese_font()


# ==================== 数据清洗工具 ====================
def clean_returns(df: pd.DataFrame, col: str = 'returns') -> pd.DataFrame:
    """
    清洗收益率数据中的异常值和缺失值。

    参数:
        df: 输入 DataFrame
        col: 收益率列名
    返回:
        清洗后的 DataFrame
    """
    df = df.copy()
    # 处理无穷值
    df = df.replace([np.inf, -np.inf], np.nan)
    # 剔除极端收益率（超过10倍标准差视为异常）
    if col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        df.loc[(df[col] - mean).abs() > 10 * std, col] = np.nan
    # 删除全空行
    df = df.dropna(how='all')
    return df


def align_dates(fund_df: pd.DataFrame, benchmark_df: pd.DataFrame,
                how: str = 'inner') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐基金与基准的日期，确保两者日期一致。

    参数:
        fund_df: 基金收益率 DataFrame，index 为日期
        benchmark_df: 基准收益率 DataFrame，index 为日期
        how: 对齐方式，'inner' 仅保留两者共同的日期
    返回:
        (对齐后的基金df, 对齐后的基准df)
    """
    # 确保索引是 DatetimeIndex
    if not isinstance(fund_df.index, pd.DatetimeIndex):
        fund_df = fund_df.copy()
        fund_df.index = pd.to_datetime(fund_df.index)
    if not isinstance(benchmark_df.index, pd.DatetimeIndex):
        benchmark_df = benchmark_df.copy()
        benchmark_df.index = pd.to_datetime(benchmark_df.index)

    # 按日期对齐
    aligned_fund, aligned_bench = fund_df.align(benchmark_df, join=how, how=how)
    return aligned_fund, aligned_bench


# ==================== 回归结果打印工具 ====================
def print_ols_summary(result, model_name: str = "") -> None:
    """
    打印 OLS 回归结果的格式化摘要（兼容 GBK 编码环境）。

    参数:
        result: statsmodels.regression.linear_model.RegressionResultsWrapper
        model_name: 模型名称（如 'T-M', 'H-M', 'C-L'）
    """
    model_name = f"[{model_name}] " if model_name else ""
    params = result.params
    pvalues = result.pvalues
    rsquared = result.rsquared
    rsquared_adj = result.rsquared_adj
    f_pvalue = result.f_pvalue

    print(f"\n{'=' * 60}")
    print(f"{model_name}OLS Regression Results")
    print(f"{'=' * 60}")
    print(f"{'Variable':<20} {'Coef':>12} {'Std Err':>10} {'t':>8} {'P>|t|':>10}")
    print(f"{'-' * 60}")
    for var in params.index:
        print(f"{var:<20} {params[var]:>12.6f} {result.bse[var]:>10.6f} "
              f"{result.tvalues[var]:>8.4f} {pvalues[var]:>10.4f}")
    print(f"{'-' * 60}")
    print(f"{'R-squared':<20} {rsquared:>12.6f}")
    print(f"{'Adj. R-squared':<20} {rsquared_adj:>12.6f}")
    print(f"{'F-statistic p-value':<20} {f_pvalue:>12.6f}")
    print(f"{'No. Observations':<20} {int(result.nobs):>12}")
    print(f"{'=' * 60}")


def results_to_dict(result, model_name: str) -> Dict[str, Any]:
    """
    将 OLS 回归结果转换为字典格式，便于序列化为 JSON。

    参数:
        result: statsmodels 回归结果对象
        model_name: 模型名称
    返回:
        包含关键指标的字典
    """
    params = result.params
    pvalues = result.pvalues

    # 计算 alpha 的 t 统计量判断显著性
    alpha_sig = "显著" if pvalues.get('alpha', 1) < 0.05 else "不显著"
    beta2_sig = "显著" if pvalues.get('beta2', 1) < 0.05 else "不显著"

    # 择时能力判断（beta2 > 0 且显著）
    timing_ability = pvalues.get('beta2', 1) < 0.05 and params.get('beta2', 0) > 0

    return {
        'model': model_name,
        'alpha': float(params.get('alpha', 0)),
        'alpha_pvalue': float(pvalues.get('alpha', 1)),
        'alpha_significant': alpha_sig,
        'beta1': float(params.get('beta1', 0)),
        'beta1_pvalue': float(pvalues.get('beta1', 1)),
        'beta2': float(params.get('beta2', 0)),
        'beta2_pvalue': float(pvalues.get('beta2', 1)),
        'beta2_significant': beta2_sig,
        'timing_ability': timing_ability,
        'r_squared': float(result.rsquared),
        'adj_r_squared': float(result.rsquared_adj),
        'f_pvalue': float(result.f_pvalue),
        'nobs': int(result.nobs),
    }


# ==================== 基金名称工具 ====================
FUND_NAME_MAP = {
    '000001': '平安上证指数',
    '000300': '沪深300',
    '000628': '大成高鑫股票A',
    '000012': '华安纯债债券A',
    '008404': '华泰紫金泰盈混合A',
    '021181': '中欧价值精选混合A',
    '021182': '中欧价值精选混合C',
    '024106': '万家裕利债券A',
}


def get_fund_name(code: str) -> str:
    """根据基金代码获取基金名称，未知则返回代码本身。"""
    return FUND_NAME_MAP.get(code.upper(), code.upper())


# ==================== 日期工具 ====================
def parse_date(date_str: str) -> pd.Timestamp:
    """解析日期字符串为 pd.Timestamp。"""
    return pd.to_datetime(date_str)


def get_trading_days(start: str, end: str) -> int:
    """估算交易日数量（粗略估算，约每年252天）。"""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    delta = (end_dt - start_dt).days
    return int(delta * 252 / 365)


# ==================== 统计汇总工具 ====================
def summary_stats(returns: pd.Series) -> Dict[str, float]:
    """
    计算收益率序列的基本统计量。

    参数:
        returns: 收益率序列
    返回:
        包含年化收益、夏普比率、最大回撤等指标的字典
    """
    import numpy as np

    # 年化收益率
    ann_return = returns.mean() * 252
    # 年化波动率
    ann_vol = returns.std() * np.sqrt(252)
    # 夏普比率（假设无风险利率为0）
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'mean': returns.mean(),
        'std': returns.std(),
        'count': len(returns),
    }
