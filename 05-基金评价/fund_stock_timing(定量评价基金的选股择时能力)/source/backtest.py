# -*- coding: utf-8 -*-
"""
回测模块 - 基金选股择时能力定量评价模型
基于 T-M、H-M、C-L 模型进行滚动回测，评估基金择时能力的时序变化。

功能：
1. 滚动窗口回归（Rolling OLS）
2. 滚动择时能力指标计算
3. 业绩归因分析
4. 分时间段回测
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .factor import TMModel, HMModel, CLModel, StockTimingEvaluator


# ==================== 滚动回测引擎 ====================
class RollingTimingBacktest:
    """
    滚动窗口回测：使用固定长度的历史窗口滚动估计 T-M/H-M/C-L 模型参数，
    观察基金经理择时能力的时序变化。

    使用方法:
        bt = RollingTimingBacktest(fund_returns, bench_returns, window=252)
        rolling_results = bt.run(model='TM', step=21)
    """

    def __init__(self,
                 fund_returns: pd.Series,
                 bench_returns: pd.Series,
                 window: int = 252,
                 risk_free_rate: float = 0.015):
        """
        初始化回测引擎。

        参数:
            fund_returns: 基金日频收益率序列
            bench_returns: 基准日频收益率序列
            window: 滚动窗口长度（默认252个交易日 = 1年）
            risk_free_rate: 年化无风险利率
        """
        self.fund_returns = fund_returns.copy()
        self.bench_returns = bench_returns.copy()
        self.window = window
        self.daily_rf = risk_free_rate / 252

        # 计算超额收益率
        self.excess_fund = self.fund_returns - self.daily_rf
        self.excess_bench = self.bench_returns - self.daily_rf

        # 对齐数据
        combined = pd.DataFrame({
            'fund': self.excess_fund,
            'bench': self.excess_bench,
        }).dropna()

        self.excess_fund = combined['fund']
        self.excess_bench = combined['bench']

        if len(self.excess_fund) < window:
            warnings.warn(
                f"数据长度（{len(self.excess_fund)}）小于窗口长度（{window}），"
                "回测结果可能不可靠"
            )

    def _rolling_ols(self, X: pd.Series, y: pd.Series,
                     window: int, step: int = 1) -> pd.DataFrame:
        """
        滚动 OLS 回归。

        参数:
            X: 解释变量（市场超额收益）
            y: 被解释变量（基金超额收益）
            window: 滚动窗口长度
            step: 每次滚动的步长（交易日数）
        返回:
            DataFrame，index 为日期，columns 为回归参数
        """
        n = len(X)
        results = []

        for i in range(window, n, step):
            window_X = X.iloc[i - window:i]
            window_y = y.iloc[i - window:i]

            # 构建 X 矩阵（T-M 模型）
            X_df = pd.DataFrame({
                'const': 1,
                'bench': window_X,
                'bench_sq': window_X ** 2,
            })

            try:
                model = sm.OLS(window_y, X_df)
                fit = model.fit()

                results.append({
                    'date': X.index[i - 1],
                    'alpha': fit.params.get('const', 0),
                    'beta1': fit.params.get('bench', 0),
                    'beta2': fit.params.get('bench_sq', 0),
                    'alpha_pv': fit.pvalues.get('const', 1),
                    'beta2_pv': fit.pvalues.get('bench_sq', 1),
                    'r_squared': fit.rsquared,
                    'nobs': fit.nobs,
                })
            except Exception:
                continue

        return pd.DataFrame(results).set_index('date') if results else pd.DataFrame()

    def run_tm_rolling(self, step: int = 21) -> pd.DataFrame:
        """
        运行 T-M 模型滚动回测。

        参数:
            step: 滚动步长（默认21个交易日 = 月度）
        返回:
            DataFrame，包含各滚动窗口的 alpha, beta1, beta2, R2 等
        """
        return self._rolling_ols(self.excess_bench, self.excess_fund,
                                  self.window, step)

    def run_hm_rolling(self, step: int = 21) -> pd.DataFrame:
        """
        运行 H-M 模型滚动回测。
        """
        n = len(self.excess_fund)
        results = []

        for i in range(self.window, n, step):
            df = pd.DataFrame({
                'fund': self.excess_fund.iloc[i - self.window:i],
                'bench': self.excess_bench.iloc[i - self.window:i],
            })

            D = (df['bench'] > 0).astype(int)
            X = pd.DataFrame({
                'const': 1,
                'bench': df['bench'],
                'bench_D': df['bench'] * D,
            })
            y = df['fund']

            try:
                fit = sm.OLS(y, X).fit()
                results.append({
                    'date': self.excess_fund.index[i - 1],
                    'alpha': fit.params.get('const', 0),
                    'beta1': fit.params.get('bench', 0),
                    'beta2': fit.params.get('bench_D', 0),
                    'alpha_pv': fit.pvalues.get('const', 1),
                    'beta2_pv': fit.pvalues.get('bench_D', 1),
                    'r_squared': fit.rsquared,
                })
            except Exception:
                continue

        return pd.DataFrame(results).set_index('date') if results else pd.DataFrame()

    def run_cl_rolling(self, step: int = 21) -> pd.DataFrame:
        """
        运行 C-L 模型滚动回测。
        """
        n = len(self.excess_fund)
        results = []

        for i in range(self.window, n, step):
            df = pd.DataFrame({
                'fund': self.excess_fund.iloc[i - self.window:i],
                'bench': self.excess_bench.iloc[i - self.window:i],
            })

            D1 = (df['bench'] <= 0).astype(int)
            D2 = (df['bench'] > 0).astype(int)
            X = pd.DataFrame({
                'const': 1,
                'bench_D1': df['bench'] * D1,
                'bench_D2': df['bench'] * D2,
            })
            y = df['fund']

            try:
                fit = sm.OLS(y, X).fit()
                results.append({
                    'date': self.excess_fund.index[i - 1],
                    'alpha': fit.params.get('const', 0),
                    'beta1': fit.params.get('bench_D1', 0),
                    'beta2': fit.params.get('bench_D2', 0),
                    'timing_diff': fit.params.get('bench_D2', 0) - fit.params.get('bench_D1', 0),
                    'alpha_pv': fit.pvalues.get('const', 1),
                    'r_squared': fit.rsquared,
                })
            except Exception:
                continue

        return pd.DataFrame(results).set_index('date') if results else pd.DataFrame()

    def run(self, model: str = 'TM', step: int = 21) -> pd.DataFrame:
        """
        运行指定模型的滚动回测。

        参数:
            model: 'TM' | 'HM' | 'CL'
            step: 滚动步长
        返回:
            滚动回归结果 DataFrame
        """
        if model == 'TM':
            return self.run_tm_rolling(step)
        elif model == 'HM':
            return self.run_hm_rolling(step)
        elif model == 'CL':
            return self.run_cl_rolling(step)
        else:
            raise ValueError(f"未知模型: {model}，请选择 TM/HM/CL")


# ==================== 业绩归因分析 ====================
class PerformanceAttribution:
    """
    基于 T-M/H-M/C-L 模型的业绩归因分析。

    将基金超额收益分解为：
        1. 选股贡献（alpha）
        2. 市场系统性风险贡献（beta1 * Rm）
        3. 择时贡献（beta2 * Rm^2 或 beta2 * Rm * D）
    """

    def __init__(self, fund_returns: pd.Series, bench_returns: pd.Series,
                 risk_free_rate: float = 0.015):
        self.fund_returns = fund_returns.copy()
        self.bench_returns = bench_returns.copy()
        self.daily_rf = risk_free_rate / 252

        combined = pd.DataFrame({
            'fund': self.fund_returns - self.daily_rf,
            'bench': self.bench_returns - self.daily_rf,
        }).dropna()

        self.excess_fund = combined['fund']
        self.excess_bench = combined['bench']

    def decompose_tm(self, tm_result: Dict[str, float]) -> pd.DataFrame:
        """
        基于 T-M 模型结果分解业绩贡献。

        返回:
            DataFrame，包含各收益来源的日频贡献序列
        """
        alpha = tm_result['alpha']
        beta1 = tm_result['beta1']
        beta2 = tm_result['beta2']
        bench = self.excess_bench

        # 分解各收益来源
        decomposition = pd.DataFrame({
            'alpha': np.full(len(bench), alpha),           # 选股收益（日频 alpha）
            'beta1_contrib': beta1 * bench,                # 市场风险贡献
            'beta2_contrib': beta2 * (bench ** 2),        # 择时贡献
            'residual': self.excess_fund - alpha - beta1 * bench - beta2 * (bench ** 2),  # 残差
            'total': self.excess_fund,                     # 总超额收益
        }, index=bench.index)

        # 累计曲线
        cumulative = decomposition.cumsum()

        return cumulative

    def decompose_summary(self, tm_result: Dict[str, float]) -> Dict[str, Any]:
        """
        返回业绩归因的汇总统计。
        """
        decomp = self.decompose_tm(tm_result)

        summary = {}
        for col in decomp.columns:
            total = decomp[col].iloc[-1]  # 累计收益
            ann = total * 252 / len(decomp)  # 年化（简化估算）
            summary[col] = {
                'cumulative_return': float(total),
                'annualized_return': float(ann),
                'contribution_pct': float(total / decomp['total'].iloc[-1] * 100)
                               if decomp['total'].iloc[-1] != 0 else 0,
            }

        return summary


# ==================== 分段回测分析 ====================
def period_backtest(fund_returns: pd.Series, bench_returns: pd.Series,
                    periods: List[Tuple[str, str]],
                    risk_free_rate: float = 0.015) -> pd.DataFrame:
    """
    分时间段回测：按指定时间区间分别评估基金择时能力。

    参数:
        fund_returns: 基金收益率
        bench_returns: 基准收益率
        periods: 时间段列表 [(start, end), ...]
        risk_free_rate: 年化无风险利率
    返回:
        包含各时间段分析结果的 DataFrame
    """
    rows = []
    for start, end in periods:
        fr = fund_returns[start:end].dropna()
        br = bench_returns[start:end].dropna()

        if len(fr) < 30:
            continue

        evaluator = StockTimingEvaluator(fr, br, risk_free_rate)
        results = evaluator.evaluate()

        for model_name, res in results.items():
            if 'error' in res:
                continue
            rows.append({
                'period_start': start,
                'period_end': end,
                'model': model_name,
                'alpha': res.get('alpha', 0),
                'alpha_pv': res.get('alpha_pvalue', 1),
                'stock_ability': res.get('stock_ability', False),
                'beta2': res.get('beta2', 0),
                'beta2_pv': res.get('beta2_pvalue', 1),
                'timing_ability': res.get('timing_ability', False),
                'r_squared': res.get('r_squared', 0),
                'nobs': res.get('nobs', 0),
            })

    return pd.DataFrame(rows)


# ==================== 单元测试 ====================
if __name__ == '__main__':
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2021-01-01', periods=n, freq='B')

    bench = pd.Series(np.random.randn(n) * 0.01, index=dates)
    fund = 0.001 + 0.8 * bench + 0.5 * bench ** 2 + np.random.randn(n) * 0.005
    fund.index = dates
    bench.index = dates

    # 测试滚动回测
    bt = RollingTimingBacktest(fund, bench, window=252, risk_free_rate=0.015)
    rolling = bt.run('TM', step=21)
    print(f"\nT-M 滚动回测结果: {len(rolling)} 个窗口")
    print(rolling.tail())

    # 测试业绩归因
    evaluator = StockTimingEvaluator(fund, bench)
    results = evaluator.evaluate()

    attr = PerformanceAttribution(fund, bench)
    decomp = attr.decompose_summary(results['TM'])
    print("\n业绩归因汇总:")
    for k, v in decomp.items():
        print(f"  {k}: {v}")
