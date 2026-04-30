# -*- coding: utf-8 -*-
"""
因子计算模块 - 基金选股择时能力定量评价模型
实现三种经典模型：T-M 模型、H-M 模型、C-L 模型

参考文献：
    华泰金工研究 | 2020-08-21
    Treynor & Mazuy (1966) - T-M Model
    Henriksson & Merton (1981) - H-M Model
    Chang & Lewellen (1984) - C-L Model
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any, Optional, Tuple
import warnings


# ==================== 核心公式（按研报原文） ====================
#
# T-M 模型（Treynor-Mazuy, 1966）:
#   R_p - R_f = alpha + beta1 * (R_m - R_f) + beta2 * (R_m - R_f)^2 + epsilon
#
# H-M 模型（Henriksson-Merton, 1981）:
#   R_p - R_f = alpha + beta1 * (R_m - R_f) + beta2 * (R_m - R_f) * D + epsilon
#   D = 1 当 R_m > R_f（牛市），D = 0 当 R_m < R_f（熊市）
#
# C-L 模型（Chang-Lewellen, 1984）:
#   R_p - R_f = alpha + beta1 * (R_m - R_f) * D1 + beta2 * (R_m - R_f) * D2 + epsilon
#   当 R_m > R_f: D1=0, D2=1
#   当 R_m < R_f: D1=1, D2=0


class StockTimingModel:
    """
    基金选股择时能力定量评价模型基类。

    属性:
        model_name: 模型名称
        result: 回归结果
    """

    def __init__(self, name: str):
        self.model_name = name
        self.result: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self._params: Dict[str, float] = {}

    def fit(self, excess_fund: pd.Series, excess_bench: pd.Series) -> 'StockTimingModel':
        """
        拟合模型。

        参数:
            excess_fund: 基金超额收益率序列（Rf - Rf）
            excess_bench: 市场基准超额收益率序列（R_m - Rf）
        返回:
            self
        """
        raise NotImplementedError("子类必须实现 fit 方法")

    def get_timing_ability(self) -> Dict[str, Any]:
        """
        获取择时能力评估结果。

        返回:
            包含 alpha、beta1、beta2 及显著性判断的字典
        """
        raise NotImplementedError("子类必须实现 get_timing_ability 方法")

    def summary(self) -> str:
        """返回模型结果的字符串摘要。"""
        if self.result is None:
            return f"<{self.model_name}: 未拟合>"
        r = self.result
        p = r.params
        pv = r.pvalues
        return (
            f"\n{'=' * 55}\n"
            f"{self.model_name} Model Results\n"
            f"{'=' * 55}\n"
            f"{'Parameter':<15} {'Coef':>12} {'Std Err':>10} {'t':>8} {'P>|t|':>10}\n"
            f"{'-' * 55}\n"
            + "".join(
                f"{v:<15} {p[v]:>12.6f} {r.bse[v]:>10.6f} "
                f"{r.tvalues[v]:>8.4f} {pv[v]:>10.4f}\n"
                for v in p.index
            )
            + f"{'-' * 55}\n"
            f"{'R-squared':<15} {r.rsquared:>12.6f}\n"
            f"{'Adj. R-squared':<15} {r.rsquared_adj:>12.6f}\n"
            f"{'F-statistic p':<15} {r.f_pvalue:>12.6f}\n"
            f"{'N Obs':<15} {int(r.nobs):>12}\n"
            f"{'=' * 55}"
        )


# ==================== T-M 模型 ====================
class TMModel(StockTimingModel):
    """
    T-M 模型（Treynor-Mazuy Model）。

    公式：
        R_p - R_f = alpha + beta1 * (R_m - R_f) + beta2 * (R_m - R_f)^2 + epsilon

    解读：
        - alpha > 0 且显著：基金经理有选股能力（择券能力）
        - beta2 > 0 且显著：基金经理有择时能力
          （能预判市场涨跌，在牛市增加仓位、熊市降低仓位）
    """

    def __init__(self):
        super().__init__("T-M")

    def fit(self, excess_fund: pd.Series, excess_bench: pd.Series) -> 'TMModel':
        """
        拟合 T-M 模型。

        参数:
            excess_fund: 基金超额收益率（R_p - R_f），index 为日期
            excess_bench: 市场基准超额收益率（R_m - R_f），index 为日期
        """
        # 对齐数据
        df = pd.DataFrame({'fund': excess_fund, 'bench': excess_bench}).dropna()

        if len(df) < 30:
            warnings.warn(f"T-M 模型数据点过少（{len(df)}），回归结果可能不可靠")

        # 构建回归变量
        # X1 = R_m - R_f（市场超额收益）
        # X2 = (R_m - R_f)^2（二次项，捕捉择时能力）
        X = pd.DataFrame({
            'bench': df['bench'],
            'bench_sq': df['bench'] ** 2,   # (R_m - R_f)^2
        })
        X = sm.add_constant(X)  # 添加常数项 alpha

        y = df['fund']

        # OLS 回归
        model = sm.OLS(y, X)
        self.result = model.fit()

        # 存储参数
        self._params = {
            'alpha': self.result.params.get('const', 0),
            'beta1': self.result.params.get('bench', 0),
            'beta2': self.result.params.get('bench_sq', 0),
        }

        return self

    def get_timing_ability(self) -> Dict[str, Any]:
        """
        评估 T-M 模型的择时能力。

        关键指标：
            - alpha: 选股（择券）能力，>0 且显著则说明有选股能力
            - beta2: 择时能力系数，>0 且显著则说明有择时能力

        返回:
            {
                'model': 'T-M',
                'alpha': float, 'alpha_pvalue': float,
                'beta1': float, 'beta1_pvalue': float,
                'beta2': float, 'beta2_pvalue': float,
                'timing_ability': bool,  # beta2 > 0 且 p < 0.05
                'stock_ability': bool,   # alpha > 0 且 p < 0.05
                'r_squared': float,
                'nobs': int
            }
        """
        if self.result is None:
            raise ValueError("模型尚未拟合，请先调用 fit()")

        p = self.result.params
        pv = self.result.pvalues

        alpha = p.get('const', 0)
        beta1 = p.get('bench', 0)
        beta2 = p.get('bench_sq', 0)
        alpha_pv = pv.get('const', 1)
        beta1_pv = pv.get('bench', 1)
        beta2_pv = pv.get('bench_sq', 1)

        return {
            'model': 'T-M',
            'alpha': float(alpha),
            'alpha_pvalue': float(alpha_pv),
            'alpha_significant': alpha_pv < 0.05,
            'stock_ability': alpha > 0 and alpha_pv < 0.05,
            'beta1': float(beta1),
            'beta1_pvalue': float(beta1_pv),
            'beta2': float(beta2),
            'beta2_pvalue': float(beta2_pv),
            'beta2_significant': beta2_pv < 0.05,
            'timing_ability': float(beta2) > 0 and float(beta2_pv) < 0.05,
            'r_squared': float(self.result.rsquared),
            'adj_r_squared': float(self.result.rsquared_adj),
            'f_pvalue': float(self.result.f_pvalue),
            'nobs': int(self.result.nobs),
        }


# ==================== H-M 模型 ====================
class HMModel(StockTimingModel):
    """
    H-M 模型（Henriksson-Merton Model）。

    公式：
        R_p - R_f = alpha + beta1 * (R_m - R_f) + beta2 * (R_m - R_f) * D + epsilon

    其中 D 为虚拟变量：
        D = 1 当 R_m > R_f（牛市，市场收益高于无风险利率）
        D = 0 当 R_m < R_f（熊市）

    解读：
        - alpha > 0 且显著：基金经理有选股能力
        - beta2 > 0 且显著：基金经理有择时能力
          （牛市时 β = beta1 + beta2 > beta1，熊市时 β = beta1）
    """

    def __init__(self):
        super().__init__("H-M")

    def fit(self, excess_fund: pd.Series, excess_bench: pd.Series) -> 'HMModel':
        """
        拟合 H-M 模型。

        参数:
            excess_fund: 基金超额收益率（R_p - R_f）
            excess_bench: 市场基准超额收益率（R_m - R_f）
        """
        df = pd.DataFrame({'fund': excess_fund, 'bench': excess_bench}).dropna()

        if len(df) < 30:
            warnings.warn(f"H-M 模型数据点过少（{len(df)}），回归结果可能不可靠")

        # 构建虚拟变量 D
        # D = 1 牛市（R_m > R_f），D = 0 熊市（R_m <= R_f）
        D = (df['bench'] > 0).astype(int)

        # X = [1, bench, bench * D]
        X = pd.DataFrame({
            'bench': df['bench'],
            'bench_D': df['bench'] * D,   # (R_m - R_f) * D
        })
        X = sm.add_constant(X)

        y = df['fund']

        model = sm.OLS(y, X)
        self.result = model.fit()

        self._params = {
            'alpha': self.result.params.get('const', 0),
            'beta1': self.result.params.get('bench', 0),
            'beta2': self.result.params.get('bench_D', 0),
        }

        return self

    def get_timing_ability(self) -> Dict[str, Any]:
        """
        评估 H-M 模型的择时能力。
        """
        if self.result is None:
            raise ValueError("模型尚未拟合，请先调用 fit()")

        p = self.result.params
        pv = self.result.pvalues

        alpha = p.get('const', 0)
        beta1 = p.get('bench', 0)
        beta2 = p.get('bench_D', 0)
        alpha_pv = pv.get('const', 1)
        beta1_pv = pv.get('bench', 1)
        beta2_pv = pv.get('bench_D', 1)

        # 牛市 Beta = beta1 + beta2，熊市 Beta = beta1
        bull_beta = beta1 + beta2
        bear_beta = beta1

        return {
            'model': 'H-M',
            'alpha': float(alpha),
            'alpha_pvalue': float(alpha_pv),
            'alpha_significant': alpha_pv < 0.05,
            'stock_ability': alpha > 0 and alpha_pv < 0.05,
            'beta1': float(beta1),      # 熊市 Beta
            'beta1_pvalue': float(beta1_pv),
            'beta2': float(beta2),      # 择时增量 Beta
            'beta2_pvalue': float(beta2_pv),
            'beta2_significant': beta2_pv < 0.05,
            'timing_ability': float(beta2) > 0 and float(beta2_pv) < 0.05,
            'bull_beta': float(bull_beta),   # 牛市 Beta
            'bear_beta': float(bear_beta),   # 熊市 Beta
            'r_squared': float(self.result.rsquared),
            'adj_r_squared': float(self.result.rsquared_adj),
            'f_pvalue': float(self.result.f_pvalue),
            'nobs': int(self.result.nobs),
        }


# ==================== C-L 模型 ====================
class CLModel(StockTimingModel):
    """
    C-L 模型（Chang-Lewellen Model）。

    公式：
        R_p - R_f = alpha + beta1 * (R_m - R_f) * D1 + beta2 * (R_m - R_f) * D2 + epsilon

    其中：
        当 R_m > R_f（多头市场）：D1 = 0, D2 = 1
        当 R_m < R_f（空头市场）：D1 = 1, D2 = 0

    解读：
        - alpha > 0 且显著：基金经理有选股能力
        - beta2 - beta1 > 0：基金经理有择时能力
          （多头市场时保持较大 beta2，空头市场时保持较小 beta1，减少损失）
    """

    def __init__(self):
        super().__init__("C-L")

    def fit(self, excess_fund: pd.Series, excess_bench: pd.Series) -> 'CLModel':
        """
        拟合 C-L 模型。

        参数:
            excess_fund: 基金超额收益率（R_p - R_f）
            excess_bench: 市场基准超额收益率（R_m - R_f）
        """
        df = pd.DataFrame({'fund': excess_fund, 'bench': excess_bench}).dropna()

        if len(df) < 30:
            warnings.warn(f"C-L 模型数据点过少（{len(df)}），回归结果可能不可靠")

        # 构建虚拟变量
        # 多头市场（D2=1, D1=0）：R_m > R_f
        # 空头市场（D1=1, D2=0）：R_m <= R_f
        D1 = (df['bench'] <= 0).astype(int)   # 空头市场虚拟变量
        D2 = (df['bench'] > 0).astype(int)    # 多头市场虚拟变量

        # X = [1, bench * D1, bench * D2]
        X = pd.DataFrame({
            'bench_D1': df['bench'] * D1,   # (R_m - R_f) * D1（空头市场）
            'bench_D2': df['bench'] * D2,   # (R_m - R_f) * D2（多头市场）
        })
        X = sm.add_constant(X)

        y = df['fund']

        model = sm.OLS(y, X)
        self.result = model.fit()

        self._params = {
            'alpha': self.result.params.get('const', 0),
            'beta1': self.result.params.get('bench_D1', 0),   # 空头市场 Beta
            'beta2': self.result.params.get('bench_D2', 0),   # 多头市场 Beta
        }

        return self

    def get_timing_ability(self) -> Dict[str, Any]:
        """
        评估 C-L 模型的择时能力。
        """
        if self.result is None:
            raise ValueError("模型尚未拟合，请先调用 fit()")

        p = self.result.params
        pv = self.result.pvalues

        alpha = p.get('const', 0)
        beta1 = p.get('bench_D1', 0)   # 空头市场 Beta
        beta2 = p.get('bench_D2', 0)   # 多头市场 Beta
        alpha_pv = pv.get('const', 1)
        beta1_pv = pv.get('bench_D1', 1)
        beta2_pv = pv.get('bench_D2', 1)

        # 择时能力判断：beta2 - beta1 > 0（多头市场 Beta > 空头市场 Beta）
        timing_diff = beta2 - beta1

        return {
            'model': 'C-L',
            'alpha': float(alpha),
            'alpha_pvalue': float(alpha_pv),
            'alpha_significant': alpha_pv < 0.05,
            'stock_ability': alpha > 0 and alpha_pv < 0.05,
            'beta1': float(beta1),      # 空头市场 Beta
            'beta1_pvalue': float(beta1_pv),
            'beta2': float(beta2),      # 多头市场 Beta
            'beta2_pvalue': float(beta2_pv),
            'timing_diff': float(timing_diff),   # beta2 - beta1
            'timing_ability': float(timing_diff) > 0,
            # 注意：C-L 模型择时能力看 beta2-beta1，而非单独的 beta2
            'timing_ability_beta2_minus_beta1': float(timing_diff) > 0,
            'r_squared': float(self.result.rsquared),
            'adj_r_squared': float(self.result.rsquared_adj),
            'f_pvalue': float(self.result.f_pvalue),
            'nobs': int(self.result.nobs),
        }


# ==================== 综合评估器 ====================
class StockTimingEvaluator:
    """
    综合评估器：同时运行 T-M、H-M、C-L 三种模型，
    并对基金选股择时能力给出综合判断。

    使用方法:
        evaluator = StockTimingEvaluator(fund_returns, benchmark_returns)
        results = evaluator.evaluate()
    """

    def __init__(self, fund_returns: pd.Series, bench_returns: pd.Series,
                 risk_free_rate: float = 0.015):
        """
        初始化评估器。

        参数:
            fund_returns: 基金日频收益率序列
            bench_returns: 基准日频收益率序列
            risk_free_rate: 年化无风险利率（默认1.5%）
        """
        self.fund_returns = fund_returns.copy()
        self.bench_returns = bench_returns.copy()
        # 年化无风险利率转换为日频
        self.daily_rf = risk_free_rate / 252

        # 计算超额收益率
        self.excess_fund = self.fund_returns - self.daily_rf
        self.excess_bench = self.bench_returns - self.daily_rf

        self.results: Dict[str, Dict] = {}

    def evaluate(self) -> Dict[str, Dict]:
        """
        运行全部三种模型，返回结果字典。
        """
        models = {
            'TM': TMModel(),
            'HM': HMModel(),
            'CL': CLModel(),
        }

        for name, model in models.items():
            try:
                model.fit(self.excess_fund, self.excess_bench)
                self.results[name] = model.get_timing_ability()
                print(f"\n{name} 模型拟合完成:")
                print(model.summary())
            except Exception as e:
                warnings.warn(f"{name} 模型拟合失败: {e}")
                self.results[name] = {'model': name, 'error': str(e)}

        return self.results

    def get_summary(self) -> pd.DataFrame:
        """
        返回三种模型的汇总对比表。
        """
        rows = []
        for name, res in self.results.items():
            if 'error' in res:
                continue
            rows.append({
                'Model': name,
                'Alpha': res.get('alpha', 0),
                'Alpha_pv': res.get('alpha_pvalue', 1),
                'StockAbility': 'Yes' if res.get('stock_ability', False) else 'No',
                'Beta2': res.get('beta2', 0),
                'Beta2_pv': res.get('beta2_pvalue', 1),
                'TimingAbility': 'Yes' if res.get('timing_ability', False) else 'No',
                'R2': res.get('r_squared', 0),
                'N': res.get('nobs', 0),
            })

        return pd.DataFrame(rows)


# ==================== 单元测试 ====================
if __name__ == '__main__':
    # 使用模拟数据快速测试模型逻辑
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2021-01-01', periods=n, freq='B')

    # 模拟市场基准超额收益（标准布朗运动）
    bench_excess = pd.Series(np.random.randn(n) * 0.01, index=dates)

    # 模拟基金超额收益（有选股能力 alpha=0.001，有择时能力 beta2=0.5）
    fund_excess = 0.001 + 0.8 * bench_excess + 0.5 * bench_excess ** 2 + np.random.randn(n) * 0.005
    fund_excess.index = dates
    bench_excess.index = dates

    print("=== T-M 模型测试 ===")
    tm = TMModel()
    tm.fit(fund_excess, bench_excess)
    res = tm.get_timing_ability()
    print(f"alpha={res['alpha']:.6f}, beta2={res['beta2']:.6f}, timing={res['timing_ability']}")

    print("\n=== H-M 模型测试 ===")
    hm = HMModel()
    hm.fit(fund_excess, bench_excess)
    res = hm.get_timing_ability()
    print(f"alpha={res['alpha']:.6f}, beta2={res['beta2']:.6f}, timing={res['timing_ability']}")

    print("\n=== C-L 模型测试 ===")
    cl = CLModel()
    cl.fit(fund_excess, bench_excess)
    res = cl.get_timing_ability()
    print(f"alpha={res['alpha']:.6f}, timing_diff={res['timing_diff']:.6f}, timing={res['timing_ability_beta2_minus_beta1']}")

    print("\n=== 综合评估器测试 ===")
    fund_ret = fund_excess + 0.015 / 252
    bench_ret = bench_excess + 0.015 / 252
    evaluator = StockTimingEvaluator(fund_ret, bench_ret)
    results = evaluator.evaluate()
    print(evaluator.get_summary())
