# -*- coding: utf-8 -*-
"""
DEPI 回测引擎模块
实现 CCR 模型求解器，对基金池进行截面 DEPI 计算与排名。

关键修复（2026-04-29）：
  1. 输入指标标准化（Z-score），消除量纲差异
  2. 过滤 R<=0 的基金（CCR 模型对负产出无解）
  3. 数值稳定性：linprog method='highs'
"""
import pandas as pd
import numpy as np
from scipy.optimize import linprog


# ============================================================
# CCR / DEPI 模型
# ============================================================
class DEPIEngine:
    """DEPI 截面分析引擎。
    
    使用 CCR 模型（Charnes-Cooper-Rhodes）对同一时期、同类基金
    进行横向对比，计算每只基金的 DEPI 相对效率值。
    
    公式（与研报完全一致）：
        max  DEPI_j = R_j / sum(w_i * x_ij)
        s.t. DEPI_k = R_k / sum(w_i * x_ik) <= 1,  for all k
             w_i >= 0
    
    其中：
        R_j   = 基金 j 的超额收益（产出指标，只有一个）
        x_ij  = 基金 j 的第 i 个投入指标
        w_i   = 第 i 个投入指标的最优权重
    
    求解方法：Charnes-Cooper 变换 → 线性规划（LP）
    
    关键处理：
        - 输入指标标准化（Z-score）消除量纲差异
        - 过滤 R_j <= 0 的基金（CCR 模型要求正产出）
        - 对极端值进行缩尾处理
    """

    def __init__(self):
        self.results_ = None

    def _standardize_inputs(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Z-score 标准化投入指标矩阵。
        
        Parameters
        ----------
        X : np.ndarray
            (n, m) 投入指标矩阵
        eps : float
            防止除零
        
        Returns
        -------
        np.ndarray
            标准化后的投入矩阵
        """
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < eps] = 1.0
        return (X - mean) / std

    def _winsorize(self, x: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
        """对输入数组进行缩尾处理，减少极端值影响。"""
        lo = np.nanpercentile(x, lower * 100)
        hi = np.nanpercentile(x, upper * 100)
        return np.clip(x, lo, hi)

    def fit_transform(self, df: pd.DataFrame,
                     output_col: str = '超额收益R',
                     input_cols: list = None) -> pd.DataFrame:
        """对基金池截面数据计算 DEPI 值。
        
        Parameters
        ----------
        df : pd.DataFrame
            因子表格，每行一只基金
        output_col : str
            产出指标列名（超额收益）
        input_cols : list
            投入指标列名列表
        
        Returns
        -------
        pd.DataFrame
            增加 DEPI, DEPI_Rank 列，并按 DEPI 降序排列
        """
        if input_cols is None:
            input_cols = ['volatility', 'fee_rate', 'timing_alpha', 'timing_beta']

        # ---- 数据准备 ----
        R = df[output_col].values.astype(float)   # 产出
        X = df[input_cols].values.astype(float)   # 投入

        # ---- 过滤 R <= 0 的基金（CCR 模型要求正产出）----
        valid_mask = R > 0
        n_valid = valid_mask.sum()
        if n_valid < 5:
            print(f'  [WARNING] 正超额收益基金仅 {n_valid} 只，DEPI 参考性有限')
            valid_mask = np.ones_like(R, dtype=bool)
        
        R_valid = R[valid_mask]
        X_valid = X[valid_mask]
        codes_valid = df['基金代码'].values[valid_mask]
        other_cols = [c for c in df.columns if c not in input_cols + ['基金代码', output_col]]

        if len(R_valid) == 0:
            out = df.copy()
            out['DEPI'] = 0.0
            out['DEPI_Rank'] = 0
            self.results_ = out
            return out

        # ---- 投入指标标准化 ----
        X_std = self._standardize_inputs(X_valid)

        # ---- 缩尾处理 ----
        for j in range(X_std.shape[1]):
            X_std[:, j] = self._winsorize(X_std[:, j])

        # ---- 替换 nan/inf ----
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=10.0, neginf=-10.0)

        # ---- 求解 LP ----
        depi_values = np.zeros(len(R_valid))
        
        # 对每只基金 j 求解 LP
        for j in range(len(R_valid)):
            rj = R_valid[j]
            if abs(rj) < 1e-10:
                depi_values[j] = 0.0
                continue

            m = X_std.shape[1]  # 投入指标数量
            # 决策变量：[w_1,...,w_m, theta]
            # 目标：min theta
            c = np.zeros(m + 1)
            c[-1] = 1.0

            # 约束：sum(w*x_k) - theta * rk <= 0  (k=1..n)
            # 即对于每只基金 k：w·x_k <= theta * rk
            A_ub_list = []
            b_ub_list = []
            for k in range(len(R_valid)):
                A_ub_list.append(np.concatenate([X_std[k], [-R_valid[k]]]))
                b_ub_list.append(0.0)

            A_ub = np.array(A_ub_list)   # (n, m+1)
            b_ub = np.array(b_ub_list)    # (n,)

            # 变量边界：w_i >= 0, theta >= 0
            bounds = [(0, None)] * m + [(0, None)]

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                             method='highs')

            if result.success:
                theta = result.x[-1]
                if theta > 1e-10:
                    depi = 1.0 / theta
                    depi_values[j] = min(depi, 5.0)  # 上限防止异常
                else:
                    depi_values[j] = 0.0
            else:
                depi_values[j] = 0.0

        # ---- 构建结果 DataFrame ----
        result_df = pd.DataFrame({'基金代码': codes_valid, output_col: R_valid})
        for i, col in enumerate(input_cols):
            result_df[col] = X_valid[:, i]
        result_df['DEPI'] = depi_values
        result_df['DEPI_Rank'] = result_df['DEPI'].rank(
            ascending=False, method='min').astype(int)

        # 按 DEPI 降序排列
        result_df = result_df.sort_values('DEPI', ascending=False).reset_index(drop=True)

        # 添加被过滤的基金（R <= 0）
        dropped_mask = ~valid_mask
        if dropped_mask.sum() > 0:
            dropped_df = df[dropped_mask].copy()
            dropped_df['DEPI'] = 0.0
            dropped_df['DEPI_Rank'] = len(result_df) + dropped_df.index  # 排在最后
            result_df = pd.concat([result_df, dropped_df], ignore_index=True)

        self.results_ = result_df
        return result_df

    def get_top_n(self, n: int = 10):
        """获取 DEPI 排名前 n 的基金。"""
        if self.results_ is None:
            raise ValueError('Please call fit_transform first.')
        return self.results_.head(n)

    def get_summary(self) -> dict:
        """获取本次 DEPI 分析的统计摘要。"""
        if self.results_ is None:
            return {}
        depi_vals = self.results_['DEPI']
        pos_depi = depi_vals[depi_vals > 0]
        return {
            '基金总数': len(self.results_),
            '正DEPI基金数': int(len(pos_depi)),
            'DEPI均值': float(depi_vals.mean()),
            'DEPI中位数': float(depi_vals.median()),
            'DEPI标准差': float(depi_vals.std()),
            'DEPI最大值': float(depi_vals.max()),
            'DEPI最小值': float(depi_vals.min()),
        }


# ============================================================
# 时间序列回测
# ============================================================
def backtest_depi(nav_dict: dict,
                  benchmark_returns: pd.Series,
                  fee_df: pd.DataFrame,
                  start: str,
                  end: str,
                  freq: str = 'Q',
                  input_cols: list = None) -> pd.DataFrame:
    """滚动 DEPI 回测：在每个调仓节点计算截面 DEPI 并记录。
    
    Parameters
    ----------
    nav_dict : dict
        {基金代码: nav_df} 净值历史
    benchmark_returns : pd.Series
        基准指数日收益率
    fee_df : pd.DataFrame
        费率数据
    start, end : str
        回测区间
    freq : str
        调仓频率，'Q' = 季度末，'M' = 月末
    input_cols : list
        投入指标列表
    
    Returns
    -------
    pd.DataFrame
        时间序列形式的 DEPI 排名结果
    """
    from .factor import build_factor_table, calc_returns

    if input_cols is None:
        input_cols = ['volatility', 'fee_rate', 'timing_alpha', 'timing_beta']

    # 生成调仓日期
    periods = pd.date_range(start=start, end=end, freq=freq)
    engine = DEPIEngine()
    records = []

    for period_end in periods:
        period_start = period_end - pd.DateOffset(months=12)
        window_nav = {}
        for code, nav_df in nav_dict.items():
            sub = nav_df[nav_df.index <= period_end]
            if len(sub) < 30:
                continue
            window_nav[code] = sub[sub.index >= period_start]

        if len(window_nav) < 10:
            continue

        bench_sub = benchmark_returns[
            (benchmark_returns.index <= period_end) &
            (benchmark_returns.index >= period_start)
        ]
        if len(bench_sub) < 30:
            continue

        factor_df = build_factor_table(window_nav, bench_sub, fee_df)
        if len(factor_df) < 10:
            continue

        result = engine.fit_transform(factor_df, input_cols=input_cols)
        result['调仓日期'] = period_end
        result['区间'] = f"{period_start.strftime('%Y-%m')}~{period_end.strftime('%Y-%m')}"
        records.append(result[['基金代码', 'DEPI', 'DEPI_Rank', '调仓日期', '区间',
                                '超额收益R', 'volatility', 'fee_rate',
                                'timing_alpha', 'timing_beta']])

    if records:
        full_df = pd.concat(records, ignore_index=True)
        print(f'  [回测] 共 {len(periods)} 个调仓节点，{len(full_df)} 条记录')
        return full_df
    else:
        return pd.DataFrame()
