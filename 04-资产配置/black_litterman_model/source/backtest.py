"""
backtest.py - Black-Litterman 回测引擎

功能:
  - 月末调仓 (每月最后一个交易日)
  - 支持 BL_S1, BL_S2, MVO, FIXED 四种策略
  - 换手率限制
  - 输出持仓权重、累计收益、绩效指标
"""

import numpy as np
import pandas as pd
from typing import Literal

from source.bl_model import BlackLittermanModel
from source.optimizer import PortfolioOptimizer


# ============================================================================
# 回测结果容器
# ============================================================================

class BacktestResult:
    """
    回测结果容器
    """

    def __init__(
        self,
        strategies: list[str],
        asset_names: list[str],
        start_date: str,
        end_date: str,
    ):
        self.strategies = strategies
        self.asset_names = asset_names
        self.start_date = start_date
        self.end_date = end_date

        self.weights: dict[str, pd.DataFrame] = {}   # {strat: date→weights}
        self.cumulative_returns: dict[str, pd.Series] = {}  # 累计收益序列
        self.daily_returns: dict[str, pd.Series] = {}      # 日频收益序列
        self.stats: dict[str, dict] = {}                    # 绩效统计
        self.yearly_stats: dict[str, pd.DataFrame] = {}    # 年度收益

    def add_strategy(
        self,
        strat: str,
        weights_df: pd.DataFrame,
        daily_rets: pd.Series,
    ):
        self.weights[strat] = weights_df
        self.daily_returns[strat] = daily_rets

        # 累计收益
        cum = (1 + daily_rets / 100).cumprod()
        self.cumulative_returns[strat] = (cum - 1) * 100

        # 绩效统计
        self.stats[strat] = self._compute_stats(daily_rets)

        # 年度收益
        self.yearly_stats[strat] = self._compute_yearly(daily_rets)

    def _compute_stats(self, rets: pd.Series) -> dict:
        """计算绩效指标"""
        r = rets.dropna() / 100
        if len(r) == 0:
            return {}

        ann_ret = r.mean() * 252 * 100
        ann_vol = r.std() * np.sqrt(252) * 100
        rf = 2.0  # 无风险利率 %/年

        # 最大回撤
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min() * 100

        # 夏普比率
        excess = r - rf / 252
        sharpe = excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

        # 收益回撤比
        ret_dd = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # 卡玛比率
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # 胜率
        win_rate = (r > 0).mean() * 100

        # 换手率 (需要权重数据)
        turn = 0.0

        return {
            '年化收益(%)': round(ann_ret, 2),
            '年化波动(%)': round(ann_vol, 2),
            '最大回撤(%)': round(max_dd, 2),
            '夏普比率':    round(sharpe, 3),
            '收益回撤比':  round(ret_dd, 2),
            '卡玛比率':    round(calmar, 2),
            '日胜率(%)':  round(win_rate, 1),
            '年化换手(%)': round(turn, 1),
        }

    def _compute_yearly(self, rets: pd.Series) -> pd.DataFrame:
        """计算各年度收益"""
        r = rets.dropna() / 100
        if len(r) == 0:
            return pd.DataFrame()
        r.index = pd.to_datetime(r.index)
        yearly = (1 + r).groupby(r.index.year).prod() - 1
        return (yearly * 100).round(2)

    # ── 摘要 ────────────────────────────────────────────────────────
    def summary_table(self) -> pd.DataFrame:
        rows = []
        for strat in self.strategies:
            if strat in self.stats and self.stats[strat]:
                rows.append({'策略': strat, **self.stats[strat]})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index('策略')

    def print_summary(self):
        """打印回测摘要"""
        print(f"\n{'='*70}")
        print(f"  Black-Litterman 回测摘要")
        print(f"  回测区间: {self.start_date} ~ {self.end_date}")
        print(f"{'='*70}")
        print()
        df = self.summary_table()
        if df is not None and not df.empty:
            print(df.to_string())
        else:
            print("  无回测数据")

    def save_results(self, output_dir: str):
        """保存所有结果到 CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 绩效统计
        df = self.summary_table()
        if df is not None and not df.empty:
            df.to_csv(os.path.join(output_dir, 'performance_summary.csv'))

        # 各策略日频收益
        for strat, rets in self.daily_returns.items():
            fname = f'daily_returns_{strat}.csv'
            rets.to_csv(os.path.join(output_dir, fname), header=['daily_return_pct'])

        # 累计收益
        for strat, cum in self.cumulative_returns.items():
            fname = f'cumulative_returns_{strat}.csv'
            cum.to_csv(os.path.join(output_dir, fname), header=['cumulative_return_pct'])

        # 权重
        for strat, wdf in self.weights.items():
            fname = f'weights_{strat}.csv'
            wdf.to_csv(os.path.join(output_dir, fname))

        print(f"  结果已保存: {output_dir}")


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngine:
    """
    Black-Litterman 月频调仓回测引擎

    Parameters
    ----------
    returns          : pd.DataFrame (T×n) 日频收益率 (%)
    rf_rate         : float 无风险利率 (%/年)
    stock_cap       : float 股票资产上限 (如 0.20)
    commodity_cap    : float 商品资产上限 (如 0.20)
    turnover_limit  : float 单期换手率上限 (如 0.50)
    rebalance_freq   : str 调仓频率 'monthly'|'weekly'
    lookback_months  : int 历史数据回望月数
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        rf_rate: float = 2.0,
        stock_cap: float = 0.20,
        commodity_cap: float = 0.20,
        turnover_limit: float = 1.0,
        rebalance_freq: str = 'monthly',
        lookback_months: int = 60,
    ):
        self.returns = returns.dropna()
        self.asset_names = list(self.returns.columns)
        self.n = len(self.asset_names)
        self.rf_rate = rf_rate
        self.stock_cap = stock_cap
        self.commodity_cap = commodity_cap
        self.turnover_limit = turnover_limit
        self.rebalance_freq = rebalance_freq
        self.lookback_months = lookback_months

        # 计算调仓日期 (月末)
        self.rebalance_dates = self._get_rebalance_dates()

    # ── 调仓日期 ────────────────────────────────────────────────────────
    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        """计算所有调仓日 (每月末)"""
        dates = self.returns.index
        if self.rebalance_freq == 'monthly':
            # 每月最后一个交易日
            months = dates.to_period('M')
            last_days = {}
            for m in months.unique():
                last_days[m] = dates[months == m].max()
            result = pd.DatetimeIndex(list(last_days.values()))
            return result.sort_values()
        elif self.rebalance_freq == 'weekly':
            weeks = dates.to_period('W')
            last_days = {}
            for w in weeks.unique():
                last_days[w] = dates[weeks == w].max()
            result = pd.DatetimeIndex(list(last_days.values()))
            return result.sort_values()
        return pd.DatetimeIndex([])

    # ── 主回测 ─────────────────────────────────────────────────────────
    def run(
        self,
        strategies: list[str],
        start_date: str,
        end_date: str,
        risk_aversion: float = 10.0,
        tau: float = 1 / 60,
    ) -> BacktestResult:
        """
        运行回测

        Parameters
        ----------
        strategies     : ['BL_S1', 'BL_S2', 'MVO', 'FIXED']
        start_date     : 回测开始日期
        end_date       : 回测结束日期
        risk_aversion  : 风险厌恶系数 (λ)
        tau            : 观点置信度参数

        Returns
        -------
        BacktestResult
        """
        print(f"\n[BacktestEngine] 回测配置:")
        print(f"  策略: {strategies}")
        print(f"  区间: {start_date} ~ {end_date}")
        print(f"  调仓频率: {self.rebalance_freq}")
        print(f"  股票上限: {self.stock_cap*100:.0f}%  商品上限: {self.commodity_cap*100:.0f}%  换手率: {self.turnover_limit*100:.0f}%")

        # 筛选回测区间
        rets = self.returns[start_date:end_date].copy()
        rets.index = pd.to_datetime(rets.index)
        rebal_dates = self.rebalance_dates[
            (self.rebalance_dates >= rets.index[0]) &
            (self.rebalance_dates <= rets.index[-1])
        ]

        result = BacktestResult(
            strategies=strategies,
            asset_names=self.asset_names,
            start_date=start_date,
            end_date=end_date,
        )

        # 逐策略回测
        for strat in strategies:
            w_df, daily_rets = self._backtest_single(
                strat=strat,
                rets=rets,
                rebal_dates=rebal_dates,
                risk_aversion=risk_aversion,
                tau=tau,
            )
            result.add_strategy(strat, w_df, daily_rets)

        return result

    # ── 单策略回测 ─────────────────────────────────────────────────────
    def _backtest_single(
        self,
        strat: str,
        rets: pd.DataFrame,
        rebal_dates: pd.DatetimeIndex,
        risk_aversion: float,
        tau: float,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        回测单个策略
        """
        n = self.n
        w_prev = np.ones(n) / n   # 上期权重
        weight_records = []         # 权重记录
        date_records = []           # 日期记录

        for rebal_date in rebal_dates:
            # 确定本期可用的历史数据 (回望 lookback_months)
            cutoff = rebal_date - pd.DateOffset(months=self.lookback_months)
            hist = rets[(rets.index >= cutoff) & (rets.index < rebal_date)]

            if len(hist) < 30:   # 至少30天数据
                w_prev = w_prev   # 保持上期权重
                continue

            # ── 求解本期权重 ───────────────────────────────────────
            if strat == 'FIXED':
                # FIXED: 精确权重 10%股+80%债+10%商品
                w_new = np.zeros(n)
                stock_i = [i for i, a in enumerate(self.asset_names)
                            if any(x in a.upper() for x in ['CSI300','SP500','HSI'])]
                bond_i  = [i for i, a in enumerate(self.asset_names)
                            if any(x in a.upper() for x in ['GOV','CORP','511'])]
                comm_i  = [i for i, a in enumerate(self.asset_names)
                            if any(x in a.upper() for x in ['NHCI','GOLD','1599','5188','商品'])]
                for i in stock_i: w_new[i] = 0.10 / len(stock_i)
                for i in bond_i:  w_new[i] = 0.80 / len(bond_i)
                for i in comm_i:  w_new[i] = 0.10 / len(comm_i)
                w_new = w_new / w_new.sum()  # 归一化

            elif strat == 'MVO':
                bl = BlackLittermanModel(
                    returns=hist,
                    market_weights='benchmark',
                    risk_aversion=risk_aversion,
                    tau=tau,
                    rf_rate=self.rf_rate,
                    view_lookback=1,
                )
                w_new = bl.get_weights_mvo()

            elif strat == 'BL_S1':
                bl = BlackLittermanModel(
                    returns=hist,
                    market_weights='benchmark',
                    risk_aversion=risk_aversion,  # λ=10
                    tau=tau,
                    rf_rate=self.rf_rate,
                    view_lookback=1,
                )
                w_new = bl.get_weights_bl()

            elif strat == 'BL_S2':
                bl = BlackLittermanModel(
                    returns=hist,
                    market_weights='benchmark',
                    risk_aversion=5.0,  # λ=5 (策略2更激进)
                    tau=tau,
                    rf_rate=self.rf_rate,
                    view_lookback=1,
                )
                w_new = bl.get_weights_bl()
            else:
                w_new = w_prev

            # ── FIXED精确权重直接记录 ───────────────────────────────
            if strat == 'FIXED':
                pass  # w_new 已在上面计算完毕

            else:
                # ── 换手率限制 ────────────────────────────────────
                if self.turnover_limit < 1.0:
                    w_new = self._apply_turnover(w_new, w_prev)
                # ── 上下限约束 ────────────────────────────────────
                w_new = self._apply_bounds(w_new)

            # 记录
            weight_records.append(w_new)
            date_records.append(rebal_date)
            w_prev = w_new.copy()

        # 构建权重 DataFrame
        w_df = pd.DataFrame(
            weight_records,
            index=pd.DatetimeIndex(date_records),
            columns=self.asset_names,
        )

        # 计算每日组合收益
        daily_pf_rets = self._compute_portfolio_returns(w_df, rets)

        return w_df, daily_pf_rets

    # ── 换手率限制 ─────────────────────────────────────────────────────
    def _apply_turnover(
        self,
        w_new: np.ndarray,
        w_prev: np.ndarray,
    ) -> np.ndarray:
        """限制单期换手"""
        change = w_new - w_prev
        total_change = np.abs(change).sum()
        if total_change > self.turnover_limit:
            scale = self.turnover_limit / total_change
            # 只收紧增加的部分，减少的不受影响
            excess = np.maximum(change - 0, 0) * (1 - scale)
            w_new = w_new - excess
        # 归一化
        w_new = np.maximum(w_new, 0)
        w_new = w_new / w_new.sum()
        return w_new

    # ── 上下限约束 ────────────────────────────────────────────────────
    def _apply_bounds(self, w: np.ndarray) -> np.ndarray:
        """应用资产类别上下限"""
        for i, name in enumerate(self.asset_names):
            n = name.upper()
            if any(x in n for x in ['CSI300', 'SP500', 'HSI']):
                w[i] = min(w[i], self.stock_cap)
            elif any(x in n for x in ['NHCI', 'GOLD', '1599', '5188', '商品']):
                w[i] = min(w[i], self.commodity_cap)
        w = np.maximum(w, 0)
        if w.sum() > 0:
            w = w / w.sum()
        return w

    # ── 组合日频收益 ───────────────────────────────────────────────────
    def _compute_portfolio_returns(
        self,
        w_df: pd.DataFrame,
        daily_rets: pd.DataFrame,
    ) -> pd.Series:
        """
        根据持仓权重计算每日组合收益
        使用前向填充权重: 调仓日确定的权重，在下一调仓日前持续使用
        """
        # 将调仓日权重对齐到所有日期
        # 先用最近一次调仓权重向后填充(覆盖调仓日前的空档期)
        # 再向前填充(覆盖未来日期,保守取最近权重)
        all_dates = daily_rets.index
        w_aligned = w_df.reindex(all_dates)
        # backward-fill 处理数据起点到第一次调仓之间的空档
        w_aligned = w_aligned.bfill().ffill()
        # 数据起点仍然无权重则用等权
        w_aligned = w_aligned.fillna(1.0 / self.n)

        # 组合日收益 = Σ(w_i × r_i)
        pf_rets = (w_aligned.values * daily_rets.values).sum(axis=1)
        return pd.Series(pf_rets, index=all_dates)
