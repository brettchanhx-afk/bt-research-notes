"""
main.py - Black-Litterman 模型一键运行脚本

国泰君安证券研究 | 大类资产配置量化模型研究系列之二
《手把手教你实现 Black-Litterman 模型》

运行:
    python main.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from source.data_loader import load_all_returns
from source.backtest import BacktestEngine
from source.plot import PlotEngine


def main():
    print("=" * 70)
    print("  Black-Litterman 模型 - 大类资产配置量化复现")
    print("  数据区间: 2006-11 ~ 2023-01 | 回测区间: 2012-01 ~ 2023-01")
    print("=" * 70)

    # ── Step 1: 数据 ──────────────────────────────────────────────
    print("\n[Step 1] 数据获取 ...")
    try:
        returns = load_all_returns(
            start=config.START_DATE,
            end=config.END_DATE,
            force=False,
        )
    except Exception as e:
        print(f"  [ERROR] 数据获取失败: {e}")
        print("  提示: 尝试 force=True 强制重新拉取")
        return

    # ── Step 2: 单期演示 ─────────────────────────────────────────
    print("\n[Step 2] BL模型单期演示 ...")
    from source.bl_model import BlackLittermanModel

    # 取最近 ~60 个月日频数据
    bl_data = returns[-(60 * 22):]
    if len(bl_data) > 500:
        print(f"  使用最近 {len(bl_data)} 条日频数据进行建模")
        bl = BlackLittermanModel(
            returns=bl_data,
            market_weights='benchmark',
            risk_aversion=10.0,
            tau=1.0 / 60,
            rf_rate=config.RF_RATE,
            view_lookback=1,
        )
        print("\n  先验/后验收益对比:")
        print(bl.summary().round(4).to_string())

        # 绘制先验 vs 后验
        class _DummyBT:
            def __init__(self, names): self.asset_names = names
        plot_eng_demo = PlotEngine(_DummyBT(bl.asset_names), config.OUTPUT_DIR)
        plot_eng_demo.plot_prior_vs_posterior(
            prior_mu=bl.pi_prior_,
            posterior_mu=bl.posterior_mu_,
            asset_names=bl.asset_names,
            title='BL模型: 先验均衡收益 vs 后验收益 (研报单期示例)',
        )

    # ── Step 3: 完整回测 ────────────────────────────────────────
    print("\n[Step 3] 启动完整回测 ...")
    bt = BacktestEngine(
        returns=returns,
        rf_rate=config.RF_RATE,
        stock_cap=config.STOCK_CAP,
        commodity_cap=config.COMMODITY_CAP,
        turnover_limit=config.TURNOVER_LIMIT,
        rebalance_freq=config.REBALANCE_FREQ,
        lookback_months=config.LOOKBACK_MONTHS,
    )
    result = bt.run(
        strategies=['BL_S1', 'BL_S2', 'MVO', 'FIXED'],
        start_date=config.BACKTEST_START,
        end_date=config.BACKTEST_END,
        risk_aversion=config.RISK_AVERSION,
        tau=1.0 / config.LOOKBACK_MONTHS,
    )

    # ── Step 4: 输出结果 ─────────────────────────────────────────
    print("\n[Step 4] 回测结果 ...")
    result.print_summary()
    result.save_results(config.OUTPUT_DIR)

    # ── Step 5: 可视化 ───────────────────────────────────────────
    print("\n[Step 5] 生成图表 ...")
    pe = PlotEngine(result, config.OUTPUT_DIR)
    pe.plot_cumulative_returns('BL模型策略 vs 基准累计收益 (2012-2023)')
    pe.plot_drawdown('BL策略1 vs MVO基准回撤对比')
    pe.plot_weights('BL_S1', 'BL策略1 资产配置权重')
    pe.plot_weights('MVO', 'MVO均值方差基准策略权重')
    pe.plot_stats_bar('各策略绩效指标对比')
    pe.plot_yearly_heatmap('各策略年度收益热力图')
    pe.plot_correlation_matrix(returns)

    print(f"\n{'='*70}")
    print(f"  ✅ 回测完成！结果已保存至 {config.OUTPUT_DIR}/")
    print(f"{'='*70}")
    return result


if __name__ == '__main__':
    try:
        result = main()
    except Exception as e:
        print(f"\n[ERROR] 程序异常: {e}")
        import traceback; traceback.print_exc()
        print("\n提示:")
        print("  1. pip install cvxopt          # 凸优化求解器 (推荐)")
        print("  2. pip install akshare        # 数据获取依赖")
        print("  3. force=True 强制重新拉取数据")
