import sys
sys.path.insert(0, 'source')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("行业景气轮动策略 - 完整回测流程 (使用缓存数据)")
print("=" * 60)

print("\n【第一步】初始化数据加载器...")
from data_loader import DataLoader
loader = DataLoader()

print("\n【第二步】获取申万一级行业列表...")
sw_list = loader.get_sw_industry_list(level=1)
print(f"获取到 {len(sw_list)} 个申万一级行业")

print("\n【第三步】设置回测时间区间...")
start_date = '20200101'
end_date = '20231231'
print(f"回测区间: {start_date} - {end_date}")

print("\n【第四步】加载行业历史价格数据 (使用缓存)...")
industry_codes = sw_list['index_code'].tolist()[:10]
hist_data_list = []
for code in industry_codes:
    cache_name = f"sw_hist_{code}_{start_date}_{end_date}"
    df = loader.load_with_cache(cache_name, lambda: pd.DataFrame())
    if df is not None and len(df) > 0:
        df['industry_code'] = code
        hist_data_list.append(df)

if hist_data_list:
    hist_data = pd.concat(hist_data_list, ignore_index=True)
    if 'trade_date' in hist_data.columns:
        hist_data['trade_date'] = pd.to_datetime(hist_data['trade_date'])
    hist_data['return'] = hist_data.groupby('industry_code')['close'].pct_change() * 100
    print(f"加载到 {len(hist_data)} 条行业历史数据")
else:
    print("缓存数据为空，生成模拟数据...")
    hist_data = loader._generate_mock_historical_data(industry_codes, start_date, end_date)
    print(f"生成 {len(hist_data)} 条模拟数据")

print("\n【第五步】生成模拟财务数据...")
financial_data = loader.get_industry_financial_aggregate(start_date, end_date, reload=True)
print(f"生成 {len(financial_data)} 条行业财务数据")

print("\n【第六步】生成模拟一致预期数据...")
consensus_data = loader.get_consensus_data(start_date, end_date, reload=True)
print(f"生成 {len(consensus_data)} 条一致预期数据")

print("\n【第七步】计算景气度指标...")
from indicators import ProsperityIndicator, ConsensusIndicator
prosperity_indicator = ProsperityIndicator()
industry_indicators = prosperity_indicator.build_industry_indicators(financial_data)
print(f"计算得到 {len(industry_indicators)} 条行业指标数据")

print("\n【第八步】计算一致预期指标...")
consensus_indicator = ConsensusIndicator()
consensus_indicators = consensus_indicator.build_consensus_indicators(consensus_data)
print(f"计算得到 {len(consensus_indicators)} 条一致预期指标")

print("\n【第九步】构建复合景气度指标...")
from composite_indicator import IndustryProsperityCalculator
prosperity_calculator = IndustryProsperityCalculator()
prosperity_data = prosperity_calculator.calculate_prosperity_index(
    financial_data,
    consensus_data,
    hist_data
)
print(f"构建得到 {len(prosperity_data)} 条复合景气度数据")

print("\n【第十步】初始化策略...")
from strategy import ProsperityRotationStrategy
strategy = ProsperityRotationStrategy(
    data_loader=loader,
    rebalance_freq='M',
    top_n=5
)
strategy.industry_list = sw_list
strategy.industry_returns = hist_data
strategy.financial_data = financial_data
strategy.consensus_data = consensus_data
strategy.prosperity_data = prosperity_data
strategy.start_date = start_date
strategy.end_date = end_date
print("策略初始化完成")

print("\n【第十一步】运行回测...")
backtest_result = strategy.run_backtest()

if backtest_result is not None:
    portfolio_values = backtest_result['portfolio_values']
    signals_df = backtest_result['signals']
    initial_capital = backtest_result['initial_capital']
    final_value = backtest_result['final_value']

    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"最终价值: {final_value:,.2f}")
    print(f"总收益: {(final_value / initial_capital - 1) * 100:.2f}%")

    print("\n【第十二步】计算业绩指标...")
    returns = portfolio_values['return'].dropna()
    portfolio_series = portfolio_values['portfolio_value']

    from performance import PerformanceAnalyzer
    performance_analyzer = PerformanceAnalyzer(risk_free_rate=0.03)
    metrics = performance_analyzer.calculate_all_metrics(portfolio_series, returns)

    print("\n" + "-" * 40)
    print("业绩指标")
    print("-" * 40)
    print(f"总收益率:      {metrics.get('total_return', 0):.2f}%")
    print(f"年化收益率:    {metrics.get('annual_return', 0):.2f}%")
    print(f"年化波动率:    {metrics.get('annual_volatility', 0):.2f}%")
    print(f"夏普比率:      {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"最大回撤:      {metrics.get('max_drawdown', 0):.2f}%")
    print(f"胜率:          {metrics.get('win_rate', 0):.2f}%")

    print("\n【第十三步】生成可视化图表...")
    output_dir = 'output'
    from utils import ensure_dir, save_to_csv
    ensure_dir(output_dir)

    from visualization import DataVisualizer
    visualizer = DataVisualizer(figsize=(12, 6))

    visualizer.plot_equity_curve(
        portfolio_series,
        title='策略权益曲线',
        save_path=f'{output_dir}/equity_curve.png',
        show=False
    )
    print("权益曲线图已保存")

    visualizer.plot_cumulative_returns(
        returns,
        title='累计收益',
        save_path=f'{output_dir}/cumulative_returns.png',
        show=False
    )
    print("累计收益图已保存")

    visualizer.plot_drawdown(
        portfolio_series,
        title='回撤分析',
        save_path=f'{output_dir}/drawdown.png',
        show=False
    )
    print("回撤分析图已保存")

    visualizer.plot_returns_distribution(
        returns,
        title='收益分布',
        save_path=f'{output_dir}/returns_distribution.png',
        show=False
    )
    print("收益分布图已保存")

    print("\n【第十四步】保存回测结果...")
    save_to_csv(portfolio_values, f'{output_dir}/portfolio_values.csv')
    save_to_csv(signals_df, f'{output_dir}/trading_signals.csv')

    report = performance_analyzer.generate_performance_report(portfolio_series, returns)
    with open(f'{output_dir}/performance_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n回测结果已保存到 {output_dir} 目录")

    print("\n【第十五步】展示部分交易信号...")
    if len(signals_df) > 0:
        print("\n最近10条交易信号:")
        print(signals_df.tail(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)

else:
    print("回测失败，请检查数据!")
