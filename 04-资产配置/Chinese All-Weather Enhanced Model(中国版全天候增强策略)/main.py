
"""
中国版全天候增强策略 - 主程序
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from source import DataLoader, RiskCalculator, StrategyBuilder, BacktestEngine, Visualizer
import config


def generate_mock_data():
    """
    生成模拟的历史数据用于测试
    """
    print("正在生成模拟数据...")
    
    # 创建日期范围
    dates = pd.date_range(start='2014-01-01', end='2024-04-30', freq='D')
    
    # 资产列表
    assets = [
        '510300.SH',  # 沪深300ETF
        '512100.SH',  # 中证1000ETF
        '512890.SH',  # 红利低波ETF
        '511260.SH',  # 十年国债ETF
        '511090.SH',  # 三十年国债ETF
        '159980.SZ',  # 有色ETF
        '159981.SZ',  # 能化ETF
        '159985.SZ',  # 豆粕ETF
        '518880.SH'   # 黄金ETF
    ]
    
    # 生成价格数据
    np.random.seed(42)
    price_data = pd.DataFrame(index=dates, columns=assets)
    
    # 初始化价格
    current_prices = {asset: 1.0 for asset in assets}
    
    # 每日回报的均值和波动率（根据资产特性设定）
    return_params = {
        '510300.SH': {'mean': 0.0003, 'std': 0.015},    # 股票
        '512100.SH': {'mean': 0.0004, 'std': 0.018},    # 股票
        '512890.SH': {'mean': 0.0003, 'std': 0.012},    # 红利
        '511260.SH': {'mean': 0.0001, 'std': 0.003},    # 债券
        '511090.SH': {'mean': 0.0001, 'std': 0.004},    # 债券
        '159980.SZ': {'mean': 0.0002, 'std': 0.018},    # 商品
        '159981.SZ': {'mean': 0.0002, 'std': 0.020},    # 商品
        '159985.SZ': {'mean': 0.0002, 'std': 0.015},    # 商品
        '518880.SH': {'mean': 0.0002, 'std': 0.010}     # 黄金
    }
    
    # 生成价格序列
    for date in dates:
        for asset in assets:
            params = return_params[asset]
            daily_return = np.random.normal(params['mean'], params['std'])
            current_prices[asset] *= (1 + daily_return)
            price_data.loc[date, asset] = current_prices[asset]
    
    # 生成收益率数据
    return_data = price_data.pct_change().dropna()
    
    return price_data, return_data


def main():
    print("=" * 60)
    print("中国版全天候增强策略 - 量化回测系统")
    print("=" * 60)
    
    output_dir = config.OUTPUT_CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/5] 正在加载数据...")
    
    # 尝试使用模拟数据
    if os.path.exists(f"{output_dir}/price_data.csv"):
        print("从本地加载数据...")
        try:
            price_data = pd.read_csv(f"{output_dir}/price_data.csv", index_col=0, parse_dates=True)
            return_data = pd.read_csv(f"{output_dir}/return_data.csv", index_col=0, parse_dates=True)
        except:
            print("本地数据加载失败，使用模拟数据...")
            price_data, return_data = generate_mock_data()
    else:
        # 使用模拟数据
        price_data, return_data = generate_mock_data()
    
    # 保存模拟数据
    price_data.to_csv(f"{output_dir}/price_data.csv")
    return_data.to_csv(f"{output_dir}/return_data.csv")
    
    print(f"数据加载完成，共 {len(return_data)} 个交易日")
    print(f"可用资产: {list(return_data.columns)}")
    
    print("\n[2/5] 正在构建四象限组合...")
    strategy_builder = StrategyBuilder(DataLoader())
    quadrant_returns = strategy_builder.build_quadrant_portfolios(return_data)
    quadrant_performance = RiskCalculator.calculate_quadrant_performance(quadrant_returns)
    
    print("\n四象限组合绩效:")
    quadrant_names = {
        'growth_above': '增长超预期',
        'growth_below': '增长不及预期',
        'inflation_above': '通胀超预期',
        'inflation_below': '通胀不及预期'
    }
    print(quadrant_performance.rename(index=quadrant_names).to_string())
    
    print("\n[3/5] 正在运行策略回测...")
    backtest_engine = BacktestEngine(
        return_data,
        rebalance_freq=config.BACKTEST_CONFIG['rebalance_freq'],
        fee_rate=config.BACKTEST_CONFIG['fee_rate']
    )
    
    results = backtest_engine.backtest_all_strategies(
        initial_capital=config.BACKTEST_CONFIG['initial_capital'],
        lookback_window=config.STRATEGY_CONFIG['lookback_window'],
        use_semicovariance=config.STRATEGY_CONFIG['use_semicovariance'],
        momentum_lookback=config.STRATEGY_CONFIG['momentum_lookback']
    )
    
    print("\n[4/5] 正在分析策略表现...")
    comparison = BacktestEngine.compare_strategies(results)
    
    print("\n策略绩效对比:")
    print(comparison.rename(index=Visualizer.LABELS).to_string())
    
    print("\n[5/5] 正在生成可视化图表...")
    
    if config.OUTPUT_CONFIG['save_plots']:
        plot_dir = f"{output_dir}/plots"
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = None
    
    # 不显示交互式图表，只保存
    import matplotlib
    matplotlib.use('Agg')
    
    Visualizer.plot_quadrant_performance(
        quadrant_performance,
        title='四象限组合绩效对比',
        save_path=f"{plot_dir}/quadrant_performance.png" if plot_dir else None
    )
    
    Visualizer.plot_portfolio_value(
        results,
        title='策略净值对比',
        save_path=f"{plot_dir}/portfolio_value.png" if plot_dir else None
    )
    
    Visualizer.plot_drawdown(
        results,
        title='策略回撤对比',
        save_path=f"{plot_dir}/drawdown.png" if plot_dir else None
    )
    
    Visualizer.plot_performance_comparison(
        comparison,
        title='策略绩效对比',
        save_path=f"{plot_dir}/performance_comparison.png" if plot_dir else None
    )
    
    for strategy_name in ['allweather', 'enhanced']:
        Visualizer.plot_weights_evolution(
            results[strategy_name]['weights_record'],
            title=f'{Visualizer.LABELS[strategy_name]} - 仓位演变',
            save_path=f"{plot_dir}/{strategy_name}_weights.png" if plot_dir else None
        )
        
        Visualizer.plot_asset_category_weights(
            results[strategy_name]['weights_record'],
            title=f'{Visualizer.LABELS[strategy_name]} - 大类资产配置',
            save_path=f"{plot_dir}/{strategy_name}_category_weights.png" if plot_dir else None
        )
        
        Visualizer.plot_yearly_performance(
            results[strategy_name]['portfolio_returns'],
            title=f'{Visualizer.LABELS[strategy_name]} - 年度收益',
            save_path=f"{plot_dir}/{strategy_name}_yearly.png" if plot_dir else None
        )
    
    # 保存回测结果
    with open(f"{output_dir}/backtest_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    comparison.to_csv(f"{output_dir}/performance_comparison.csv")
    
    print(f"\n回测完成！")
    print("=" * 60)
    print(f"结果已保存到 {output_dir} 目录")
    print(f"\n主要输出:")
    print(f"  - {plot_dir}/portfolio_value.png: 策略净值对比")
    print(f"  - {plot_dir}/drawdown.png: 策略回撤对比")
    print(f"  - {output_dir}/performance_comparison.csv: 策略绩效对比表")
    
    return results, comparison


if __name__ == "__main__":
    results, comparison = main()
