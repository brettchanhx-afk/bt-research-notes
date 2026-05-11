"""
主程序入口
运行完整的资产配置回测
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from source.data_loader import fetch_global_index_data
from source.models import get_model_weights
from source.backtest import run_backtest, calculate_metrics, run_multi_model_backtest, compare_results
from source.plot import plot_nav_curve, plot_annual_returns, plot_correlation_matrix
from config import DATA_CONFIG, BACKTEST_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG


def main():
    """主函数"""
    print("=" * 60)
    print("半衰主成分风险平价模型 (HPCRP) 全球资产配置策略回测")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n[1] 获取全球指数数据...")
    data_path = os.path.join(OUTPUT_CONFIG['DATA_DIR'], 'returns_data.csv')
    
    if os.path.exists(data_path):
        print(f"从缓存读取数据: {data_path}")
        returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        returns = fetch_global_index_data()
        os.makedirs(OUTPUT_CONFIG['DATA_DIR'], exist_ok=True)
        returns.to_csv(data_path)
        print(f"数据已保存: {data_path}")
    
    print(f"\n数据信息:")
    print(f"  时间范围: {returns.index.min()} 至 {returns.index.max()}")
    print(f"  交易日数: {len(returns)}")
    print(f"  指数: {list(returns.columns)}")
    
    # 2. 运行回测
    print("\n[2] 运行回测...")
    from source.backtest import run_multi_model_backtest
    
    models = MODEL_CONFIG['MODELS']
    
    # 创建权重函数
    def weights_func(model_name, hist_data):
        if model_name == 'HPCRP':
            return get_model_weights(model_name, hist_data, half_life=BACKTEST_CONFIG['HALF_LIFE'])
        else:
            return get_model_weights(model_name, hist_data)
    
    # 运行所有模型回测
    results = {}
    
    for model in models:
        print(f"\n  回测模型: {model}")
        
        result = run_backtest(
            returns, 
            weights_func, 
            model,
            rebalance_freq=BACKTEST_CONFIG['REBALANCE_FREQ'],
            window=BACKTEST_CONFIG['WINDOW'],
            start_date=BACKTEST_CONFIG['BACKTEST_START'],
            end_date=BACKTEST_CONFIG['BACKTEST_END']
        )
        
        # 计算指标
        metrics = calculate_metrics(result['returns'], result['nav'])
        result['metrics'] = metrics
        
        results[model] = result
        
        print(f"    年化收益: {metrics['annual_return']:.2%}")
        print(f"    年化波动: {metrics['annual_vol']:.2%}")
        print(f"    夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"    最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"    Calmar比率: {metrics['calmar_ratio']:.3f}")
    
    # 3. 结果汇总
    print("\n[3] 回测结果汇总...")
    print(compare_results(results).to_string(index=False))
    
    # 4. 绘图
    print("\n[4] 生成图表...")
    os.makedirs(OUTPUT_CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # 净值曲线
    plot_nav_curve(results, os.path.join(OUTPUT_CONFIG['OUTPUT_DIR'], 'nav_curve.png'))
    
    # 年度收益
    plot_annual_returns(results, os.path.join(OUTPUT_CONFIG['OUTPUT_DIR'], 'annual_returns.png'))
    
    # 相关系数矩阵
    plot_correlation_matrix(returns, returns.columns.tolist(), 
                          os.path.join(OUTPUT_CONFIG['OUTPUT_DIR'], 'correlation_matrix.png'))
    
    # 5. 保存结果
    print("\n[5] 保存结果...")
    
    # 保存净值数据
    nav_data = {}
    for model, result in results.items():
        nav_data[model] = result['nav']
    
    nav_df = pd.DataFrame(nav_data)
    nav_df.to_csv(os.path.join(OUTPUT_CONFIG['OUTPUT_DIR'], 'nav_data.csv'))
    
    # 保存指标
    metrics_data = []
    for model, result in results.items():
        m = result['metrics']
        m['Model'] = model
        metrics_data.append(m)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(OUTPUT_CONFIG['OUTPUT_DIR'], 'metrics.csv'), index=False)
    
    print("\n完成!")
    print(f"结果保存在: {OUTPUT_CONFIG['OUTPUT_DIR']}")
    
    return results


if __name__ == '__main__':
    main()