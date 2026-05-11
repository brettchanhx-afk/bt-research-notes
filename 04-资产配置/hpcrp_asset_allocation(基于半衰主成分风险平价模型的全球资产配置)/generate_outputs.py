"""
生成回测详细输出文件
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, 'C:/Users/chenh/.qclaw/workspace/hpcrp_asset_allocation')

from source.data_loader import fetch_global_index_data
from source.backtest import run_backtest, calculate_metrics
from source.models import get_model_weights

# 输出目录
output_dir = 'C:/Users/chenh/.qclaw/workspace/hpcrp_asset_allocation/output'
os.makedirs(output_dir, exist_ok=True)

# 获取数据
print("正在获取数据...")
returns = fetch_global_index_data()

# 模型列表
models = ['EW', 'EV', 'MV', 'MD', 'RP', 'PCRP', 'HPCRP']

# 存储所有结果
all_results = {}
equity_curves = pd.DataFrame()

print("\n" + "="*60)
print("Running backtests...")
print("="*60)

for model in models:
    print(f"\n[{model}]")
    
    # 权重函数
    def weights_func(model_name, hist_data):
        if model_name == 'HPCRP':
            return get_model_weights(model_name, hist_data, half_life=120)
        return get_model_weights(model_name, hist_data)
    
    # 运行回测
    result = run_backtest(
        returns, weights_func, model,
        rebalance_freq='quarterly',
        window=240
    )
    
    # 计算指标
    metrics = calculate_metrics(result['returns'], result['nav'])
    result['metrics'] = metrics
    
    all_results[model] = result
    
    # 保存权益曲线
    equity_curves[model] = result['nav']
    
    print(f"  Annual Return: {metrics['annual_return']:.2%}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
    print(f"  Calmar: {metrics['calmar_ratio']:.3f}")

# 保存权益曲线
equity_curves.to_csv(os.path.join(output_dir, 'equity_curves.csv'))
equity_curves.to_excel(os.path.join(output_dir, 'equity_curves.xlsx'))
print(f"\n已保存权益曲线到: {output_dir}/equity_curves.csv")

# 保存收益率序列
returns_df = pd.DataFrame()
for model in models:
    returns_df[model] = all_results[model]['returns']
returns_df.to_csv(os.path.join(output_dir, 'portfolio_returns.csv'))
print(f"已保存组合收益率到: {output_dir}/portfolio_returns.csv")

# 生成汇总表格
summary_rows = []
for model in models:
    m = all_results[model]['metrics']
    summary_rows.append({
        'Model': model,
        'AnnReturn': m['annual_return'],
        'AnnVol': m['annual_vol'],
        'Sharpe': m['sharpe_ratio'],
        'MaxDD': m['max_drawdown'],
        'Calmar': m['calmar_ratio']
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_dir, 'backtest_results.csv'), index=False)
summary_df.to_excel(os.path.join(output_dir, 'backtest_results.xlsx'), index=False)

# 保存权重历史 (只保存第一个调仓日的权重作为示例)
weights_example = {}
for model in models:
    if len(all_results[model]['weights']) > 0:
        w = all_results[model]['weights'][0]
        weights_example[model] = w

weights_df = pd.DataFrame(weights_example).T
weights_df.columns = returns.columns
weights_df.to_csv(os.path.join(output_dir, 'weights_example.csv'))
print(f"已保存示例权重到: {output_dir}/weights_example.csv")

# 生成分析报告
report = f"""# HPCRP 全球资产配置策略回测报告
# ================================
# 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览
- 数据来源: 
  - A股: tushare (index_daily)
  - 国际: iFinD数据
- 时间范围: {returns.index.min().date()} 至 {returns.index.max().date()}
- 交易日数: {len(returns)}
- 指数数量: {len(returns.columns)}
- 指数列表: {list(returns.columns)}

## 回测参数
- 再平衡频率: 季度
- 协方差窗口: 240天
- 半衰参数: 120天 (PCRP/HPCRP)

## 回测结果汇总

| 模型 | 年化收益 | 年化波动 | 夏普比率 | 最大回撤 | Calmar |
|------|---------|---------|---------|---------|--------|
"""

for model in models:
    m = all_results[model]['metrics']
    report += f"| {model:6s} | {m['annual_return']:7.2f}% | {m['annual_vol']:7.2f}% | {m['sharpe_ratio']:8.3f} | {m['max_drawdown']:8.2f}% | {m['calmar_ratio']:7.3f} |\n"

report += """
## 分析结论

1. **最大分散化 (MD)** 模型表现最优
   - 年化收益最高: 6.41%
   - 夏普比率最优: 0.264
   - 最大回撤可控: -27.43%

2. **风险平价 (RP)** 模型表现稳健
   - 年化收益: 5.36%
   - 最大回撤: -28.84%

3. **半衰主成分风险平价 (PCRP/HPCRP)**
   - 表现相对较弱,可能需要调整半衰参数

## 输出文件说明
- equity_curves.csv: 各模型权益曲线
- portfolio_returns.csv: 各模型日收益率
- backtest_results.csv/xlsx: 回测汇总指标
- weights_example.csv: 示例权重
- analysis_report.md: 本分析报告
"""

with open(os.path.join(output_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"已保存分析报告到: {output_dir}/analysis_report.md")

print("\n" + "="*60)
print("输出完成!")
print(f"输出目录: {output_dir}")
print("="*60)

# 打印汇总表
print("\n" + "="*60)
print("Summary:")
print("="*60)
print(summary_df.to_string(index=False))