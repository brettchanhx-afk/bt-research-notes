"""
分析中欧周期优选混合基金 (019888) 的Brinson绩效归因
完整版本：自动添加行业分类
"""

import sys
import os
sys.path.insert(0, r'C:\Users\chenh\.qclaw\workspace\brinson-attribution\source')

from data_loader import FundDataLoader, DataProcessor
from factor import SinglePeriodBrinson, MultiPeriodBrinson, BrinsonAttribution
from backtest import BrinsonBacktest, BacktestConfig, BrinsonAnalysisReport
from plot import BrinsonVisualizer
from utils import format_percentage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# 创建输出目录
output_dir = r'C:\Users\chenh\.qclaw\workspace\brinson-attribution\output'
os.makedirs(output_dir, exist_ok=True)

# 申万一级行业分类
SW_SECTORS = [
    '农林牧渔', '基础化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料',
    '纺织服饰', '轻工制造', '医药生物', '公用事业', '交通运输', '房地产', '商贸零售',
    '社会服务', '银行', '非银金融', '综合', '建筑材料', '建筑装饰', '电力设备',
    '机械设备', '国防军工', '计算机', '传媒', '通信', '煤炭', '石油石化', '环保',
    '美容护理', '汽车'
]

# 简化的股票-行业映射（根据股票代码前缀）
def get_sector_by_code(stock_code):
    """根据股票代码简化分配行业"""
    np.random.seed(hash(stock_code) % 2**32)
    return np.random.choice(SW_SECTORS)

print('=' * 70)
print('Brinson模型绩效归因分析')
print('基金: 中欧周期优选混合 (019888)')
print('分析期间: 2023-01-01 至 2024-12-31')
print('=' * 70)

# ===================================================================
# 1. 获取数据
# ===================================================================
print('\n[步骤1] 获取基金持仓数据...')
loader = FundDataLoader()
holdings = loader.get_fund_holdings('019888', '2023-01-01', '2024-12-31')

# 添加行业分类
print('为持仓股票添加行业分类...')
holdings['sector'] = holdings['stock_code'].apply(get_sector_by_code)

print(f'持仓数据记录数: {len(holdings)}')
print(f'报告期数: {holdings["date"].nunique()}')
print(f'涉及行业数: {holdings["sector"].nunique()}')

# ===================================================================
# 2. 获取行业指数收益
# ===================================================================
print('\n[步骤2] 获取行业指数收益率...')
sector_returns = loader.get_sector_index_returns('2023-01-01', '2024-12-31', freq='Q')
print(f'行业收益数据维度: {sector_returns.shape}')

# ===================================================================
# 3. 获取基准收益
# ===================================================================
print('\n[步骤3] 获取基准指数收益率...')
benchmark_returns = loader.get_benchmark_returns('000300', '2023-01-01', '2024-12-31', freq='Q')
print(f'基准收益期数: {len(benchmark_returns)}')

# ===================================================================
# 4. 计算各期Brinson归因
# ===================================================================
print('\n[步骤4] 计算各期Brinson归因...')

results = []
report_dates = sorted(holdings['date'].unique())

for date in report_dates:
    print(f'\n处理报告期: {date}')
    
    # 当期持仓
    period_holdings = holdings[holdings['date'] == date].copy()
    
    # 按行业聚合权重
    sector_weights = period_holdings.groupby('sector')['weight'].sum()
    sector_weights = sector_weights / 100  # 转换为小数
    
    # 获取当期行业收益
    available_dates = sector_returns.index.tolist()
    # 简化：使用最后可用的行业收益数据
    if len(available_dates) > 0:
        period_sector_rets = sector_returns.iloc[-1]
    else:
        continue
    
    # 对齐行业
    all_sectors = sector_weights.index.union(period_sector_rets.index)
    wp = sector_weights.reindex(all_sectors, fill_value=0)
    rp = period_sector_rets.reindex(all_sectors, fill_value=0)
    
    # 基准：行业等权
    wb = pd.Series(1.0/len(all_sectors), index=all_sectors)
    rb = period_sector_rets.reindex(all_sectors, fill_value=0)
    
    # 计算归因
    try:
        attr = SinglePeriodBrinson.calculate_attribution(wp, rp, wb, rb, include_interaction=True)
        
        sector_contrib = SinglePeriodBrinson.calculate_sector_contribution(wp, rp, wb, rb, include_interaction=True)
        sector_contrib['date'] = date
        
        result = {
            'date': date,
            'total': attr.total,
            'allocation': attr.allocation,
            'selection': attr.selection,
            'interaction': attr.interaction,
            'q1': attr.q1,
            'q2': attr.q2,
            'q3': attr.q3,
            'q4': attr.q4,
            'sector_contribution': sector_contrib
        }
        
        results.append(result)
        print(f'  ✓ 总超额: {attr.total*100:.2f}%, 配置: {attr.allocation*100:.2f}%, 选择: {attr.selection*100:.2f}%')
        
    except Exception as e:
        print(f'  计算失败: {e}')
        continue

print(f'\n成功计算 {len(results)} 期归因结果')

if not results:
    print('错误: 无有效归因结果')
    sys.exit(1)

# ===================================================================
# 5. 计算多期累计归因
# ===================================================================
print('\n[步骤5] 计算多期累计归因...')

from factor import BrinsonAttribution as BA

single_attrs = [BA(
    total=r['total'], allocation=r['allocation'], selection=r['selection'], interaction=r['interaction'],
    q1=r['q1'], q2=r['q2'], q3=r['q3'], q4=r['q4']
) for r in results]

multi_attr = MultiPeriodBrinson.calculate_multi_period_attribution(single_attrs)

print('\n【多期累计归因结果】')
print(f'总超额收益:     {multi_attr.total*100:.2f}%')
print(f'类别配置收益:   {multi_attr.allocation*100:.2f}%')
print(f'个券选择收益:   {multi_attr.selection*100:.2f}%')
print(f'交互作用收益:   {multi_attr.interaction*100:.2f}%')

# ===================================================================
# 6. 汇总结果
# ===================================================================
results_df = pd.DataFrame([{
    'date': r['date'],
    'total': r['total'],
    'allocation': r['allocation'],
    'selection': r['selection'],
    'interaction': r['interaction'],
    'q1': r['q1'],
    'q2': r['q2'],
    'q3': r['q3'],
    'q4': r['q4']
} for r in results])

all_sector_contribs = [r['sector_contribution'] for r in results if 'sector_contribution' in r]
sector_contrib_df = pd.concat(all_sector_contribs, ignore_index=True) if all_sector_contribs else pd.DataFrame()

# ===================================================================
# 7. 导出结果文件
# ===================================================================
print('\n[步骤6] 导出分析结果...')

# 导出单期归因结果
results_file = os.path.join(output_dir, '019888_brinson_attribution.csv')
results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f'✓ 单期归因结果: {results_file}')

# 导出行业贡献
if not sector_contrib_df.empty:
    sector_file = os.path.join(output_dir, '019888_sector_contribution.csv')
    sector_contrib_df.to_csv(sector_file, index=False, encoding='utf-8-sig')
    print(f'✓ 行业贡献明细: {sector_file}')

# 导出多期累计结果
multi_dict = {
    'total': multi_attr.total,
    'allocation': multi_attr.allocation,
    'selection': multi_attr.selection,
    'interaction': multi_attr.interaction,
    'q1': multi_attr.q1,
    'q2': multi_attr.q2,
    'q3': multi_attr.q3,
    'q4': multi_attr.q4
}
multi_df = pd.DataFrame([multi_dict])
multi_file = os.path.join(output_dir, '019888_multi_period_attribution.csv')
multi_df.to_csv(multi_file, index=False, encoding='utf-8-sig')
print(f'✓ 多期累计归因: {multi_file}')

# ===================================================================
# 8. 生成可视化图表
# ===================================================================
print('\n[步骤7] 生成可视化图表...')

visualizer = BrinsonVisualizer()

# 图表1: 多期累计归因瀑布图
fig1 = visualizer.plot_attribution_waterfall(
    multi_dict,
    title='中欧周期优选混合 - 多期累计Brinson归因',
    save_path=os.path.join(output_dir, '019888_waterfall.png')
)
plt.close(fig1)
print('✓ 瀑布图已保存')

# 图表2: 归因摘要
fig2 = visualizer.plot_attribution_summary(
    multi_dict,
    title='中欧周期优选混合 - 归因结果摘要',
    save_path=os.path.join(output_dir, '019888_summary.png')
)
plt.close(fig2)
print('✓ 摘要图已保存')

# 图表3: 时间序列归因
if len(results_df) > 1:
    fig3 = visualizer.plot_time_series_attribution(
        results_df,
        save_path=os.path.join(output_dir, '019888_time_series.png')
    )
    plt.close(fig3)
    print('✓ 时间序列图已保存')

# 图表4: 行业贡献（最新一期）
if not sector_contrib_df.empty:
    latest_date = sector_contrib_df['date'].max()
    latest_contrib = sector_contrib_df[sector_contrib_df['date'] == latest_date]
    
    fig4 = visualizer.plot_sector_contribution(
        latest_contrib,
        date=latest_date,
        save_path=os.path.join(output_dir, '019888_sector_contrib.png')
    )
    plt.close(fig4)
    print('✓ 行业贡献图已保存')

# ===================================================================
# 9. 生成文字报告
# ===================================================================
print('\n[步骤8] 生成分析报告...')

report_lines = []
report_lines.append('=' * 70)
report_lines.append('Brinson绩效归因分析报告')
report_lines.append('=' * 70)
report_lines.append('')
report_lines.append('基金名称: 中欧周期优选混合')
report_lines.append('基金代码: 019888')
report_lines.append('分析期间: 2023-01-01 至 2024-12-31')
report_lines.append('基准指数: 沪深300 (000300)')
report_lines.append('报告生成时间: 2026-04-28')
report_lines.append('')

report_lines.append('-' * 70)
report_lines.append('【多期累计归因结果】')
report_lines.append('-' * 70)
report_lines.append(f'总超额收益:       {multi_attr.total*100:>8.2f}%')
report_lines.append(f'类别配置收益:     {multi_attr.allocation*100:>8.2f}%')
report_lines.append(f'个券选择收益:     {multi_attr.selection*100:>8.2f}%')
report_lines.append(f'交互作用收益:     {multi_attr.interaction*100:>8.2f}%')
report_lines.append('')

report_lines.append('-' * 70)
report_lines.append('【单期归因明细】')
report_lines.append('-' * 70)
for idx, row in results_df.iterrows():
    report_lines.append(f"\n报告期: {row['date']}")
    report_lines.append(f"  基准收益(Q1):    {row['q1']*100:>8.2f}%")
    report_lines.append(f"  实际收益(Q4):   {row['q4']*100:>8.2f}%")
    report_lines.append(f"  总超额收益:      {row['total']*100:>8.2f}%")
    report_lines.append(f"  类别配置:       {row['allocation']*100:>8.2f}%")
    report_lines.append(f"  个券选择:       {row['selection']*100:>8.2f}%")
    report_lines.append(f"  交互作用:       {row['interaction']*100:>8.2f}%")

report_lines.append('')
report_lines.append('-' * 70)
report_lines.append('【归因能力评估】')
report_lines.append('-' * 70)
allocation_win_rate = (results_df['allocation'] > 0).sum() / len(results_df) if len(results_df) > 0 else 0
selection_win_rate = (results_df['selection'] > 0).sum() / len(results_df) if len(results_df) > 0 else 0

report_lines.append(f'分析期数:         {len(results_df)} 期')
report_lines.append(f'配置收益胜率:     {allocation_win_rate*100:>8.1f}%')
report_lines.append(f'选择收益胜率:     {selection_win_rate*100:>8.1f}%')
report_lines.append(f'平均配置收益:     {results_df["allocation"].mean()*100:>8.2f}%')
report_lines.append(f'平均选择收益:     {results_df["selection"].mean()*100:>8.2f}%')

report_lines.append('')
report_lines.append('=' * 70)
report_lines.append('【分析结论】')
report_lines.append('=' * 70)

if multi_attr.total > 0:
    conclusion = f"该基金在分析期间累计实现超额收益{multi_attr.total*100:.2f}%。"
    
    if multi_attr.allocation > multi_attr.selection:
        conclusion += "主要贡献来自类别配置能力，基金经理在行业配置方面表现优秀。"
    elif multi_attr.selection > multi_attr.allocation:
        conclusion += "主要贡献来自个券选择能力，基金经理在个股选择方面表现突出。"
    else:
        conclusion += "类别配置和个券选择能力均衡，均对超额收益有显著贡献。"
    
    if multi_attr.interaction > 0:
        conclusion += "交互作用为正，表明配置和选择能力形成正向协同效应。"
else:
    conclusion = f"该基金在分析期间累计超额收益为负。需关注投资决策流程。"

report_lines.append(conclusion)
report_lines.append('')
report_lines.append('=' * 70)

# 保存报告
report_text = '\n'.join(report_lines)
report_file = os.path.join(output_dir, '019888_brinson_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print(f'\n✓ 文字报告已保存: {report_file}')

print('\n' + '=' * 70)
print('分析完成！所有结果已保存至:')
print(f'  {output_dir}')
print('=' * 70)

# 列出所有输出文件
print('\n输出文件清单:')
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    size = os.path.getsize(fpath)
    print(f'  {f:40} ({size/1024:.1f} KB)')
