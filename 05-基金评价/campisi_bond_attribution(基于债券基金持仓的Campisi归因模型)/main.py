# -*- coding: utf-8 -*-
"""
Campisi 债券基金归因模型 - 主程序

功能：
  1. 获取债券基金持仓数据
  2. 获取债券基本信息（久期、YTM、评级）
  3. 获取国债收益率曲线
  4. 执行Campisi归因分析
  5. 输出结果和图表
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入模块
from config import OUTPUT_DIR, DATA_DIR, setup_chinese_font
from source.data_loader import (
    get_fund_bond_holdings,
    get_bond_info,
    get_treasury_yield_curve,
    get_fund_nav_history,
)
from source.yield_curve import YieldCurve
from source.campisi_model import CampisiAttribution, rolling_attribution
from source.plot import (
    plot_attribution_pie,
    plot_attribution_timeseries,
    plot_bond_contribution,
    plot_duration_distribution,
    plot_attribution_report,
)

# 设置中文字体
setup_chinese_font()


# ============================================================
# 示例债券基金列表（债券型基金）
# ============================================================
SAMPLE_BOND_FUNDS = [
    '110017',  # 易方达增强回报A
    '050011',  # 博时信用债券A
    '070009',  # 嘉实超短债
    '240001',  # 宝康债券
    '519667',  # 银河银泰理财
]


# ============================================================
# 主函数
# ============================================================
def main():
    """主程序入口。"""
    print('=' * 60)
    print('Campisi 债券基金业绩归因模型')
    print('基于华泰金工研报复现')
    print('数据来源：tushare > efinance > akshare')
    print('=' * 60)
    
    # ============================================================
    # Step 1: 选择分析基金
    # ============================================================
    print('\n[Step 1/5] 选择分析基金...')
    
    # 使用示例基金
    fund_code = '110017'  # 易方达增强回报A
    fund_name = '易方达增强回报A'
    
    print(f'  分析基金: {fund_code} - {fund_name}')
    
    # ============================================================
    # Step 2: 获取基金持仓数据
    # ============================================================
    print('\n[Step 2/5] 获取基金持仓数据...')
    
    # 获取最新持仓
    holdings = get_fund_bond_holdings(fund_code)
    
    if len(holdings) == 0:
        print('  [WARNING] 未获取到持仓数据，使用模拟数据演示...')
        holdings = _generate_sample_holdings()
    
    print(f'  持仓债券: {len(holdings)} 只')
    print(f'  持仓权重合计: {holdings["weight"].sum():.2f}%')
    
    # ============================================================
    # Step 3: 获取债券基本信息
    # ============================================================
    print('\n[Step 3/5] 获取债券基本信息...')
    
    bond_codes = holdings['bond_code'].tolist()
    bond_info = get_bond_info(bond_codes)
    
    if len(bond_info) == 0:
        print('  [WARNING] 未获取到债券信息，使用模拟数据演示...')
        bond_info = _generate_sample_bond_info(bond_codes)
    
    print(f'  债券信息: {len(bond_info)} 只')
    
    # ============================================================
    # Step 4: 获取收益率曲线
    # ============================================================
    print('\n[Step 4/5] 获取收益率曲线...')
    
    # 分析区间
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    
    # 获取期初和期末收益率曲线
    curve_start_df = get_treasury_yield_curve(start_date, curve_type='国债')
    curve_end_df = get_treasury_yield_curve(end_date, curve_type='国债')
    
    if len(curve_start_df) == 0 or len(curve_end_df) == 0:
        print('  [WARNING] 未获取到收益率曲线，使用模拟数据演示...')
        curve_start_df, curve_end_df = _generate_sample_yield_curves()
    
    # 构建YieldCurve对象
    yc_start = YieldCurve(
        curve_start_df['term'].values,
        curve_start_df['yield_rate'].values,
        method='cubic'
    )
    yc_end = YieldCurve(
        curve_end_df['term'].values,
        curve_end_df['yield_rate'].values,
        method='cubic'
    )
    
    print(f'  收益率曲线期限数: {len(curve_start_df)}')
    print(f'  1年期收益率变化: {yc_end.get_yield(1) - yc_start.get_yield(1):.2f} bp')
    
    # ============================================================
    # Step 5: 执行Campisi归因分析
    # ============================================================
    print('\n[Step 5/5] 执行Campisi归因分析...')
    
    analyzer = CampisiAttribution()
    
    results = analyzer.analyze(
        holdings=holdings,
        bond_info=bond_info,
        treasury_curve_start=yc_start,
        treasury_curve_end=yc_end,
        holding_period_days=90
    )
    
    summary = analyzer.get_summary()
    
    # ============================================================
    # 输出结果
    # ============================================================
    print('\n' + '=' * 60)
    print('归因分析结果')
    print('=' * 60)
    
    if summary:
        print(f'\n总收益:          {summary["total_return"]:.4f} ({summary["total_return"]*100:.2f}%)')
        print(f'\n票息效应:        {summary["coupon_contrib"]:.4f} ({summary["coupon_pct"]:.1f}%)')
        print(f'国债利率效应:    {summary["treasury_contrib"]:.4f} ({summary["treasury_pct"]:.1f}%)')
        print(f'信用利差效应:    {summary["credit_contrib"]:.4f} ({summary["credit_pct"]:.1f}%)')
        print(f'\n持仓债券数:      {summary["n_bonds"]}')
        print(f'平均久期:        {summary["avg_duration"]:.2f}')
        print(f'平均YTM:         {summary["avg_ytm"]*100:.2f}%')
    
    # ============================================================
    # 保存结果
    # ============================================================
    print('\n[保存结果]')
    
    # 保存详细结果
    if len(results) > 0:
        results_path = os.path.join(OUTPUT_DIR, 'campisi_detailed_results.csv')
        results.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f'  详细结果: {results_path}')
    
    # 保存摘要
    if summary:
        summary_path = os.path.join(OUTPUT_DIR, 'campisi_summary.csv')
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f'  归因摘要: {summary_path}')
    
    # ============================================================
    # 绘制图表
    # ============================================================
    print('\n[绘制图表]')
    
    # 1. 归因饼图
    pie_path = os.path.join(OUTPUT_DIR, 'attribution_pie.png')
    plot_attribution_pie(summary, title=f'{fund_name} - Campisi归因分解', save_path=pie_path)
    
    # 2. 债券贡献图
    for effect in ['coupon', 'treasury', 'credit']:
        contrib_path = os.path.join(OUTPUT_DIR, f'{effect}_contribution.png')
        plot_bond_contribution(results, effect=effect, top_n=10, save_path=contrib_path)
    
    # 3. 久期分布图
    duration_path = os.path.join(OUTPUT_DIR, 'duration_distribution.png')
    plot_duration_distribution(results, save_path=duration_path)
    
    # 4. 综合报告
    report_path = os.path.join(OUTPUT_DIR, 'attribution_report.png')
    plot_attribution_report(summary, results, fund_name=fund_name, save_path=report_path)
    
    print('\n' + '=' * 60)
    print('分析完成！')
    print(f'结果目录: {OUTPUT_DIR}')
    print('=' * 60)
    
    return results, summary


# ============================================================
# 模拟数据生成（用于演示）
# ============================================================
def _generate_sample_holdings() -> pd.DataFrame:
    """生成模拟持仓数据。"""
    np.random.seed(42)
    
    n_bonds = 20
    bond_codes = [f'{100000 + i:06d}' for i in range(n_bonds)]
    
    # 随机权重（归一化到100%）
    weights = np.random.dirichlet(np.ones(n_bonds)) * 100
    
    return pd.DataFrame({
        'bond_code': bond_codes,
        'bond_name': [f'债券{i+1}' for i in range(n_bonds)],
        'weight': weights,
        'amount': weights * 100,  # 假设总规模1亿
        'bond_type': np.random.choice(['treasury', 'corporate', 'financial'], n_bonds),
    })


def _generate_sample_bond_info(bond_codes: list) -> pd.DataFrame:
    """生成模拟债券信息。"""
    np.random.seed(42)
    
    n = len(bond_codes)
    
    return pd.DataFrame({
        'bond_code': bond_codes,
        'bond_name': [f'债券{i+1}' for i in range(n)],
        'issue_date': ['2020-01-01'] * n,
        'maturity_date': ['2025-12-31'] * n,
        'coupon_rate': np.random.uniform(0.03, 0.06, n),
        'duration': np.random.uniform(2, 7, n),
        'modified_duration': np.random.uniform(2, 6.5, n),
        'convexity': np.random.uniform(5, 50, n),
        'credit_rating': np.random.choice(['AAA', 'AA', 'A'], n),
        'bond_type': np.random.choice(['treasury', 'corporate', 'financial'], n),
        'ytm': np.random.uniform(0.025, 0.055, n),
    })


def _generate_sample_yield_curves():
    """生成模拟收益率曲线。"""
    terms = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    
    # 期初曲线
    yields_start = np.array([2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.3, 3.4, 3.6, 3.7])
    
    # 期末曲线（利率上行10bp）
    yields_end = yields_start + 0.1
    
    curve_start = pd.DataFrame({'term': terms, 'yield_rate': yields_start})
    curve_end = pd.DataFrame({'term': terms, 'yield_rate': yields_end})
    
    return curve_start, curve_end


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    results, summary = main()
