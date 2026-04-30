# -*- coding: utf-8 -*-
"""
基金评价体系 - 主程序

完整复现华泰金工研报《基金评价因子及基金评价体系》
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    OUTPUT_DIR, BACKTEST_START, BACKTEST_END, 
    WINDOWS, COMPOSITE_FACTORS, SCREENING_RULES,
    setup_chinese_font
)
from source.data_loader import (
    get_fund_nav, get_multiple_fund_nav, 
    get_market_index, calculate_peer_median_returns
)
from source.factor_calculator import calc_all_factors
from source.backtest_engine import (
    calc_rank_ic, evaluate_factor_effectiveness,
    generate_backtest_report
)
from source.factor_composite import equal_weight_composite
from source.fund_scorer import FundScorer
from source.plot import (
    plot_ic_series, plot_factor_effectiveness,
    plot_radar_chart
)

setup_chinese_font()


# ============================================================
# 示例基金池
# ============================================================
SAMPLE_FUNDS = [
    '110011',  # 易方达中小盘
    '000751',  # 嘉实新兴产业
    '070017',  # 嘉实量化阿尔法
    '161005',  # 富国天惠成长
    '163406',  # 兴全合润
    '000697',  # 汇添富移动互联
    '260108',  # 景顺长城新兴成长
    '000527',  # 华夏复兴
    '040008',  # 华安策略优选
    '519678',  # 银河稳健成长
]


# ============================================================
# 主函数
# ============================================================
def main():
    """主程序入口"""
    print('=' * 60)
    print('基金评价因子及基金评价体系')
    print('基于华泰金工研报复现')
    print('=' * 60)
    
    # ============================================================
    # Step 1: 数据获取
    # ============================================================
    print('\n[Step 1/5] 获取基金净值数据...')
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    fund_navs = get_multiple_fund_nav(SAMPLE_FUNDS, start_date, end_date)
    
    if len(fund_navs) == 0:
        print('  [WARNING] 使用模拟数据演示...')
        fund_navs = _generate_sample_data()
    
    # 获取市场指数
    market_index = get_market_index('000001', start_date, end_date)
    
    print(f'  获取基金数: {len(fund_navs)}')
    
    # ============================================================
    # Step 2: 因子计算
    # ============================================================
    print('\n[Step 2/5] 计算基金评价因子...')
    
    # 计算同类基金收益率中位数作为基准
    peer_median = calculate_peer_median_returns(fund_navs, start_date, end_date)
    
    # 计算各基金因子
    factor_results = {}
    
    for code, nav_df in fund_navs.items():
        if len(nav_df) < 252:
            continue
        
        try:
            factors = calc_all_factors(
                nav_df['nav'],
                peer_median if len(peer_median) > 0 else nav_df['daily_return'],
                window=252
            )
            
            if len(factors) > 0:
                factor_results[code] = factors
        except Exception as e:
            print(f'  [ERROR] {code}: {e}')
    
    factor_df = pd.DataFrame(factor_results).T
    
    print(f'  成功计算: {len(factor_df)} 只基金, {len(factor_df.columns)} 个因子')
    
    # ============================================================
    # Step 3: 因子有效性分析
    # ============================================================
    print('\n[Step 3/5] 因子有效性分析...')
    
    effectiveness_results = []
    
    for factor_name in factor_df.columns[:10]:  # 分析前10个因子
        factor_values = factor_df[factor_name].dropna()
        
        # 简单有效性评估（均值和标准差）
        effectiveness = {
            'factor': factor_name,
            'mean': factor_values.mean(),
            'std': factor_values.std(),
            'median': factor_values.median(),
        }
        
        effectiveness_results.append(effectiveness)
    
    effectiveness_df = pd.DataFrame(effectiveness_results)
    
    print(effectiveness_df.to_string(index=False))
    
    # ============================================================
    # Step 4: 因子复合与评分
    # ============================================================
    print('\n[Step 4/5] 因子复合与基金评分...')
    
    # 选取可用因子
    available_factors = [f for f in COMPOSITE_FACTORS if f in factor_df.columns]
    
    if len(available_factors) > 0:
        # 复合因子
        composite = equal_weight_composite(factor_df[available_factors])
        factor_df['复合因子'] = composite
        
        # 五维评分
        scorer = FundScorer()
        score_df = scorer.calculate_scores(factor_df)
        
        print('\nTop 5 基金评分:')
        print(score_df[['综合得分'] + available_factors].head().to_string())
    else:
        print('  [WARNING] 缺少复合所需因子')
        score_df = pd.DataFrame()
    
    # ============================================================
    # Step 5: 输出结果
    # ============================================================
    print('\n[Step 5/5] 保存结果...')
    
    # 保存因子数据
    factor_path = os.path.join(OUTPUT_DIR, 'fund_factors.csv')
    factor_df.to_csv(factor_path, encoding='utf-8-sig')
    print(f'  因子数据: {factor_path}')
    
    # 保存评分结果
    if len(score_df) > 0:
        score_path = os.path.join(OUTPUT_DIR, 'fund_scores.csv')
        score_df.to_csv(score_path, encoding='utf-8-sig')
        print(f'  评分结果: {score_path}')
    
    # 绘制图表
    if len(effectiveness_df) > 0:
        plot_data = effectiveness_df.set_index('factor')[['mean']].copy()
        plot_data.columns = ['mean_ic']
        plot_factor_effectiveness(
            plot_data,
            save_path=os.path.join(OUTPUT_DIR, 'factor_effectiveness.png')
        )
    
    print('\n' + '=' * 60)
    print('分析完成！')
    print(f'结果目录: {OUTPUT_DIR}')
    print('=' * 60)
    
    return factor_df, score_df


# ============================================================
# 模拟数据生成
# ============================================================
def _generate_sample_data() -> dict:
    """生成模拟基金净值数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
    n = len(dates)
    
    results = {}
    
    for code in SAMPLE_FUNDS[:5]:
        returns = np.random.normal(0.0005, 0.015, n)
        nav = 1.0 * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'nav': nav,
            'nav_acc': nav * 1.1,
            'daily_return': returns
        }, index=dates)
        
        results[code] = df
    
    return results


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    factor_df, score_df = main()
