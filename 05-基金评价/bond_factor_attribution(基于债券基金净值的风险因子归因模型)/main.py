# -*- coding: utf-8 -*-
"""
主程序 - 基于净值的债券基金风险因子归因模型
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, BOND_FUND_POOL, REGRESSION_WINDOW, setup_chinese_font
from source.data_loader import (
    get_fund_nav, get_bond_index_data, get_convertible_bond_index
)
from source.factor_builder import build_all_factors
from source.collinearity import diagnose_collinearity
from source.factor_model import (
    FactorRegressionModel, rolling_regression, calculate_factor_contribution
)
from source.plot import plot_factor_exposure, plot_rolling_exposure, plot_factor_contribution

setup_chinese_font()


def main():
    print('=' * 60)
    print('基于净值的债券基金风险因子归因模型')
    print('风险因子：利率(久期+凸性)、信用利差、可转债')
    print('=' * 60)
    
    # 参数设置
    fund_code = '110017'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f'\n分析基金: {fund_code}')
    print(f'分析区间: {start_date} ~ {end_date}')
    
    # Step 1: 获取基金净值
    print('\n[Step 1/5] 获取基金净值数据...')
    fund_nav = get_fund_nav(fund_code, start_date, end_date)
    
    if len(fund_nav) == 0:
        print('  [WARNING] 使用模拟数据演示...')
        fund_nav = _generate_sample_nav(start_date, end_date)
    
    # Step 2: 获取指数数据
    print('\n[Step 2/5] 获取债券指数数据...')
    treasury_idx = get_bond_index_data('treasury', start_date, end_date)
    corporate_idx = get_bond_index_data('corporate_bond', start_date, end_date)
    convertible_idx = get_convertible_bond_index(start_date, end_date)
    
    if len(treasury_idx) == 0:
        print('  [WARNING] 使用模拟指数数据...')
        treasury_idx, corporate_idx, convertible_idx = _generate_sample_indices(fund_nav)
    
    # Step 3: 构建风险因子
    print('\n[Step 3/5] 构建风险因子...')
    factors = build_all_factors(treasury_idx, corporate_idx, convertible_idx)
    
    if len(factors) == 0:
        print('  [ERROR] 因子构建失败')
        return
    
    # Step 4: 共线性诊断
    print('\n[Step 4/5] 共线性诊断...')
    factor_names = ['duration_factor', 'convexity_factor', 'credit_factor', 'convertible_factor']
    available_factors = [f for f in factor_names if f in factors.columns]
    
    diagnosis = diagnose_collinearity(factors[available_factors])
    
    print(f'  VIF检验: {len(diagnosis["vif"])} 个因子')
    if diagnosis['has_collinearity']:
        print(f'  [WARNING] 存在共线性问题')
        for rec in diagnosis['recommendation']:
            print(f'    - {rec}')
    else:
        print('  [OK] 未发现明显共线性')
    
    # Step 5: 因子回归
    print('\n[Step 5/5] 执行因子回归...')
    model = FactorRegressionModel()
    results = model.fit(fund_nav['daily_return'], factors, available_factors)
    
    if not results:
        print('  [ERROR] 回归失败')
        return
    
    # 输出结果
    print('\n' + '=' * 60)
    print('归因分析结果')
    print('=' * 60)
    print(f'\nAlpha:          {results["alpha"]*100:.4f}%')
    print(f'R-squared:       {results["r_squared"]:.4f}')
    print(f'Adj R-squared:   {results["adj_r_squared"]:.4f}')
    print(f'观测数:         {results["n_obs"]}')
    
    print('\n因子暴露:')
    for name, beta in results['factor_exposures'].items():
        tstat = results['factor_tstats'].get(name, 0)
        pval = results['factor_pvalues'].get(name, 1)
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f'  {name:20s}: {beta:8.4f} (t={tstat:6.2f}) {sig}')
    
    # 计算贡献度
    contrib = calculate_factor_contribution(fund_nav['daily_return'], factors, results)
    print('\n因子贡献度:')
    print(contrib.to_string(index=False))
    
    # 保存结果
    print('\n[保存结果]')
    
    # 保存回归结果
    results_df = pd.DataFrame([results])
    results_path = os.path.join(OUTPUT_DIR, 'factor_regression_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f'  回归结果: {results_path}')
    
    # 保存贡献度
    contrib_path = os.path.join(OUTPUT_DIR, 'factor_contribution.csv')
    contrib.to_csv(contrib_path, index=False, encoding='utf-8-sig')
    print(f'  贡献度: {contrib_path}')
    
    # 绘图
    print('\n[绘制图表]')
    plot_factor_exposure(results, save_path=os.path.join(OUTPUT_DIR, 'factor_exposure.png'))
    plot_factor_contribution(contrib, save_path=os.path.join(OUTPUT_DIR, 'factor_contribution_pie.png'))
    
    print('\n' + '=' * 60)
    print('分析完成！')
    print(f'结果目录: {OUTPUT_DIR}')
    print('=' * 60)
    
    return results, contrib


def _generate_sample_nav(start_date, end_date):
    """生成模拟净值数据"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.005, len(dates))
    nav = 1.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'nav': nav,
        'daily_return': returns
    }, index=dates)


def _generate_sample_indices(fund_nav):
    """生成模拟指数数据"""
    dates = fund_nav.index
    n = len(dates)
    np.random.seed(42)
    
    treasury = pd.DataFrame({
        'close': 100 * (1 + np.random.normal(0.0001, 0.003, n)).cumprod(),
        'daily_return': np.random.normal(0.0001, 0.003, n)
    }, index=dates)
    
    corporate = pd.DataFrame({
        'close': 100 * (1 + np.random.normal(0.0002, 0.004, n)).cumprod(),
        'daily_return': np.random.normal(0.0002, 0.004, n)
    }, index=dates)
    
    convertible = pd.DataFrame({
        'close': 100 * (1 + np.random.normal(0.0004, 0.008, n)).cumprod(),
        'daily_return': np.random.normal(0.0004, 0.008, n)
    }, index=dates)
    
    return treasury, corporate, convertible


if __name__ == '__main__':
    results, contrib = main()
