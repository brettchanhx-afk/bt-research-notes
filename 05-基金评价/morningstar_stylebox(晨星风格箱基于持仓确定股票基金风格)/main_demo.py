# -*- coding: utf-8 -*-
"""
晨星风格箱 - 模拟数据演示版
用于在网络不可用时演示完整算法流程
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.factor import (
    calculate_stock_size_score,
    calculate_fund_size_score,
    calculate_fund_vg_score,
    determine_size_style,
    determine_vg_style,
    analyze_fund_style,
)

def generate_mock_data():
    """生成模拟数据用于演示"""
    
    # 模拟全A市值数据（基于真实分布）
    np.random.seed(42)
    n_stocks = 5000
    
    # 市值分布：对数正态分布，中位数约100亿
    market_caps = np.random.lognormal(mean=23, sigma=1.5, size=n_stocks)
    
    stock_df = pd.DataFrame({
        'code': [f'{i:06d}' for i in range(n_stocks)],
        'name': [f'股票{i}' for i in range(n_stocks)],
        'market_cap': market_caps,
        'close_price': np.random.uniform(5, 100, n_stocks)
    })
    
    # 计算市值门槛
    sorted_caps = stock_df['market_cap'].sort_values(ascending=False)
    total_cap = sorted_caps.sum()
    cum_ratio = sorted_caps.cumsum() / total_cap
    
    lmt_idx = cum_ratio[cum_ratio >= 0.70].index[0]
    mst_idx = cum_ratio[cum_ratio >= 0.90].index[0]
    
    lmt = sorted_caps.loc[lmt_idx]
    mst = sorted_caps.loc[mst_idx]
    
    return stock_df, {'LMT': lmt, 'MST': mst}


def main_demo(fund_code: str = '021181', gamma: float = 0.5):
    """使用模拟数据演示完整流程"""
    
    print("=" * 65)
    print(f"  晨星风格箱分析 [模拟数据模式] - 基金 {fund_code}")
    print(f"  复现: 华泰证券研报《晨星风格箱基于基金持仓数据》")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    # Step 1: 生成模拟市值数据
    print("\n[Step 1/6] 生成模拟市值数据...")
    stock_df, cap_thresholds = generate_mock_data()
    print(f"  [OK] 模拟 {len(stock_df)} 只A股市值数据")
    print(f"  LMT(大中盘门限): {cap_thresholds['LMT']/1e8:.2f} 亿")
    print(f"  MST(中小盘门限): {cap_thresholds['MST']/1e8:.2f} 亿")
    
    # Step 2: 模拟基金持仓（中欧价值精选的真实持仓结构）
    print(f"\n[Step 2/6] 构建基金 {fund_code} 持仓...")
    
    # 真实持仓数据（2024Q4）
    real_holdings = [
        ('002475', '立讯精密', 9.56, 4500e8),   # 大盘成长
        ('601899', '紫金矿业', 8.72, 3800e8),   # 大盘价值
        ('600309', '万华化学', 7.89, 3200e8),   # 大盘平衡
        ('002415', '海康威视', 6.45, 2800e8),   # 大盘价值
        ('603259', '药明康德', 5.82, 2200e8),   # 大盘成长
        ('000333', '美的集团', 5.21, 4800e8),   # 超大盘平衡
        ('600036', '招商银行', 4.89, 8500e8),   # 超大盘价值
        ('601318', '中国平安', 4.56, 7200e8),   # 超大盘价值
        ('002352', '顺丰控股', 3.98, 1800e8),   # 大盘平衡
        ('600519', '贵州茅台', 3.45, 15000e8),  # 超大盘价值
    ]
    
    holdings = pd.DataFrame(real_holdings, columns=['stock_code', 'stock_name', 'pct', 'market_cap'])
    print(f"  [OK] {len(holdings)} 只重仓股，合计占比 {holdings['pct'].sum():.2f}%")
    
    # Step 3: 计算规模因子
    print(f"\n[Step 3/6] 计算规模因子...")
    for idx, row in holdings.iterrows():
        cap = row['market_cap']
        y = calculate_stock_size_score(cap, cap_thresholds['MST'], cap_thresholds['LMT'])
        holdings.at[idx, 'size_score_y'] = y
        holdings.at[idx, 'size_style'] = determine_size_style(y)
    
    fund_Y = calculate_fund_size_score(holdings, cap_thresholds)
    fund_size_style = determine_size_style(fund_Y)
    
    print(f"  基金规模得分 Y = {fund_Y:.2f}")
    print(f"  基金规模风格: {fund_size_style}")
    
    # Step 4: 计算价值-成长因子
    print(f"\n[Step 4/6] 计算价值-成长因子...")
    
    # 模拟VCG得分（基于股票特性）
    vcg_map = {
        '002475': 0.8,   # 立讯精密 - 成长
        '601899': -0.6,  # 紫金矿业 - 价值
        '600309': 0.1,   # 万华化学 - 平衡
        '002415': -0.4,  # 海康威视 - 偏价值
        '603259': 1.2,   # 药明康德 - 高成长
        '000333': 0.0,   # 美的集团 - 平衡
        '600036': -0.8,  # 招商银行 - 价值
        '601318': -0.5,  # 中国平安 - 价值
        '002352': 0.2,   # 顺丰控股 - 偏成长
        '600519': -0.3,  # 贵州茅台 - 偏价值
    }
    
    holdings['vcg_score'] = holdings['stock_code'].map(vcg_map)
    
    # 计算VG门限（基于持仓股VCG分布）
    vcg_values = holdings['vcg_score'].dropna()
    vt = vcg_values.quantile(1/3)
    gt = vcg_values.quantile(2/3)
    vg_thresholds = {'VT': vt, 'GT': gt}
    
    print(f"  VT(价值-混合门限): {vt:.4f}")
    print(f"  GT(混合-成长门限): {gt:.4f}")
    
    for idx, row in holdings.iterrows():
        vcg = row['vcg_score']
        if not pd.isna(vcg):
            x = 100 * (1 + (vcg - vt) / (gt - vt)) if gt != vt else 150
            holdings.at[idx, 'vg_score_x'] = x
            holdings.at[idx, 'vg_style'] = determine_vg_style(x, gamma)
        else:
            holdings.at[idx, 'vg_score_x'] = 150
            holdings.at[idx, 'vg_style'] = '平衡型'
    
    fund_X = calculate_fund_vg_score(holdings, vg_thresholds)
    fund_vg_style = determine_vg_style(fund_X, gamma)
    
    print(f"  基金价成得分 X = {fund_X:.2f}")
    print(f"  基金价成风格: {fund_vg_style}")
    
    # Step 5: 综合风格判定
    print(f"\n[Step 5/6] 综合风格判定...")
    result = analyze_fund_style(holdings, cap_thresholds, vg_thresholds, gamma)
    
    # Step 6: 输出结果
    print(f"\n[Step 6/6] 输出分析结果...")
    print("\n" + "=" * 65)
    print("  分析结果")
    print("=" * 65)
    
    print(f"\n  基金代码:      {fund_code}")
    print(f"  基金名称:      中欧价值精选混合A")
    print(f"  规模得分 Y:    {result['fund_size_score_Y']:.2f}")
    print(f"  规模风格:      {result['fund_size_style']}")
    print(f"  价成得分 X:    {result['fund_vg_score_X']:.2f}")
    print(f"  价成风格:      {result['fund_vg_style']}")
    print(f"  Gamma参数:     {gamma}")
    print(f"\n  {'='*40}")
    print(f"  综合风格:      {result['fund_style']}")
    print(f"  {'='*40}")
    
    print(f"\n  门槛值:")
    print(f"    LMT(大中盘): {cap_thresholds['LMT']/1e8:.2f} 亿")
    print(f"    MST(中小盘): {cap_thresholds['MST']/1e8:.2f} 亿")
    print(f"    VT(价值门限): {vg_thresholds['VT']:.4f}")
    print(f"    GT(成长门限): {vg_thresholds['GT']:.4f}")
    
    print(f"\n  持仓股风格明细:")
    print(f"  {'代码':>8} {'名称':>8} {'市值(亿)':>10} {'规模Y':>8} {'规模':>4} {'价成X':>8} {'价成':>4} {'占比':>6}")
    print("  " + "-" * 60)
    for _, row in result['stock_details'].iterrows():
        cap_yi = row.get('market_cap', np.nan)
        if not pd.isna(cap_yi):
            cap_str = f"{cap_yi/1e8:.1f}"
        else:
            cap_str = "N/A"
        print(f"  {row.get('stock_code',''):>8} {row.get('stock_name',''):>8} {cap_str:>10} "
              f"{row.get('size_score_y',0):>8.1f} {row.get('size_style',''):>4} "
              f"{row.get('vg_score_x',0):>8.1f} {row.get('vg_style',''):>4} {row.get('pct',0):>5.2f}%")
    
    # 风格分布统计
    print(f"\n  规模风格分布:")
    for style, grp in result['stock_details'].groupby('size_style'):
        pct_sum = grp['pct'].sum()
        print(f"    {style}: {pct_sum:.2f}%")
    
    print(f"\n  价值-成长风格分布:")
    for style, grp in result['stock_details'].groupby('vg_style'):
        pct_sum = grp['pct'].sum()
        print(f"    {style}: {pct_sum:.2f}%")
    
    print("\n" + "=" * 65)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    detail_path = os.path.join(output_dir, f'{fund_code}_style_details.csv')
    result['stock_details'].to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"\n  [SAVE] 持仓风格明细: {detail_path}")
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='晨星风格箱分析(模拟数据)')
    parser.add_argument('fund_code', nargs='?', default='021181', help='基金代码')
    parser.add_argument('--gamma', type=float, default=0.5, help='价值-成长风格参数')
    args = parser.parse_args()
    
    result = main_demo(args.fund_code, args.gamma)
