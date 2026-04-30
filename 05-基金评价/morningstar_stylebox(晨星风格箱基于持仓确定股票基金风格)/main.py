# -*- coding: utf-8 -*-
"""
晨星风格箱 - 主程序入口
复现华泰证券研报《晨星风格箱基于基金持仓数据并根据规模、价值成长特性确定基金风格》(2020-08-21)

使用示例:
    python main.py 021181
    python main.py 021181 --gamma 0.5
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.data_loader import (
    get_all_stock_market_cap,
    get_fund_holdings,
    get_stock_financial_data,
    calculate_market_cap_thresholds,
    calculate_vg_thresholds,
)
from source.factor import (
    calculate_stock_size_score,
    calculate_value_score_from_market,
    calculate_growth_score_from_market,
    calculate_vcg_score,
    calculate_stock_vg_score,
    calculate_fund_size_score,
    calculate_fund_vg_score,
    determine_size_style,
    determine_vg_style,
    analyze_fund_style,
)
from source.backtest import (
    calculate_returns,
    calculate_cumulative_return,
    calculate_annualized_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_win_rate,
)
from source.utils import clean_stock_code, format_percentage


def main(fund_code: str = '021181', gamma: float = 0.5):
    """
    主函数：完整分析基金风格
    
    Args:
        fund_code: 基金代码
        gamma: 价值-成长风格判定参数
    """
    print("=" * 65)
    print(f"  晨星风格箱分析 — 基金 {fund_code}")
    print(f"  复现: 华泰证券研报《晨星风格箱基于基金持仓数据》")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    # ========== Step 1: 获取全A股票市值数据 ==========
    print("\n[Step 1/6] 获取全A股票市值数据...")
    try:
        stock_df = get_all_stock_market_cap()
        print(f"  [OK] 获取到 {len(stock_df)} 只A股市值数据")
    except Exception as e:
        print(f"  [ERR] 获取市值数据失败: {e}")
        return None
    
    # ========== Step 2: 计算市值门槛值 ==========
    print("\n[Step 2/6] 计算市值门槛值...")
    cap_thresholds = calculate_market_cap_thresholds(stock_df['market_cap'])
    print(f"  LMT(大中盘门限): {cap_thresholds['LMT']/1e8:.2f} 亿")
    print(f"  MST(中小盘门限): {cap_thresholds['MST']/1e8:.2f} 亿")
    
    # ========== Step 3: 获取基金持仓 ==========
    print(f"\n[Step 3/6] 获取基金 {fund_code} 持仓数据...")
    try:
        holdings = get_fund_holdings(fund_code)
        print(f"  [OK] 获取到 {len(holdings)} 只重仓股")
        for _, row in holdings.iterrows():
            print(f"     {row.get('stock_code',''):>8} {row.get('stock_name',''):>8} "
                  f"占比:{row.get('pct',0):.2f}%")
    except Exception as e:
        print(f"  [ERR] 获取持仓数据失败: {e}")
        return None
    
    # ========== Step 4: 合并市值数据 & 获取财务指标 ==========
    print(f"\n[Step 4/6] 合并市值 & 财务数据...")
    
    # 合并市值
    stock_codes = holdings['stock_code'].tolist()
    stock_cap_map = dict(zip(stock_df['code'], stock_df['market_cap']))
    stock_name_map = dict(zip(stock_df['code'], stock_df['name']))
    
    holdings['market_cap'] = holdings['stock_code'].map(stock_cap_map)
    
    # 补充缺失的市值
    missing_cap = holdings['market_cap'].isna().sum()
    if missing_cap > 0:
        print(f"  [WARN] {missing_cap} 只股票市值缺失，尝试实时获取...")
        try:
            fin_data = get_stock_financial_data(
                holdings[holdings['market_cap'].isna()]['stock_code'].tolist()
            )
            if not fin_data.empty and 'total_mv' in fin_data.columns:
                mv_map = dict(zip(fin_data['code'], fin_data['total_mv'] * 1e4))  # 万元→元
                for idx, row in holdings[holdings['market_cap'].isna()].iterrows():
                    if row['stock_code'] in mv_map:
                        holdings.at[idx, 'market_cap'] = mv_map[row['stock_code']]
        except Exception:
            pass
    
    # 获取财务指标
    print(f"  获取 {len(stock_codes)} 只股票的财务指标...")
    try:
        fin_data = get_stock_financial_data(stock_codes)
    except Exception as e:
        print(f"  [WARN] 获取财务指标失败: {e}，使用简化计算")
        fin_data = pd.DataFrame({'code': stock_codes})
    
    # ========== Step 5: 计算因子 & 风格得分 ==========
    print(f"\n[Step 5/6] 计算规模因子 & 价值-成长因子...")
    
    # 5.1 规模得分
    for idx, row in holdings.iterrows():
        cap = row.get('market_cap', np.nan)
        if not pd.isna(cap) and cap > 0:
            y = calculate_stock_size_score(cap, cap_thresholds['MST'], cap_thresholds['LMT'])
            holdings.at[idx, 'size_score_y'] = y
            holdings.at[idx, 'size_style'] = determine_size_style(y)
        else:
            holdings.at[idx, 'size_score_y'] = np.nan
            holdings.at[idx, 'size_style'] = '未知'
    
    # 5.2 价值-成长得分
    if not fin_data.empty and len(fin_data) > 0:
        # 建立代码映射
        fin_map = dict(zip(fin_data['code'], range(len(fin_data))))
        
        # 计算价值得分和成长得分
        value_scores = calculate_value_score_from_market(fin_data)
        growth_scores = calculate_growth_score_from_market(fin_data)
        vcg_scores = calculate_vcg_score(value_scores, growth_scores)
        
        fin_data['value_score'] = value_scores.values
        fin_data['growth_score'] = growth_scores.values
        fin_data['vcg_score'] = vcg_scores.values
        
        # 计算VG门限
        vg_thresholds = calculate_vg_thresholds(fin_data)
        print(f"  VT(价值-混合门限): {vg_thresholds['VT']:.4f}")
        print(f"  GT(混合-成长门限): {vg_thresholds['GT']:.4f}")
        
        # 映射到持仓
        vcg_map = dict(zip(fin_data['code'], fin_data['vcg_score']))
        for idx, row in holdings.iterrows():
            vcg = vcg_map.get(row['stock_code'], np.nan)
            holdings.at[idx, 'vcg_score'] = vcg
            if not pd.isna(vcg):
                x = calculate_stock_vg_score(vcg, vg_thresholds['VT'], vg_thresholds['GT'])
                holdings.at[idx, 'vg_score_x'] = x
                holdings.at[idx, 'vg_style'] = determine_vg_style(x, gamma)
            else:
                holdings.at[idx, 'vg_score_x'] = np.nan
                holdings.at[idx, 'vg_style'] = '未知'
    else:
        vg_thresholds = {'VT': -0.5, 'GT': 0.5}
        holdings['vcg_score'] = np.nan
        holdings['vg_score_x'] = 150.0
        holdings['vg_style'] = '平衡型'
    
    # ========== Step 6: 输出风格分析结果 ==========
    print(f"\n[Step 6/6] 风格分析结果:")
    print("-" * 65)
    
    # 完整风格分析
    result = analyze_fund_style(holdings, cap_thresholds, vg_thresholds, gamma)
    
    # 输出
    print(f"  基金代码:      {fund_code}")
    print(f"  规模得分 Y:    {result['fund_size_score_Y']:.2f}")
    print(f"  规模风格:      {result['fund_size_style']}")
    print(f"  价成得分 X:    {result['fund_vg_score_X']:.2f}")
    print(f"  价成风格:      {result['fund_vg_style']}")
    print(f"  ─────────────────────────────")
    print(f"  综合风格:      {result['fund_style']}")
    print(f"  Gamma参数:     {gamma}")
    
    print(f"\n  门槛值:")
    print(f"    LMT(大中盘): {cap_thresholds['LMT']/1e8:.2f} 亿")
    print(f"    MST(中小盘): {cap_thresholds['MST']/1e8:.2f} 亿")
    print(f"    VT(价值门限): {vg_thresholds['VT']:.4f}")
    print(f"    GT(成长门限): {vg_thresholds['GT']:.4f}")
    
    # 持仓股详情
    if not result['stock_details'].empty:
        print(f"\n  持仓股风格明细:")
        print(f"  {'代码':>8} {'名称':>8} {'市值(亿)':>10} {'规模Y':>8} "
              f"{'规模':>4} {'价成X':>8} {'价成':>4} {'占比':>6}")
        print("  " + "-" * 62)
        for _, row in result['stock_details'].iterrows():
            cap_yi = row.get('market_cap', np.nan)
            if not pd.isna(cap_yi):
                cap_yi = cap_yi / 1e8
            print(f"  {row.get('stock_code',''):>8} {row.get('stock_name',''):>8} "
                  f"{cap_yi if isinstance(cap_yi, str) else f'{cap_yi:>10.2f}'} "
                  f"{row.get('size_score_y',np.nan):>8.1f} "
                  f"{row.get('size_style',''):>4} "
                  f"{row.get('vg_score_x',np.nan):>8.1f} "
                  f"{row.get('vg_style',''):>4} "
                  f"{row.get('pct',0):>5.2f}%")
    
    print("\n" + "=" * 65)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存持仓风格明细
    if not result['stock_details'].empty:
        detail_path = os.path.join(output_dir, f'{fund_code}_style_details.csv')
        result['stock_details'].to_csv(detail_path, index=False, encoding='utf-8-sig')
        print(f"  [SAVE] 持仓风格明细: {detail_path}")
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='晨星风格箱分析')
    parser.add_argument('fund_code', nargs='?', default='021181', help='基金代码')
    parser.add_argument('--gamma', type=float, default=0.5, help='价值-成长风格参数')
    args = parser.parse_args()
    
    result = main(args.fund_code, args.gamma)
