import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import settings
from source import DataLoader, NorthboundFunds, MarginFunds, ETFFunds
from source import IndustrialCapital, IndicatorCalculator
from source import IndustryRotationStrategy, CompositeIndicator

def save_data(df, filename, subfolder=''):
    if len(df) == 0:
        print(f"[SKIP] {filename} - No data")
        return
    if subfolder:
        output_dir = os.path.join('data', subfolder)
    else:
        output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[SAVE] {filepath} - {len(df)} rows")

def main():
    print("=" * 60)
    print("行业配置策略：资金流向视角 - 数据获取与回测")
    print("=" * 60)

    print("\n[1/8] 初始化数据加载器...")
    dl = DataLoader()
    ic = IndicatorCalculator(dl)
    north = NorthboundFunds(dl)
    margin = MarginFunds(dl)
    etf = ETFFunds(dl)
    ic_capital = IndustrialCapital(dl)
    strategy = IndustryRotationStrategy(dl, ic)
    composite = CompositeIndicator(dl, ic)
    print("[OK] 初始化完成")

    print("\n[2/8] 获取北向资金数据...")
    try:
        north_df = north.get_northbound_net_inflow()
        save_data(north_df, 'northbound_net_inflow.csv')
    except Exception as e:
        print(f"[ERROR] 北向资金数据获取失败: {e}")
        north_df = pd.DataFrame()

    try:
        north_holdings = north.get_northbound_holdings()
        save_data(north_holdings, 'northbound_holdings.csv')
    except Exception as e:
        print(f"[ERROR] 北向持股数据获取失败: {e}")
        north_holdings = pd.DataFrame()

    print("\n[3/8] 获取两融资金数据...")
    try:
        margin_df = margin.get_margin_summary()
        save_data(margin_df, 'margin_summary.csv')
    except Exception as e:
        print(f"[ERROR] 两融数据获取失败: {e}")
        margin_df = pd.DataFrame()

    print("\n[4/8] 获取ETF数据...")
    try:
        etf_list = etf.get_etf_list()
        save_data(etf_list, 'etf_list.csv')
    except Exception as e:
        print(f"[ERROR] ETF列表获取失败: {e}")
        etf_list = pd.DataFrame()

    try:
        sector_etfs = etf.get_sector_etf_list()
        save_data(sector_etfs, 'sector_etf_list.csv')
    except Exception as e:
        print(f"[ERROR] 行业ETF列表获取失败: {e}")
        sector_etfs = pd.DataFrame()

    print("\n[5/8] 获取产业资本数据...")
    try:
        seo_preplan = ic_capital.get_seo_preplan()
        save_data(seo_preplan, 'seo_preplan.csv')
    except Exception as e:
        print(f"[ERROR] 定增预案获取失败: {e}")
        seo_preplan = pd.DataFrame()

    try:
        float_calendar = ic_capital.get_float_calendar()
        save_data(float_calendar, 'float_calendar.csv')
    except Exception as e:
        print(f"[ERROR] 限售解禁日历获取失败: {e}")
        float_calendar = pd.DataFrame()

    try:
        repurchase = ic_capital.get_repurchase()
        save_data(repurchase, 'repurchase.csv')
    except Exception as e:
        print(f"[ERROR] 回购数据获取失败: {e}")
        repurchase = pd.DataFrame()

    print("\n[6/8] 获取中信行业数据...")
    try:
        industry_list = ic.get_industry_list(level='L1')
        save_data(industry_list, 'industry_list_l1.csv')
        print(f"一级行业数量: {len(industry_list)}")
    except Exception as e:
        print(f"[ERROR] 行业列表获取失败: {e}")
        industry_list = pd.DataFrame()

    print("\n[7/8] 获取基准指数数据...")
    try:
        benchmark = strategy.get_benchmark_returns()
        save_data(benchmark, 'benchmark_returns.csv')
    except Exception as e:
        print(f"[ERROR] 基准收益获取失败: {e}")
        benchmark = pd.DataFrame()

    print("\n[8/8] 构建资金流向指标...")
    north_indicators = []
    margin_indicators = []
    seo_indicators = []
    float_indicators = []
    repurchase_indicators = []

    if len(north_df) > 0:
        try:
            north_indicators = north.build_north_indicators(freq='W')
            for i, ind in enumerate(north_indicators):
                if len(ind) > 0:
                    save_data(ind, f'north_indicator_w_{i}.csv')
        except Exception as e:
            print(f"[ERROR] 北向指标构建失败: {e}")

        try:
            north_indicators_m = north.build_north_indicators(freq='M')
            for i, ind in enumerate(north_indicators_m):
                if len(ind) > 0:
                    save_data(ind, f'north_indicator_m_{i}.csv')
        except Exception as e:
            print(f"[ERROR] 北向月度指标构建失败: {e}")

    if len(margin_df) > 0:
        try:
            margin_indicators = margin.build_margin_indicators(freq='W')
            for i, ind in enumerate(margin_indicators):
                if len(ind) > 0:
                    save_data(ind, f'margin_indicator_w_{i}.csv')
        except Exception as e:
            print(f"[ERROR] 两融指标构建失败: {e}")

        try:
            margin_indicators_m = margin.build_margin_indicators(freq='M')
            for i, ind in enumerate(margin_indicators_m):
                if len(ind) > 0:
                    save_data(ind, f'margin_indicator_m_{i}.csv')
        except Exception as e:
            print(f"[ERROR] 两融月度指标构建失败: {e}")

    if len(seo_preplan) > 0:
        try:
            seo_indicators = ic_capital.build_seo_indicators(freq='W')
            for i, ind in enumerate(seo_indicators):
                if len(ind) > 0:
                    save_data(ind, f'seo_indicator_w_{i}.csv')
        except Exception as e:
            print(f"[ERROR] 定增指标构建失败: {e}")

    if len(float_calendar) > 0:
        try:
            float_indicators = ic_capital.build_float_indicators(freq='W')
            for i, ind in enumerate(float_indicators):
                if len(ind) > 0:
                    save_data(ind, f'float_indicator_w_{i}.csv')
        except Exception as e:
            print(f"[ERROR] 限售解禁指标构建失败: {e}")

    if len(repurchase) > 0:
        try:
            repurchase_indicators = ic_capital.build_repurchase_indicators(freq='W')
            for i, ind in enumerate(repurchase_indicators):
                if len(ind) > 0:
                    save_data(ind, f'repurchase_indicator_w_{i}.csv')
        except Exception as e:
            print(f"[ERROR] 回购指标构建失败: {e}")

    print("\n" + "=" * 60)
    print("数据获取完成！")
    print("=" * 60)
    print(f"\n数据已保存到 data/ 文件夹")
    print(f"图表将保存到 output/charts/ 文件夹")

    return {
        'north_df': north_df,
        'margin_df': margin_df,
        'north_indicators': north_indicators,
        'margin_indicators': margin_indicators,
        'industry_list': industry_list,
        'benchmark': benchmark
    }

if __name__ == "__main__":
    results = main()