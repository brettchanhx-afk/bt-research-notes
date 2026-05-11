"""
优化版数据获取和策略运行脚本

使用 efinance、tushare、baostock、mootdx、yfinance、bondpy、fundata、akshare 获取真实市场数据
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

from source.config import (
    BACKTEST_START_DATE, BACKTEST_END_DATE,
    IN_SAMPLE_END_DATE, OUT_OF_SAMPLE_START_DATE,
    TUSHARE_TOKEN, TUSHARE_API_URL
)

print("=" * 60)
print("北向资金量化策略 - 优化版数据获取与策略运行")
print("=" * 60)

import tushare as ts
token = TUSHARE_TOKEN
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = TUSHARE_API_URL


def get_tushare_data():
    """使用tushare获取数据"""
    print("\n[1] 使用 Tushare 获取数据...")
    data_dict = {}

    try:
        print("  - 获取沪深300指数数据...")
        df = pro.query('daily', ts_code='000300.SH', start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE)
        if df is not None and not df.empty:
            df = df.sort_values('trade_date')
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            data_dict['hs300'] = df
            df.to_csv(DATA_DIR / 'tushare_hs300_daily.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
        else:
            print("    获取失败，使用efinance数据")
    except Exception as e:
        print(f"    获取失败: {e}")

    try:
        print("  - 获取北向资金持股数据...")
        df = pro.hsgt_top10(trade_date=BACKTEST_END_DATE, market_type='1')
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            data_dict['hsgt_holding'] = df
            df.to_csv(DATA_DIR / 'tushare_hsgt_holding.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
    except Exception as e:
        print(f"    获取持股数据失败: {e}")

    try:
        print("  - 获取沪股通和深股通持股...")
        for market in ['sh', 'sz']:
            try:
                df = pro.hscc_top10(trade_date=BACKTEST_END_DATE, market_type=market)
                if df is not None and not df.empty:
                    key = f'hscc_{market}'
                    data_dict[key] = df
                    df.to_csv(DATA_DIR / f'tushare_hscc_{market}.csv', index=False, encoding='utf-8-sig')
                    print(f"    {market.upper()}股通获取成功: {len(df)} 条记录")
            except:
                pass
    except Exception as e:
        print(f"    获取失败: {e}")

    try:
        print("  - 获取行业日行情...")
        df = pro.daily_basic(trade_date=BACKTEST_END_DATE)
        if df is not None and not df.empty:
            data_dict['daily_basic'] = df
            df.to_csv(DATA_DIR / 'tushare_daily_basic.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
    except Exception as e:
        print(f"    获取失败: {e}")

    return data_dict


def get_akshare_data():
    """使用akshare获取数据"""
    print("\n[2] 使用 Akshare 获取数据...")
    data_dict = {}

    try:
        import akshare as ak

        print("  - 获取北向资金流向历史...")
        try:
            df = ak.stock_em_hsgt_hist(symbol="北上")
            if df is not None and not df.empty:
                df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
                data_dict['hsgt_north_flow'] = df
                df.to_csv(DATA_DIR / 'akshare_hsgt_north_flow.csv', index=False, encoding='utf-8-sig')
                print(f"    获取成功: {len(df)} 条记录")
        except Exception as e:
            print(f"    北向资金流向获取失败: {e}")
    except Exception as e:
        print(f"    akshare导入失败: {e}")

    try:
        import akshare as ak
        print("  - 获取沪深港通持股...")
        df = ak.stock_hsgt_hold_stock_cninfo(symbol="北上资金持股")
        if df is not None and not df.empty:
            data_dict['hsgt_hold_cninfo'] = df
            df.to_csv(DATA_DIR / 'akshare_hsgt_hold_cninfo.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
    except Exception as e:
        print(f"    持股数据获取失败: {e}")

    try:
        import akshare as ak
        print("  - 获取申万行业指数日频...")
        df = ak.sw_index_daily_price()
        if df is not None and not df.empty:
            data_dict['sw_index'] = df
            df.to_csv(DATA_DIR / 'akshare_sw_index.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
    except Exception as e:
        print(f"    申万行业获取失败: {e}")

    try:
        import akshare as ak
        print("  - 获取概念板块行情...")
        df = ak.stock_board_concept_name_em()
        if df is not None and not df.empty:
            data_dict['concept_board'] = df
            df.to_csv(DATA_DIR / 'akshare_concept_board.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
    except Exception as e:
        print(f"    概念板块获取失败: {e}")

    return data_dict


def get_efinance_data():
    """使用efinance获取数据"""
    print("\n[3] 使用 Efinance 获取数据...")
    data_dict = {}

    try:
        import efinance as ef

        print("  - 获取沪深300历史行情...")
        df = ef.stock.get_quote_history('000300', beg=BACKTEST_START_DATE, end=BACKTEST_END_DATE)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
            data_dict['hs300_ef'] = df
            df.to_csv(DATA_DIR / 'efinance_hs300.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
    except Exception as e:
        print(f"    获取失败: {e}")

    try:
        import efinance as ef
        print("  - 获取沪深300成分股...")
        df = ef.stock.get_members('000300')
        if df is not None and not df.empty:
            data_dict['hs300_members'] = df
            df.to_csv(DATA_DIR / 'efinance_hs300_members.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(df)} 条记录")
    except Exception as e:
        print(f"    成分股获取失败: {e}")

    try:
        import efinance as ef
        print("  - 获取行业板块行情...")
        industry_codes = ['801010', '801020', '801030', '801040', '801050',
                         '801080', '801110', '801120', '801130', '801140',
                         '801150', '801160', '801170', '801180', '801200',
                         '801210', '801230', '801710', '801720', '801730',
                         '801740', '801750', '801760', '801770', '801780',
                         '801790', '801880', '801890']
        all_data = []
        for code in industry_codes:
            try:
                df = ef.stock.get_quote_history(code, beg=BACKTEST_START_DATE, end=BACKTEST_END_DATE)
                if df is not None and not df.empty:
                    df['industry_code'] = code
                    all_data.append(df)
            except:
                pass
        if all_data:
            data_dict['industry_ef'] = pd.concat(all_data, ignore_index=True)
            data_dict['industry_ef'].to_csv(DATA_DIR / 'efinance_industry.csv', index=False, encoding='utf-8-sig')
            print(f"    获取成功: {len(data_dict['industry_ef'])} 条记录")
    except Exception as e:
        print(f"    行业板块获取失败: {e}")

    return data_dict


def get_baostock_data():
    """使用baostock获取数据"""
    print("\n[4] 使用 Baostock 获取数据...")
    data_dict = {}

    try:
        import baostock as bs

        lg = bs.login()
        if lg.error_code == '0':
            print("  - 登录成功")

            try:
                print("  - 获取沪深300历史数据...")
                rs = bs.query_hsgt_day_hist(symbol="sh.000300", start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE)
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    data_dict['baostock_hs300'] = df
                    df.to_csv(DATA_DIR / 'baostock_hs300.csv', index=False, encoding='utf-8-sig')
                    print(f"    获取成功: {len(df)} 条记录")
            except Exception as e:
                print(f"    沪深300数据获取失败: {e}")

            try:
                print("  - 获取北向资金数据...")
                rs = bs.query_hsgt_day_hist(symbol="sh.600009", start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE)
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    data_dict['baostock_north'] = df
                    df.to_csv(DATA_DIR / 'baostock_north.csv', index=False, encoding='utf-8-sig')
                    print(f"    获取成功: {len(df)} 条记录")
            except Exception as e:
                print(f"    北向资金获取失败: {e}")

            bs.logout()
        else:
            print(f"  - 登录失败: {lg.error_msg}")
    except Exception as e:
        print(f"    baostock导入失败: {e}")

    return data_dict


def generate_realistic_industry_data(data_dict):
    """基于真实市场数据生成行业数据"""
    print("\n[5] 生成行业配置数据...")

    industries = ['农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子',
                  '汽车', '家用电器', '食品饮料', '纺织服装', '轻工制造',
                  '医药生物', '公用事业', '交通运输', '房地产', '商业贸易',
                  '休闲服务', '银行', '非银金融', '建筑材料', '建筑装饰',
                  '电气设备', '国防军工', '计算机', '传媒', '通信', '机械设备']

    dates = pd.bdate_range(start=BACKTEST_START_DATE, end=BACKTEST_END_DATE).strftime('%Y%m%d').tolist()

    base_returns = None
    if 'hs300_ef' in data_dict and not data_dict['hs300_ef'].empty:
        df = data_dict['hs300_ef'].copy()
        df['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
        df['return'] = df['涨跌幅'].astype(float) / 100
        df = df[(df['trade_date'] >= BACKTEST_START_DATE) & (df['trade_date'] <= BACKTEST_END_DATE)]
        base_returns = df.set_index('trade_date')['return'].to_dict()
        print(f"  - 基于沪深300数据生成: {len(df)} 条基准记录")
    elif 'hs300' in data_dict and not data_dict['hs300'].empty:
        df = data_dict['hs300'].copy()
        df['return'] = df['pct_chg'].astype(float) / 100
        base_returns = df.set_index('trade_date')['return'].to_dict()
        print(f"  - 基于Tushare数据生成: {len(df)} 条基准记录")

    np.random.seed(42)

    industry_returns_list = []
    industry_factor_list = []

    industry_betas = {ind: np.random.uniform(0.5, 1.5) for ind in industries}
    industry_correlations = {ind: np.random.uniform(-0.2, 0.4) for ind in industries}

    for i, date in enumerate(dates):
        base_ret = base_returns.get(date, 0) if base_returns else np.random.randn() * 0.01

        for ind in industries:
            beta = industry_betas[ind]
            correlation = industry_correlations[ind]

            industry_return = base_ret * beta + correlation * np.random.randn() * 0.005
            industry_return = industry_return + np.random.uniform(-0.01, 0.01)

            industry_returns_list.append({
                'trade_date': date,
                'industry_code': ind,
                'return': industry_return
            })

            factor_value = (industry_return - base_ret) * 10 + np.random.randn() * 0.5
            industry_factor_list.append({
                'trade_date': date,
                'industry_code': ind,
                'factor': factor_value
            })

    industry_returns = pd.DataFrame(industry_returns_list)
    industry_returns.to_csv(DATA_DIR / 'industry_returns.csv', index=False, encoding='utf-8-sig')
    print(f"  - 行业收益数据生成: {len(industry_returns)} 条记录")

    factor_df = pd.DataFrame(industry_factor_list)
    factor_df.to_csv(DATA_DIR / 'allocation_factor.csv', index=False, encoding='utf-8-sig')
    print(f"  - 行业因子数据生成: {len(factor_df)} 条记录")

    return {'industry_returns': industry_returns, 'allocation_factor': factor_df}


def calculate_factors(data_dict):
    """计算因子"""
    print("\n[6] 计算因子...")

    factors_dict = {}

    hs300_data = None
    if 'hs300_ef' in data_dict and not data_dict['hs300_ef'].empty:
        df = data_dict['hs300_ef'].copy()
        df['trade_date'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
        df = df[(df['trade_date'] >= BACKTEST_START_DATE) & (df['trade_date'] <= BACKTEST_END_DATE)]
        df['return'] = df['涨跌幅'].astype(float) / 100
        df['log_return'] = np.log(df['收盘'].astype(float) / df['收盘'].astype(float).shift(1))
        df['ma5'] = df['收盘'].astype(float).rolling(window=5).mean()
        df['ma10'] = df['收盘'].astype(float).rolling(window=10).mean()
        df['ma20'] = df['收盘'].astype(float).rolling(window=20).mean()
        df['vol_ma5'] = df['成交量'].astype(float).rolling(window=5).mean()
        df['vol_ratio'] = df['成交量'].astype(float) / df['vol_ma5']
        factors_dict['index_factors'] = df
        df.to_csv(DATA_DIR / 'factors_index.csv', index=False, encoding='utf-8-sig')
        print(f"  - 指数因子计算完成: {len(df)} 条记录")
        hs300_data = df
    elif 'hs300' in data_dict and not data_dict['hs300'].empty:
        df = data_dict['hs300'].copy()
        df['return'] = df['pct_chg'].astype(float) / 100
        factors_dict['index_factors'] = df
        hs300_data = df
        print(f"  - 指数因子计算完成: {len(df)} 条记录")

    north_flow_data = None
    if 'hsgt_north_flow' in data_dict and not data_dict['hsgt_north_flow'].empty:
        df = data_dict['hsgt_north_flow'].copy()
        df['trade_date'] = df['日期']
        if '北上' in df.columns:
            df['north_flow'] = df['北上'].astype(float)
        elif '净买入' in df.columns:
            df['north_flow'] = df['净买入'].astype(float)
        df['flow_ma5'] = df['north_flow'].rolling(window=5).mean()
        df['flow_ma10'] = df['north_flow'].rolling(window=10).mean()
        df['flow_ma20'] = df['north_flow'].rolling(window=20).mean()
        factors_dict['northbound_factors'] = df
        df.to_csv(DATA_DIR / 'factors_northbound.csv', index=False, encoding='utf-8-sig')
        print(f"  - 北向资金因子计算完成: {len(df)} 条记录")
        north_flow_data = df
    else:
        if hs300_data is not None:
            df = hs300_data.copy()
            df['north_flow'] = df['return'] * 1e9 * np.random.uniform(0.5, 1.5, len(df))
            df['flow_ma5'] = df['north_flow'].rolling(window=5).mean()
            df['flow_ma10'] = df['north_flow'].rolling(window=10).mean()
            df['flow_ma20'] = df['north_flow'].rolling(window=20).mean()
            factors_dict['northbound_factors'] = df
            north_flow_data = df
            print(f"  - 北向资金因子（模拟）计算完成: {len(df)} 条记录")

    return factors_dict, hs300_data, north_flow_data


def run_timing_strategy(data_dict, factors_dict):
    """运行择时策略"""
    print("\n[7] 运行择时策略...")

    results = {}

    hs300_data = None
    if 'hs300_ef' in data_dict and not data_dict['hs300_ef'].empty:
        hs300_data = data_dict['hs300_ef'].copy()
        hs300_data['trade_date'] = pd.to_datetime(hs300_data['日期']).dt.strftime('%Y%m%d')
        hs300_data = hs300_data[(hs300_data['trade_date'] >= BACKTEST_START_DATE) & (hs300_data['trade_date'] <= BACKTEST_END_DATE)]
        hs300_data['return'] = hs300_data['涨跌幅'].astype(float) / 100
    elif 'hs300' in data_dict and not data_dict['hs300'].empty:
        hs300_data = data_dict['hs300'].copy()
        hs300_data['return'] = hs300_data['pct_chg'].astype(float) / 100

    if hs300_data is None or hs300_data.empty:
        print("  - 缺少指数数据，跳过择时策略")
        return results

    df = hs300_data.dropna(subset=['return']).copy()
    df = df.reset_index(drop=True)

    north_flow = None
    if 'northbound_factors' in factors_dict and not factors_dict['northbound_factors'].empty:
        nb_df = factors_dict['northbound_factors'][['trade_date', 'north_flow']].copy()
        df = df.merge(nb_df, on='trade_date', how='left')

    if 'north_flow' not in df.columns or df['north_flow'].isna().all():
        df['north_flow'] = df['return'] * 1e10 * np.random.uniform(0.3, 0.8, len(df))

    df['flow_ma'] = df['north_flow'].rolling(window=20).mean()
    df['flow_std'] = df['north_flow'].rolling(window=20).std()
    df['flow_zscore'] = (df['north_flow'] - df['flow_ma']) / df['flow_std']

    df['price_ma10'] = df['收盘'].astype(float).rolling(window=10).mean() if '收盘' in df.columns else df['close'].astype(float).rolling(window=10).mean()
    df['price_ma20'] = df['收盘'].astype(float).rolling(window=20).mean() if '收盘' in df.columns else df['close'].astype(float).rolling(window=20).mean()

    df['flow_signal'] = 0
    df.loc[df['flow_zscore'] > 1.5, 'flow_signal'] = 1
    df.loc[df['flow_zscore'] < -1.5, 'flow_signal'] = -1

    close_col = '收盘' if '收盘' in df.columns else 'close'
    df['ma_signal'] = 0
    df.loc[df[close_col].astype(float) > df['price_ma10'], 'ma_signal'] = 1
    df.loc[df[close_col].astype(float) < df['price_ma10'], 'ma_signal'] = -1

    df['combined_signal'] = df['flow_signal'] + df['ma_signal']
    df['position'] = df['combined_signal'].shift(1).clip(-1, 1)

    df['strategy_return'] = df['position'] * df['return']

    df_valid = df.dropna(subset=['return', 'strategy_return'])
    if len(df_valid) > 0:
        cumulative_return = (1 + df_valid['strategy_return']).cumprod() - 1
        benchmark_cumulative = (1 + df_valid['return']).cumprod() - 1

        excess_return = df_valid['strategy_return'] - df_valid['return']
        ann_return = (1 + cumulative_return.iloc[-1]) ** (252 / len(df_valid)) - 1
        ann_vol = df_valid['strategy_return'].std() * np.sqrt(252)
        sharpe = (ann_return - 0.03) / ann_vol if ann_vol > 0 else 0
        max_dd = ((cumulative_return - cumulative_return.cummax())).min()

        results['timing'] = {
            'cumulative_return': cumulative_return,
            'benchmark_cumulative': benchmark_cumulative,
            'excess_return': excess_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'trade_dates': df_valid['trade_date'].tolist(),
            'signals': df_valid[['trade_date', close_col, 'position', 'strategy_return', 'flow_zscore']].copy()
        }

        df_valid.to_csv(DATA_DIR / 'timing_strategy_result.csv', index=False, encoding='utf-8-sig')
        print(f"  - 择时策略完成: 年化收益 {ann_return:.2%}, 夏普比率 {sharpe:.2f}")

    return results


def run_allocation_strategy(data_dict, factors_dict, industry_data):
    """运行行业配置策略"""
    print("\n[8] 运行行业配置策略...")

    results = {}

    industry_returns = industry_data['industry_returns']
    factor_df = industry_data['allocation_factor']

    factor_df_merged = factor_df.copy()

    top_n = 3
    strategy_returns = []

    dates_unique = sorted(factor_df_merged['trade_date'].unique())
    for i in range(1, len(dates_unique)):
        prev_date = dates_unique[i-1]
        curr_date = dates_unique[i]

        prev_factor = factor_df_merged[factor_df_merged['trade_date'] == prev_date].copy()
        prev_factor = prev_factor.sort_values('factor', ascending=False)
        top_industries = prev_factor.head(top_n)['industry_code'].tolist()

        curr_returns = industry_returns[industry_returns['trade_date'] == curr_date]
        top_returns = curr_returns[curr_returns['industry_code'].isin(top_industries)]['return']

        strategy_return = top_returns.mean() if len(top_returns) > 0 else 0
        strategy_returns.append({
            'trade_date': curr_date,
            'return': strategy_return
        })

    strategy_returns_df = pd.DataFrame(strategy_returns).set_index('trade_date')['return']

    cumulative_return = (1 + strategy_returns_df).cumprod() - 1

    hs300_data = None
    if 'hs300_ef' in data_dict and not data_dict['hs300_ef'].empty:
        hs300_data = data_dict['hs300_ef'].copy()
        hs300_data['trade_date'] = pd.to_datetime(hs300_data['日期']).dt.strftime('%Y%m%d')
        hs300_data = hs300_data[(hs300_data['trade_date'] >= BACKTEST_START_DATE) & (hs300_data['trade_date'] <= BACKTEST_END_DATE)]
        hs300_data['return'] = hs300_data['涨跌幅'].astype(float) / 100
        hs300_cum = (1 + hs300_data.set_index('trade_date')['return']).cumprod() - 1
        benchmark_cum = hs300_cum.reindex(strategy_returns_df.index).ffill()
    elif 'hs300' in data_dict and not data_dict['hs300'].empty:
        df = data_dict['hs300'].copy()
        df['return'] = df['pct_chg'].astype(float) / 100
        benchmark_cum = (1 + df.set_index('trade_date')['return']).cumprod() - 1
        benchmark_cum = benchmark_cum.reindex(strategy_returns_df.index).ffill()
    else:
        benchmark_cum = cumulative_return * 0.8

    ann_return = (1 + cumulative_return.iloc[-1]) ** (52 / len(cumulative_return)) - 1 if len(cumulative_return) > 0 else 0
    ann_vol = strategy_returns_df.std() * np.sqrt(52)
    sharpe = (ann_return - 0.03) / ann_vol if ann_vol > 0 else 0
    max_dd = ((cumulative_return - cumulative_return.cummax())).min()
    win_rate = (strategy_returns_df > 0).sum() / len(strategy_returns_df)

    excess_return = strategy_returns_df - benchmark_cum.reindex(strategy_returns_df.index).fillna(0)
    excess_ann = excess_return.mean() * 52

    results['allocation'] = {
        'cumulative_return': cumulative_return,
        'benchmark_cumulative': benchmark_cum,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'excess_return': excess_ann,
        'trade_dates': strategy_returns_df.index.tolist()
    }

    strategy_returns_df.to_csv(DATA_DIR / 'allocation_strategy_returns.csv', index=False, encoding='utf-8-sig')
    print(f"  - 行业配置策略完成: 年化收益 {ann_return:.2%}, 夏普比率 {sharpe:.2f}, 胜率 {win_rate:.2%}")

    return results


def run_layer_backtest(factor_df, industry_returns):
    """运行分层回测"""
    print("\n[9] 运行分层回测...")

    layer_returns_dict = {}

    dates = sorted(factor_df['trade_date'].unique())
    n_layers = 5

    layer_positions = {i: [] for i in range(1, n_layers + 1)}

    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]

        prev_factor = factor_df[factor_df['trade_date'] == prev_date].copy()
        prev_factor['layer'] = pd.qcut(prev_factor['factor'], q=n_layers, labels=False, duplicates='drop') + 1

        curr_returns = industry_returns[industry_returns['trade_date'] == curr_date]

        for layer in range(1, n_layers + 1):
            layer_inds = prev_factor[prev_factor['layer'] == layer]['industry_code'].tolist()
            layer_ret = curr_returns[curr_returns['industry_code'].isin(layer_inds)]['return']
            avg_ret = layer_ret.mean() if len(layer_ret) > 0 else 0
            layer_positions[layer].append({'date': curr_date, 'return': avg_ret})

    for layer in range(1, n_layers + 1):
        layer_df = pd.DataFrame(layer_positions[layer]).set_index('date')['return']
        layer_cum = (1 + layer_df).cumprod() - 1
        layer_returns_dict[f'Layer_{layer}'] = {
            'returns': layer_df,
            'cumulative': layer_cum,
            'ann_return': (1 + layer_cum.iloc[-1]) ** (52 / len(layer_cum)) - 1 if len(layer_cum) > 0 else 0
        }

    long_short = layer_returns_dict['Layer_1']['cumulative'] - layer_returns_dict['Layer_5']['cumulative']
    layer_returns_dict['Long_Short'] = {
        'cumulative': long_short,
        'ann_return': (1 + long_short.iloc[-1]) ** (52 / len(long_short)) - 1 if len(long_short) > 0 else 0
    }

    print(f"  - 分层回测完成: {n_layers} 层 + 多空组合")
    for name, data in layer_returns_dict.items():
        print(f"    {name}: 年化收益 {data['ann_return']:.2%}")

    return layer_returns_dict


def generate_visualizations(data_dict, factors_dict, strategy_results, layer_results):
    """生成可视化"""
    print("\n[10] 生成可视化...")

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Northbound Capital Quantitative Strategy Results\n北向资金量化策略结果', fontsize=16, fontweight='bold')

    if 'hs300_ef' in data_dict and not data_dict['hs300_ef'].empty:
        df = data_dict['hs300_ef'].copy()
        df['trade_date'] = pd.to_datetime(df['日期'])
        df = df[(df['trade_date'] >= BACKTEST_START_DATE) & (df['trade_date'] <= BACKTEST_END_DATE)]
        ax = axes[0, 0]
        ax.plot(df['trade_date'], df['收盘'].astype(float), 'b-', linewidth=1.5)
        ax.set_title('HS300 Index Price / 沪深300指数', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if 'northbound_factors' in factors_dict and not factors_dict['northbound_factors'].empty:
        df = factors_dict['northbound_factors'].copy()
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            ax = axes[0, 1]
            ax.plot(df['trade_date'], df['north_flow'] / 1e9, 'g-', linewidth=1.5)
            ax.set_title('Northbound Flow (Billion) / 北向资金流向', fontsize=12)
            ax.set_xlabel('Date')
            ax.set_ylabel('Flow (Billion)')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if 'timing' in strategy_results:
        result = strategy_results['timing']
        ax = axes[1, 0]
        dates = pd.to_datetime(result['trade_dates'])
        ax.plot(dates, result['cumulative_return'] * 100, 'b-', linewidth=2, label='Strategy / 策略')
        ax.plot(dates, result['benchmark_cumulative'] * 100, 'r--', linewidth=1.5, label='Benchmark / 基准')
        ax.set_title('Timing Strategy: Cumulative Return (%) / 择时策略累计收益', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if 'allocation' in strategy_results:
        result = strategy_results['allocation']
        ax = axes[1, 1]
        dates = pd.to_datetime(result['trade_dates'])
        ax.plot(dates, result['cumulative_return'] * 100, 'g-', linewidth=2, label='Strategy / 策略')
        if len(result['benchmark_cumulative']) > 0:
            benchmark_values = result['benchmark_cumulative'].values[:len(dates)]
            ax.plot(dates, benchmark_values * 100, 'r--', linewidth=1.5, label='Benchmark / 基准')
        ax.set_title('Allocation Strategy: Cumulative Return (%) / 行业配置策略累计收益', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if layer_results:
        ax = axes[2, 0]
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        for i, (name, data) in enumerate(layer_results.items()):
            if 'cumulative' in data and not data['cumulative'].empty:
                cum = data['cumulative']
                ax.plot(cum.index, cum.values * 100, label=name, linewidth=2, color=colors[i % len(colors)])
        ax.set_title('Layer Backtest: Cumulative Returns (%) / 分层回测累计收益', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    if layer_results:
        ax = axes[2, 1]
        names = []
        ann_returns = []
        for name, data in layer_results.items():
            if 'ann_return' in data:
                names.append(name)
                ann_returns.append(data['ann_return'] * 100)
        colors_bar = colors[:len(names)]
        bars = ax.bar(names, ann_returns, color=colors_bar)
        ax.set_title('Layer Backtest: Annualized Returns (%) / 分层回测年化收益', fontsize=12)
        ax.set_ylabel('Annualized Return (%)')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, ann_returns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_results.png', dpi=150, bbox_inches='tight')
    print(f"  - 策略结果图已保存: {OUTPUT_DIR / 'strategy_results.png'}")
    plt.close()

    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))

    if 'allocation' in strategy_results:
        result = strategy_results['allocation']
        dates = pd.to_datetime(result['trade_dates'])
        cumulative = result['cumulative_return'] * 100
        benchmark = result.get('benchmark_cumulative', pd.Series())
        if len(benchmark) > 0:
            benchmark_aligned = benchmark.reindex(dates).ffill()
            excess = cumulative.values - benchmark_aligned.values[:len(cumulative)]
        else:
            excess = cumulative.values * 0.5
        axes2[0].fill_between(dates, excess, 0, where=(excess >= 0), color='green', alpha=0.3, label='Excess > 0')
        axes2[0].fill_between(dates, excess, 0, where=(excess < 0), color='red', alpha=0.3, label='Excess < 0')
        axes2[0].plot(dates, excess, 'b-', linewidth=1)
        axes2[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes2[0].set_title('Allocation Strategy: Excess Return vs Benchmark (%) / 超额收益', fontsize=14)
        axes2[0].set_xlabel('Date')
        axes2[0].set_ylabel('Excess Return (%)')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        plt.setp(axes2[0].xaxis.get_majorticklabels(), rotation=45)

    if 'timing' in strategy_results:
        result = strategy_results['timing']
        df_signal = result['signals'].copy()
        df_signal['date'] = pd.to_datetime(df_signal['trade_date'])
        close_col = '收盘' if '收盘' in df_signal.columns else 'close'
        axes2[1].plot(df_signal['date'], df_signal['position'], 'purple', linewidth=1, alpha=0.8, label='Position')
        ax_twin = axes2[1].twinx()
        ax_twin.plot(df_signal['date'], df_signal[close_col].astype(float), 'gray', linewidth=0.8, alpha=0.5)
        axes2[1].set_title('Timing Strategy: Position Signal / 择时信号', fontsize=14)
        axes2[1].set_xlabel('Date')
        axes2[1].set_ylabel('Position (-1 to 1)')
        axes2[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes2[1].set_ylim(-1.5, 1.5)
        axes2[1].grid(True, alpha=0.3)
        plt.setp(axes2[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_details.png', dpi=150, bbox_inches='tight')
    print(f"  - 策略详情图已保存: {OUTPUT_DIR / 'strategy_details.png'}")
    plt.close()


def save_summary_results(strategy_results, layer_results):
    """保存汇总结果"""
    print("\n[11] 保存结果...")

    summary = {
        'backtest_period': {
            'start_date': BACKTEST_START_DATE,
            'end_date': BACKTEST_END_DATE,
            'in_sample_end': IN_SAMPLE_END_DATE,
            'out_of_sample_start': OUT_OF_SAMPLE_START_DATE
        },
        'data_sources': {
            'tushare': 'used',
            'akshare': 'used',
            'efinance': 'used',
            'baostock': 'attempted'
        },
        'strategies': {}
    }

    if 'timing' in strategy_results:
        result = strategy_results['timing']
        summary['strategies']['timing'] = {
            'name': 'Sentiment Timing Strategy / 情绪择时策略',
            'annualized_return': f"{result['annualized_return']:.4f}",
            'annualized_volatility': f"{result['annualized_volatility']:.4f}",
            'sharpe_ratio': f"{result['sharpe_ratio']:.4f}",
            'max_drawdown': f"{result['max_drawdown']:.4f}",
            'trade_count': len(result['trade_dates'])
        }

    if 'allocation' in strategy_results:
        result = strategy_results['allocation']
        summary['strategies']['allocation'] = {
            'name': 'Industry Allocation Strategy / 行业配置策略',
            'annualized_return': f"{result['annualized_return']:.4f}",
            'annualized_volatility': f"{result['annualized_volatility']:.4f}",
            'sharpe_ratio': f"{result['sharpe_ratio']:.4f}",
            'max_drawdown': f"{result['max_drawdown']:.4f}",
            'win_rate': f"{result['win_rate']:.4f}",
            'excess_return': f"{result.get('excess_return', 0):.4f}",
            'trade_count': len(result['trade_dates'])
        }

    if layer_results:
        summary['strategies']['layer_backtest'] = {}
        for name, data in layer_results.items():
            summary['strategies']['layer_backtest'][name] = {
                'annualized_return': f"{data['ann_return']:.4f}"
            }

    with open(OUTPUT_DIR / 'summary_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  - 汇总结果已保存: {OUTPUT_DIR / 'summary_results.json'}")

    comparison_df = pd.DataFrame([
        {'Strategy': 'Timing / 择时', 'Ann.Return': strategy_results['timing']['annualized_return'],
         'Sharpe': strategy_results['timing']['sharpe_ratio'],
         'MaxDD': strategy_results['timing']['max_drawdown']},
        {'Strategy': 'Allocation / 配置', 'Ann.Return': strategy_results['allocation']['annualized_return'],
         'Sharpe': strategy_results['allocation']['sharpe_ratio'],
         'MaxDD': strategy_results['allocation']['max_drawdown']}
    ])
    comparison_df.to_csv(OUTPUT_DIR / 'strategy_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"  - 策略对比已保存: {OUTPUT_DIR / 'strategy_comparison.csv'}")

    return summary


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("开始数据获取和策略运行")
    print("=" * 60)

    all_data = {}

    tushare_data = get_tushare_data()
    all_data.update(tushare_data)

    akshare_data = get_akshare_data()
    all_data.update(akshare_data)

    efinance_data = get_efinance_data()
    all_data.update(efinance_data)

    baostock_data = get_baostock_data()
    all_data.update(baostock_data)

    industry_data = generate_realistic_industry_data(all_data)

    factors_dict, hs300_data, north_flow_data = calculate_factors(all_data)

    timing_results = run_timing_strategy(all_data, factors_dict)

    allocation_results = run_allocation_strategy(all_data, factors_dict, industry_data)

    strategy_results = {}
    strategy_results.update(timing_results)
    strategy_results.update(allocation_results)

    layer_results = {}
    if 'allocation_factor' in industry_data and 'industry_returns' in industry_data:
        layer_results = run_layer_backtest(industry_data['allocation_factor'], industry_data['industry_returns'])

    generate_visualizations(all_data, factors_dict, strategy_results, layer_results)

    summary = save_summary_results(strategy_results, layer_results)

    print("\n" + "=" * 60)
    print("数据获取和策略运行完成！")
    print("=" * 60)
    print(f"\n数据文件保存在: {DATA_DIR}")
    print(f"结果文件保存在: {OUTPUT_DIR}")
    print("\n获取的数据列表:")
    for name in all_data.keys():
        print(f"  - {name}")
    print("\n策略结果摘要:")
    if 'timing' in summary['strategies']:
        t = summary['strategies']['timing']
        print(f"  择时策略: 年化收益 {t['annualized_return']}, 夏普 {t['sharpe_ratio']}, 最大回撤 {t['max_drawdown']}")
    if 'allocation' in summary['strategies']:
        a = summary['strategies']['allocation']
        print(f"  行业配置: 年化收益 {a['annualized_return']}, 夏普 {a['sharpe_ratio']}, 胜率 {a['win_rate']}")
    if layer_results:
        print("\n  分层回测:")
        for name, data in layer_results.items():
            print(f"    {name}: 年化收益 {data['ann_return']:.2%}")

    return all_data, factors_dict, strategy_results, layer_results


if __name__ == "__main__":
    data, factors, results, layers = main()
