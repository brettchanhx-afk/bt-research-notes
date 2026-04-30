# -*- coding: utf-8 -*-
"""
数据获取模块 - 基金选股择时能力定量评价模型
支持多数据源：efinance > akshare > baostock
严格从开源库实时拉取真实市场数据，不编造数据。

数据口径：
- 基金净值：日频单位净值（涨跌幅列直接用，不自行计算）
- 基准：沪深300指数日收盘价
- 无风险利率：存款基准利率 1.5%（年化）
"""

import os
import sys
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ==================== 数据源1：efinance ====================
def load_fund_nav_efinance(fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 efinance 获取基金日频净值数据。
    接口：efinance.fund.get_quote_history(fund_code)

    参数:
        fund_code: 基金代码（如 '021181'）
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
    返回:
        DataFrame，index='日期'，columns=['净值', '收益率']
    """
    try:
        import efinance.fund as ef_fund

        # 获取全部历史数据（efinance 不支持日期范围过滤，需自行切片）
        df = ef_fund.get_quote_history(fund_code)

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # efinance 返回的列名：['日期', '单位净值', '累计净值', '涨跌幅']
        date_col = '日期'
        nav_col = '单位净值'
        pct_col = '涨跌幅'

        if date_col not in df.columns or nav_col not in df.columns:
            return pd.DataFrame()

        # 转换日期（字符串 'YYYY-MM-DD' -> pd.Timestamp）
        df[date_col] = pd.to_datetime(df[date_col])

        # 过滤日期范围（使用字符串比较，efinance 格式已经是 YYYY-MM-DD）
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values(date_col).reset_index(drop=True)

        # 收益率：直接使用涨跌幅列（efinance 已计算好，需处理 '--' 缺失值）
        # 注意：涨跌幅是百分比形式（如 0.52 表示 0.52%），转换为小数
        # 使用 .values 避免 Series 索引与 DataFrame 索引对齐问题
        raw_returns = pd.to_numeric(df[pct_col].replace('--', np.nan), errors='coerce').values
        raw_returns = raw_returns / 100.0

        result = pd.DataFrame({
            '净值': pd.to_numeric(df[nav_col], errors='coerce').values,
            '收益率': raw_returns,
        }, index=df[date_col])
        result.index.name = '日期'

        # 删除收益率为空的行
        result = result[~np.isnan(result['收益率'])]

        return result

    except ImportError:
        warnings.warn("efinance 未安装，尝试下一数据源")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"efinance 获取失败: {e}")
        return pd.DataFrame()


def load_benchmark_efinance(benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 efinance 获取基准指数（沪深300）日频行情数据。
    接口：efinance.stock.get_quote_history(code, beg='YYYYMMDD', end='YYYYMMDD')

    参数:
        benchmark: 指数代码（如 '000300'）
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
    返回:
        DataFrame，index='日期'，columns=['收盘价', '收益率']
    """
    try:
        import efinance as ef

        # efinance 股票接口需要 YYYYMMDD 格式
        beg_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')

        # 如果代码带 .SH/.SZ 后缀，去掉
        bm_code = benchmark.replace('.SH', '').replace('.SZ', '').replace('.CSI', '')

        df = ef.stock.get_quote_history(bm_code, beg=beg_str, end=end_str)

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # 标准化列名
        date_col = '日期'
        close_col = '收盘'

        if date_col not in df.columns or close_col not in df.columns:
            # 尝试找其他列名
            for col in df.columns:
                if '日期' in col:
                    date_col = col
                if col in ['收盘', '收盘价', 'close']:
                    close_col = col

        if date_col not in df.columns or close_col not in df.columns:
            return pd.DataFrame()

        # 转换日期
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # 计算日收益率（使用 .values 避免索引对齐问题）
        closes = pd.to_numeric(df[close_col], errors='coerce').values
        returns = np.diff(closes) / closes[:-1]  # 日收益率

        result = pd.DataFrame({
            '收盘价': closes,
            '收益率': np.concatenate([[np.nan], returns]),
        }, index=df[date_col])
        result.index.name = '日期'

        return result[~np.isnan(result['收益率'])]

    except Exception as e:
        warnings.warn(f"efinance 基准获取失败: {e}")
        return pd.DataFrame()


# ==================== 数据源2：akshare ====================
def load_fund_nav_akshare(fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 akshare 获取基金日频净值数据。
    """
    try:
        import akshare as ak

        symbol = fund_code.zfill(6)
        try:
            # akshare 基金净值接口
            df = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")
        except Exception:
            try:
                df = ak.fund_individual_basic_info_xq(symbol=symbol)
            except Exception:
                return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # 找日期列和净值列
        date_col = df.columns[0]
        nav_col = None
        for col in df.columns:
            if any(k in col for k in ['净值', '单位净值', 'nav']):
                nav_col = col
                break
        if nav_col is None:
            nav_col = df.columns[1]

        # 处理日期
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values(date_col).reset_index(drop=True)

        result = pd.DataFrame({
            '净值': pd.to_numeric(df[nav_col], errors='coerce'),
            '收益率': pd.to_numeric(df[nav_col], errors='coerce').pct_change(fill_method=None),
        }, index=df[date_col])
        result.index.name = '日期'

        return result.dropna(how='all')

    except ImportError:
        warnings.warn("akshare 未安装")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"akshare 获取失败: {e}")
        return pd.DataFrame()


def load_benchmark_akshare(benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 akshare 获取沪深300指数数据。
    """
    try:
        import akshare as ak

        symbol = "000300"
        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
        except Exception:
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values('date').reset_index(drop=True)

        closes = pd.to_numeric(df['close'], errors='coerce')
        returns = closes.pct_change(fill_method=None)

        result = pd.DataFrame({
            '收盘价': closes,
            '收益率': returns,
        }, index=df['date'])
        result.index.name = '日期'

        return result.dropna(how='all')

    except Exception as e:
        warnings.warn(f"akshare 基准获取失败: {e}")
        return pd.DataFrame()


# ==================== 数据源3：baostock ====================
def load_benchmark_baostock(benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 baostock 获取指数日线数据。
    """
    try:
        import baostock as bs

        bs.login()
        bs_code = f"sh.{benchmark.replace('.SH', '').replace('.SZ', '').replace('.CSI', '')}"
        df = bs.query_history_k_data_plus(
            bs_code,
            "date,close",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        bs.logout()

        if df is None or df.data is None or len(df.data) == 0:
            return pd.DataFrame()

        records = df.data
        dates = [r[0] for r in records]
        closes = [float(r[1]) for r in records]

        result = pd.DataFrame({
            '收盘价': closes,
            '收益率': pd.Series(closes).pct_change(fill_method=None).values,
        }, index=pd.to_datetime(dates))
        result.index.name = '日期'

        return result.dropna(how='all')

    except ImportError:
        warnings.warn("baostock 未安装")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"baostock 基准获取失败: {e}")
        return pd.DataFrame()


def load_fund_nav_baostock(fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    baostock 主要提供股票数据，基金数据支持有限。
    """
    warnings.warn("baostock 基金数据支持有限，建议使用 efinance 或 akshare")
    return pd.DataFrame()


# ==================== 统一数据加载入口 ====================
def load_fund_nav(fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    统一入口：从多个数据源依次尝试获取基金净值数据。

    数据源优先级：efinance > akshare > baostock

    参数:
        fund_code: 基金代码
        start_date: 开始日期
        end_date: 结束日期
    返回:
        DataFrame with columns ['净值', '收益率'], index='日期' (DatetimeIndex)
    """
    sources = [
        ("efinance", lambda: load_fund_nav_efinance(fund_code, start_date, end_date)),
        ("akshare",  lambda: load_fund_nav_akshare(fund_code, start_date, end_date)),
        ("baostock", lambda: load_fund_nav_baostock(fund_code, start_date, end_date)),
    ]

    for name, loader in sources:
        df = loader()
        if df is not None and not df.empty and len(df) > 30:
            print(f"[{name}] 成功获取基金 {fund_code} 数据: {len(df)} 条记录")
            return df

    warnings.warn(f"所有数据源均无法获取基金 {fund_code} 数据，请检查代码是否正确")
    return pd.DataFrame()


def load_benchmark(benchmark: str = '000300.SH',
                   start_date: str = '2018-01-01',
                   end_date: str = '2026-04-28') -> pd.DataFrame:
    """
    统一入口：从多个数据源获取基准指数数据。

    数据源优先级：efinance > akshare > baostock

    参数:
        benchmark: 指数代码（默认沪深300: 000300.SH）
        start_date: 开始日期
        end_date: 结束日期
    返回:
        DataFrame with columns ['收盘价', '收益率'], index='日期' (DatetimeIndex)
    """
    bm_code = benchmark.replace('.SH', '').replace('.SZ', '').replace('.CSI', '')

    sources = [
        ("efinance", lambda: load_benchmark_efinance(bm_code, start_date, end_date)),
        ("akshare",  lambda: load_benchmark_akshare(bm_code, start_date, end_date)),
        ("baostock", lambda: load_benchmark_baostock(bm_code, start_date, end_date)),
    ]

    for name, loader in sources:
        df = loader()
        if df is not None and not df.empty and len(df) > 30:
            print(f"[{name}] 成功获取基准 {benchmark} 数据: {len(df)} 条记录")
            return df

    warnings.warn(f"所有数据源均无法获取基准 {benchmark} 数据")
    return pd.DataFrame()


def load_all_data(fund_code: str,
                  start_date: str = '2018-01-01',
                  end_date: str = '2026-04-28',
                  benchmark: str = '000300.SH') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    一次性加载基金净值和基准数据，并对齐日期。

    返回:
        (fund_returns, benchmark_returns) - 两个 DataFrame，index 为对齐的日期
    """
    # 1. 加载基金净值
    fund_nav = load_fund_nav(fund_code, start_date, end_date)
    if fund_nav.empty:
        raise ValueError(f"无法获取基金 {fund_code} 的净值数据")

    # 2. 加载基准数据
    bench_df = load_benchmark(benchmark, start_date, end_date)
    if bench_df.empty:
        raise ValueError(f"无法获取基准 {benchmark} 的数据")

    # 3. 对齐日期（inner join，只保留两者共同的交易日）
    # align() 返回 (Series, Series)，各自有对齐后的日期索引
    fund_aligned, bench_aligned = fund_nav['收益率'].align(
        bench_df['收益率'],
        join='inner',
    )
    fund_returns_df = pd.DataFrame({'基金收益率': fund_aligned.values}, index=fund_aligned.index)
    bench_returns_df = pd.DataFrame({'基准收益率': bench_aligned.values}, index=bench_aligned.index)
    print(f"日期对齐完成：{len(fund_returns_df)} 个共同交易日")
    return fund_returns_df, bench_returns_df


# ==================== 单元测试 ====================
if __name__ == '__main__':
    fund_code = '021181'
    start_date = '2021-01-01'
    end_date = '2026-04-28'
    benchmark = '000300.SH'

    print(f"正在加载基金 {fund_code} 和基准 {benchmark} 数据...")

    fund_df, bench_df = load_all_data(fund_code, start_date, end_date, benchmark)

    if not fund_df.empty:
        print(f"\n基金数据预览（前5行）:\n{fund_df.head()}")
        print(f"\n数据范围: {fund_df.index[0]} 至 {fund_df.index[-1]}")
        print(f"数据条数: {len(fund_df)}")
