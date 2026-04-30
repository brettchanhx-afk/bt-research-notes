# -*- coding: utf-8 -*-
"""
数据获取模块
数据源优先级：efinance > akshare > baostock
"""
import os
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


def get_fund_list(n: int = 200) -> pd.DataFrame:
    """获取股票型基金列表（efinance）。"""
    try:
        import efinance as ef
        df = ef.fund.get_fund_codes()
        df.columns = ['基金代码', '基金名称']
        df = df.dropna(subset=['基金代码'])
        df['基金代码'] = df['基金代码'].astype(str).str.zfill(6)
        if len(df) > n:
            df = df.sample(n=n, random_state=42).reset_index(drop=True)
        print(f'  [efinance] 基金列表 {len(df)} 只')
        return df.reset_index(drop=True)
    except Exception as e:
        print(f'  [efinance] 获取基金列表失败: {e}')
        return pd.DataFrame(columns=['基金代码', '基金名称'])


def _parse_nav_return_col(raw_vals) -> pd.Series:
    """将任意格式的涨跌幅列转换为数值日收益率。
    
    支持格式：
      - 数值小数：0.0123
      - 百分数字符串：'1.23' 或 '1.23%'
      - 缺失标记：'--' 或 nan
    """
    str_vals = raw_vals.astype(str)
    str_vals = str_vals.replace('--', 'nan').replace('None', 'nan').replace('', 'nan')
    str_vals = str_vals.str.replace('%', '', regex=False).str.strip()
    num_vals = pd.to_numeric(str_vals, errors='coerce')
    if num_vals.abs().mean() > 1:
        num_vals = num_vals / 100.0
    return num_vals


def get_fund_nav_history(fund_codes: list, start: str, end: str) -> dict:
    """获取多只基金的历史净值（efinance，批量请求）。"""
    import efinance as ef
    
    result = {}
    batch_size = 10
    
    for i in range(0, len(fund_codes), batch_size):
        batch = fund_codes[i:i + batch_size]
        try:
            nav_dict = ef.fund.get_quote_history_multi(batch)
            if nav_dict is None:
                continue
            for code, raw_df in nav_dict.items():
                if raw_df is None or len(raw_df) == 0:
                    continue
                df = raw_df.copy()
                df.columns = [str(c).strip() for c in df.columns]
                
                date_col = df.columns[0]
                nav_col  = df.columns[1]
                ret_col  = df.columns[-1]
                
                nav_vals = pd.to_numeric(df[nav_col], errors='coerce')
                ret_vals = _parse_nav_return_col(df[ret_col])
                
                dates = pd.to_datetime(df[date_col], errors='coerce')
                out = pd.DataFrame({
                    '净值': nav_vals.values,
                    '日增长率': ret_vals.values,
                }, index=dates)
                out = out.sort_index()
                
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                out = out[(out.index >= start_dt) & (out.index <= end_dt)]
                
                if len(out) > 0 and out['净值'].notna().sum() > 30:
                    result[str(code).zfill(6)] = out[['净值', '日增长率']]
                    
        except Exception as e:
            print(f'  [efinance] 净值batch{i//batch_size}失败: {e}')
        time.sleep(0.3)

    print(f'  [efinance] 净值历史 {len(result)} 只基金')
    return result


def get_fund_fee_rate(fund_codes: list) -> pd.DataFrame:
    """获取基金费率（管理费+托管费，年化，akshare）。"""
    try:
        import akshare as ak
        results = []
        for code in fund_codes[:50]:
            try:
                info = ak.fund_financial_summary_ths(symbol=str(code).zfill(6))
                if info is not None and len(info) > 0:
                    mng_fee = info['管理费率'].iloc[0] if '管理费率' in info.columns else None
                    trust_fee = info['托管费率'].iloc[0] if '托管费率' in info.columns else None
                    results.append({
                        '基金代码': str(code).zfill(6),
                        '管理费率': float(mng_fee) if mng_fee is not None else 0.015,
                        '托管费率': float(trust_fee) if trust_fee is not None else 0.0025,
                    })
            except Exception:
                pass
        if results:
            df = pd.DataFrame(results)
            df['总费率'] = df['管理费率'] + df['托管费率']
            print(f'  [akshare] 费率数据 {len(df)} 只基金')
            return df
    except Exception as e:
        print(f'  [akshare] 费率获取失败: {e}')
    
    default_df = pd.DataFrame({'基金代码': fund_codes})
    default_df['管理费率'] = 0.015
    default_df['托管费率'] = 0.0025
    default_df['总费率'] = default_df['管理费率'] + default_df['托管费率']
    print(f'  [费率] 使用默认值 {len(default_df)} 只基金')
    return default_df


def get_benchmark_history(benchmark_code: str = '000300',
                          start: str = '2019-01-01',
                          end: str = '2024-12-31') -> pd.DataFrame:
    """获取基准指数（沪深300）历史数据（akshare）。"""
    # ---- 方法1：akshare ----
    try:
        import akshare as ak
        # 沪深300: sh000300
        df = ak.stock_zh_index_daily(symbol='sh000300')
        if df is not None and len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df['日收益率'] = df['close'].pct_change()
            # 日期过滤
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            print(f'  [akshare] 沪深300历史 {len(df)} 条')
            return df
    except Exception as e:
        print(f'  [akshare] 沪深300获取失败: {e}')

    # ---- 方法2：baostock 备选 ----
    try:
        import baostock as bs
        bs.login()
        rs = bs.query_history_k_data_plus(
            f'sh.{benchmark_code}',
            'date,close',
            start_date=start.replace('-', ''),
            end_date=end.replace('-', ''),
            frequency='d'
        )
        if rs is not None:
            data_list = []
            while rs.next():
                row = rs.get_row_data()
                if row:
                    data_list.append(row)
            bs.logout()
            if data_list:
                df = pd.DataFrame(data_list, columns=['date', 'close'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['日收益率'] = df['close'].pct_change()
                print(f'  [baostock] 沪深300历史 {len(df)} 条（备选）')
                return df
        else:
            bs.logout()
    except Exception as e:
        print(f'  [baostock] 备选失败: {e}')
        try:
            import baostock as bs
            bs.logout()
        except Exception:
            pass

    print(f'  [WARNING] 基准指数获取失败，使用零收益基准')
    dates = pd.bdate_range(start=start, end=end)
    return pd.DataFrame({'close': 1.0, '日收益率': 0.0}, index=dates, dtype=float)


def get_fund_info(fund_codes: list) -> pd.DataFrame:
    """获取基金基本信息（成立时间，akshare）。"""
    try:
        import akshare as ak
        info_list = []
        for code in fund_codes[:100]:
            try:
                info = ak.fund_info_sina(symbol=str(code).zfill(6))
                if info is not None:
                    info_list.append({
                        '基金代码': str(code).zfill(6),
                        '成立日期': info.get('成立日期', None),
                        '基金类型': info.get('基金类型', '股票型'),
                    })
            except Exception:
                pass
        if info_list:
            df = pd.DataFrame(info_list)
            print(f'  [akshare] 基金信息 {len(df)} 只')
            return df
    except Exception as e:
        print(f'  [akshare] 基金信息获取失败: {e}')
    return pd.DataFrame({'基金代码': fund_codes})
