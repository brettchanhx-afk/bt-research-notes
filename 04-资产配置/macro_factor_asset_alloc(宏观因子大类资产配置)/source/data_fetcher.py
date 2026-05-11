"""
数据获取模块 - 从各数据源获取市场数据
支持: tushare, efinance, akshare, baostock等
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import (
    TUSHARE_TOKEN, TUSHARE_URL, ASSET_CONFIG, ALL_ASSETS,
    BACKTEST_CONFIG, OUTPUT_DIR
)


def init_tushare():
    """初始化tushare Pro API"""
    token = TUSHARE_TOKEN
    pro = ts.pro_api(token)
    pro._DataApi__token = token
    pro._DataApi__http_url = TUSHARE_URL
    return pro


def get_tushare_index_data(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从tushare获取指数行情数据

    Parameters:
        ts_code: tushare指数代码，如 '000300.SH'
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYYMMDD'

    Returns:
        DataFrame with columns: [trade_date, close, pct_chg]
    """
    pro = init_tushare()
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values('trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')
    return df[['close', 'pct_chg']].rename(columns={'pct_chg': 'return'})


def get_tushare_stock_data(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从tushare获取股票/可转债行情数据

    Parameters:
        ts_code: tushare股票代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with columns: [trade_date, close, pct_chg]
    """
    pro = init_tushare()

    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date')
    df = df.set_index('trade_date')

    df['return'] = df['close'].pct_change()
    return df[['close', 'return']]


def get_efinance_index_data(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从efinance获取债券/商品指数数据

    Parameters:
        index_code: efinance指数代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with columns: [date, close, return]
    """
    try:
        import efinance as ef

        df = ef.bond.get_hist(index_code, start=start_date, end=end_date)
        if df is None or df.empty:
            df = ef.index.get_hist(index_code, start=start_date, end=end_date)

        if df is None or df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['日期'] if '日期' in df.columns else df.index)
        df = df.sort_values('date')
        df = df.set_index('date')

        close_col = '收盘' if '收盘' in df.columns else 'close'
        df['return'] = df[close_col].pct_change()

        return df[[close_col, 'return']].rename(columns={close_col: 'close'})

    except Exception as e:
        print(f"efinance获取数据失败 {index_code}: {e}")
        return pd.DataFrame()


def get_akshare_commodity_data(commodity_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从akshare获取大宗商品数据

    Parameters:
        commodity_name: 商品名称，如 '布伦特原油', '沪金'
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with columns: [date, close, return]
    """
    try:
        import akshare as ak

        symbol_map = {
            '布伦特原油': 'oil_brent',
            '沪金': 'futures_cu_main'
        }

        symbol = symbol_map.get(commodity_name, commodity_name)

        if commodity_name == '布伦特原油':
            df = ak.oil_brent()
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        elif commodity_name == '沪金':
            df = ak.futures_cu_main_sina()
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        else:
            return pd.DataFrame()

        df = df.sort_values('date')
        df = df.set_index('date')
        df['return'] = df['close'].pct_change()

        return df[['close', 'return']]

    except Exception as e:
        print(f"akshare获取商品数据失败 {commodity_name}: {e}")
        return pd.DataFrame()


def get_akshare_forex_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从akshare获取汇率数据

    Parameters:
        symbol: 货币对符号，如 'USDCNY'
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with columns: [date, close, return]
    """
    try:
        import akshare as ak

        if symbol == 'USDCNY':
            df = ak.currency_us_cny_hist(start_date=start_date, end_date=end_date)
        else:
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['日期'])
        df = df.sort_values('date')
        df = df.set_index('date')
        df['return'] = df['收盘'].pct_change()

        return df[['收盘', 'return']].rename(columns={'收盘': 'close'})

    except Exception as e:
        print(f"akshare获取汇率数据失败 {symbol}: {e}")
        return pd.DataFrame()


def get_asset_price_data(asset_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取单个资产的价格数据（统一接口）

    Parameters:
        asset_name: 资产名称
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with columns: [date, close, return]
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    for category, assets in ASSET_CONFIG.items():
        if asset_name in assets:
            config = assets[asset_name]
            source = config['source']

            if source == 'tushare':
                if category == 'stock_indices':
                    ts_code = config['ts_code']
                    start_str = start_date.replace('-', '')
                    end_str = end_date.replace('-', '')
                    return get_tushare_index_data(ts_code, start_str, end_str)
                elif category == 'bonds':
                    if 'ts_code' in config:
                        ts_code = config['ts_code']
                        start_str = start_date.replace('-', '')
                        end_str = end_date.replace('-', '')
                        return get_tushare_stock_data(ts_code, start_str, end_str)
                    else:
                        index_code = config['index_code']
                        return get_efinance_index_data(index_code, start_date, end_date)

            elif source == 'efinance':
                index_code = config['index_code']
                return get_efinance_index_data(index_code, start_date, end_date)

            elif source == 'akshare':
                if category == 'commodities':
                    commodity_code = config['commodity_code']
                    return get_akshare_commodity_data(commodity_code, start_date, end_date)
                elif category == 'forex':
                    symbol = config['symbol']
                    return get_akshare_forex_data(symbol, start_date, end_date)

    return pd.DataFrame()


def get_all_assets_data(start_date: str, end_date: str) -> dict:
    """
    获取所有资产的价格数据

    Parameters:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        dict: {asset_name: DataFrame}
    """
    all_data = {}

    for asset in ALL_ASSETS:
        print(f"正在获取 {asset} 数据...")
        df = get_asset_price_data(asset, start_date, end_date)
        if not df.empty:
            all_data[asset] = df
            print(f"  {asset}: 获取 {len(df)} 条数据")
        else:
            print(f"  {asset}: 数据获取失败")

    return all_data


def resample_to_monthly(data: pd.DataFrame) -> pd.DataFrame:
    """
    将日频数据转换为月频数据

    Parameters:
        data: DataFrame with DateIndex

    Returns:
        DataFrame with monthly frequency
    """
    monthly = data.resample('M').last()
    monthly['return'] = data['close'].resample('M').apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1 if len(x) > 1 else 0)
    return monthly.dropna()


def get_macro_indicators(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取宏观经济指标（用于原始因子构建）

    Parameters:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with macro indicators
    """
    pro = init_tushare()

    indicators = pd.DataFrame()

    try:
        gdp = pro.cn_gdp(quarter_start='201001', quarter_end='202304')
        if gdp is not None and not gdp.empty:
            gdp['quarter'] = pd.to_datetime(gdp['quarter'])
            gdp = gdp[(gdp['quarter'] >= start_date) & (gdp['quarter'] <= end_date)]
            indicators['GDP'] = gdp.set_index('quarter')['gdp']
    except Exception as e:
        print(f"获取GDP数据失败: {e}")

    try:
        cpi = pro.cn_cpi(month_start='201001', month_end='202305')
        if cpi is not None and not cpi.empty:
            cpi['month'] = pd.to_datetime(cpi['month'])
            cpi = cpi[(cpi['month'] >= start_date) & (cpi['month'] <= end_date)]
            indicators['CPI'] = cpi.set_index('month')['cpi']
    except Exception as e:
        print(f"获取CPI数据失败: {e}")

    return indicators


def save_data(data: dict, filename: str):
    """
    保存数据到文件

    Parameters:
        data: 数据字典
        filename: 文件名
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    combined = {}
    for asset, df in data.items():
        if not df.empty:
            combined[asset] = df['return']

    if combined:
        result = pd.DataFrame(combined)
        result.to_pickle(filepath)
        print(f"数据已保存至: {filepath}")


def load_data(filename: str) -> pd.DataFrame:
    """
    从文件加载数据

    Parameters:
        filename: 文件名

    Returns:
        DataFrame
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_pickle(filepath)
    return pd.DataFrame()


if __name__ == "__main__":
    start = BACKTEST_CONFIG['start_date']
    end = BACKTEST_CONFIG['end_date']

    print("开始获取资产数据...")
    data = get_all_assets_data(start, end)

    if data:
        save_data(data, 'asset_returns.pkl')
        print(f"\n成功获取 {len(data)} 个资产的数据")
    else:
        print("警告: 未能获取任何资产数据")