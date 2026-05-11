"""
数据获取模块 - 获取六类底层资产数据
用于桥水全天候策略和风险平价模型

底层资产：
1. 沪深300指数 (CSI300)
2. 标普500指数 (SPX)
3. 恒生指数 (HSI)
4. 中债-企业债总财富(总值)指数 (CBCE)
5. 南华商品指数 (NHCI)
6. COMEX黄金期货 (GC)
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"

try:
    pro = ts.pro_api(TUSHARE_TOKEN)
    pro._DataApi__token = TUSHARE_TOKEN
    pro._DataApi__http_url = "http://jiaoch.site"
except Exception as e:
    print(f"Tushare初始化失败: {e}")
    pro = None

ASSET_CONFIG = {
    'CSI300': {
        'name': '沪深300指数',
        'tushare_code': '000300.SH',
        'data_source': 'tushare',
        'type': 'stock'
    },
    'SPX': {
        'name': '标普500指数',
        'tushare_code': 'SPX',
        'data_source': 'yfinance',
        'type': 'stock'
    },
    'HSI': {
        'name': '恒生指数',
        'tushare_code': 'HSI',
        'data_source': 'yfinance',
        'type': 'stock'
    },
    'CBCE': {
        'name': '中债-企业债总财富指数',
        'tushare_code': 'CBA00203.CS',
        'data_source': 'tushare',
        'type': 'bond'
    },
    'NHCI': {
        'name': '南华商品指数',
        'tushare_code': 'NHCI',
        'data_source': 'tushare',
        'type': 'commodity'
    },
    'GC': {
        'name': 'COMEX黄金',
        'tushare_code': 'GC00Y',
        'data_source': 'yfinance',
        'type': 'commodity'
    }
}


def get_tushare_data(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从Tushare获取数据

    Parameters:
    -----------
    ts_code : str
        Tushare代码
    start_date : str
        开始日期，格式YYYYMMDD
    end_date : str
        结束日期，格式YYYYMMDD

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    if pro is None:
        print("Tushare未初始化")
        return pd.DataFrame()

    try:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        return df[['close']]
    except Exception as e:
        print(f"Tushare获取{ts_code}数据失败: {e}")
        return pd.DataFrame()


def get_yfinance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从yfinance获取数据

    Parameters:
    -----------
    ticker : str
        Yahoo Finance代码
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            return pd.DataFrame()
        data = data.sort_index()
        return data[['Close']]
    except Exception as e:
        print(f"yfinance获取{ticker}数据失败: {e}")
        return pd.DataFrame()


def get_cbce_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取中债-企业债总财富指数数据
    优先使用本地补充数据，其次使用AKShare或Tushare

    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    # 优先使用本地补充数据
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'CBCE.csv')
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path, index_col=0, parse_dates=True)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            df.columns = ['close']
            print(f"使用本地中债指数数据: {len(df)} 条")
            return df
        except Exception as e:
            print(f"读取本地中债指数数据失败: {e}")

    try:
        import akshare as ak
        df = ak.bond_zhihui_quotes()
        if df is not None and not df.empty:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            return df[['收盘']]
    except Exception as e:
        print(f"AKShare获取企业债数据失败: {e}")

    try:
        df = get_tushare_data('CBA00203.CS', start_date.replace('-', ''), end_date.replace('-', ''))
        if not df.empty:
            return df
    except Exception:
        pass

    print("注意：企业债指数使用模拟数据，请使用真实数据源替换")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(42)
    prices = 100 * (1 + np.cumsum(np.random.normal(0.0002, 0.001, len(dates))))
    df = pd.DataFrame({'close': prices}, index=dates)
    return df


def get_nhci_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取南华商品指数数据
    优先使用本地补充数据，其次使用yfinance或Tushare

    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    # 优先使用本地补充数据
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'NHCI.csv')
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path, index_col=0, parse_dates=True)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            df.columns = ['close']
            print(f"使用本地南华商品指数数据: {len(df)} 条")
            return df
        except Exception as e:
            print(f"读取本地南华商品指数数据失败: {e}")

    try:
        import yfinance as yf
        ticker = yf.Ticker("CNYA=MYB")
        data = ticker.history(start=start_date, end=end_date)
        if data is not None and not data.empty:
            data = data.sort_index()
            return data[['Close']]
    except Exception:
        pass

    try:
        df = get_tushare_data('NHCI', start_date.replace('-', ''), end_date.replace('-', ''))
        if not df.empty:
            return df
    except Exception:
        pass

    print("注意：南华商品指数使用模拟数据，请使用真实数据源替换")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(123)
    prices = 1000 * (1 + np.cumsum(np.random.normal(0.0002, 0.015, len(dates))))
    df = pd.DataFrame({'close': prices}, index=dates)
    return df


def get_spx_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取标普500指数数据

    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    df = get_yfinance_data('^GSPC', start_date, end_date)
    if df.empty:
        try:
            df = get_tushare_data('SPX', start_date.replace('-', ''), end_date.replace('-', ''))
        except Exception:
            pass
    return df


def get_hsi_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取恒生指数数据

    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    df = get_yfinance_data('^HSI', start_date, end_date)
    if df.empty:
        try:
            df = get_tushare_data('HSI', start_date.replace('-', ''), end_date.replace('-', ''))
        except Exception:
            pass
    return df


def get_gc_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取COMEX黄金期货数据

    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    df = get_yfinance_data('GC=F', start_date, end_date)
    if df.empty:
        try:
            df = get_tushare_data('GC00Y', start_date.replace('-', ''), end_date.replace('-', ''))
        except Exception:
            pass
    return df


def get_csi300_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取沪深300指数数据

    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期

    Returns:
    --------
    pd.DataFrame
        包含日期和收盘价的数据框
    """
    return get_tushare_data('000300.SH', start_date.replace('-', ''), end_date.replace('-', ''))


def fetch_all_assets(start_date: str = '20080101', end_date: str = '20230430') -> pd.DataFrame:
    """
    获取所有六类底层资产的数据

    Parameters:
    -----------
    start_date : str
        开始日期，格式YYYYMMDD或YYYY-MM-DD
    end_date : str
        结束日期，格式YYYYMMDD或YYYY-MM-DD

    Returns:
    --------
    pd.DataFrame
        多列收盘价数据框，列为各资产代码，索引为日期
    """
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')

    start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    result_dict = {}

    print("开始获取数据...")
    print(f"时间范围: {start_str} 至 {end_str}")

    fetch_funcs = {
        'CSI300': get_csi300_data,
        'SPX': get_spx_data,
        'HSI': get_hsi_data,
        'CBCE': get_cbce_data,
        'NHCI': get_nhci_data,
        'GC': get_gc_data
    }

    for asset_code, fetch_func in fetch_funcs.items():
        print(f"正在获取 {ASSET_CONFIG[asset_code]['name']} ({asset_code})...")
        df = fetch_func(start_str, end_str)
        if not df.empty:
            result_dict[asset_code] = df['close']
            print(f"  成功获取 {len(df)} 条数据")
        else:
            print(f"  获取失败，使用模拟数据")
            if asset_code == 'CBCE':
                dates = pd.date_range(start=start_str, end=end_str, freq='B')
                np.random.seed(42)
                prices = 100 * (1 + np.cumsum(np.random.normal(0.0002, 0.001, len(dates))))
                result_dict[asset_code] = pd.Series(prices, index=dates)
            elif asset_code == 'NHCI':
                dates = pd.date_range(start=start_str, end=end_str, freq='B')
                np.random.seed(123)
                prices = 1000 * (1 + np.cumsum(np.random.normal(0.0002, 0.015, len(dates))))
                result_dict[asset_code] = pd.Series(prices, index=dates)
            else:
                dates = pd.date_range(start=start_str, end=end_str, freq='B')
                np.random.seed(asset_code.__hash__() % 1000)
                prices = 100 * (1 + np.cumsum(np.random.normal(0.0003, 0.015, len(dates))))
                result_dict[asset_code] = pd.Series(prices, index=dates)

    if not result_dict:
        raise ValueError("所有资产数据获取失败")

    prices_df = pd.DataFrame(result_dict)
    prices_df = prices_df.sort_index()
    prices_df = prices_df.dropna()

    print(f"\n数据获取完成!")
    print(f"最终数据范围: {prices_df.index[0].strftime('%Y-%m-%d')} 至 {prices_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"数据条数: {len(prices_df)}")

    return prices_df


def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算收益率数据

    Parameters:
    -----------
    prices_df : pd.DataFrame
        价格数据框

    Returns:
    --------
    pd.DataFrame
        日收益率数据框
    """
    returns_df = prices_df.pct_change().dropna()
    return returns_df


def calculate_monthly_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算月度收益率数据

    Parameters:
    -----------
    prices_df : pd.DataFrame
        价格数据框

    Returns:
    --------
    pd.DataFrame
        月度收益率数据框
    """
    monthly_prices = prices_df.resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def save_data(prices_df: pd.DataFrame, filepath: str = None):
    """
    保存价格数据到CSV文件

    Parameters:
    -----------
    prices_df : pd.DataFrame
        价格数据框
    filepath : str, optional
        保存路径，默认为output目录下
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'asset_prices.csv')

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    prices_df.to_csv(filepath)
    print(f"数据已保存至: {filepath}")


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    从CSV文件加载价格数据

    Parameters:
    -----------
    filepath : str, optional
        文件路径

    Returns:
    --------
    pd.DataFrame
        价格数据框
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'asset_prices.csv')

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"已从 {filepath} 加载数据，共 {len(df)} 条")
        return df
    else:
        print(f"文件 {filepath} 不存在")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("桥水全天候策略 - 数据获取模块测试")
    print("=" * 60)

    prices_df = fetch_all_assets('2008-01-01', '2023-04-30')
    print("\n数据预览:")
    print(prices_df.head())
    print(f"\n数据形状: {prices_df.shape}")

    returns_df = calculate_returns(prices_df)
    print("\n收益率预览:")
    print(returns_df.head())

    save_data(prices_df)