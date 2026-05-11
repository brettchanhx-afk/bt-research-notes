"""
数据获取模块 - 支持多种数据源
优先使用: tushare > akshare > baostock > efinance > yfinance
本地数据文件 > 模拟数据
"""
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

import tushare as ts
import akshare as ak
import baostock as bs
import efinance as ef
import yfinance as yf

import config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class DataFetcher:
    def __init__(self, token=None, api_url=None):
        self.tushare_available = True
        self.akshare_available = True
        self.baostock_available = True
        self.efinance_available = True
        self.yfinance_available = True

        if token is None:
            token = config.TUSHARE_TOKEN
        if api_url is None:
            api_url = config.TUSHARE_API_URL

        self.token = token
        try:
            self.pro = ts.pro_api(token)
            self.pro._DataApi__token = token
            self.pro._DataApi__http_url = api_url
            self.tushare_available = True
        except Exception as e:
            print(f"tushare初始化失败: {e}")
            self.tushare_available = False

        try:
            bs.login()
            self.baostock_available = True
        except Exception as e:
            print(f"baostock初始化失败: {e}")
            self.baostock_available = False

    def __del__(self):
        try:
            bs.logout()
        except:
            pass

    def get_stock_daily(self, ts_code, start_date=None, end_date=None):
        """获取股票日线数据"""
        df = self._get_from_tushare(ts_code, start_date, end_date)
        if not df.empty:
            return df

        df = self._get_from_akshare(ts_code, start_date, end_date)
        if not df.empty:
            return df

        df = self._get_from_baostock(ts_code, start_date, end_date)
        return df

    def _get_from_tushare(self, ts_code, start_date, end_date):
        """从tushare获取数据"""
        if not self.tushare_available:
            return pd.DataFrame()

        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def _get_from_akshare(self, symbol, start_date, end_date):
        """从akshare获取数据"""
        if not self.akshare_available:
            return pd.DataFrame()

        try:
            if start_date:
                start_date = start_date.replace('-', '')
            if end_date:
                end_date = end_date.replace('-', '')

            symbol_clean = symbol.split('.')[0]

            df = ak.stock_zh_a_hist(
                symbol=symbol_clean,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'
            )

            if not df.empty:
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'vol',
                    '成交额': 'amount'
                })
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df['ts_code'] = symbol
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def _get_from_baostock(self, ts_code, start_date, end_date):
        """从baostock获取数据"""
        if not self.baostock_available:
            return pd.DataFrame()

        try:
            if start_date:
                start_date = start_date.replace('-', '')
            if end_date:
                end_date = end_date.replace('-', '')

            bs_code = ts_code.replace('.SH', '.sh').replace('.SZ', '.sz')

            rs = bs.query_history_k_data_plus(
                bs_code,
                'date,open,high,low,close,volume,amount',
                start_date=start_date,
                end_date=end_date,
                frequency='d'
            )

            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df = df.rename(columns={
                    'date': 'trade_date',
                    'volume': 'vol'
                })
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df['ts_code'] = ts_code
                df = df.sort_values('trade_date')
                return df
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()

    def get_index_daily(self, ts_code, start_date=None, end_date=None):
        """获取指数日线数据"""
        df = self._get_index_from_efinance(ts_code, start_date, end_date)
        if not df.empty:
            return df

        df = self._get_index_from_tushare(ts_code, start_date, end_date)
        if not df.empty:
            return df

        df = self._get_index_from_akshare(ts_code, start_date, end_date)
        return df

    def _get_index_from_efinance(self, ts_code, start_date, end_date):
        """从efinance获取指数数据"""
        if not self.efinance_available:
            return pd.DataFrame()

        try:
            symbol = ts_code.replace('.SH', '').replace('.SZ', '')

            if ts_code.startswith('000') or ts_code.startswith('399'):
                df = ef.stock.get_quote_history(symbol)
            else:
                return pd.DataFrame()

            if not df.empty and '日期' in df.columns:
                df = df.rename(columns={'日期': 'trade_date'})
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                if start_date:
                    df = df[df['trade_date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['trade_date'] <= pd.to_datetime(end_date)]
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def _get_index_from_tushare(self, ts_code, start_date, end_date):
        """从tushare获取指数数据"""
        if not self.tushare_available:
            return pd.DataFrame()

        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                api=self.pro
            )
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def _get_index_from_akshare(self, ts_code, start_date, end_date):
        """从akshare获取指数数据"""
        if not self.akshare_available:
            return pd.DataFrame()

        try:
            if start_date:
                start_date = start_date.replace('-', '')
            if end_date:
                end_date = end_date.replace('-', '')

            symbol = ts_code.replace('.SH', '').replace('.SZ', '')

            df = ak.stock_zh_index_daily(symbol=symbol)

            if not df.empty:
                if '日期' in df.columns:
                    df = df.rename(columns={'日期': 'trade_date'})
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    if start_date:
                        df = df[df['trade_date'] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df['trade_date'] <= pd.to_datetime(end_date)]
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_index_monthly(self, ts_code, start_date=None, end_date=None):
        """获取指数月线数据"""
        daily_df = self.get_index_daily(ts_code, start_date, end_date)

        if daily_df.empty:
            return pd.DataFrame()

        if 'close' not in daily_df.columns and '收盘' not in daily_df.columns:
            return pd.DataFrame()

        close_col = 'close' if 'close' in daily_df.columns else '收盘'

        daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'])
        daily_df = daily_df.set_index('trade_date')

        monthly = daily_df.resample('M').last().reset_index()
        monthly['ts_code'] = ts_code

        return monthly

    def get_macro_data(self, ts_code, start_date=None, end_date=None):
        """获取宏观数据（使用tushare）"""
        if not self.tushare_available:
            return self._get_macro_simulated(ts_code, start_date, end_date)

        try:
            df = self.pro.cn_macro(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return self._get_macro_simulated(ts_code, start_date, end_date)

    def _get_macro_simulated(self, ts_code, start_date, end_date):
        """生成模拟宏观数据"""
        print(f"警告: 无法获取 {ts_code} 真实数据，使用模拟数据")

        dates = pd.date_range(start=start_date or '20100101',
                             end=end_date or '20210630',
                             freq='M')

        np.random.seed(hash(ts_code) % (2**31))

        base_value = np.random.uniform(50, 100)
        trend = np.linspace(0, 10, len(dates))
        seasonal = 5 * np.sin(np.arange(len(dates)) * np.pi / 6)
        cycle = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 42)
        noise = np.random.randn(len(dates)) * 3

        values = base_value + trend + seasonal + cycle + noise

        df = pd.DataFrame({
            'trade_date': dates,
            'ts_code': ts_code,
            'value': values
        })

        return df

    def get_PMI(self, start_date=None, end_date=None):
        """获取PMI数据（优先使用本地真实数据）"""
        df = self._get_macro_from_local(start_date, end_date)
        if not df.empty and 'PMI' in df.columns:
            result = pd.DataFrame({
                'trade_date': df.index,
                'pmi': df['PMI'].values
            })
            return result
        df = self.get_macro_data('M0017126', start_date, end_date)
        if not df.empty:
            df = df.rename(columns={'value': 'pmi'})
        return df

    def get_cpi(self, start_date=None, end_date=None):
        """获取CPI数据（优先使用本地真实数据）"""
        df = self._get_macro_from_local(start_date, end_date)
        if not df.empty and 'CPI' in df.columns:
            result = pd.DataFrame({
                'trade_date': df.index,
                'cpi': df['CPI'].values
            })
            return result
        df = self.get_macro_data('M0000705', start_date, end_date)
        if not df.empty:
            df = df.rename(columns={'value': 'cpi'})
        return df

    def get_ppi(self, start_date=None, end_date=None):
        """获取PPI数据（优先使用本地真实数据）"""
        df = self._get_macro_from_local(start_date, end_date)
        if not df.empty and 'PPI' in df.columns:
            result = pd.DataFrame({
                'trade_date': df.index,
                'ppi': df['PPI'].values
            })
            return result
        df = self.get_macro_data('M0049160', start_date, end_date)
        if not df.empty:
            df = df.rename(columns={'value': 'ppi'})
        return df

    def get_m1(self, start_date=None, end_date=None):
        """获取M1数据（优先使用本地真实数据）"""
        df = self._get_macro_from_local(start_date, end_date)
        if not df.empty and 'M1' in df.columns:
            result = pd.DataFrame({
                'trade_date': df.index,
                'm1': df['M1'].values
            })
            return result
        return self.get_macro_data('M0001382', start_date, end_date)

    def get_m2(self, start_date=None, end_date=None):
        """获取M2数据（优先使用本地真实数据）"""
        df = self._get_macro_from_local(start_date, end_date)
        if not df.empty and 'M2' in df.columns:
            result = pd.DataFrame({
                'trade_date': df.index,
                'm2': df['M2'].values
            })
            return result
        return self.get_macro_data('M0001384', start_date, end_date)

    def get_社融(self, start_date=None, end_date=None):
        """获取社会融资规模数据（优先使用本地真实数据）"""
        df = self._get_macro_from_local(start_date, end_date)
        if not df.empty and '社融' in df.columns:
            result = pd.DataFrame({
                'trade_date': df.index,
                'sr': df['社融'].values
            })
            return result
        return self.get_macro_data('M5525755', start_date, end_date)

    def _get_macro_from_local(self, start_date, end_date):
        """从本地文件获取宏观经济数据"""
        try:
            macro_file = os.path.join(DATA_DIR, '宏观经济数据.csv')
            if os.path.exists(macro_file):
                df = pd.read_excel(macro_file)
                df = df.rename(columns={'Unnamed: 0': 'date', '制造业PMI': 'PMI', 'CPI同比': 'CPI', 'PPI同比': 'PPI', '社融存量': '社融'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]

                return df
        except Exception as e:
            print(f"加载本地宏观经济数据失败: {e}")
        return pd.DataFrame()

    def get_bond_yield(self, start_date=None, end_date=None):
        """获取国债收益率数据"""
        df = self._get_bond_from_akshare(start_date, end_date)
        if not df.empty:
            return df

        return self._get_bond_simulated(start_date, end_date)

    def _get_bond_from_akshare(self, start_date, end_date):
        """从akshare获取国债收益率"""
        if not self.akshare_available:
            return pd.DataFrame()

        try:
            if start_date:
                start_date = start_date.replace('-', '')
            if end_date:
                end_date = end_date.replace('-', '')

            df = ak.bond_china_yield(start_date=start_date, end_date=end_date)

            if not df.empty and '日期' in df.columns:
                df = df.rename(columns={'日期': 'trade_date'})
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def _get_bond_simulated(self, start_date, end_date):
        """生成模拟国债收益率数据"""
        dates = pd.date_range(start=start_date or '20100101',
                             end=end_date or '20210630',
                             freq='M')

        np.random.seed(42)

        base_rate = 3.0
        trend = np.linspace(0, 0.5, len(dates))
        cycle = 1.5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 42)
        noise = np.random.randn(len(dates)) * 0.3

        rates = base_rate + trend + cycle + noise
        rates = np.clip(rates, 1.0, 5.0)

        df = pd.DataFrame({
            'trade_date': dates,
            '1年': rates,
            '10年': rates + 1.5
        })

        return df

    def get_industry_index(self, industry_code, start_date=None, end_date=None):
        """获取行业指数数据"""
        df = self._get_industry_from_akshare(industry_code, start_date, end_date)
        if not df.empty:
            return df

        return self._get_industry_simulated(industry_code, start_date, end_date)

    def _get_industry_from_akshare(self, industry_code, start_date, end_date):
        """从akshare获取行业数据"""
        if not self.akshare_available:
            return pd.DataFrame()

        try:
            if start_date:
                start_date = start_date.replace('-', '')
            if end_date:
                end_date = end_date.replace('-', '')

            df = ak.stock_zh_index_daily(symbol=industry_code.replace('.SI', ''))

            if not df.empty:
                if '日期' in df.columns:
                    df = df.rename(columns={'日期': 'trade_date'})
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    if start_date:
                        df = df[df['trade_date'] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df['trade_date'] <= pd.to_datetime(end_date)]
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def _get_industry_simulated(self, industry_code, start_date, end_date):
        """生成模拟行业数据"""
        dates = pd.date_range(start=start_date or '20110101',
                             end=end_date or '20210630',
                             freq='M')

        np.random.seed(hash(industry_code) % (2**31))

        base_value = 1000
        trend = np.linspace(0, 500, len(dates))
        cycle = 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 42)
        noise = np.random.randn(len(dates)) * 30

        values = base_value + trend + cycle + noise

        df = pd.DataFrame({
            'trade_date': dates,
            'close': values,
            'open': values * 0.98,
            'high': values * 1.02,
            'low': values * 0.96,
            'vol': np.random.randint(1000000, 10000000, len(dates))
        })

        return df

    def get_commodity_data(self, commodity_code, start_date=None, end_date=None):
        """获取大宗商品数据"""
        df = self._get_commodity_from_yfinance(commodity_code, start_date, end_date)
        if not df.empty:
            return df

        return self._get_commodity_simulated(commodity_code, start_date, end_date)

    def _get_commodity_from_yfinance(self, commodity_code, start_date, end_date):
        """从yfinance获取大宗商品数据"""
        if not self.yfinance_available:
            return pd.DataFrame()

        try:
            ticker = yf.Ticker(commodity_code)
            df = ticker.history(start=start_date, end=end_date, interval='1mo')

            if not df.empty:
                df = df.reset_index()
                df['trade_date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('trade_date')
            return df
        except Exception as e:
            return pd.DataFrame()

    def _get_commodity_simulated(self, commodity_code, start_date, end_date):
        """生成模拟大宗商品数据"""
        dates = pd.date_range(start=start_date or '20110101',
                             end=end_date or '20210630',
                             freq='M')

        np.random.seed(hash(commodity_code) % (2**31))

        base_value = 100
        trend = np.linspace(0, 50, len(dates))
        cycle = 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 42)
        noise = np.random.randn(len(dates)) * 5

        values = base_value + trend + cycle + noise

        df = pd.DataFrame({
            'trade_date': dates,
            'close': values,
            'open': values * 0.98,
            'high': values * 1.03,
            'low': values * 0.97
        })

        return df

    def resample_to_monthly(self, df, date_col='trade_date', value_col='close'):
        """将日线数据转换为月频数据"""
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        if value_col in df.columns:
            monthly = df[value_col].resample('M').last()
        else:
            return pd.DataFrame()

        result = pd.DataFrame({
            'trade_date': monthly.index,
            value_col: monthly.values
        })
        result['trade_date'] = pd.to_datetime(result['trade_date'])
        return result

    def get_trade_dates(self, start_date=None, end_date=None):
        """获取交易日历"""
        try:
            df = self.pro.trade_cal(
                start_date=start_date,
                end_date=end_date,
                is_open='1'
            )
            return df
        except Exception as e:
            dates = pd.date_range(start=start_date or '20110101',
                                 end=end_date or '20210630',
                                 freq='B')
            return pd.DataFrame({'trade_date': dates})

    def generate_simulation_data(self, start_date, end_date, factor_type='growth'):
        """生成模拟宏观因子数据"""
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        n = len(dates)
        t = np.arange(n)

        np.random.seed(42 if factor_type == 'growth' else 123)

        if factor_type == 'growth':
            base = 0
            amplitude = 5
            phase = 0
            trend = 0.02 * t
        elif factor_type == 'inflation':
            base = 0
            amplitude = 3
            phase = 1
            trend = 0.01 * t
        elif factor_type == 'credit':
            base = 0
            amplitude = 3
            phase = 2
            trend = 0.015 * t
        else:
            base = 0
            amplitude = 1.5
            phase = 3
            trend = -0.01 * t

        values = base + amplitude * np.sin(2 * np.pi * t / 42 + phase) + \
                 amplitude * 0.5 * np.sin(4 * np.pi * t / 42 + phase) + \
                 trend + np.random.randn(n) * (amplitude * 0.2)

        series = pd.Series(values, index=dates)
        return series

    def get_all_macro_factors(self, start_date='20100101', end_date='20210630'):
        """获取所有宏观因子（优先使用本地真实数据）"""
        factors = {}
        real_data_loaded = False

        try:
            factor_file = os.path.join(DATA_DIR, '宏观因子指数.csv')
            if os.path.exists(factor_file):
                df = pd.read_excel(factor_file)
                df = df.rename(columns={'Unnamed: 0': 'date'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                mask = (df.index >= start_dt) & (df.index <= end_dt)
                df_filtered = df.loc[mask]

                if not df_filtered.empty:
                    factors['growth'] = df_filtered['增长'].dropna()
                    factors['inflation'] = df_filtered['通胀'].dropna()
                    factors['credit'] = df_filtered['信用'].dropna()
                    factors['monetary'] = df_filtered['货币'].dropna()
                    real_data_loaded = True
                    print(f"宏观因子数据已从本地文件加载: {len(df_filtered)} 条记录")
        except Exception as e:
            print(f"加载本地宏观因子数据失败: {e}")

        if not real_data_loaded:
            print("使用模拟宏观因子数据")
            factors['growth'] = self.generate_simulation_data(start_date, end_date, 'growth')
            factors['inflation'] = self.generate_simulation_data(start_date, end_date, 'inflation')
            factors['credit'] = self.generate_simulation_data(start_date, end_date, 'credit')
            factors['monetary'] = self.generate_simulation_data(start_date, end_date, 'monetary')

        return factors

    def get_asset_returns(self, start_date='20110101', end_date='20210630'):
        """获取大类资产收益率"""
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        n = len(dates)

        np.random.seed(42)

        assets = {
            '沪深300': 0.008 + 0.05 * np.random.randn(n),
            '中证500': 0.010 + 0.06 * np.random.randn(n),
            '创业板指': 0.012 + 0.07 * np.random.randn(n),
            '国债指数': 0.003 + 0.01 * np.random.randn(n),
            '工业品指数': 0.005 + 0.04 * np.random.randn(n),
            '黄金指数': 0.004 + 0.03 * np.random.randn(n)
        }

        returns_df = pd.DataFrame(assets, index=dates)
        return returns_df

    def get_industry_returns(self, start_date='20110101', end_date='20210630'):
        """获取行业收益率"""
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        n = len(dates)

        np.random.seed(123)

        industries = [
            '农林牧渔', '采掘', '化工', '钢铁', '有色金属',
            '电子', '汽车', '家用电器', '食品饮料', '纺织服装',
            '轻工制造', '医药生物', '公用事业', '交通运输', '房地产',
            '商业贸易', '休闲服务', '银行', '非银金融', '建筑材料',
            '建筑装饰', '电气设备', '国防军工', '计算机', '传媒',
            '通信', '机械设备'
        ]

        industry_returns = {}
        for industry in industries:
            seed = hash(industry) % (2**31)
            np.random.seed(seed)
            mean_return = np.random.uniform(-0.005, 0.015)
            volatility = np.random.uniform(0.03, 0.08)
            industry_returns[industry] = mean_return + volatility * np.random.randn(n)

        returns_df = pd.DataFrame(industry_returns, index=dates)
        return returns_df
