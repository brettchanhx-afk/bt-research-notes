import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
pro = ts.pro_api(TOKEN)
pro._DataApi__token = TOKEN
pro._DataApi__http_url = "http://jiaoch.site"

class DataFetcher:
    def __init__(self):
        self.pro = pro

    def get_index_daily(self, ts_code, start_date=None, end_date=None):
        df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df.set_index('trade_date', inplace=True)
            df['close'] = df['close'].astype(float)
        return df

    def get_stock_daily(self, ts_code, start_date=None, end_date=None):
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df.set_index('trade_date', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'vol']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
        return df

    def get_etf_daily(self, ts_code, start_date=None, end_date=None):
        try:
            df = self.pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)
                df['close'] = df['close'].astype(float)
                return df
        except Exception as e:
            pass
        return None

    def get_bond_yield(self, start_date=None, end_date=None):
        try:
            df = self.pro.bond_zh(start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            pass
        return None

    def get_interest_rate(self, start_date=None, end_date=None):
        try:
            df = self.pro.cn_gdp()
            return df
        except Exception as e:
            pass
        return None

    def get_monthly_data(self, price_dict, start_date=None, end_date=None):
        result = {}
        for name, ts_code in price_dict.items():
            if 'SH' in ts_code or 'SZ' in ts_code:
                if '399' in ts_code or '000' in ts_code or '399' in ts_code:
                    df = self.get_index_daily(ts_code, start_date, end_date)
                elif ts_code.startswith('93') or ts_code.startswith('98'):
                    df = self.get_index_daily(ts_code, start_date, end_date)
                else:
                    df = self.get_stock_daily(ts_code, start_date, end_date)
            elif '.SH' in ts_code or '.SZ' in ts_code:
                df = self.get_etf_daily(ts_code.replace('.SH', '').replace('.SZ', ''), start_date, end_date)
            else:
                df = self.get_index_daily(ts_code, start_date, end_date)

            if df is not None and len(df) > 0:
                monthly = df['close'].resample('M').last()
                result[name] = monthly
        return pd.DataFrame(result)

    def get_china_stock_indices(self, start_date=None, end_date=None):
        indices = {
            '上证指数': '000001.SH',
            '深证成指': '399001.SZ',
            '沪深300成长': '000918.SH',
            '沪深300价值': '000919.SH',
            '中证500成长': '399620.SZ',
            '中证500价值': '399621.SZ',
            '中证800汽车': 'H30015.CSI',
            '中证全指电力': 'H30199.CSI',
            '中证全指运输': 'H30171.CSI',
            '建筑材料': '931009.CSI',
            '中证军工': '399967.SZ',
            '中证银行': '399986.SZ',
            '中证煤炭': '399998.SZ',
            '中证钢铁': '930606.CSI',
            '中证医疗': '399989.SZ',
            '中证消费': '000932.SH',
            '中证传媒': '399971.SZ',
            '中证新能': '399808.SZ',
            '动漫游戏': '930901.CSI',
            'CS计算机': '930651.CSI',
            'CS人工智能': '930713.CSI',
            '国证芯片': '980017.CNI',
            '消费电子': '980030.CNI',
            '国证通信': '399389.SZ',
            'CS食品饮料': '930653.CSI',
            '中证酒': '399987.SZ',
            '中证畜牧': '930707.CSI',
            '家用电器': '930697.CSI',
            '中证旅游': '930633.CSI',
            '中证全指房地产': '931775.CSI',
            '光伏产业': '931151.CSI',
            'CS创新药': '931152.CSI',
            'CS稀有金属': '930632.CSI',
            '中证化工': '000813.CSI',
        }
        return self.get_monthly_data(indices, start_date, end_date)

    def get_global_stock_indices(self, start_date=None, end_date=None):
        indices = {
            '上证指数': '000001.SH',
            '深证成指': '399001.SZ',
            '恒生指数': 'HSI.HK',
            '道琼斯工业': 'DJI.GI',
            '标普500': 'SPX.GI',
            '纳斯达克': 'NDX.GI',
            '日经225': 'N225.GI',
            '韩国综合': 'KS11.GI',
            '澳洲标普': 'AS51.GI',
            '印度SENSEX': 'SENSEX.GI',
            'MSCI发达': '990100.GI',
            'MSCI新兴': '891600.GI',
            'MSCI欧洲': 'SXXP.GI',
            'MSCI亚太': 'MSEAFE.GI',
        }
        return self.get_monthly_data(indices, start_date, end_date)

    def get_china_bond_yields(self, start_date=None, end_date=None):
        yields = {}
        bond_codes = {
            '中债国债1Y': 'NDBI1Y',
            '中债国债10Y': 'NDBI10Y',
            '中债国开债1Y': 'EDBI1Y',
            '中债国开债10Y': 'EDBI10Y',
            '中债企业债AAA1Y': 'CEDBIAAA1Y',
            '中债企业债AAA10Y': 'CEDBIAAA10Y',
        }
        try:
            for name in bond_codes:
                try:
                    df = self.pro.bond_zh(name, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
                    if df is not None:
                        yields[name] = df['close'].resample('M').last()
                except:
                    pass
        except Exception as e:
            pass
        return pd.DataFrame(yields) if yields else None

    def get_commodity_indices(self, start_date=None, end_date=None):
        commodities = {
            '黄金': 'AU9999.SGE',
            '南华商品': 'NHCI.NHF',
            '原油': 'CL.F',
        }
        return self.get_monthly_data(commodities, start_date, end_date)

    def get_industry_indices(self, index_type='sw', start_date=None, end_date=None):
        if index_type == 'sw':
            try:
                df = self.pro.index_classify(level='L1', src='SW')
                result = {}
                for _, row in df.iterrows():
                    code = row['index_code']
                    name = row['industry_name']
                    try:
                        daily = self.get_index_daily(code, start_date, end_date)
                        if daily is not None:
                            result[name] = daily['close'].resample('M').last()
                    except:
                        pass
                return pd.DataFrame(result)
            except Exception as e:
                pass
        return None

    def get_etf_tracking_indices(self, include_dev=False):
        etf_tracking = {
            '科技龙头': '931087.CSI',
            '中证传媒': '399971.SZ',
            '动漫游戏': '930901.CSI',
            '国证芯片': '980017.CNI',
            '消费电子': '980030.CNI',
            'CS计算机': '930651.CSI',
            'CS人工智能': '930713.CSI',
            '国证通信': '399389.SZ',
            '中证新能': '399808.SZ',
            '光伏产业': '931151.CSI',
            '中证医疗': '399989.SZ',
            'CS创新药': '931152.CSI',
            '中证消费': '000932.SH',
            'CS食品饮料': '930653.CSI',
            '中证酒': '399987.SZ',
            '中证畜牧': '930707.CSI',
            '家用电器': '930697.CSI',
            '中证800汽车': 'H30015.CSI',
            '中证旅游': '930633.CSI',
            '中证全指房地产': '931775.CSI',
            '基建工程': '399995.SZ',
            '中证银行': '399986.SZ',
            '证券公司': '399975.SZ',
            '中证煤炭': '399998.SZ',
            '中证钢铁': '930606.CSI',
            '中证全指电力': 'H30199.CSI',
            '中证全指运输': 'H30171.CSI',
            '建筑材料': '931009.CSI',
            '细分化工': '000813.CSI',
            '装备产业': 'H11054.CSI',
            '中证军工': '399967.SZ',
            '红利指数': '000015.SH',
            '红利低波': 'H30269.CSI',
            '结构调整': '000860.CSI',
            '央企创新': '000861.CSI',
            '恒生科技': 'HSTECH.HI',
            '有色金属': '000819.SH',
            '上证50': '000016.SH',
            '沪深300': '000300.SH',
            '中证500': '000905.SH',
            '中证1000': '000852.SH',
            '中证2000': '932000.CSI',
            '创业板50': '399673.SZ',
            '黄金': 'AU9999.SGE',
            '豆粕': 'DCE.M2209',
            '有色金属指数': 'IMCI.SHF',
        }
        if include_dev:
            dev_indices = {
                '日经225': 'N225.GI',
                '标普500': 'SPX.GI',
                '纳斯达克100': 'NDX.GI',
                '德国DAX': 'GDAXI.GI',
                '法国CAC40': 'FCHI.GI',
            }
            etf_tracking.update(dev_indices)
        return etf_tracking

    def convert_to_yoy(self, df):
        return df.pct_change(periods=12)

    def convert_to_mom(self, df):
        return df.pct_change(periods=1)

    def calculate_volatility(self, returns, window=12):
        return returns.rolling(window=window).std()

    def download_all_data(self, start_date='20060101', end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        data = {}
        data['china_stocks'] = self.get_china_stock_indices(start_date, end_date)
        data['china_bonds'] = self.get_china_bond_yields(start_date, end_date)
        data['commodities'] = self.get_commodity_indices(start_date, end_date)

        return data