import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import tushare as ts
import efinance as ef
import baostock as bs
import akshare as ak

BASE_DIR = r'd:\Documents\trae_projects\industry_residual_momentum_pricing'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

TUSHARE_TOKEN = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"

class MultiSourceDataFetcher:
    def __init__(self):
        self.tushare_pro = None
        self.baostock_login = False
        self.init_tushare()
        self.init_baostock()

    def init_tushare(self):
        try:
            self.tushare_pro = ts.pro_api(TUSHARE_TOKEN)
            self.tushare_pro._DataApi__token = TUSHARE_TOKEN
            self.tushare_pro._DataApi__http_url = "http://jiaoch.site"
            print("[tushare] 初始化成功")
        except Exception as e:
            print(f"[tushare] 初始化失败: {e}")
            self.tushare_pro = None

    def init_baostock(self):
        try:
            lg = bs.login()
            if lg.error_code == '0':
                self.baostock_login = True
                print("[baostock] 登录成功")
            else:
                print(f"[baostock] 登录失败: {lg.error_msg}")
        except Exception as e:
            print(f"[baostock] 初始化失败: {e}")

    def fetch_china_indices_tushare(self):
        result = {}
        indices = {
            '上证指数': '000001.SH',
            '深证成指': '399001.SZ',
            '沪深300': '000300.SH',
            '中证500': '000905.SH',
            '中证1000': '000852.SH',
            '创业板指': '399006.SZ',
            '上证50': '000016.SH',
            '科创50': '000688.SH',
        }

        if not self.tushare_pro:
            return pd.DataFrame()

        for name, code in indices.items():
            try:
                df = self.tushare_pro.index_monthly(ts_code=code, start_date='20100101', end_date='20240131')
                if df is not None and len(df) > 0:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df = df.set_index('trade_date')
                    df = df.sort_index()
                    result[name] = df['close'].astype(float)
                    print(f"[tushare] 获取 {name}({code}) 成功, 长度: {len(df)}")
            except Exception as e:
                print(f"[tushare] 获取 {name}({code}) 失败: {e}")

        if result:
            return pd.DataFrame(result)
        return pd.DataFrame()

    def fetch_sw_industry_efinance(self):
        result = {}

        sw_codes = [
            '801010', '801020', '801030', '801040', '801050',
            '801080', '801110', '801120', '801130', '801140',
            '801150', '801160', '801170', '801180', '801200',
            '801210', '801230', '801710', '801720', '801730',
            '801740', '801750', '801760', '801770', '801780',
            '801790', '801880', '801890'
        ]

        for code in sw_codes:
            try:
                df = ef.stock.get_quote_history(code, start='2010-01-01', end='2024-01-31')
                if df is not None and len(df) > 0:
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.set_index('日期')
                    df = df.sort_index()
                    result[code] = df['收盘']
            except:
                pass

        if result:
            df = pd.DataFrame(result)
            print(f"[efinance] 获取申万行业成功, 形状: {df.shape}")
            return df
        return pd.DataFrame()

    def fetch_industry_etf_tushare(self):
        result = {}
        etf_list = [
            ('科技龙头', '931087.CSI'),
            ('中证传媒', '399971.SZ'),
            ('动漫游戏', '930901.CSI'),
            ('国证芯片', '980017.CNI'),
            ('消费电子', '980030.CNI'),
            ('CS计算机', '930651.CSI'),
            ('CS人工智能', '930713.CSI'),
            ('国证通信', '399389.SZ'),
            ('中证新能', '399808.SZ'),
            ('光伏产业', '931151.CSI'),
            ('中证医疗', '399989.SZ'),
            ('CS创新药', '931152.CSI'),
            ('中证消费', '000932.SH'),
            ('CS食品饮料', '930653.CSI'),
            ('中证酒', '399987.SZ'),
            ('中证畜牧', '930707.CSI'),
            ('家用电器', '930697.CSI'),
            ('中证旅游', '930633.CSI'),
            ('中证银行', '399986.SZ'),
            ('证券公司', '399975.SZ'),
            ('中证煤炭', '399998.SZ'),
            ('中证钢铁', '930606.CSI'),
            ('中证军工', '399967.SZ'),
            ('红利指数', '000015.SH'),
            ('恒生科技', 'HSTECH.HI'),
            ('有色金属', '000819.SH'),
            ('基建工程', '399995.SZ'),
            ('中证房地产', '931775.CSI'),
            ('中证电力', 'H30199.CSI'),
        ]

        if not self.tushare_pro:
            return pd.DataFrame()

        for name, code in etf_list:
            try:
                if code.endswith('.CSI') or code.endswith('.SZ') or code.endswith('.SH'):
                    ts_code = code.replace('.CSI', '.CSI').replace('.SZ', '.SZ').replace('.SH', '.SH')

                    df = self.tushare_pro.index_monthly(ts_code=code, start_date='20100101', end_date='20240131')
                    if df is not None and len(df) > 0:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df = df.set_index('trade_date')
                        df = df.sort_index()
                        result[name] = df['close'].astype(float)
                        print(f"[tushare] 获取 {name}({code}) 成功")
            except Exception as e:
                print(f"[tushare] 获取 {name}({code}) 失败")

        if result:
            return pd.DataFrame(result)
        return pd.DataFrame()

    def fetch_commodities_efinance(self):
        result = {}

        commodity_codes = [
            ('AU9999', '黄金现货'),
            ('AG9999', '白银现货'),
        ]

        for code, name in commodity_codes:
            try:
                df = ef.stock.get_quote_history(code, start='2010-01-01', end='2024-01-31')
                if df is not None and len(df) > 0:
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.set_index('日期')
                    df = df.sort_index()
                    result[name] = df['收盘']
                    print(f"[efinance] 获取 {name}({code}) 成功")
            except Exception as e:
                print(f"[efinance] 获取 {name}({code}) 失败")

        try:
            df = ef.stock.get_quote_history('NHCI', start='2010-01-01', end='2024-01-31')
            if df is not None and len(df) > 0:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期')
                df = df.sort_index()
                result['南华商品'] = df['收盘']
                print(f"[efinance] 获取 南华商品(NHCI) 成功")
        except Exception as e:
            print(f"[efinance] 获取 南华商品 失败: {e}")

        if result:
            return pd.DataFrame(result)
        return pd.DataFrame()

    def fetch_bond_yields_akshare(self):
        result = {}

        try:
            df = ak.bond_zh_us_yield(start_date="20100101", end_date="20240131")
            if df is not None and len(df) > 0:
                print(f"[akshare] 美债数据获取成功, 形状: {df.shape}")
        except Exception as e:
            print(f"[akshare] 美债数据失败: {e}")

        try:
            df = ak.bond_zh_cn_yield(start_date="20100101", end_date="20240131")
            if df is not None and len(df) > 0:
                print(f"[akshare] 中债数据获取成功, 形状: {df.shape}")
        except Exception as e:
            print(f"[akshare] 中债数据失败: {e}")

        return result

    def save_data(self, df, filename):
        if df is not None and not df.empty:
            filepath = os.path.join(DATA_DIR, filename)
            df.to_pickle(filepath)
            print(f"[保存] {filename}, 形状: {df.shape}")
            return filepath
        return None

    def load_data(self, filename):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            return pd.read_pickle(filepath)
        return None


def main():
    print("=" * 60)
    print("开始获取多源市场数据...")
    print("=" * 60)

    fetcher = MultiSourceDataFetcher()

    print("\n[1/5] 获取宽基指数数据...")
    broad_indices = fetcher.fetch_china_indices_tushare()
    fetcher.save_data(broad_indices, 'broad_indices.pkl')

    print("\n[2/5] 获取申万行业指数...")
    sw_industry = fetcher.fetch_sw_industry_efinance()
    fetcher.save_data(sw_industry, 'sw_industry.pkl')

    print("\n[3/5] 获取行业ETF跟踪指数...")
    industry_etf = fetcher.fetch_industry_etf_tushare()
    fetcher.save_data(industry_etf, 'industry_etf.pkl')

    print("\n[4/5] 获取商品数据...")
    commodities = fetcher.fetch_commodities_efinance()
    fetcher.save_data(commodities, 'commodities.pkl')

    print("\n[5/5] 获取债券数据...")
    bonds = fetcher.fetch_bond_yields_akshare()
    fetcher.save_data(pd.DataFrame(bonds) if bonds else pd.DataFrame(), 'bonds.pkl')

    print("\n" + "=" * 60)
    print("数据获取完成!")
    print("=" * 60)

    if fetcher.baostock_login:
        bs.logout()

    return fetcher


if __name__ == '__main__':
    fetcher = main()