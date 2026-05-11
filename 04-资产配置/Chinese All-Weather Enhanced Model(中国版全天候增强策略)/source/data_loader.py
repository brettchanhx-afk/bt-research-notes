
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    数据加载器，用于获取策略所需的各类资产数据
    """
    
    ASSET_POOL = {
        'stocks': {
            '510300.SH': '沪深300ETF',
            '512100.SH': '中证1000ETF'
        },
        'high_dividend': {
            '512890.SH': '红利低波ETF'
        },
        'bonds': {
            '511260.SH': '十年国债ETF',
            '511090.SH': '三十年国债ETF'
        },
        'commodities': {
            '159980.SZ': '有色ETF',
            '159981.SZ': '能化ETF',
            '159985.SZ': '豆粕ETF'
        },
        'gold': {
            '518880.SH': '黄金ETF'
        }
    }
    
    QUADRANT_ASSETS = {
        'growth_above': ['510300.SH', '512100.SH', '159980.SZ', '159981.SZ', '159985.SZ'],
        'growth_below': ['511260.SH', '511090.SH', '518880.SH'],
        'inflation_above': ['159980.SZ', '159981.SZ', '159985.SZ', '518880.SH'],
        'inflation_below': ['511260.SH', '511090.SH', '518880.SH', '512890.SH']
    }
    
    def __init__(self, start_date='20131231', end_date='20250430'):
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = None
        self.return_data = None
        
    def get_etf_data(self, symbol, period='daily'):
        """
        使用akshare获取ETF数据
        """
        try:
            code = symbol.split('.')[0]
            df = ak.fund_etf_hist_em(symbol=code, period=period, 
                                    start_date=self.start_date, 
                                    end_date=self.end_date, 
                                    adjust="qfq")
            df = df.sort_values('日期')
            df = df.set_index('日期')
            df.index = pd.to_datetime(df.index)
            return df['收盘']
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def get_index_data(self, index_code):
        """
        获取指数数据作为ETF的替代
        """
        try:
            df = ak.index_zh_a_hist(symbol=index_code, 
                                  period='daily',
                                  start_date=self.start_date,
                                  end_date=self.end_date)
            df = df.sort_values('日期')
            df = df.set_index('日期')
            df.index = pd.to_datetime(df.index)
            return df['收盘']
        except Exception as e:
            print(f"获取指数 {index_code} 数据失败: {e}")
            return None
    
    def load_all_data(self):
        """
        加载所有资产数据
        """
        all_symbols = []
        for category in self.ASSET_POOL.values():
            all_symbols.extend(category.keys())
        
        price_data = pd.DataFrame()
        
        for symbol in all_symbols:
            print(f"正在获取 {symbol} 数据...")
            prices = self.get_etf_data(symbol)
            if prices is not None:
                price_data[symbol] = prices
        
        self.price_data = price_data
        self.return_data = price_data.pct_change().dropna()
        
        return self.price_data, self.return_data
    
    def get_quadrant_returns(self):
        """
        获取四象限组合的收益率
        """
        if self.return_data is None:
            self.load_all_data()
        
        quadrant_returns = pd.DataFrame()
        
        for quadrant, assets in self.QUADRANT_ASSETS.items():
            valid_assets = [a for a in assets if a in self.return_data.columns]
            if len(valid_assets) > 0:
                quadrant_returns[quadrant] = self.return_data[valid_assets].mean(axis=1)
        
        return quadrant_returns
    
    def get_macro_data(self):
        """
        获取宏观数据（用于预期共振动量）
        简化版：主要获取价格数据来拟合宏观预期
        """
        macro_data = {}
        
        try:
            pmi_data = ak.macro_china_pmi_monthly()
            ppi_data = ak.macro_china_ppi_yearly()
            macro_data['pmi'] = pmi_data
            macro_data['ppi'] = ppi_data
        except Exception as e:
            print(f"获取宏观数据失败: {e}")
        
        return macro_data
    
    def save_data(self, output_dir='output'):
        """
        保存数据到本地
        """
        if self.price_data is not None:
            self.price_data.to_csv(f'{output_dir}/price_data.csv')
        if self.return_data is not None:
            self.return_data.to_csv(f'{output_dir}/return_data.csv')
    
    @staticmethod
    def load_saved_data(input_dir='output'):
        """
        从本地加载已保存的数据
        """
        loader = DataLoader()
        try:
            loader.price_data = pd.read_csv(f'{input_dir}/price_data.csv', index_col=0, parse_dates=True)
            loader.return_data = pd.read_csv(f'{input_dir}/return_data.csv', index_col=0, parse_dates=True)
        except:
            print("本地数据不存在，将从网络获取")
        return loader

