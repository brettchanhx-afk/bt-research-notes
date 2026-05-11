"""
趋势追踪行业配置策略 - 数据获取模块
使用tushare获取市场数据
"""

import pandas as pd
import numpy as np
import tushare as ts
from typing import Optional, Dict, List

TOKEN = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
pro = ts.pro_api(TOKEN)
pro._DataApi__token = TOKEN
pro._DataApi__http_url = "http://jiaoch.site"

INDEX_CODES = {
    "上证50": "000016.SH",
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "恒生指数": "HSI.HI",
    "标普500": "SPX.GI",
    "上证国债指数": "000012.SH",
    "黄金": "SPTAUUSDOZ.IDC",
    "布伦特原油": "S0031525",
    "LME铜": "CA.LME",
}

INDUSTRY_CODES = {
    "石油石化": "CI005001.WI",
    "煤炭": "CI005002.WI",
    "有色金属": "CI005003.WI",
    "电力及公用事业": "CI005004.WI",
    "钢铁": "CI005005.WI",
    "基础化工": "CI005006.WI",
    "建筑": "CI005007.WI",
    "建材": "CI005008.WI",
    "轻工制造": "CI005009.WI",
    "机械": "CI005010.WI",
    "电力设备及新能源": "CI005011.WI",
    "国防军工": "CI005012.WI",
    "汽车": "CI005013.WI",
    "商贸零售": "CI005014.WI",
    "消费者服务": "CI005015.WI",
    "家电": "CI005016.WI",
    "纺织服装": "CI005017.WI",
    "医药": "CI005018.WI",
    "农林牧渔": "CI005020.WI",
    "银行": "CI005021.WI",
    "房地产": "CI005023.WI",
    "交通运输": "CI005024.WI",
    "电子": "CI005025.WI",
    "通信": "CI005026.WI",
    "计算机": "CI005027.WI",
    "传媒": "CI005028.WI",
    "酒类": "CI005156.WI",
    "饮料": "CI005822.WI",
    "食品": "CI005823.WI",
    "证券": "CI005165.WI",
    "保险": "CI005166.WI",
}

class DataLoader:
    def __init__(self):
        self.pro = pro

    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df.rename(columns={'pct_chg': 'returns'}, inplace=True)
        return df

    def get_industry_daily(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df.rename(columns={'pct_chg': 'returns'}, inplace=True)
        return df

    def get_multiple_indices(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        result = {}
        for name, code in INDEX_CODES.items():
            try:
                df = self.get_index_daily(code, start_date, end_date)
                if len(df) > 0:
                    result[name] = df
                    print(f"成功获取 {name} 数据，共 {len(df)} 条记录")
                else:
                    print(f"警告: {name} 无数据")
            except Exception as e:
                print(f"获取 {name} 数据失败: {e}")
        return result

    def get_multiple_industries(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        result = {}
        for name, code in INDUSTRY_CODES.items():
            try:
                df = self.get_industry_daily(code, start_date, end_date)
                if len(df) > 0:
                    result[name] = df
                    print(f"成功获取 {name} 数据，共 {len(df)} 条记录")
                else:
                    print(f"警告: {name} 无数据")
            except Exception as e:
                print(f"获取 {name} 数据失败: {e}")
        return result

    def calculate_assets_statistics(self, assets_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        stats_list = []
        for name, df in assets_data.items():
            daily_return = df['returns'].mean()
            daily_vol = df['returns'].std()
            stats_list.append({
                'asset_name': name,
                'daily_return': daily_return,
                'daily_volatility': daily_vol,
                'annual_return': daily_return * 252,
                'annual_volatility': daily_vol * np.sqrt(252),
                'return_vol_ratio': (daily_return * 252) / (daily_vol * np.sqrt(252)) if daily_vol > 0 else np.nan
            })
        return pd.DataFrame(stats_list)

    def calculate_correlation_matrix(self, assets_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        returns_dict = {}
        for name, df in assets_data.items():
            df_temp = df.set_index('trade_date')['returns']
            returns_dict[name] = df_temp

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(0)
        corr_matrix = returns_df.corr()
        return corr_matrix

def load_asset_data(asset_type: str = "index", start_date: str = "20100101",
                   end_date: str = "20200731") -> Dict[str, pd.DataFrame]:
    loader = DataLoader()
    if asset_type == "index":
        return loader.get_multiple_indices(start_date, end_date)
    elif asset_type == "industry":
        return loader.get_multiple_industries(start_date, end_date)
    else:
        raise ValueError("asset_type must be 'index' or 'industry'")

if __name__ == "__main__":
    loader = DataLoader()
    print("测试数据获取功能...")
    test_data = loader.get_index_daily("000300.SH", "20200101", "20200731")
    print(f"测试数据形状: {test_data.shape}")
    print(test_data.head())