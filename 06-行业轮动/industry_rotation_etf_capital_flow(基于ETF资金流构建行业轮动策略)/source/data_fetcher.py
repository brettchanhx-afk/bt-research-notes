import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    TUSHARE_TOKEN,
    TUSHARE_API_URL,
    INDUSTRY_INDEX_MAPPING,
    INDUSTRY_NAME_MAPPING,
    INDEX_BASE_DATE,
)

warnings.filterwarnings("ignore")


class MultiSourceDataFetcher:
    def __init__(self, token=None, api_url=None):
        self.token = token if token else TUSHARE_TOKEN
        self.api_url = api_url if api_url else TUSHARE_API_URL
        self.tushare_pro = None
        self._init_tushare()

    def _init_tushare(self):
        try:
            import tushare as ts
            self.tushare_pro = ts.pro_api(self.token)
            self.tushare_pro._DataApi__token = self.token
            self.tushare_pro._DataApi__http_url = self.api_url
            print("[Tushare] 初始化成功")
        except Exception as e:
            print(f"[Tushare] 初始化失败: {e}")
            self.tushare_pro = None

    def get_etf_list_tushare(self) -> pd.DataFrame:
        if self.tushare_pro is None:
            return pd.DataFrame()
        try:
            df = self.tushare_pro.fund_basic(
                exchange="", market="E", status="L", type="ETF"
            )
            if df is not None and len(df) > 0:
                df = df[df["name"].notna()].reset_index(drop=True)
                print(f"[Tushare] ETF列表获取成功: {len(df)} 只")
                return df
        except Exception as e:
            print(f"[Tushare] ETF列表获取失败: {e}")
        return pd.DataFrame()

    def get_etf_list_akshare(self) -> pd.DataFrame:
        try:
            import akshare as ak
            df = ak.fund_etf_spot_em()
            if df is not None and len(df) > 0:
                df = df[['代码', '名称', '最新价', '成交量', '成交额', '最新份额', '数据日期']].copy()
                df.columns = ['ts_code', 'name', 'close', 'volume', 'amount', 'shares', 'data_date']
                print(f"[AkShare] ETF列表获取成功: {len(df)} 只")
                return df
        except Exception as e:
            print(f"[AkShare] ETF列表获取失败: {e}")
        return pd.DataFrame()

    def get_etf_daily_tushare(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.tushare_pro is None:
            return pd.DataFrame()
        try:
            df = self.tushare_pro.fund_daily(
                ts_code=ts_code, start_date=start_date, end_date=end_date
            )
            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date").reset_index(drop=True)
                print(f"[Tushare] ETF {ts_code} 日线数据获取成功: {len(df)} 条")
                return df
        except Exception as e:
            print(f"[Tushare] ETF {ts_code} 日线数据获取失败: {e}")
        return pd.DataFrame()

    def get_etf_hist_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import akshare as ak
            df = ak.fund_etf_hist_em(symbol=symbol, period='daily', start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '涨跌幅': 'pct_change',
                    '换手率': 'turnover'
                })
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                print(f"[AkShare] ETF {symbol} 历史数据获取成功: {len(df)} 条")
                return df
        except Exception as e:
            print(f"[AkShare] ETF {symbol} 历史数据获取失败: {e}")
        return pd.DataFrame()

    def get_index_daily_tushare(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.tushare_pro is None:
            return pd.DataFrame()
        if index_code.startswith("h") or index_code.startswith("H"):
            print(f"[Tushare] 指数 {index_code} 非标准tushare代码格式")
            return pd.DataFrame()
        try:
            df = self.tushare_pro.index_daily(
                ts_code=index_code, start_date=start_date, end_date=end_date
            )
            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date").reset_index(drop=True)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                print(f"[Tushare] 指数 {index_code} 数据获取成功: {len(df)} 条")
                return df
        except Exception as e:
            print(f"[Tushare] 指数 {index_code} 数据获取失败: {e}")
        return pd.DataFrame()

    def get_index_daily_baostock(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import baostock as bs

            bs.login()

            bs_code = self._convert_to_baostock_code(index_code)
            if bs_code is None:
                bs.logout()
                return pd.DataFrame()

            rs = bs.query_history_k_data_plus(
                bs_code,
                'date,code,open,high,low,close,volume,pctChg',
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                frequency='d',
                adjustflag='3'
            )

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            bs.logout()

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                df['pct_change'] = pd.to_numeric(df['pctChg'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                print(f"[BaoStock] 指数 {index_code} 数据获取成功: {len(df)} 条")
                return df

        except Exception as e:
            print(f"[BaoStock] 指数 {index_code} 数据获取失败: {e}")
        return pd.DataFrame()

    def get_index_daily_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import akshare as ak
            ak_code = self._convert_to_akshare_code(symbol)
            if ak_code is None:
                return pd.DataFrame()

            df = ak.stock_zh_index_daily(symbol=ak_code)
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    'date': 'trade_date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
                df['pct_change'] = df['close'].pct_change() * 100
                print(f"[AkShare] 指数 {symbol} 数据获取成功: {len(df)} 条")
                return df
        except Exception as e:
            print(f"[AkShare] 指数 {symbol} 数据获取失败: {e}")
        return pd.DataFrame()

    def _convert_to_baostock_code(self, index_code: str) -> Optional[str]:
        code_mapping = {
            '399975.SZ': 'sz.399975',
            '930651.CSI': 'sh.930651',
            '399971.SZ': 'sz.399971',
            '980017.CNI': 'sh.980017',
            '000949.CSI': 'sh.000949',
            '000815.CSI': 'sh.000815',
            '930697.CSI': 'sh.930697',
            '931775.CSI': 'sh.931775',
            '931160.CSI': 'sh.931160',
            '000819.SH': 'sh.000819',
            'h30199.CSI': None,
            '000813.CSI': 'sh.000813',
            'h30171.CSI': None,
            '931009.CSI': 'sh.931009',
            '930606.CSI': 'sh.930606',
            'h30015.CSI': None,
            '399989.SZ': 'sz.399989',
            '399967.SZ': 'sz.399967',
            '399808.SZ': 'sz.399808',
            '399986.SZ': 'sz.399986',
            '399998.SZ': 'sz.399998',
        }
        return code_mapping.get(index_code)

    def _convert_to_akshare_code(self, index_code: str) -> Optional[str]:
        code_mapping = {
            '399975.SZ': 'sz399975',
            '930651.CSI': 'sh930651',
            '399971.SZ': 'sz399971',
            '980017.CNI': 'sh980017',
            '000949.CSI': 'sh000949',
            '000815.CSI': 'sh000815',
            '930697.CSI': 'sh930697',
            '931775.CSI': 'sh931775',
            '931160.CSI': 'sh931160',
            '000819.SH': 'sh000819',
            'h30199.CSI': None,
            '000813.CSI': 'sh000813',
            'h30171.CSI': None,
            '931009.CSI': 'sh931009',
            '930606.CSI': 'sh930606',
            'h30015.CSI': None,
            '399989.SZ': 'sz399989',
            '399967.SZ': 'sz399967',
            '399808.SZ': 'sz399808',
            '399986.SZ': 'sz399986',
            '399998.SZ': 'sz399998',
        }
        return code_mapping.get(index_code)

    def get_sw_industry_daily_tushare(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.tushare_pro is None:
            return pd.DataFrame()
        try:
            df = self.tushare_pro.sw_daily(trade_date="", start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df
        except Exception as e:
            print(f"[Tushare] 申万行业数据获取失败: {e}")
        return pd.DataFrame()

    def get_all_index_data(
        self,
        start_date: str = "20150101",
        end_date: str = "20240831"
    ) -> Dict[str, pd.DataFrame]:
        all_index_data = {}

        for industry_name, index_code in INDUSTRY_INDEX_MAPPING.items():
            print(f"\n获取 {industry_name} ({index_code}) 指数数据...")

            df = self.get_index_daily_tushare(index_code, start_date, end_date)
            if df is None or len(df) == 0:
                print(f"  -> Tushare失败,尝试BaoStock...")
                df = self.get_index_daily_baostock(index_code, start_date, end_date)
            if df is None or len(df) == 0:
                print(f"  -> BaoStock失败,尝试AkShare...")
                df = self.get_index_daily_akshare(index_code, start_date, end_date)

            if df is not None and len(df) > 0:
                df['industry'] = industry_name
                df['index_code'] = index_code
                df['index_name'] = INDUSTRY_NAME_MAPPING.get(index_code, "")
                all_index_data[industry_name] = df
            else:
                print(f"  ⚠️  {industry_name} ({index_code}) 无法获取数据")

        return all_index_data

    def get_industry_etf_list_tushare(self, industry_name: str) -> pd.DataFrame:
        if self.tushare_pro is None:
            return pd.DataFrame()
        try:
            df = self.tushare_pro.fund_basic(
                exchange="", market="E", status="L", type="ETF"
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()

            industry_keywords = {
                "非银金融": ["非银", "证券", "保险", "多元金融"],
                "计算机": ["计算机", "软件", "科技", "互联网"],
                "传媒": ["传媒", "文化", "娱乐"],
                "电子": ["电子", "半导体", "芯片", "光电子"],
                "农林牧渔": ["农业", "农林", "畜牧", "渔业", "牧业"],
                "食品饮料": ["食品", "饮料", "酒", "白酒", "乳业"],
                "家用电器": ["家电", "电器", "空调", "洗衣机", "冰箱"],
                "房地产": ["房地产", "地产", "物业"],
                "通信": ["通信", "电信", "5G", "通信设备"],
                "有色金属": ["有色", "金属", "铜", "铝", "稀土"],
                "公用事业": ["公用事业", "电力", "燃气", "水务", "电网"],
                "基础化工": ["化工", "化学", "石化"],
                "交通运输": ["交通", "运输", "航空", "航运", "港口", "铁路", "公路"],
                "建筑材料": ["建材", "水泥", "玻璃"],
                "钢铁": ["钢铁", "钢材"],
                "汽车": ["汽车", "整车", "零部件", "新能源车"],
                "医药生物": ["医药", "医疗", "生物", "中药", "化药", "医疗器械"],
                "国防军工": ["军工", "国防", "航天", "航空", "船舶", "兵器"],
                "电力设备": ["电力设备", "新能源", "光伏", "风电", "储能", "电池"],
                "银行": ["银行"],
                "煤炭": ["煤炭", "煤碳"],
            }

            keywords = industry_keywords.get(industry_name, [])
            if not keywords:
                return pd.DataFrame()

            pattern = "|".join(keywords)
            df_filtered = df[df["name"].str.contains(pattern, na=False)].copy()
            print(f"[Tushare] {industry_name}行业ETF列表: {len(df_filtered)} 只")
            return df_filtered
        except Exception as e:
            print(f"[Tushare] 获取{industry_name}行业ETF列表失败: {e}")
            return pd.DataFrame()

    def get_industry_etf_data(
        self,
        industry_name: str,
        start_date: str = "20180101",
        end_date: str = "20240731"
    ) -> pd.DataFrame:
        etf_list = self.get_industry_etf_list_tushare(industry_name)

        if etf_list is None or len(etf_list) == 0:
            print(f"  Tushare无{industry_name}行业ETF,使用AkShare...")
            industry_etf_map = {
                "非银金融": ["512880", "159993"],
                "计算机": ["512720", "159998"],
                "传媒": ["512980", "159805"],
                "电子": ["512760", "515050"],
                "农林牧渔": ["159825", "010632"],
                "食品饮料": ["512690", "159928"],
                "家用电器": ["159996", "561170"],
                "房地产": ["512200", "159918"],
                "通信": ["515050", "512220"],
                "有色金属": ["512400", "159876"],
                "公用事业": ["159509", "560890"],
                "基础化工": ["510620", "159870"],
                "交通运输": ["159659", "512480"],
                "建筑材料": ["159619", "512780"],
                "钢铁": ["515210", "159628"],
                "汽车": ["516110", "159806"],
                "医药生物": ["512010", "159829"],
                "国防军工": ["512660", "159713"],
                "电力设备": ["515050", "159825"],
                "银行": ["512800", "159887"],
                "煤炭": ["515220", "161032"],
            }
            etf_codes = industry_etf_map.get(industry_name, [])
        else:
            etf_codes = etf_list['ts_code'].str[:6].tolist()[:2]

        all_data = []
        for code in etf_codes:
            df = self.get_etf_daily_tushare(code, start_date, end_date)
            if df is None or len(df) == 0:
                df = self.get_etf_hist_akshare(code, start_date, end_date)
            if df is not None and len(df) > 0:
                df['etf_code'] = code
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


class TushareDataFetcher:
    def __init__(self, token=None, api_url=None):
        if token is None:
            token = TUSHARE_TOKEN
        if api_url is None:
            api_url = TUSHARE_API_URL

        self.token = token
        self.api_url = api_url
        self.pro = ts.pro_api(token)
        self.pro._DataApi__token = token
        self.pro._DataApi__http_url = api_url

    def get_etf_basic_info(self):
        try:
            df = self.pro.fund_basic(
                exchange="", market="E", status="L", type="ETF"
            )
            if df is not None and len(df) > 0:
                df = df[df["name"].notna()].reset_index(drop=True)
            return df
        except Exception as e:
            print(f"获取ETF基本信息失败: {e}")
            return pd.DataFrame()

    def get_etf_daily_flow(self, ts_code, start_date, end_date):
        try:
            df = self.pro.fund_daily(
                ts_code=ts_code, start_date=start_date, end_date=end_date
            )
            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date").reset_index(drop=True)
            return df
        except Exception as e:
            print(f"获取ETF日线数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_index_daily(self, index_code, start_date, end_date):
        try:
            if index_code.startswith("h") or index_code.startswith("H"):
                print(f"警告: 指数 {index_code} 可能不是标准tushare指数代码格式")
                return pd.DataFrame()

            df = self.pro.index_daily(
                ts_code=index_code, start_date=start_date, end_date=end_date
            )
            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date").reset_index(drop=True)
            return df
        except Exception as e:
            print(f"获取指数日线数据失败 {index_code}: {e}")
            return pd.DataFrame()

    def get_sw_industry_daily(self, start_date, end_date):
        try:
            df = self.pro.sw_daily(trade_date="", start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
            return df
        except Exception as e:
            print(f"获取申万行业日线数据失败: {e}")
            return pd.DataFrame()

    def get_industry_etf_list(self, industry_name):
        try:
            df = self.pro.fund_basic(
                exchange="", market="E", status="L", type="ETF"
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()

            industry_keywords = {
                "非银金融": ["非银", "证券", "保险", "多元金融"],
                "计算机": ["计算机", "软件", "科技", "互联网"],
                "传媒": ["传媒", "文化", "娱乐"],
                "电子": ["电子", "半导体", "芯片", "光电子"],
                "农林牧渔": ["农业", "农林", "畜牧", "渔业", "牧业"],
                "食品饮料": ["食品", "饮料", "酒", "白酒", "乳业"],
                "家用电器": ["家电", "电器", "空调", "洗衣机", "冰箱"],
                "房地产": ["房地产", "地产", "物业"],
                "通信": ["通信", "电信", "5G", "通信设备"],
                "有色金属": ["有色", "金属", "铜", "铝", "稀土"],
                "公用事业": ["公用事业", "电力", "燃气", "水务", "电网"],
                "基础化工": ["化工", "化学", "石化"],
                "交通运输": ["交通", "运输", "航空", "航运", "港口", "铁路", "公路"],
                "建筑材料": ["建材", "水泥", "玻璃"],
                "钢铁": ["钢铁", "钢材"],
                "汽车": ["汽车", "整车", "零部件", "新能源车"],
                "医药生物": ["医药", "医疗", "生物", "中药", "化药", "医疗器械"],
                "国防军工": ["军工", "国防", "航天", "航空", "船舶", "兵器"],
                "电力设备": ["电力设备", "新能源", "光伏", "风电", "储能", "电池"],
                "银行": ["银行"],
                "煤炭": ["煤炭", "煤碳"],
            }

            keywords = industry_keywords.get(industry_name, [])
            if not keywords:
                return pd.DataFrame()

            pattern = "|".join(keywords)
            df_filtered = df[df["name"].str.contains(pattern, na=False)].copy()

            return df_filtered
        except Exception as e:
            print(f"获取{industry_name}行业ETF列表失败: {e}")
            return pd.DataFrame()


def load_or_fetch_data(
    data_dir: Path,
    force_refresh: bool = False,
    start_date: str = "20150101",
    end_date: str = "20240831"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    print("=" * 60)
    print("多数据源数据获取 (优先Tushare)")
    print("=" * 60)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    etf_list_file = data_dir / "etf_list.csv"
    index_data_file = data_dir / "index_returns.csv"
    etf_flow_file = data_dir / "etf_flow_data.pkl"

    if not force_refresh:
        if etf_list_file.exists() and index_data_file.exists():
            print("\n从缓存加载数据...")
            try:
                etf_df = pd.read_csv(etf_list_file)
                index_returns = pd.read_csv(index_data_file)
                if 'trade_date' in index_returns.columns:
                    index_returns['trade_date'] = pd.to_datetime(index_returns['trade_date'])
                print(f"  ETF列表: {len(etf_df)} 只")
                print(f"  指数收益: {len(index_returns)} 条")
                return etf_df, index_returns, {}
            except Exception as e:
                print(f"加载缓存失败: {e}")

    print("\n从多数据源获取数据 (优先Tushare)...")

    fetcher = MultiSourceDataFetcher()

    print("\n[1/3] 获取ETF列表 (Tushare优先)...")
    etf_df = fetcher.get_etf_list_tushare()
    if etf_df is None or len(etf_df) == 0:
        print("  Tushare失败,使用AkShare...")
        etf_df = fetcher.get_etf_list_akshare()
    if etf_df is not None and len(etf_df) > 0:
        etf_df.to_csv(etf_list_file, index=False)
        print(f"  ETF列表已保存: {etf_list_file}")

    print("\n[2/3] 获取行业指数数据 (Tushare优先)...")
    all_index_data = fetcher.get_all_index_data(start_date, end_date)

    if all_index_data:
        all_index_dfs = []
        for industry, df in all_index_data.items():
            if df is not None and len(df) > 0:
                all_index_dfs.append(df)

        if all_index_dfs:
            index_returns = pd.concat(all_index_dfs, ignore_index=True)
            index_returns = index_returns.sort_values(['trade_date', 'industry'])
            index_returns.to_csv(index_data_file, index=False)
            print(f"\n  指数收益数据已保存: {index_data_file}")
            print(f"  总计: {len(index_returns)} 条记录")

    print("\n[3/3] 提示: ETF资金流数据需要从Wind/Choice获取")
    print("  当前使用模拟数据进行演示")

    industry_etf_flow = {}
    for industry in INDUSTRY_INDEX_MAPPING.keys():
        dates = pd.date_range("2018-01-01", "2024-07-31", freq="D")
        np.random.seed(hash(industry) % (2**32))
        industry_etf_flow[industry] = pd.DataFrame({
            'trade_date': dates,
            'net_flow': np.random.randn(len(dates)) * 1000000,
            'nav': np.random.uniform(0.9, 1.1, len(dates)),
            'vol': np.random.randint(1000, 100000, len(dates)),
        })
        industry_etf_flow[industry]['trade_date'] = pd.to_datetime(industry_etf_flow[industry]['trade_date'])

    return etf_df, index_returns if 'index_returns' in dir() else pd.DataFrame(), industry_etf_flow


if __name__ == "__main__":
    print("多数据源数据获取模块测试 (优先Tushare)...")
    fetcher = MultiSourceDataFetcher()

    print("\n1. 测试获取ETF列表...")
    etf_df = fetcher.get_etf_list_tushare()
    if etf_df is None or len(etf_df) == 0:
        etf_df = fetcher.get_etf_list_akshare()
    print(f"ETF总数: {len(etf_df) if etf_df is not None else 0}")

    print("\n2. 测试获取指数数据...")
    test_index = fetcher.get_index_daily_tushare("399975.SZ", "20230101", "20231231")
    if test_index is None or len(test_index) == 0:
        print("  Tushare失败,尝试BaoStock...")
        test_index = fetcher.get_index_daily_baostock("399975.SZ", "20230101", "20231231")
    if test_index is None or len(test_index) == 0:
        print("  BaoStock失败,尝试AkShare...")
        test_index = fetcher.get_index_daily_akshare("399975.SZ", "20230101", "20231231")
    print(f"非银金融指数: {len(test_index) if test_index is not None else 0} 条")

    print("\n数据获取模块测试完成")
