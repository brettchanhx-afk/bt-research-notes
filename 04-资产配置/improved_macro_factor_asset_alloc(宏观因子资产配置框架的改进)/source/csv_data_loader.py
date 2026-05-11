import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


class CSVDataLoader:
    def __init__(self):
        self.data_dir = DATA_DIR
        self._load_all_data()

    def _load_all_data(self):
        self.asset_prices = self._load_asset_prices()
        self.raw_factors = self._load_raw_factors()
        self.high_freq_factors = self._load_high_freq_factors()

    def _load_asset_prices(self) -> pd.DataFrame:
        file_path = self.data_dir / "seven_assets_price_2013_PCA.csv"
        df = pd.read_csv(file_path, index_col=0, encoding='gbk')

        df.index = pd.to_datetime(df.index, errors='coerce')
        df.index.name = 'date'

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.replace('--', np.nan)
        df = df.sort_index()

        return df

    def _load_raw_factors(self) -> pd.DataFrame:
        file_path = self.data_dir / "original_macro_factor_2013.csv"
        df = pd.read_csv(file_path, index_col=0, encoding='utf-8')

        df.index = pd.to_datetime(df.index, errors='coerce')
        df.index.name = 'date'

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_index()

        return df

    def _load_high_freq_factors(self) -> pd.DataFrame:
        file_path = self.data_dir / "high_frequency_macro_factor_portfolio.csv"
        df = pd.read_csv(file_path, index_col=0, encoding='utf-8')

        df.index = pd.to_datetime(df.index, errors='coerce')
        df.index.name = 'date'

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_index()

        return df

    def get_asset_prices(self) -> pd.DataFrame:
        return self.asset_prices.copy()

    def get_asset_returns(self) -> pd.DataFrame:
        returns = self.asset_prices.pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan)
        return returns

    def get_raw_factors(self) -> pd.DataFrame:
        return self.raw_factors.copy()

    def get_high_freq_factors(self) -> pd.DataFrame:
        return self.high_freq_factors.copy()

    def get_processed_high_freq_factors(self) -> pd.DataFrame:
        df = self.high_freq_factors.copy()

        df = df.rename(columns={
            '恒生指数': 'HSI',
            'CRB现货指数:工业': 'CRBIndustrial',
            '南华铜指数': 'SouthwestCopper',
            '申万行业指数:房地产开发': 'RealEstate',
            '市场价:生猪(外三元):全国均价': 'PorkPrice',
            '期货收盘价(连续):布伦特原油:ICE': 'BrentOil',
            '现货价:螺纹钢': 'SteelRebar',
            '中债国债总指数(总值)净价指数': 'ChinaGovBond',
            '中债信用债总指数(3-5年)财富指数': 'CorpBondAA',
            '中债3-5年期国债指数(总值)财富指数': 'ChinaGovBond3Y',
            '美国:美元指数': 'USDIndex',
            '市盈率(成份股计算):申万大盘指数': 'SWLargeCapPE',
            '市盈率(成份股计算):申万小盘指数': 'SWSmallCapPE',
        })

        return df

    def get_processed_raw_factors(self) -> pd.DataFrame:
        df = self.raw_factors.copy()

        df = df.rename(columns={
            '制造业PMI': 'PMI',
            '固定资产投资(不含农户)完成额:累计同比': 'FAI',
            '社会消费品零售总额:当月同比': 'RetailSales',
            '进出口金额(人民币计价):中国:当月同比': 'ExportImport',
            'CPI:当月同比': 'CPI',
            'PPI:当月同比': 'PPI',
            '中债国债到期收益率:10年:月:平均值': 'TenYearYield',
            '中债中短期票据到期收益率(AA):3年:月:平均值': 'AA3YYield',
            '中债国开债到期收益率:3年:月:平均值': 'GovBond3YYield',
            '美国:美元指数:月:平均值': 'USDIndex',
            'M2(货币和准货币):同比': 'M2YoY',
            '社会融资规模增量:当月同比': 'SocialFinYoY',
        })

        df['CreditSpread'] = df['AA3YYield'] - df['GovBond3YYield']

        return df

    def get_processed_asset_mapping(self) -> Dict[str, str]:
        return {
            'CSI300': '000300.SH',
            'CSI500': '000905.SH',
            'NHCI': 'NHCI.SL',
            'GOV_BOND': 'CBA00601.CB',
            'CORP_BOND': 'CBA02001.CB',
            'BRENT': 'BRN0Y.ICE',
        }

    def get_data_summary(self) -> Dict:
        summary = {
            'asset_prices': {
                'start_date': str(self.asset_prices.index.min()),
                'end_date': str(self.asset_prices.index.max()),
                'columns': list(self.asset_prices.columns),
                'shape': self.asset_prices.shape,
            },
            'raw_factors': {
                'start_date': str(self.raw_factors.index.min()),
                'end_date': str(self.raw_factors.index.max()),
                'columns': list(self.raw_factors.columns),
                'shape': self.raw_factors.shape,
            },
            'high_freq_factors': {
                'start_date': str(self.high_freq_factors.index.min()),
                'end_date': str(self.high_freq_factors.index.max()),
                'columns': list(self.high_freq_factors.columns),
                'shape': self.high_freq_factors.shape,
            },
        }
        return summary


if __name__ == "__main__":
    loader = CSVDataLoader()
    summary = loader.get_data_summary()

    print("=" * 60)
    print("数据加载成功")
    print("=" * 60)

    print("\n【资产价格数据】")
    print(f"时间范围: {summary['asset_prices']['start_date']} ~ {summary['asset_prices']['end_date']}")
    print(f"资产列表: {summary['asset_prices']['columns']}")

    print("\n【原始宏观因子】")
    print(f"时间范围: {summary['raw_factors']['start_date']} ~ {summary['raw_factors']['end_date']}")
    print(f"因子列表: {summary['raw_factors']['columns']}")

    print("\n【高频化因子】")
    print(f"时间范围: {summary['high_freq_factors']['start_date']} ~ {summary['high_freq_factors']['end_date']}")
    print(f"因子列表: {summary['high_freq_factors']['columns']}")
