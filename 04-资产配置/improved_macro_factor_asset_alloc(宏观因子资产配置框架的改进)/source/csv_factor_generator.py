import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from .csv_data_loader import CSVDataLoader
from .config import (
    FACTOR_CONFIG,
    ASSETS_CONFIG,
    HIGH_FREQ_FACTOR_COLUMNS,
    RAW_FACTOR_COLUMNS,
)


class CSVFactorGenerator:
    def __init__(self, csv_loader: Optional[CSVDataLoader] = None):
        self.csv_loader = csv_loader if csv_loader else CSVDataLoader()

    def generate_all_factors(self) -> Dict[str, pd.DataFrame]:
        factors = {}

        high_freq = self.csv_loader.get_processed_high_freq_factors()
        raw = self.csv_loader.get_processed_raw_factors()

        high_freq = high_freq[~high_freq.index.duplicated(keep='first')]
        raw = raw[~raw.index.duplicated(keep='first')]

        factors["Growth"] = self._generate_growth_factor(high_freq, raw)
        factors["Inflation"] = self._generate_inflation_factor(high_freq, raw)
        factors["IntRate"] = self._generate_interest_rate_factor(high_freq, raw)
        factors["Credit"] = self._generate_credit_factor(high_freq, raw)
        factors["ExchRate"] = self._generate_exchange_rate_factor(high_freq, raw)
        factors["Liquidity"] = self._generate_liquidity_factor(high_freq, raw)

        return factors

    def _generate_growth_factor(
        self, high_freq: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        common_idx = high_freq.index.intersection(raw.index)

        if len(common_idx) == 0:
            raw_growth = pd.Series(dtype=float)
        else:
            raw_subset = raw.loc[common_idx]
            if "PMI" in raw_subset.columns and "FAI" in raw_subset.columns:
                raw_growth = raw_subset[["PMI", "FAI"]].mean(axis=1)
            else:
                raw_growth = pd.Series(dtype=float)

        crb = high_freq["CRBIndustrial"].pct_change() if "CRBIndustrial" in high_freq.columns else pd.Series(dtype=float)
        copper = high_freq["SouthwestCopper"].pct_change() if "SouthwestCopper" in high_freq.columns else pd.Series(dtype=float)
        re = high_freq["RealEstate"].pct_change() if "RealEstate" in high_freq.columns else pd.Series(dtype=float)

        weights = FACTOR_CONFIG["Growth"]["weights"]

        crb_weighted = crb * weights.get("CRBIndustrial", 0.61)
        copper_weighted = copper * weights.get("SouthwestCopper", 0.24)
        re_weighted = re * weights.get("RealEstate", 0.15)

        high_freq_growth = (crb_weighted + copper_weighted + re_weighted).replace([np.inf, -np.inf], np.nan)

        result = pd.DataFrame({
            "raw_growth": raw_growth,
            "high_freq_growth": high_freq_growth,
        })

        return result.dropna(how='all')

    def _generate_inflation_factor(
        self, high_freq: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        common_idx = high_freq.index.intersection(raw.index)

        if len(common_idx) == 0:
            raw_inflation = pd.Series(dtype=float)
        else:
            raw_subset = raw.loc[common_idx]
            if "CPI" in raw_subset.columns and "PPI" in raw_subset.columns:
                raw_inflation = raw_subset[["CPI", "PPI"]].mean(axis=1)
            else:
                raw_inflation = pd.Series(dtype=float)

        weights = FACTOR_CONFIG["Inflation"]["weights"]

        pork = high_freq["PorkPrice"].pct_change() * weights.get("PorkPrice", 0.35) if "PorkPrice" in high_freq.columns else pd.Series(dtype=float)
        oil = high_freq["BrentOil"].pct_change() * weights.get("BrentOil", 0.22) if "BrentOil" in high_freq.columns else pd.Series(dtype=float)
        steel = high_freq["SteelRebar"].pct_change() * weights.get("SteelRebar", 0.43) if "SteelRebar" in high_freq.columns else pd.Series(dtype=float)

        high_freq_inflation = (pork + oil + steel).replace([np.inf, -np.inf], np.nan)

        result = pd.DataFrame({
            "raw_inflation": raw_inflation,
            "high_freq_inflation": high_freq_inflation,
        })

        return result.dropna(how='all')

    def _generate_interest_rate_factor(
        self, high_freq: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        common_idx = high_freq.index.intersection(raw.index)

        if len(common_idx) == 0:
            raw_rate = pd.Series(dtype=float)
        else:
            raw_subset = raw.loc[common_idx]
            raw_rate = raw_subset["TenYearYield"] if "TenYearYield" in raw_subset.columns else pd.Series(dtype=float)

        gov_bond = high_freq["ChinaGovBond"].pct_change() if "ChinaGovBond" in high_freq.columns else pd.Series(dtype=float)

        result = pd.DataFrame({
            "raw_interest_rate": raw_rate,
            "high_freq_interest_rate": gov_bond,
        })

        return result.dropna(how='all')

    def _generate_credit_factor(
        self, high_freq: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        common_idx = high_freq.index.intersection(raw.index)

        if len(common_idx) == 0:
            raw_credit = pd.Series(dtype=float)
        else:
            raw_subset = raw.loc[common_idx]
            if "CreditSpread" in raw_subset.columns:
                raw_credit = raw_subset["CreditSpread"]
            else:
                raw_credit = pd.Series(dtype=float)

        corp = high_freq["CorpBondAA"].pct_change() if "CorpBondAA" in high_freq.columns else pd.Series(dtype=float)
        gov = high_freq["ChinaGovBond3Y"].pct_change() if "ChinaGovBond3Y" in high_freq.columns else pd.Series(dtype=float)
        high_freq_credit = corp - gov

        result = pd.DataFrame({
            "raw_credit": raw_credit,
            "high_freq_credit": high_freq_credit,
        })

        return result.dropna(how='all')

    def _generate_exchange_rate_factor(
        self, high_freq: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        common_idx = high_freq.index.intersection(raw.index)

        if len(common_idx) == 0:
            raw_fx = pd.Series(dtype=float)
        else:
            raw_subset = raw.loc[common_idx]
            raw_fx = raw_subset["USDIndex"] if "USDIndex" in raw_subset.columns else pd.Series(dtype=float)

        hsi = high_freq["HSI"].pct_change() if "HSI" in high_freq.columns else pd.Series(dtype=float)

        result = pd.DataFrame({
            "raw_exchange_rate": raw_fx,
            "high_freq_exchange_rate": hsi,
        })

        return result.dropna(how='all')

    def _generate_liquidity_factor(
        self, high_freq: pd.DataFrame, raw: pd.DataFrame
    ) -> pd.DataFrame:
        common_idx = high_freq.index.intersection(raw.index)

        if len(common_idx) == 0:
            raw_liquidity = pd.Series(dtype=float)
        else:
            raw_subset = raw.loc[common_idx]
            m2 = raw_subset["M2YoY"] if "M2YoY" in raw_subset.columns else pd.Series(dtype=float)
            social_fin = raw_subset["SocialFinYoY"] if "SocialFinYoY" in raw_subset.columns else pd.Series(dtype=float)

            if len(m2) > 0 and len(social_fin) > 0:
                raw_liquidity = m2 - social_fin
            else:
                raw_liquidity = pd.Series(dtype=float)

        large_pe = high_freq["SWLargeCapPE"].pct_change() if "SWLargeCapPE" in high_freq.columns else pd.Series(dtype=float)
        small_pe = high_freq["SWSmallCapPE"].pct_change() if "SWSmallCapPE" in high_freq.columns else pd.Series(dtype=float)

        high_freq_liquidity = large_pe - small_pe

        result = pd.DataFrame({
            "raw_liquidity": raw_liquidity,
            "high_freq_liquidity": high_freq_liquidity,
        })

        return result.dropna(how='all')

    def get_high_freq_factors(self) -> pd.DataFrame:
        factors = self.generate_all_factors()

        high_freq_dict = {}
        for factor_name, factor_df in factors.items():
            if len(factor_df) > 0:
                col_name = "high_freq_" + factor_name.lower()
                if col_name in factor_df.columns:
                    high_freq_dict[factor_name] = factor_df[col_name]
                elif len(factor_df.columns) > 0:
                    high_freq_dict[factor_name] = factor_df.iloc[:, 0]

        if high_freq_dict:
            high_freq_df = pd.DataFrame(high_freq_dict)
            return high_freq_df.dropna(how='all')
        else:
            return pd.DataFrame()

    def get_raw_factors(self) -> pd.DataFrame:
        factors = self.generate_all_factors()

        raw_dict = {}
        for factor_name, factor_df in factors.items():
            if len(factor_df) > 0:
                col_name = "raw_" + factor_name.lower()
                if col_name in factor_df.columns:
                    raw_dict[factor_name] = factor_df[col_name]
                elif len(factor_df.columns) > 1:
                    raw_dict[factor_name] = factor_df.iloc[:, 1]

        if raw_dict:
            raw_df = pd.DataFrame(raw_dict)
            return raw_df.dropna(how='all')
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    generator = CSVFactorGenerator()
    factors = generator.generate_all_factors()

    print("=" * 60)
    print("因子生成成功")
    print("=" * 60)

    for name, df in factors.items():
        if len(df) > 0:
            print(f"\n【{name}】")
            print(f"  数据条数: {len(df)}")
            print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")
            print(f"  列名: {list(df.columns)}")
