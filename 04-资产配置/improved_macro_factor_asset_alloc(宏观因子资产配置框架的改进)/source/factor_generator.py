import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings

from .data_loader import DataLoader
from .config import (
    FACTOR_CONFIG,
    RAW_FACTOR_INDICATORS,
    HIGH_FREQ_ASSET_CODES,
    ASSETS_CONFIG,
)

warnings.filterwarnings("ignore")


class FactorGenerator:
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader if data_loader else DataLoader()

    def generate_growth_factor(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        pmi_df = self.data_loader.get_macro_data(
            RAW_FACTOR_INDICATORS["PMI"]["code"], start_date, end_date
        )
        if len(pmi_df) > 0 and "data_value" in pmi_df.columns:
            pmi = pmi_df["data_value"].resample("M").last()
            pmi_yoy = pmi.pct_change(12) * 100
        else:
            pmi_yoy = pd.Series(dtype=float)

        fai_df = self.data_loader.get_macro_data(
            RAW_FACTOR_INDICATORS["FAI"]["code"], start_date, end_date
        )
        if len(fai_df) > 0 and "data_value" in fai_df.columns:
            fai = fai_df["data_value"].resample("M").last()
            fai_yoy = fai.pct_change(12) * 100
        else:
            fai_yoy = pd.Series(dtype=float)

        retail_df = self.data_loader.get_macro_data(
            RAW_FACTOR_INDICATORS["RetailSales"]["code"], start_date, end_date
        )
        if len(retail_df) > 0 and "data_value" in retail_df.columns:
            retail = retail_df["data_value"].resample("M").last()
            retail_yoy = retail.pct_change(12) * 100
        else:
            retail_yoy = pd.Series(dtype=float)

        raw_growth = pd.concat([pmi_yoy, fai_yoy, retail_yoy], axis=1)
        if raw_growth.shape[1] == 3:
            raw_growth = raw_growth.mean(axis=1)
        else:
            raw_growth = pmi_yoy if len(pmi_yoy) > 0 else fai_yoy

        high_freq_growth = self._generate_high_freq_growth(start_date, end_date)

        result = pd.DataFrame({
            "raw_growth": raw_growth,
            "high_freq_growth": high_freq_growth,
        })
        return result.dropna()

    def _generate_high_freq_growth(
        self, start_date: str, end_date: str
    ) -> pd.Series:
        crb_df = self.data_loader.get_commodity_futures(
            HIGH_FREQ_ASSET_CODES["CRBIndustrial"]["code"], start_date, end_date
        )
        copper_df = self.data_loader.get_commodity_futures(
            HIGH_FREQ_ASSET_CODES["SouthwestCopper"]["code"], start_date, end_date
        )
        realestate_df = self.data_loader.get_industry_index(
            HIGH_FREQ_ASSET_CODES["RealEstate"]["code"], start_date, end_date
        )

        weights = FACTOR_CONFIG["Growth"]["weights"]

        returns_list = []
        if len(crb_df) > 0 and "pct_change" in crb_df.columns:
            crb_ret = crb_df["pct_change"] / 100
            crb_vol = crb_ret.rolling(60).std()
            crb_weight = 1 / crb_vol
            crb_weighted = (crb_ret * crb_weight).dropna()
            returns_list.append(crb_weighted * weights.get("CRBIndustrial", 0.61))

        if len(copper_df) > 0 and "pct_change" in copper_df.columns:
            copper_ret = copper_df["pct_change"] / 100
            copper_vol = copper_ret.rolling(60).std()
            copper_weight = 1 / copper_vol
            copper_weighted = (copper_ret * copper_weight).dropna()
            returns_list.append(copper_weighted * weights.get("SouthwestCopper", 0.24))

        if len(realestate_df) > 0 and "pct_change" in realestate_df.columns:
            re_ret = realestate_df["pct_change"] / 100
            re_vol = re_ret.rolling(60).std()
            re_weight = 1 / re_vol
            re_weighted = (re_ret * re_weight).dropna()
            returns_list.append(re_weighted * weights.get("RealEstate", 0.15))

        if returns_list:
            aligned_returns = []
            for ret in returns_list:
                aligned_returns.append(ret.reindex(pd.date_range(
                    start=min(r.index.min() for r in returns_list),
                    end=max(r.index.max() for r in returns_list),
                    freq="D"
                ).intersection(ret.index)))

            high_freq = pd.concat(aligned_returns, axis=1).sum(axis=1)
            high_freq = high_freq.resample("M").last()
            return high_freq
        else:
            return pd.Series(dtype=float)

    def generate_inflation_factor(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cpi_df = self.data_loader.get_cpi_data(start_date, end_date)
        if len(cpi_df) > 0 and "data_value" in cpi_df.columns:
            cpi = cpi_df["data_value"].resample("M").last()
        else:
            cpi = pd.Series(dtype=float)

        ppi_df = self.data_loader.get_ppi_data(start_date, end_date)
        if len(ppi_df) > 0 and "data_value" in ppi_df.columns:
            ppi = ppi_df["data_value"].resample("M").last()
        else:
            ppi = pd.Series(dtype=float)

        raw_inflation = (cpi + ppi) / 2 if len(cpi) > 0 and len(ppi) > 0 else cpi if len(cpi) > 0 else ppi

        high_freq_inflation = self._generate_high_freq_inflation(start_date, end_date)

        result = pd.DataFrame({
            "raw_inflation": raw_inflation,
            "high_freq_inflation": high_freq_inflation,
        })
        return result.dropna()

    def _generate_high_freq_inflation(
        self, start_date: str, end_date: str
    ) -> pd.Series:
        pork_df = self.data_loader.get_macro_data(
            "PORK", start_date, end_date
        )
        oil_df = self.data_loader.get_commodity_futures(
            "Brent Crude", start_date, end_date
        )
        steel_df = self.data_loader.get_commodity_futures(
            "RB", start_date, end_date
        )

        weights = FACTOR_CONFIG["Inflation"]["weights"]

        returns_list = []
        if len(pork_df) > 0:
            if "pct_change" in pork_df.columns:
                pork_ret = pork_df["pct_change"] / 100
            elif "close" in pork_df.columns:
                pork_ret = pork_df["close"].pct_change()
            else:
                pork_ret = pork_df.iloc[:, 0].pct_change()

            pork_vol = pork_ret.rolling(60).std()
            pork_weight = 1 / pork_vol
            pork_weighted = (pork_ret * pork_weight).dropna()
            returns_list.append(pork_weighted * weights.get("PorkPrice", 0.35))

        if len(oil_df) > 0:
            if "pct_change" in oil_df.columns:
                oil_ret = oil_df["pct_change"] / 100
            elif "close" in oil_df.columns:
                oil_ret = oil_df["close"].pct_change()
            else:
                oil_ret = oil_df.iloc[:, 0].pct_change()

            oil_vol = oil_ret.rolling(60).std()
            oil_weight = 1 / oil_vol
            oil_weighted = (oil_ret * oil_weight).dropna()
            returns_list.append(oil_weighted * weights.get("BrentOil", 0.22))

        if len(steel_df) > 0:
            if "pct_change" in steel_df.columns:
                steel_ret = steel_df["pct_change"] / 100
            elif "close" in steel_df.columns:
                steel_ret = steel_df["close"].pct_change()
            else:
                steel_ret = steel_df.iloc[:, 0].pct_change()

            steel_vol = steel_ret.rolling(60).std()
            steel_weight = 1 / steel_vol
            steel_weighted = (steel_ret * steel_weight).dropna()
            returns_list.append(steel_weighted * weights.get("SteelRebar", 0.43))

        if returns_list:
            high_freq = pd.concat(returns_list, axis=1).sum(axis=1)
            high_freq = high_freq.resample("M").last()
            return high_freq
        else:
            return pd.Series(dtype=float)

    def generate_interest_rate_factor(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        rate_df = self.data_loader.get_interest_rate(start_date, end_date)

        if len(rate_df) > 0:
            if "10Y" in rate_df.columns:
                ten_year = rate_df["10Y"]
            elif "rate" in rate_df.columns:
                ten_year = rate_df["rate"]
            else:
                ten_year = rate_df.iloc[:, 0]
        else:
            shibor_df = self.data_loader.get_macro_data(
                "M0010101", start_date, end_date
            )
            if len(shibor_df) > 0 and "data_value" in shibor_df.columns:
                ten_year = shibor_df["data_value"].resample("M").last()
            else:
                ten_year = pd.Series(dtype=float)

        bond_df = self.data_loader.get_bond_daily(
            "CBA11001.CS", start_date, end_date
        )
        if len(bond_df) > 0:
            if "pct_change" in bond_df.columns:
                bond_ret = bond_df["pct_change"] / 100
            elif "close" in bond_df.columns:
                bond_ret = bond_df["close"].pct_change()
            else:
                bond_ret = pd.Series(dtype=float)

            high_freq_rate = -bond_ret.resample("M").last()
        else:
            high_freq_rate = pd.Series(dtype=float)

        result = pd.DataFrame({
            "raw_interest_rate": ten_year,
            "high_freq_interest_rate": high_freq_rate,
        })
        return result.dropna()

    def generate_credit_factor(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        spread_df = self.data_loader.get_macro_data(
            "G0010103", start_date, end_date
        )

        if len(spread_df) > 0 and "data_value" in spread_df.columns:
            credit_spread = spread_df["data_value"].resample("M").last()
        else:
            corp_bond_df = self.data_loader.get_bond_daily(
                "CBA11003.CS", start_date, end_date
            )
            gov_bond_df = self.data_loader.get_bond_daily(
                "CBA11001.CS", start_date, end_date
            )

            if len(corp_bond_df) > 0 and len(gov_bond_df) > 0:
                corp_ret = corp_bond_df["close"].pct_change().resample("M").last() if "close" in corp_bond_df.columns else pd.Series(dtype=float)
                gov_ret = gov_bond_df["close"].pct_change().resample("M").last() if "close" in gov_bond_df.columns else pd.Series(dtype=float)
                credit_spread = corp_ret - gov_ret
            else:
                credit_spread = pd.Series(dtype=float)

        corp_bond_df = self.data_loader.get_bond_daily(
            "CBA11003.CS", start_date, end_date
        )
        gov_bond_df = self.data_loader.get_bond_daily(
            "CBA11001.CS", start_date, end_date
        )

        if len(corp_bond_df) > 0 and len(gov_bond_df) > 0:
            corp_ret = corp_bond_df["close"].pct_change().resample("M").last() if "close" in corp_bond_df.columns else pd.Series(dtype=float)
            gov_ret = gov_bond_df["close"].pct_change().resample("M").last() if "close" in gov_bond_df.columns else pd.Series(dtype=float)
            high_freq_credit = corp_ret - gov_ret
        else:
            high_freq_credit = pd.Series(dtype=float)

        result = pd.DataFrame({
            "raw_credit": credit_spread,
            "high_freq_credit": high_freq_credit,
        })
        return result.dropna()

    def generate_exchange_rate_factor(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        fx_df = self.data_loader.get_fx_daily(
            "USDCNY", start_date, end_date
        )

        if len(fx_df) > 0:
            if "close" in fx_df.columns:
                raw_fx = fx_df["close"].resample("M").last()
            else:
                raw_fx = fx_df.iloc[:, 0].resample("M").last()
        else:
            raw_fx = pd.Series(dtype=float)

        sh_gold_df = self.data_loader.get_gold_price(start_date, end_date)

        try:
            import yfinance as yf
            come_gold = yf.download("GC=F", start=start_date, end=end_date)
            if len(come_gold) > 0:
                come_gold_ret = come_gold["Adj Close"].pct_change()
            else:
                come_gold_ret = pd.Series(dtype=float)
        except Exception:
            come_gold_ret = pd.Series(dtype=float)

        if len(sh_gold_df) > 0 and "pct_change" in sh_gold_df.columns:
            sh_gold_ret = sh_gold_df["pct_change"] / 100
        elif len(sh_gold_df) > 0 and "close" in sh_gold_df.columns:
            sh_gold_ret = sh_gold_df["close"].pct_change()
        else:
            sh_gold_ret = pd.Series(dtype=float)

        if len(sh_gold_ret) > 0 and len(come_gold_ret) > 0:
            aligned_sh = sh_gold_ret.reindex(come_gold_ret.index, method="ffill")
            high_freq_fx = (aligned_sh - come_gold_ret).resample("M").last()
        else:
            high_freq_fx = pd.Series(dtype=float)

        result = pd.DataFrame({
            "raw_exchange_rate": raw_fx,
            "high_freq_exchange_rate": high_freq_fx,
        })
        return result.dropna()

    def generate_liquidity_factor(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        m2_df = self.data_loader.get_money_supply(start_date, end_date)
        sf_df = self.data_loader.get_social_financing(start_date, end_date)

        if len(m2_df) > 0 and "m2_yoy" in m2_df.columns:
            m2 = m2_df["m2_yoy"].resample("M").last()
        elif len(m2_df) > 0 and "data_value" in m2_df.columns:
            m2 = m2_df["data_value"].resample("M").last()
        else:
            m2 = pd.Series(dtype=float)

        if len(sf_df) > 0 and "data_value" in sf_df.columns:
            social_fin = sf_df["data_value"].resample("M").last()
        else:
            social_fin = pd.Series(dtype=float)

        raw_liquidity = m2 - social_fin if len(m2) > 0 and len(social_fin) > 0 else m2

        high_freq_liquidity = self._generate_high_freq_liquidity(start_date, end_date)

        result = pd.DataFrame({
            "raw_liquidity": raw_liquidity,
            "high_freq_liquidity": high_freq_liquidity,
        })
        return result.dropna()

    def _generate_high_freq_liquidity(
        self, start_date: str, end_date: str
    ) -> pd.Series:
        large_cap_df = self.data_loader.get_index_daily(
            "000300.SH", start_date, end_date
        )
        small_cap_df = self.data_loader.get_index_daily(
            "000852.SH", start_date, end_date
        )

        returns_list = []
        if len(large_cap_df) > 0:
            if "pct_change" in large_cap_df.columns:
                large_ret = large_cap_df["pct_change"] / 100
            elif "close" in large_cap_df.columns:
                large_ret = large_cap_df["close"].pct_change()
            else:
                large_ret = pd.Series(dtype=float)

            large_vol = large_ret.rolling(60).std()
            large_weight = 1 / large_vol
            large_weighted = (large_ret * large_weight).dropna()
            returns_list.append(large_weighted)

        if len(small_cap_df) > 0:
            if "pct_change" in small_cap_df.columns:
                small_ret = small_cap_df["pct_change"] / 100
            elif "close" in small_cap_df.columns:
                small_ret = small_cap_df["close"].pct_change()
            else:
                small_ret = pd.Series(dtype=float)

            small_vol = small_ret.rolling(60).std()
            small_weight = 1 / small_vol
            small_weighted = (small_ret * small_weight).dropna()
            returns_list.append(-small_weighted)

        if returns_list:
            high_freq = pd.concat(returns_list, axis=1).sum(axis=1)
            high_freq = high_freq.resample("M").last()
            return high_freq
        else:
            return pd.Series(dtype=float)

    def generate_all_factors(
        self, start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        factors = {}

        try:
            growth = self.generate_growth_factor(start_date, end_date)
            factors["Growth"] = growth
        except Exception as e:
            print(f"Error generating Growth factor: {e}")
            factors["Growth"] = pd.DataFrame()

        try:
            inflation = self.generate_inflation_factor(start_date, end_date)
            factors["Inflation"] = inflation
        except Exception as e:
            print(f"Error generating Inflation factor: {e}")
            factors["Inflation"] = pd.DataFrame()

        try:
            interest_rate = self.generate_interest_rate_factor(start_date, end_date)
            factors["IntRate"] = interest_rate
        except Exception as e:
            print(f"Error generating Interest Rate factor: {e}")
            factors["IntRate"] = pd.DataFrame()

        try:
            credit = self.generate_credit_factor(start_date, end_date)
            factors["Credit"] = credit
        except Exception as e:
            print(f"Error generating Credit factor: {e}")
            factors["Credit"] = pd.DataFrame()

        try:
            exchange_rate = self.generate_exchange_rate_factor(start_date, end_date)
            factors["ExchRate"] = exchange_rate
        except Exception as e:
            print(f"Error generating Exchange Rate factor: {e}")
            factors["ExchRate"] = pd.DataFrame()

        try:
            liquidity = self.generate_liquidity_factor(start_date, end_date)
            factors["Liquidity"] = liquidity
        except Exception as e:
            print(f"Error generating Liquidity factor: {e}")
            factors["Liquidity"] = pd.DataFrame()

        return factors

    def get_high_freq_factors(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        factors = self.generate_all_factors(start_date, end_date)

        high_freq_dict = {}
        for factor_name, factor_df in factors.items():
            if len(factor_df) > 0 and "high_freq_" + factor_name.lower() in factor_df.columns:
                high_freq_dict[factor_name] = factor_df["high_freq_" + factor_name.lower()]
            elif len(factor_df) > 0:
                high_freq_dict[factor_name] = factor_df.iloc[:, 0]

        if high_freq_dict:
            high_freq_df = pd.DataFrame(high_freq_dict)
            return high_freq_df
        else:
            return pd.DataFrame()
