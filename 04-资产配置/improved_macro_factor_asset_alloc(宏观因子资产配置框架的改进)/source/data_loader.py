import pandas as pd
import numpy as np
import tushare as ts
from typing import Dict, List, Optional, Tuple
import warnings
import os
import pickle
from datetime import datetime, timedelta

from .config import (
    TUSHARE_TOKEN,
    TUSHARE_HTTP_URL,
    ASSETS_CONFIG,
    HIGH_FREQ_ASSET_CODES,
    RAW_FACTOR_INDICATORS,
    DATA_DIR,
)

warnings.filterwarnings("ignore")


class DataLoader:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_dir = DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_tushare()

    def _init_tushare(self):
        self.token = TUSHARE_TOKEN
        self.pro = ts.pro_api(self.token)
        self.pro._DataApi__token = self.token
        self.pro._DataApi__http_url = TUSHARE_HTTP_URL

    def _get_cache_path(self, name: str) -> os.PathLike:
        return self.cache_dir / f"{name}.pkl"

    def _load_from_cache(self, name: str) -> Optional[pd.DataFrame]:
        if not self.use_cache:
            return None
        cache_path = self._get_cache_path(name)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, name: str, data: pd.DataFrame):
        if self.use_cache:
            cache_path = self._get_cache_path(name)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

    def get_trading_days(self, start_date: str, end_date: str) -> pd.DataFrame:
        cache_name = f"trading_days_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        df = self.pro.trade_cal(
            exchange="SSE",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            is_open="1",
        )
        df = df.sort_values("cal_date")
        self._save_to_cache(cache_name, df)
        return df

    def get_index_daily(
        self, index_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"index_daily_{index_code}_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            if "." in index_code:
                ts_code = index_code
            else:
                ts_code = index_code

            df = self.pro.index_dailybasic(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching index daily for {index_code}: {e}")

        return pd.DataFrame()

    def get_index_weights(
        self, index_code: str, date: str
    ) -> Optional[pd.DataFrame]:
        try:
            df = self.pro.index_weight(
                index_code=index_code, trade_date=date.replace("-", "")
            )
            return df
        except Exception as e:
            print(f"Error fetching index weights for {index_code} on {date}: {e}")
            return None

    def get_bond_daily(self, bond_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_name = f"bond_daily_{bond_code}_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            if bond_code.startswith("CBA"):
                df = self.pro.cb_dailybasic(
                    ts_code=bond_code,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )
            else:
                df = self.pro.bond_daily(
                    ts_code=bond_code,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )

            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching bond daily for {bond_code}: {e}")

        return pd.DataFrame()

    def get_macro_data(
        self, indicator_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"macro_{indicator_code}_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.macro_data(
                indicator=indicator_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("period")
                df["period"] = pd.to_datetime(df["period"])
                df.set_index("period", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching macro data for {indicator_code}: {e}")

        return pd.DataFrame()

    def get_gdp_data(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"gdp_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.cn_gdp(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("end_date")
                df["end_date"] = pd.to_datetime(df["end_date"])
                df.set_index("end_date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching GDP data: {e}")

        return pd.DataFrame()

    def get_cpi_data(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"cpi_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.cpi(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("month")
                df["month"] = pd.to_datetime(df["month"])
                df.set_index("month", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching CPI data: {e}")

        return pd.DataFrame()

    def get_ppi_data(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"ppi_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.ppi(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("month")
                df["month"] = pd.to_datetime(df["month"])
                df.set_index("month", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching PPI data: {e}")

        return pd.DataFrame()

    def get_fx_daily(
        self, currency_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"fx_daily_{currency_code}_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            if currency_code == "USDCNY":
                df = self.pro.fx_daily(
                    ts_code="USDCNY.Ex",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )
            else:
                df = self.pro.fx_daily(
                    ts_code=currency_code,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )

            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching FX daily for {currency_code}: {e}")

        return pd.DataFrame()

    def get_money_supply(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"money_supply_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.cn_m(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("month")
                df["month"] = pd.to_datetime(df["month"])
                df.set_index("month", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching money supply data: {e}")

        return pd.DataFrame()

    def get_social_financing(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"social_financing_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.cn_sa(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("month")
                df["month"] = pd.to_datetime(df["month"])
                df.set_index("month", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching social financing data: {e}")

        return pd.DataFrame()

    def get_interest_rate(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"interest_rate_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.shibor(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("date")
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching interest rate data: {e}")

        return pd.DataFrame()

    def get_gold_price(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"gold_price_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.daily(
                ts_code="AU9999.SHF",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching gold price data: {e}")

        return pd.DataFrame()

    def get_commodity_futures(
        self, futures_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"commodity_{futures_code}_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.realtime_daily(
                ts_code=futures_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is None or len(df) == 0:
                df = self.pro.fut_daily(
                    ts_code=futures_code,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )

            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching commodity futures for {futures_code}: {e}")

        return pd.DataFrame()

    def get_industry_index(
        self, industry_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_name = f"industry_{industry_code}_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_name)
        if cached is not None:
            return cached

        try:
            df = self.pro.index_daily(
                ts_code=industry_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )

            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date")
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
                self._save_to_cache(cache_name, df)
                return df
        except Exception as e:
            print(f"Error fetching industry index for {industry_code}: {e}")

        return pd.DataFrame()

    def clear_cache(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def load_asset_returns(
        self, asset_list: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        returns_dict = {}

        for asset_key in asset_list:
            if asset_key not in ASSETS_CONFIG:
                print(f"Asset {asset_key} not found in config")
                continue

            asset_info = ASSETS_CONFIG[asset_key]
            code = asset_info["code"]
            asset_type = asset_info["type"]

            if asset_type == "index":
                df = self.get_index_daily(code, start_date, end_date)
                if len(df) > 0 and "pct_change" in df.columns:
                    returns_dict[asset_key] = df["pct_change"] / 100
                elif len(df) > 0 and "close" in df.columns:
                    returns_dict[asset_key] = df["close"].pct_change()
            elif asset_type == "bond":
                df = self.get_bond_daily(code, start_date, end_date)
                if len(df) > 0 and "pct_change" in df.columns:
                    returns_dict[asset_key] = df["pct_change"] / 100
                elif len(df) > 0 and "close" in df.columns:
                    returns_dict[asset_key] = df["close"].pct_change()
            elif asset_type == "commodity":
                if "AU" in code:
                    df = self.get_gold_price(start_date, end_date)
                    if len(df) > 0 and "pct_change" in df.columns:
                        returns_dict[asset_key] = df["pct_change"] / 100
                    elif len(df) > 0 and "close" in df.columns:
                        returns_dict[asset_key] = df["close"].pct_change()
                else:
                    df = self.get_commodity_futures(code, start_date, end_date)
                    if len(df) > 0 and "pct_change" in df.columns:
                        returns_dict[asset_key] = df["pct_change"] / 100
                    elif len(df) > 0 and "close" in df.columns:
                        returns_dict[asset_key] = df["close"].pct_change()

        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
            returns_df = returns_df.dropna(how="all")
            return returns_df
        else:
            return pd.DataFrame()
