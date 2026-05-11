import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class NorthboundFunds:
    def __init__(self, data_loader):
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE
        self.dl = data_loader

    def get_northbound_net_inflow(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.hk_hold_data(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching northbound net inflow: {e}")
            return pd.DataFrame()

    def get_northbound_holdings(self, ts_code="", start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.hk_hold(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching northbound holdings: {e}")
            return pd.DataFrame()

    def get_hsgt_north_net_inflow(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.hkto_cn_hsgt_north_net_inflow(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching HSGT north net inflow: {e}")
            return pd.DataFrame()

    def get_hsgt_hold_stock(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.hsgt_top10(
                start_date=start_date,
                end_date=end_date,
                market="SH"
            )
            return df
        except Exception as e:
            print(f"Error fetching HSGT hold stock: {e}")
            return pd.DataFrame()

    def calculate_north_change_amount(self, freq="W"):
        df = self.get_northbound_net_inflow(self.start_date, self.end_date)

        if len(df) == 0:
            return pd.DataFrame()

        df = df.sort_values("trade_date")

        if freq == "W":
            df["period"] = df["trade_date"].dt.to_period("W")
            agg_func = "sum"
        elif freq == "M":
            df["period"] = df["trade_date"].dt.to_period("M")
            agg_func = "sum"
        else:
            df["period"] = df["trade_date"]
            agg_func = "last"

        result = df.groupby("period").agg({
            "buy_amount": agg_func,
            "sell_amount": agg_func,
            "net_amount": agg_func
        }).reset_index()

        return result

    def calculate_north_holdings_float(self, freq="W"):
        df = self.get_northbound_holdings(start_date=self.start_date, end_date=self.end_date)

        if len(df) == 0:
            return pd.DataFrame()

        df = df.sort_values("trade_date")

        if freq == "W":
            df["period"] = df["trade_date"].dt.to_period("W")
            agg_func = "last"
        elif freq == "M":
            df["period"] = df["trade_date"].dt.to_period("M")
            agg_func = "last"
        else:
            df["period"] = df["trade_date"]
            agg_func = "last"

        result = df.groupby(["period", "ts_code"]).agg({
            "holdings": agg_func,
            "holdings_ratio": agg_func
        }).reset_index()

        return result

    def normalize_by_turnover(self, df, turnover_df):
        if len(df) == 0 or len(turnover_df) == 0:
            return df

        df = df.merge(turnover_df, on="period", how="left")
        df["normalized"] = df["net_amount"] / df["turnover"]
        return df

    def normalize_by_market_cap(self, df, market_cap_df):
        if len(df) == 0 or len(market_cap_df) == 0:
            return df

        df = df.merge(market_cap_df, on="period", how="left")
        df["normalized"] = df["net_amount"] / df["market_cap"]
        return df

    def calculate_yoy_change(self, df, value_col="net_amount"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_yoy"] = df[value_col].pct_change(periods=12)
        return df

    def calculate_qoq_change(self, df, value_col="net_amount"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_qoq"] = df[value_col].pct_change(periods=1)
        return df

    def build_north_indicators(self, freq="W"):
        indicators = []

        df_amount = self.calculate_north_change_amount(freq=freq)
        if len(df_amount) > 0:
            df_orig = df_amount.copy()
            df_orig["indicator_name"] = f"north_change_amount_{freq}_orig"

            df_yoy = self.calculate_yoy_change(df_amount.copy(), "net_amount")
            df_yoy["indicator_name"] = f"north_change_amount_{freq}_yoy"

            df_qoq = self.calculate_qoq_change(df_amount.copy(), "net_amount")
            df_qoq["indicator_name"] = f"north_change_amount_{freq}_qoq"

            indicators.extend([df_orig, df_yoy, df_qoq])

        df_holdings = self.calculate_north_holdings_float(freq=freq)
        if len(df_holdings) > 0:
            df_hold_orig = df_holdings.copy()
            df_hold_orig["indicator_name"] = f"north_holdings_float_{freq}_orig"

            df_hold_yoy = self.calculate_yoy_change(df_holdings.copy(), "holdings_ratio")
            df_hold_yoy["indicator_name"] = f"north_holdings_float_{freq}_yoy"

            df_hold_qoq = self.calculate_qoq_change(df_holdings.copy(), "holdings_ratio")
            df_hold_qoq["indicator_name"] = f"north_holdings_float_{freq}_qoq"

            indicators.extend([df_hold_orig, df_hold_yoy, df_hold_qoq])

        return indicators

    def get_industry_north_flow(self, industry_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.hk_hold(
                ts_code=industry_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching industry north flow for {industry_code}: {e}")
            return pd.DataFrame()

    def aggregate_industry_north_flow(self, industry_members_df, freq="W"):
        result = []

        for _, row in industry_members_df.iterrows():
            code = row.get("index_code", "")
            if not code:
                continue

            df = self.get_industry_north_flow(code)

            if len(df) > 0:
                if freq == "W":
                    df["period"] = df["trade_date"].dt.to_period("W")
                    agg_df = df.groupby("period").agg({
                        "net_amount": "sum",
                        "holdings": "last"
                    }).reset_index()
                elif freq == "M":
                    df["period"] = df["trade_date"].dt.to_period("M")
                    agg_df = df.groupby("period").agg({
                        "net_amount": "sum",
                        "holdings": "last"
                    }).reset_index()
                else:
                    agg_df = df.copy()
                    agg_df["period"] = agg_df["trade_date"]

                agg_df["industry_code"] = code
                result.append(agg_df)

        if len(result) > 0:
            return pd.concat(result, ignore_index=True)
        return pd.DataFrame()