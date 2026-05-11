import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class MarginFunds:
    def __init__(self, data_loader):
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE
        self.dl = data_loader

    def get_margin_summary(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.margin(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching margin summary: {e}")
            return pd.DataFrame()

    def get_margin_detail(self, ts_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.stk_margindetail(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching margin detail for {ts_code}: {e}")
            return pd.DataFrame()

    def get_margin_balances(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.margin_weight(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching margin balances: {e}")
            return pd.DataFrame()

    def get_margin_sectors(self, trade_date=None):
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")

        try:
            df = self.pro.margin_sectors(
                trade_date=trade_date
            )
            return df
        except Exception as e:
            print(f"Error fetching margin sectors: {e}")
            return pd.DataFrame()

    def calculate_margin_indicators(self, freq="W"):
        df = self.get_margin_summary(self.start_date, self.end_date)

        if len(df) == 0:
            return pd.DataFrame()

        df = df.sort_values("trade_date")

        df = df.rename(columns={
            'rzye': 'balance',
            'rzmre': 'buy_amount',
            'rzche': 'repay_amount',
            'rqye': 'securities_balance',
            'rqmcl': 'securities_volume'
        })

        if 'balance' not in df.columns and 'buy_amount' not in df.columns:
            return pd.DataFrame()

        if freq == "W":
            df["period"] = df["trade_date"].dt.to_period("W")
            agg_funcs = {
                "buy_amount": "sum",
                "repay_amount": "sum",
            }
            if 'balance' in df.columns:
                agg_funcs["balance"] = "last"
        elif freq == "M":
            df["period"] = df["trade_date"].dt.to_period("M")
            agg_funcs = {
                "buy_amount": "sum",
                "repay_amount": "sum",
            }
            if 'balance' in df.columns:
                agg_funcs["balance"] = "last"
        else:
            df["period"] = df["trade_date"]
            agg_funcs = {
                "buy_amount": "sum",
                "repay_amount": "sum",
            }
            if 'balance' in df.columns:
                agg_funcs["balance"] = "last"

        result = df.groupby("period").agg(agg_funcs).reset_index()
        return result

    def normalize_by_turnover(self, df, turnover_df):
        if len(df) == 0 or len(turnover_df) == 0:
            return df

        df = df.merge(turnover_df, on="period", how="left")
        df["buy_amount_norm"] = df["buy_amount"] / df["turnover"]
        df["repay_amount_norm"] = df["repay_amount"] / df["turnover"]
        df["net_amount_norm"] = df["net_amount"] / df["turnover"]
        return df

    def normalize_by_market_cap(self, df, market_cap_df):
        if len(df) == 0 or len(market_cap_df) == 0:
            return df

        df = df.merge(market_cap_df, on="period", how="left")
        df["balance_norm"] = df["balance"] / df["market_cap"]
        return df

    def calculate_yoy_change(self, df, value_col="balance"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_yoy"] = df[value_col].pct_change(periods=12)
        return df

    def calculate_qoq_change(self, df, value_col="balance"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_qoq"] = df[value_col].pct_change(periods=1)
        return df

    def build_margin_indicators(self, freq="W"):
        indicators = []

        df = self.calculate_margin_indicators(freq=freq)
        if len(df) > 0:
            df_orig = df.copy()
            df_orig["indicator_name"] = f"margin_tr_balance_orig_{freq}_orig"

            df_yoy = self.calculate_yoy_change(df.copy(), "balance")
            df_yoy["indicator_name"] = f"margin_tr_balance_orig_{freq}_yoy"

            df_qoq = self.calculate_qoq_change(df.copy(), "balance")
            df_qoq["indicator_name"] = f"margin_tr_balance_orig_{freq}_qoq"

            indicators.extend([df_orig, df_yoy, df_qoq])

        df_amount = self.calculate_margin_indicators(freq=freq)
        if len(df_amount) > 0:
            df_buy_orig = df_amount.copy()
            df_buy_orig["indicator_name"] = f"margin_borrow_amount_{freq}_orig"

            df_buy_yoy = self.calculate_yoy_change(df_amount.copy(), "buy_amount")
            df_buy_yoy["indicator_name"] = f"margin_borrow_amount_{freq}_yoy"

            df_buy_qoq = self.calculate_qoq_change(df_amount.copy(), "buy_amount")
            df_buy_qoq["indicator_name"] = f"margin_borrow_amount_{freq}_qoq"

            indicators.extend([df_buy_orig, df_buy_yoy, df_buy_qoq])

            df_repay = df_amount.copy()
            df_repay["repay_amount"] = -df_repay["repay_amount"]
            df_repay_yoy = self.calculate_yoy_change(df_repay.copy(), "repay_amount")
            df_repay_yoy["indicator_name"] = f"margin_repay_amount_{freq}_yoy"

            indicators.append(df_repay_yoy)

        return indicators

    def get_industry_margin_flow(self, industry_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.margin_sectors(
                trade_date=end_date,
                indices_code=industry_code
            )
            if len(df) > 0:
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching industry margin flow for {industry_code}: {e}")
            return pd.DataFrame()

    def aggregate_industry_margin_flow(self, industry_list, freq="W"):
        result = []

        for code in industry_list:
            df = self.get_industry_margin_flow(code)

            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])

                if freq == "W":
                    df["period"] = df["trade_date"].dt.to_period("W")
                    agg_df = df.groupby("period").agg({
                        "balance": "last",
                        "buy_amount": "sum",
                        "repay_amount": "sum"
                    }).reset_index()
                elif freq == "M":
                    df["period"] = df["trade_date"].dt.to_period("M")
                    agg_df = df.groupby("period").agg({
                        "balance": "last",
                        "buy_amount": "sum",
                        "repay_amount": "sum"
                    }).reset_index()
                else:
                    agg_df = df.copy()
                    agg_df["period"] = agg_df["trade_date"]

                agg_df["industry_code"] = code
                result.append(agg_df)

        if len(result) > 0:
            return pd.concat(result, ignore_index=True)
        return pd.DataFrame()

    def calculate_market_share(self, df, value_col="balance"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        total = df.groupby("period")[value_col].transform("sum")
        df[f"{value_col}_share"] = df[value_col] / total
        return df

    def calculate_net_margin_flow(self, df):
        if len(df) == 0:
            return df

        if "buy_amount" in df.columns and "repay_amount" in df.columns:
            df["net_flow"] = df["buy_amount"] - df["repay_amount"]

        if "balance" in df.columns:
            df["balance_change"] = df.groupby("industry_code")["balance"].diff()
            df["balance_change_yoy"] = df.groupby("industry_code")["balance_change"].transform(
                lambda x: x.pct_change(periods=12)
            )

        return df