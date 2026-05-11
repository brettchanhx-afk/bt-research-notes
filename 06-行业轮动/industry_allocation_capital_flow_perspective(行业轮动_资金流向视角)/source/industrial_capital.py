import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class IndustrialCapital:
    def __init__(self, data_loader):
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE
        self.dl = data_loader

    def get_seo_detail(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.seo_detail(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "ann_date" in df.columns:
                df["ann_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("ann_date")
            return df
        except Exception as e:
            print(f"Error fetching SEO detail: {e}")
            return pd.DataFrame()

    def get_seo_preplan(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.seo_preplan(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "ann_date" in df.columns:
                df["ann_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("ann_date")
            return df
        except Exception as e:
            print(f"Error fetching SEO preplan: {e}")
            return pd.DataFrame()

    def get_seo_pass(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.seo_pass(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "pass_date" in df.columns:
                df["pass_date"] = pd.to_datetime(df["pass_date"])
                df = df.sort_values("pass_date")
            return df
        except Exception as e:
            print(f"Error fetching SEO pass: {e}")
            return pd.DataFrame()

    def get_seo_offering(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.seo_offering(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "offer_date" in df.columns:
                df["offer_date"] = pd.to_datetime(df["offer_date"])
                df = df.sort_values("offer_date")
            return df
        except Exception as e:
            print(f"Error fetching SEO offering: {e}")
            return pd.DataFrame()

    def calculate_seo_amount(self, df, amount_col="raise_amount"):
        if len(df) == 0:
            return df

        if amount_col not in df.columns:
            return df

        df[amount_col] = df[amount_col].fillna(0)
        return df

    def aggregate_seo_by_industry(self, df, date_col, industry_col="industry", freq="W"):
        if len(df) == 0 or date_col not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["trade_date"] = df[date_col]

        if freq == "W":
            df["period"] = df["trade_date"].dt.to_period("W")
        elif freq == "M":
            df["period"] = df["trade_date"].dt.to_period("M")
        else:
            df["period"] = df["trade_date"]

        agg_df = df.groupby(["period", industry_col]).agg({
            "raise_amount": "sum"
        }).reset_index()

        return agg_df

    def get_share_float(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.float_holder_change(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "ann_date" in df.columns:
                df["ann_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("ann_date")
            return df
        except Exception as e:
            print(f"Error fetching share float: {e}")
            return pd.DataFrame()

    def get_float_calendar(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.float_calendar(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                if "list_date" in df.columns:
                    df["list_date"] = pd.to_datetime(df["list_date"])
                if "ann_date" in df.columns:
                    df["ann_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("list_date")
            return df
        except Exception as e:
            print(f"Error fetching float calendar: {e}")
            return pd.DataFrame()

    def get_repurchase(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.stk_repurchase(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "ann_date" in df.columns:
                df["ann_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("ann_date")
            return df
        except Exception as e:
            print(f"Error fetching repurchase: {e}")
            return pd.DataFrame()

    def get_top10_holders(self, ts_code=None, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            if ts_code:
                df = self.pro.top10_holders(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                df = pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error fetching top 10 holders: {e}")
            return pd.DataFrame()

    def get_major_shareholder_change(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.major_shareholder_change(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "ann_date" in df.columns:
                df["ann_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("ann_date")
            return df
        except Exception as e:
            print(f"Error fetching major shareholder change: {e}")
            return pd.DataFrame()

    def get_shareholder_reduction(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.top10_holders(
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0 and "ann_date" in df.columns:
                df["ann_date"] = pd.to_datetime(df["ann_date"])
                df = df.sort_values("ann_date")
            return df
        except Exception as e:
            print(f"Error fetching shareholder reduction: {e}")
            return pd.DataFrame()

    def build_seo_indicators(self, freq="W"):
        indicators = []

        df_preplan = self.get_seo_preplan(self.start_date, self.end_date)
        if len(df_preplan) > 0:
            df_preplan_agg = self.aggregate_seo_by_industry(df_preplan, "ann_date", freq=freq)
            if len(df_preplan_agg) > 0:
                df_preplan_agg["indicator_name"] = f"AShareSEO_preplan_recent_orig_{freq}"
                indicators.append(df_preplan_agg)

        df_pass = self.get_seo_pass(self.start_date, self.end_date)
        if len(df_pass) > 0:
            df_pass_agg = self.aggregate_seo_by_industry(df_pass, "pass_date", freq=freq)
            if len(df_pass_agg) > 0:
                df_pass_agg["indicator_name"] = f"AShareSEO_pass_recent_orig_{freq}"
                indicators.append(df_pass_agg)

        df_offering = self.get_seo_offering(self.start_date, self.end_date)
        if len(df_offering) > 0:
            df_offering_agg = self.aggregate_seo_by_industry(df_offering, "offer_date", freq=freq)
            if len(df_offering_agg) > 0:
                df_offering_agg["indicator_name"] = f"AShareSEO_offering_recent_orig_{freq}"
                indicators.append(df_offering_agg)

        return indicators

    def build_float_indicators(self, freq="W"):
        indicators = []

        df_calendar = self.get_float_calendar(self.start_date, self.end_date)
        if len(df_calendar) > 0:
            df_calendar_ann = self.aggregate_seo_by_industry(
                df_calendar, "ann_date", freq=freq
            )
            if len(df_calendar_ann) > 0:
                df_calendar_ann["indicator_name"] = "AShareFreeFloatCalendar_anndt_recent_orig"
                indicators.append(df_calendar_ann)

            df_calendar_list = self.aggregate_seo_by_industry(
                df_calendar, "list_date", freq=freq
            )
            if len(df_calendar_list) > 0:
                df_calendar_list["indicator_name"] = "AShareFreeFloatCalendar_listdt_next_orig"
                indicators.append(df_calendar_list)

        return indicators

    def build_repurchase_indicators(self, freq="W"):
        indicators = []

        df_repurchase = self.get_repurchase(self.start_date, self.end_date)
        if len(df_repurchase) > 0 and "ann_date" in df_repurchase.columns:
            df_repurchase["period"] = df_repurchase["ann_date"].dt.to_period(freq) if freq in ["W", "M"] else df_repurchase["ann_date"]
            df_repurchase_agg = df_repurchase.groupby(["period"]).agg({
                "repurch_amount": "sum"
            }).reset_index()
            df_repurchase_agg["indicator_name"] = "AShareRepurchase_recent_orig"
            indicators.append(df_repurchase_agg)

        return indicators

    def build_holder_trade_indicators(self, freq="W"):
        indicators = []

        df_increase = self.get_major_shareholder_change(self.start_date, self.end_date)
        if len(df_increase) > 0 and "ann_date" in df_increase.columns:
            df_increase["period"] = df_increase["ann_date"].dt.to_period(freq) if freq in ["W", "M"] else df_increase["ann_date"]
            df_increase_agg = df_increase.groupby(["period"]).agg({
                "change_amount": "sum"
            }).reset_index()
            df_increase_agg["indicator_name"] = "MjrHolderTrade_add_recent_amount"
            indicators.append(df_increase_agg)

        df_decrease = self.get_shareholder_reduction(self.start_date, self.end_date)
        if len(df_decrease) > 0 and "ann_date" in df_decrease.columns:
            df_decrease["period"] = df_decrease["ann_date"].dt.to_period(freq) if freq in ["W", "M"] else df_decrease["ann_date"]
            df_decrease_agg = df_decrease.groupby(["period"]).agg({
                "change_amount": "sum"
            }).reset_index()
            df_decrease_agg["indicator_name"] = "MjrHolderTrade_under_recent_amount"
            indicators.append(df_decrease_agg)

        return indicators

    def normalize_by_turnover(self, df, turnover_df):
        if len(df) == 0 or len(turnover_df) == 0:
            return df

        df = df.merge(turnover_df, on="period", how="left")
        if "raise_amount" in df.columns:
            df["norm_by_turnover"] = df["raise_amount"] / df["turnover"]
        elif "repurch_amount" in df.columns:
            df["norm_by_turnover"] = df["repurch_amount"] / df["turnover"]
        elif "change_amount" in df.columns:
            df["norm_by_turnover"] = df["change_amount"] / df["turnover"]
        return df

    def normalize_by_market_cap(self, df, market_cap_df):
        if len(df) == 0 or len(market_cap_df) == 0:
            return df

        df = df.merge(market_cap_df, on="period", how="left")
        if "raise_amount" in df.columns:
            df["norm_by_mc"] = df["raise_amount"] / df["market_cap"]
        elif "repurch_amount" in df.columns:
            df["norm_by_mc"] = df["repurch_amount"] / df["market_cap"]
        elif "change_amount" in df.columns:
            df["norm_by_mc"] = df["change_amount"] / df["market_cap"]
        return df

    def calculate_yoy_change(self, df, value_col="amount"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_yoy"] = df[value_col].pct_change(periods=12)
        return df

    def calculate_qoq_change(self, df, value_col="amount"):
        if len(df) == 0 or value_col not in df.columns:
            return df

        df = df.sort_values("period")
        df[f"{value_col}_qoq"] = df[value_col].pct_change(periods=1)
        return df