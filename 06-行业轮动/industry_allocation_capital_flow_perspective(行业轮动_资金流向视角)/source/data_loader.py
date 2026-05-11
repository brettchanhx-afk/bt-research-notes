import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class DataLoader:
    def __init__(self):
        self.pro = settings.pro
        self.start_date = settings.START_DATE
        self.end_date = settings.END_DATE

    def get_index_daily(self, index_code=settings.INDEX_CODE):
        df = self.pro.index_daily(
            ts_code=index_code,
            start_date=self.start_date,
            end_date=self.end_date,
            fields="trade_date, close"
        )
        df = df.sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def get_industry_list(self, level="SW"):
        if level == "SW":
            df = self.pro.index_classify(
                src="SW2021",
                level="L1"
            )
            return df["name"].tolist() if "name" in df.columns else []
        return []

    def get_industry_daily(self, industry_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.index_daily(
                ts_code=industry_code,
                start_date=start_date,
                end_date=end_date
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df.sort_values("trade_date")
        except Exception as e:
            print(f"Error fetching industry {industry_code}: {e}")
            return pd.DataFrame()

    def get_sw_industry_members(self):
        try:
            df = self.pro.index_member(ts_code="801010.SI")
            return df
        except Exception as e:
            print(f"Error fetching SW industry members: {e}")
            return pd.DataFrame()

    def get_stock_industry(self, ts_code):
        try:
            df = self.pro.stock_basic(
                ts_code=ts_code,
                fields="ts_code, name, industry"
            )
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_concept_stock(self, trade_date=None):
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")
        try:
            df = self.pro.hk_basic(
                trade_date=trade_date,
                limit=1000
            )
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_northbound_history(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.hk_hold(
                ts_code="",
                start_date=start_date,
                end_date=end_date,
                fields="trade_date, ts_code, name, close, holdings, holdings_ratio"
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            print(f"Error fetching northbound data: {e}")
            return pd.DataFrame()

    def get_northbound_daily(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.macro_hk_mon(
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            print(f"Error fetching northbound daily: {e}")
            return pd.DataFrame()

    def get_margin_detail(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.stk_margindetail(
                trade_date=end_date
            )
            return df
        except Exception as e:
            print(f"Error fetching margin detail: {e}")
            return pd.DataFrame()

    def get_margin_ratio(self, ts_code=None, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            if ts_code:
                df = self.pro.stk_margins(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                df = self.pro.margin(
                    start_date=start_date,
                    end_date=end_date
                )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            print(f"Error fetching margin ratio: {e}")
            return pd.DataFrame()

    def get_index_weight(self, index_code=settings.INDEX_CODE, trade_date=None):
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")

        try:
            df = self.pro.index_weight(
                index_code=index_code,
                trade_date=trade_date
            )
            return df
        except Exception as e:
            print(f"Error fetching index weight: {e}")
            return pd.DataFrame()

    def get_etf_basic(self):
        try:
            df = self.pro.fund_basic(
                market="E"
            )
            return df
        except Exception as e:
            print(f"Error fetching ETF basic: {e}")
            return pd.DataFrame()

    def get_etf_daily(self, ts_code, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.fund_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")
            return df
        except Exception as e:
            print(f"Error fetching ETF daily: {e}")
            return pd.DataFrame()

    def get_block_trade(self, ts_code=None, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            if ts_code:
                df = self.pro.block_trade(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                df = pd.DataFrame()
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_share_float(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro. share_float(
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_float_delta(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.float_delta(
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_stock_repurchase(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        try:
            df = self.pro.stk_repurchase(
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_major_holders(self, ts_code=None, start_date=None, end_date=None):
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
            return pd.DataFrame()

    def get_top_list(self, trade_date=None):
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")

        try:
            df = self.pro.top_list(trade_date=trade_date)
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_pledge_stat(self, ts_code=None):
        try:
            if ts_code:
                df = self.pro.pledge_stat(
                    ts_code=ts_code
                )
            else:
                df = pd.DataFrame()
            return df
        except Exception as e:
            return pd.DataFrame()

    def save_data(self, df, filename):
        output_path = os.path.join(settings.DATA_OUTPUT_DIR, filename)
        os.makedirs(settings.DATA_OUTPUT_DIR, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")