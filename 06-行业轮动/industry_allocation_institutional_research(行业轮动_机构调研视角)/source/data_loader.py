import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False

try:
    import efinance as ef
    EFINANCE_AVAILABLE = True
except ImportError:
    EFINANCE_AVAILABLE = False


class DataLoader:
    def __init__(self, token="5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"):
        self.token = token
        self.pro = ts.pro_api(token)
        self.pro._DataApi__token = token
        self.pro._DataApi__http_url = "http://jiaoch.site"

    def get_index_data(self, index_code='000985.SH', start_date='20150101', end_date='20210228'):
        df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df.set_index('trade_date', inplace=True)
        return df

    def get_stock_basic_data(self):
        df = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        return df

    def get_industry_classification(self):
        df = self.pro.sw_daily(sec_code='001002.SH', start_date='20210101', end_date='20210131')
        if df is not None:
            industry_df = self.pro.sw_csi(sec_code=df['ts_code'].iloc[0] if len(df) > 0 else None)
        if industry_df is None:
            industry_df = self.pro.stock_basic(fields='ts_code,industry')
        return industry_df

    def get_daily_stock_data(self, ts_code, start_date, end_date):
        df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, api=self.pro)
        if df is not None:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
        return df

    def get_index_weight(self, index_code='000985.SH', trade_date=None):
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
        try:
            df = self.pro.index_weight(index_code=index_code, trade_date=trade_date)
            return df
        except Exception as e:
            print(f"获取指数权重失败: {e}")
            return None

    def get_money_flow(self, ts_code=None, trade_date=None):
        if trade_date:
            df = self.pro.moneyflow(trade_date=trade_date)
        elif ts_code:
            df = self.pro.moneyflow(ts_code=ts_code)
        else:
            return None
        return df

    def get_stock_float_free_float(self, ts_code, start_date, end_date):
        df = ts.pro_bar(ts_code=ts_code, start_date=start_date, end_date=end_date, api=self.pro,
                        adj='qfq')
        return df

    def get_concept_stock(self, code=None):
        try:
            if code:
                df = self.pro.concept_detail(id=code)
            else:
                df = self.pro.concept()
            return df
        except Exception as e:
            print(f"获取概念股失败: {e}")
            return None

    def get_ashare_survey_data(self, ts_code=None, start_date=None, end_date=None):
        try:
            if AKSHARE_AVAILABLE:
                survey_df = ak.stock_survey_summary_em()
                if survey_df is not None and len(survey_df) > 0:
                    survey_df.columns = ['股票代码', '股票名称', '调研日期', '调研机构数', '调研方式', '调研人',
                                        '股票代码_2', '类型']
                    survey_df['调研日期'] = pd.to_datetime(survey_df['调研日期'])
                    survey_df = survey_df.rename(columns={
                        '调研机构数': 'institutions_count',
                        '调研日期': 'survey_date',
                        '股票代码': 'ts_code'
                    })
                    survey_df['ts_code'] = survey_df['ts_code'].astype(str).str.zfill(6)
                    survey_df['ts_code'] = survey_df['ts_code'].apply(
                        lambda x: x + '.SZ' if x.startswith('0') or x.startswith('3') else x + '.SH' if x.startswith('6') else x
                    )
                    if start_date:
                        survey_df = survey_df[survey_df['survey_date'] >= pd.to_datetime(start_date)]
                    if end_date:
                        survey_df = survey_df[survey_df['survey_date'] <= pd.to_datetime(end_date)]
                    return survey_df
        except Exception as e:
            print(f"使用akshare获取机构调研数据失败: {e}")

        try:
            df = self.pro.ashare_survey(trade_date=start_date, ts_code=ts_code)
            return df
        except Exception as e:
            print(f"获取机构调研数据失败: {e}")
            return None

    def get_nearly_two_years_survey(self, symbol='000001'):
        try:
            if AKSHARE_AVAILABLE:
                df = ak.stock_survey_detail_em(symbol=symbol)
                return df
        except Exception as e:
            print(f"获取近两年机构调研数据失败: {e}")
        return None

    def get_financial_data(self, ts_code, start_date, end_date):
        try:
            df = self.pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            print(f"获取财务数据失败: {e}")
            return None

    def get_history_survey_count(self, ts_code=None, start_date='20120101', end_date='20210312'):
        try:
            if AKSHARE_AVAILABLE:
                df = ak.stock_history_survey_em(symbol='all')
                if df is not None and len(df) > 0:
                    if '股票代码' in df.columns:
                        df = df.rename(columns={
                            '股票代码': 'ts_code',
                            '调研日期': 'survey_date',
                            '调研机构数量': 'institutions_count'
                        })
                        df['ts_code'] = df['ts_code'].astype(str).str.zfill(6)
                        df['ts_code'] = df['ts_code'].apply(
                            lambda x: x + '.SZ' if x.startswith('0') or x.startswith('3') else x + '.SH' if x.startswith('6') else x
                        )
                        if start_date:
                            df = df[df['survey_date'] >= pd.to_datetime(start_date)]
                        if end_date:
                            df = df[df['survey_date'] <= pd.to_datetime(end_date)]
                        if ts_code:
                            df = df[df['ts_code'] == ts_code]
                        return df
        except Exception as e:
            print(f"获取历史调研数据失败: {e}")
        return None

    def get_all_survey_data(self, start_date='20120101', end_date='20210312'):
        try:
            if AKSHARE_AVAILABLE:
                df = ak.stock_history_survey_em(symbol='all')
                if df is not None and len(df) > 0:
                    df = df.rename(columns={
                        '股票代码': 'ts_code',
                        '调研日期': 'survey_date',
                        '调研机构数量': 'institutions_count',
                        '股票名称': 'stock_name'
                    })
                    df['ts_code'] = df['ts_code'].astype(str).str.zfill(6)
                    df['ts_code'] = df['ts_code'].apply(
                        lambda x: x + '.SZ' if x.startswith('0') or x.startswith('3') else x + '.SH' if x.startswith('6') else x
                    )
                    if start_date:
                        df = df[df['survey_date'] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df['survey_date'] <= pd.to_datetime(end_date)]
                    return df
        except Exception as e:
            print(f"获取全部机构调研数据失败: {e}")
        return None

    def get_index_components(self, index_code='000985.SH', trade_date=None):
        try:
            if trade_date:
                df = self.pro.index_weight(index_code=index_code, trade_date=trade_date)
                if df is not None:
                    return df['con_code'].tolist()
        except Exception as e:
            print(f"获取指数成分股失败: {e}")
        return []

    def get_stock_industry(self, ts_code):
        try:
            df = self.pro.stock_basic(ts_code=ts_code, fields='ts_code,industry')
            if df is not None and len(df) > 0:
                return df['industry'].iloc[0]
        except Exception as e:
            print(f"获取股票行业失败: {e}")
        return None

    def get_concept_detail(self, symbol='AI'):
        try:
            if AKSHARE_AVAILABLE:
                df = ak.stock_board_concept_cons_em(symbol=symbol)
                return df
        except Exception as e:
            print(f"获取概念板块成分股失败: {e}")
        return None

    def get_stock_cninfo(self, ts_code):
        try:
            if AKSHARE_AVAILABLE:
                df = ak.stock_info_cninfo(symbol=ts_code)
                return df
        except Exception as e:
            print(f"获取股票基本信息失败: {e}")
        return None

    def get_macro_data(self, indicator='CPI'):
        try:
            if AKSHARE_AVAILABLE:
                df = ak.macro_china_cpi()
                return df
        except Exception as e:
            print(f"获取宏观数据失败: {e}")
        return None


class SurveyDataAggregator:
    def __init__(self, data_loader):
        self.dl = data_loader

    def aggregate_survey_by_stock(self, survey_df):
        if survey_df is None or len(survey_df) == 0:
            return None
        daily_survey = survey_df.groupby(['ts_code', 'survey_date']).agg({
            'institutions_count': 'sum'
        }).reset_index()
        return daily_survey

    def aggregate_survey_by_date(self, survey_df):
        if survey_df is None or len(survey_df) == 0:
            return None
        daily_survey = survey_df.groupby('survey_date').agg({
            'institutions_count': 'sum'
        }).reset_index()
        return daily_survey

    def aggregate_survey_by_industry(self, survey_df, stock_industry_df):
        if survey_df is None or len(survey_df) == 0:
            return None
        merged_df = survey_df.merge(stock_industry_df, on='ts_code', how='left')
        industry_survey = merged_df.groupby(['industry', 'survey_date']).agg({
            'institutions_count': 'sum'
        }).reset_index()
        return industry_survey

    def calculate_rolling_survey_count(self, survey_df, stock_list, lookback_days=20):
        result = []
        survey_df = survey_df.sort_values(['ts_code', 'survey_date'])
        for stock in stock_list:
            stock_survey = survey_df[survey_df['ts_code'] == stock].copy()
            if len(stock_survey) == 0:
                continue
            stock_survey = stock_survey.set_index('survey_date')
            date_range = pd.date_range(stock_survey.index.min(), stock_survey.index.max())
            stock_survey = stock_survey.reindex(date_range, fill_value=0)
            stock_survey['rolling_count'] = stock_survey['institutions_count'].rolling(
                window=lookback_days, min_periods=1).sum()
            stock_survey['ts_code'] = stock
            result.append(stock_survey.reset_index().rename(columns={'index': 'trade_date'}))
        if len(result) == 0:
            return None
        return pd.concat(result, ignore_index=True)


def load_or_fetch_survey_data(data_loader, cache_path=None):
    if cache_path:
        try:
            df = pd.read_parquet(cache_path)
            print(f"从缓存加载机构调研数据: {len(df)} 条记录")
            return df
        except:
            pass
    survey_df = data_loader.get_all_survey_data()
    if survey_df is not None and cache_path:
        survey_df.to_parquet(cache_path)
        print(f"保存机构调研数据到缓存: {len(survey_df)} 条记录")
    return survey_df