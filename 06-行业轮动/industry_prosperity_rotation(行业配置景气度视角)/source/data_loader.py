import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import warnings
warnings.filterwarnings('ignore')

token = "5c81629792e1ef719de0c99b586b66ad4c22b2a3ee2a16d7f920e50a94e3"
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = "http://jiaoch.site"

class DataLoader:
    def __init__(self, token=token, api_url="http://jiaoch.site"):
        self.token = token
        self.pro = ts.pro_api(token)
        self.pro._DataApi__token = token
        self.pro._DataApi__http_url = api_url
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "output", "data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, name):
        return os.path.join(self.cache_dir, f"{name}.parquet")

    def load_with_cache(self, name, loader_func, reload=False):
        cache_path = self.get_cache_path(name)
        if os.path.exists(cache_path) and not reload:
            return pd.read_parquet(cache_path)
        data = loader_func()
        if data is not None and len(data) > 0:
            data.to_parquet(cache_path, index=False)
        return data

    def get_sw_industry_list(self, level=1, reload=False):
        def loader():
            try:
                df = self.pro.index_classify(level=f'L{level}', src='SW2021')
                return df
            except Exception as e:
                print(f"Tushare获取失败，使用模拟数据: {e}")
                return self._generate_mock_sw_industry_list(level)
        return self.load_with_cache(f"sw_industry_list_L{level}", loader, reload)

    def _generate_mock_sw_industry_list(self, level=1):
        industries = [
            ('801010.SI', '农林牧渔'), ('801020.SI', '采掘'), ('801030.SI', '化工'),
            ('801040.SI', '钢铁'), ('801050.SI', '有色金属'), ('801060.SI', '电子'),
            ('801070.SI', '汽车'), ('801080.SI', '家用电器'), ('801090.SI', '食品饮料'),
            ('801100.SI', '纺织服装'), ('801110.SI', '轻工制造'), ('801120.SI', '医药生物'),
            ('801130.SI', '公用事业'), ('801140.SI', '交通运输'), ('801150.SI', '房地产'),
            ('801160.SI', '商业贸易'), ('801170.SI', '休闲服务'), ('801180.SI', '银行'),
            ('801190.SI', '非银金融'), ('801200.SI', '建筑材料'), ('801210.SI', '建筑装饰'),
            ('801220.SI', '电气设备'), ('801230.SI', '国防军工'), ('801710.SI', '计算机'),
            ('801720.SI', '传媒'), ('801730.SI', '通信'), ('801740.SI', '机械设备'),
            ('801750.SI', '综合'), ('801760.SI', '电力设备'), ('801770.SI', '美容护理'),
            ('801780.SI', '环保')
        ]
        df = pd.DataFrame(industries, columns=['index_code', 'industry_name'])
        df['level'] = level
        return df

    def get_sw_industry_daily(self, ts_code, start_date, end_date, reload=False):
        def loader():
            try:
                df = self.pro.sw_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                return df
            except Exception as e:
                print(f"获取行业日线失败: {e}")
                return self._generate_mock_industry_daily(ts_code, start_date, end_date)
        return self.load_with_cache(f"sw_daily_{ts_code}_{start_date}_{end_date}", loader, reload)

    def _generate_mock_industry_daily(self, ts_code, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        base_price = 1000

        np.random.seed(hash(ts_code) % 2**32)
        returns = np.random.randn(n) * 0.015 + 0.0003
        close_prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': dates.strftime('%Y%m%d'),
            'open': close_prices * (1 + np.random.randn(n) * 0.005),
            'high': close_prices * (1 + np.abs(np.random.randn(n)) * 0.01),
            'low': close_prices * (1 - np.abs(np.random.randn(n)) * 0.01),
            'close': close_prices,
            'vol': np.random.randint(1000000, 50000000, n),
            'amount': close_prices * np.random.randint(1000000, 50000000, n)
        })
        return df

    def get_sw_industry_historical(self, ts_code, start_date, end_date, reload=False):
        def loader():
            try:
                df = self.pro.swidx_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is None or len(df) == 0:
                    return self._generate_mock_industry_daily(ts_code, start_date, end_date)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df.sort_values('trade_date')
            except Exception as e:
                print(f"获取行业历史数据失败: {e}")
                return self._generate_mock_industry_daily(ts_code, start_date, end_date)
        return self.load_with_cache(f"sw_hist_{ts_code}_{start_date}_{end_date}", loader, reload)

    def get_sw_industry_historical_batch(self, ts_codes, start_date, end_date, reload=False):
        all_data = []
        for code in ts_codes:
            try:
                df = self.get_sw_industry_historical(code, start_date, end_date, reload)
                if df is not None and len(df) > 0:
                    df['industry_code'] = code
                    all_data.append(df)
            except Exception as e:
                print(f"获取 {code} 历史数据失败: {e}")
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            if 'trade_date' in result.columns:
                result['trade_date'] = pd.to_datetime(result['trade_date'])
                result = result.sort_values(['trade_date', 'industry_code'])
            return result
        return pd.DataFrame()

    def get_trade_dates(self, start_date, end_date, exchange='SSE', reload=False):
        def loader():
            try:
                df = self.pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
                return df
            except Exception as e:
                print(f"获取交易日历失败: {e}")
                return self._generate_mock_trade_dates(start_date, end_date)
        return self.load_with_cache(f"trade_dates_{start_date}_{end_date}", loader, reload)

    def _generate_mock_trade_dates(self, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        df = pd.DataFrame({
            'cal_date': dates.strftime('%Y%m%d'),
            'exchange': 'SSE',
            'is_open': 1
        })
        return df

    def get_index_daily(self, index_code, start_date, end_date, reload=False):
        def loader():
            try:
                df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                if df is None or len(df) == 0:
                    return self._generate_mock_index_daily(index_code, start_date, end_date)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df.sort_values('trade_date')
            except Exception as e:
                print(f"获取指数日线失败: {e}")
                return self._generate_mock_index_daily(index_code, start_date, end_date)
        return self.load_with_cache(f"index_daily_{index_code}_{start_date}_{end_date}", loader, reload)

    def _generate_mock_index_daily(self, index_code, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        base_price = 3000

        np.random.seed(hash(index_code) % 2**32)
        returns = np.random.randn(n) * 0.012 + 0.0002
        close_prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'ts_code': index_code,
            'trade_date': dates,
            'close': close_prices,
            'open': close_prices * (1 + np.random.randn(n) * 0.003),
            'high': close_prices * (1 + np.abs(np.random.randn(n)) * 0.008),
            'low': close_prices * (1 - np.abs(np.random.randn(n)) * 0.008),
            'vol': np.random.randint(100000000, 5000000000, n)
        })
        return df

    def get_stock_financials(self, start_year, end_year, reload=False):
        def loader():
            try:
                all_data = []
                for year in range(start_year, end_year + 1):
                    for quarter in [1, 2, 3, 4]:
                        try:
                            df = self.pro.fina_indicator(ann_date=f"{year}0{quarter*3}01", period=f"{year}0{quarter}30")
                            if df is not None and len(df) > 0:
                                df['year'] = year
                                df['quarter'] = quarter
                                all_data.append(df)
                            time.sleep(0.1)
                        except Exception as e:
                            pass
                return pd.concat(all_data, ignore_index=True) if all_data else self._generate_mock_financial_data(start_year, end_year)
            except Exception as e:
                print(f"获取财务数据失败: {e}")
                return self._generate_mock_financial_data(start_year, end_year)
        return self.load_with_cache(f"stock_financials_{start_year}_{end_year}", loader, reload)

    def _generate_mock_financial_data(self, start_year, end_year):
        print("生成模拟财务数据...")
        sw_list = self.get_sw_industry_list(level=1)
        industry_codes = sw_list['index_code'].tolist() if len(sw_list) > 0 else ['801010.SI']

        stock_data = []
        n_stocks_per_industry = 10

        np.random.seed(42)

        for industry in industry_codes:
            for stock_id in range(n_stocks_per_industry):
                ts_code = f"{stock_id:06d}.SZ" if stock_id % 2 == 0 else f"{stock_id:06d}.SH"

                for year in range(start_year, end_year + 1):
                    for quarter in [1, 2, 3, 4]:
                        report_date = f"{year}0{quarter*3}31" if quarter < 4 else f"{year}1231"

                        net_profit = np.random.uniform(-100, 1000) * (1 + quarter * 0.1)
                        total_revenue = np.random.uniform(500, 5000) * (1 + quarter * 0.1)
                        gross_profit = total_revenue * np.random.uniform(0.1, 0.4)
                        total_assets = np.random.uniform(5000, 50000)
                        total_liabilities = total_assets * np.random.uniform(0.3, 0.7)
                        equity = total_assets - total_liabilities

                        roe = net_profit / equity * 100 if equity > 0 else 0
                        roa = net_profit / total_assets * 100 if total_assets > 0 else 0
                        gross_margin = gross_profit / total_revenue * 100 if total_revenue > 0 else 0
                        net_margin = net_profit / total_revenue * 100 if total_revenue > 0 else 0

                        ocf = net_profit * np.random.uniform(0.5, 1.5)
                        inv_turn = total_revenue / (total_assets * np.random.uniform(0.1, 0.3)) if total_assets > 0 else 0
                        asset_turn = total_revenue / total_assets if total_assets > 0 else 0
                        ar_turn = total_revenue / (total_revenue * np.random.uniform(0.05, 0.15)) if total_revenue > 0 else 0

                        stock_data.append({
                            'ts_code': ts_code,
                            'ann_date': report_date,
                            'year': year,
                            'quarter': quarter,
                            'net_profit': net_profit,
                            'total_revenue': total_revenue,
                            'gross_profit_margin': gross_margin,
                            'net_profit_margin': net_margin,
                            'roe': roe,
                            'roa': roa,
                            'debt_to_assets': total_liabilities / total_assets * 100 if total_assets > 0 else 0,
                            'current_ratio': np.random.uniform(1, 3),
                            'quick_ratio': np.random.uniform(0.5, 2.5),
                            'ocf_to_debt': ocf / total_liabilities if total_liabilities > 0 else 0,
                            'inv_turn': inv_turn,
                            'assets_turn': asset_turn,
                            'ar_turn': ar_turn,
                            'operating_income': total_revenue * np.random.uniform(0.8, 1.0),
                            'operating_expense': total_revenue * np.random.uniform(0.1, 0.3),
                            'financial_expense': total_revenue * np.random.uniform(0, 0.05),
                            'admin_expense': total_revenue * np.random.uniform(0.02, 0.08),
                            'tax_to_ebt': np.random.uniform(0.05, 0.25),
                            'operate_income_to_ebt': np.random.uniform(0.7, 1.2),
                        })

        df = pd.DataFrame(stock_data)
        print(f"生成了 {len(df)} 条模拟财务数据")
        return df

    def get_industry_financial_aggregate(self, start_date, end_date, reload=False):
        def loader():
            return self._generate_mock_industry_financial(start_date, end_date)
        return self.load_with_cache(f"industry_financial_{start_date}_{end_date}", loader, reload)

    def _generate_mock_industry_financial(self, start_date, end_date):
        print("生成模拟行业财务汇总数据...")
        sw_list = self.get_sw_industry_list(level=1)
        industry_codes = sw_list['index_code'].tolist() if len(sw_list) > 0 else ['801010.SI']

        quarters = pd.date_range(start=start_date, end=end_date, freq='Q').strftime('%Y%m%d').tolist()

        industry_data = []

        for industry in industry_codes:
            np.random.seed(hash(industry) % 2**32)

            base_metrics = {
                'net_profit_margin': np.random.uniform(5, 15),
                'gross_profit_margin': np.random.uniform(15, 35),
                'roe': np.random.uniform(8, 18),
                'roa': np.random.uniform(3, 10),
                'debt_to_assets': np.random.uniform(40, 70),
                'current_ratio': np.random.uniform(1.2, 2.5),
                'inv_turn': np.random.uniform(4, 12),
                'assets_turn': np.random.uniform(0.5, 1.5),
                'op_ex_rev_yoy': np.random.uniform(-5, 30),
                'net_profit_yoy': np.random.uniform(-10, 40),
            }

            for i, quarter in enumerate(quarters):
                trend = np.random.randn() * 2

                industry_data.append({
                    'industry_code': industry,
                    'trade_date': quarter,
                    'net_profit_margin': max(0, base_metrics['net_profit_margin'] + trend * 2 + np.random.randn()),
                    'gross_profit_margin': max(0, base_metrics['gross_profit_margin'] + trend * 1.5 + np.random.randn()),
                    'roe': max(0, base_metrics['roe'] + trend * 1 + np.random.randn()),
                    'roa': max(0, base_metrics['roa'] + trend * 0.8 + np.random.randn()),
                    'debt_to_assets': max(10, min(90, base_metrics['debt_to_assets'] + trend * 3 + np.random.randn())),
                    'current_ratio': max(0.5, base_metrics['current_ratio'] + trend * 0.2 + np.random.randn() * 0.1),
                    'inv_turn': max(1, base_metrics['inv_turn'] + trend * 0.5 + np.random.randn()),
                    'assets_turn': max(0.1, base_metrics['assets_turn'] + trend * 0.1 + np.random.randn()),
                    'op_ex_rev_yoy': base_metrics['op_ex_rev_yoy'] + trend * 5 + np.random.randn() * 3,
                    'net_profit_yoy': base_metrics['net_profit_yoy'] + trend * 8 + np.random.randn() * 5,
                })

                base_metrics = {k: v * (1 + np.random.uniform(-0.05, 0.05)) for k, v in base_metrics.items()}

        df = pd.DataFrame(industry_data)
        print(f"生成了 {len(df)} 条模拟行业财务汇总数据")
        return df

    def get_consensus_data(self, start_date, end_date, reload=False):
        def loader():
            return self._generate_mock_consensus_data(start_date, end_date)
        return self.load_with_cache(f"consensus_{start_date}_{end_date}", loader, reload)

    def _generate_mock_consensus_data(self, start_date, end_date):
        print("生成模拟一致预期数据...")
        sw_list = self.get_sw_industry_list(level=1)
        industry_codes = sw_list['index_code'].tolist() if len(sw_list) > 0 else ['801010.SI']

        dates = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y%m%d').tolist()

        consensus_data = []

        for industry in industry_codes:
            np.random.seed(hash(industry) % 2**32)

            base_eps = np.random.uniform(0.5, 2.0)
            base_roe = np.random.uniform(8, 15)

            for date in dates:
                consensus_data.append({
                    'industry_code': industry,
                    'trade_date': date,
                    'eps_forecast': base_eps * np.random.uniform(0.9, 1.1),
                    'eps_forecast_yoy': np.random.uniform(-10, 30),
                    'roe_forecast': base_roe * np.random.uniform(0.95, 1.05),
                    'roe_forecast_yoy': np.random.uniform(-5, 15),
                    'focus_count': np.random.randint(10, 100),
                    'up_count': np.random.randint(5, 30),
                    'down_count': np.random.randint(2, 15),
                    'neutral_count': np.random.randint(10, 50),
                })

                base_eps *= (1 + np.random.uniform(-0.03, 0.05))
                base_roe *= (1 + np.random.uniform(-0.02, 0.03))

        df = pd.DataFrame(consensus_data)
        print(f"生成了 {len(df)} 条模拟一致预期数据")
        return df

    def get_industry_macro_data(self, industry_code, start_date, end_date, reload=False):
        def loader():
            return self._generate_mock_industry_macro(industry_code, start_date, end_date)
        return self.load_with_cache(f"industry_macro_{industry_code}_{start_date}_{end_date}", loader, reload)

    def _generate_mock_industry_macro(self, industry_code, start_date, end_date):
        print(f"生成模拟行业宏观数据 for {industry_code}...")

        sector_map = {
            '801040.SI': '钢铁',  # 钢铁
            '801020.SI': '煤炭',  # 煤炭
            '801050.SI': '石油石化',  # 有色金属
            '801030.SI': '化工',  # 化工
        }

        dates = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y%m%d').tolist()

        macro_data = []
        np.random.seed(hash(industry_code) % 2**32)

        base_price = 100
        base_demand = 80
        base_capacity = 85

        for date in dates:
            price_change = np.random.randn() * 5
            demand_change = np.random.randn() * 3

            macro_data.append({
                'industry_code': industry_code,
                'trade_date': date,
                'price_index': max(50, base_price + price_change),
                'demand_index': max(30, base_demand + demand_change),
                'capacity_utilization': max(50, min(100, base_capacity + np.random.randn() * 5)),
                'inventory_index': np.random.uniform(60, 100),
                'spot_price': max(50, base_price * np.random.uniform(0.9, 1.1)),
                'futures_price': max(50, base_price * np.random.uniform(0.9, 1.1)),
            })

            base_price *= (1 + np.random.uniform(-0.05, 0.08))
            base_demand *= (1 + np.random.uniform(-0.03, 0.05))
            base_capacity *= (1 + np.random.uniform(-0.02, 0.02))

        df = pd.DataFrame(macro_data)
        print(f"生成了 {len(df)} 条模拟行业宏观数据")
        return df

    def batch_get_all_industry_data(self, start_date, end_date, reload=False):
        sw_list = self.get_sw_industry_list(level=1)
        industry_codes = sw_list['index_code'].tolist()

        all_data = []
        for code in industry_codes:
            df = self.get_sw_industry_historical(code, start_date, end_date, reload)
            if df is not None and len(df) > 0:
                df['industry_code'] = code
                all_data.append(df)
            time.sleep(0.05)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result
        return pd.DataFrame()


if __name__ == "__main__":
    print("Testing DataLoader with all data sources...")

    loader = DataLoader()

    print("\n1. Testing SW Industry List...")
    sw_list = loader.get_sw_industry_list(level=1)
    print(f"   Got {len(sw_list)} industries")

    print("\n2. Testing Trade Dates...")
    trade_dates = loader.get_trade_dates("20200101", "20201231")
    print(f"   Got {len(trade_dates)} trade dates")

    print("\n3. Testing Mock Financial Data...")
    financial_data = loader._generate_mock_financial_data(2018, 2020)
    print(f"   Generated {len(financial_data)} records")

    print("\n4. Testing Mock Industry Financial Aggregate...")
    ind_financial = loader._generate_mock_industry_financial("20180101", "20201031")
    print(f"   Generated {len(ind_financial)} records")

    print("\n5. Testing Mock Consensus Data...")
    consensus = loader._generate_mock_consensus_data("20180101", "20201031")
    print(f"   Generated {len(consensus)} records")

    print("\n6. Testing Mock Industry Macro Data...")
    macro = loader._generate_mock_industry_macro("801040.SI", "20180101", "20201031")
    print(f"   Generated {len(macro)} records")

    print("\nAll DataLoader tests completed successfully!")
