"""
数据获取模块

提供基金净值数据、市场数据、无风险利率等数据的获取功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FundDataLoader:
    """
    基金数据加载器
    
    用于获取基金净值数据并计算收益率
    """
    
    def __init__(self):
        self.data_cache = {}
    
    def get_fund_nav(
        self,
        fund_code: str,
        start_date: str,
        end_date: str,
        frequency: str = 'monthly'
    ) -> pd.DataFrame:
        """
        获取基金净值数据
        
        Parameters
        ----------
        fund_code : str
            基金代码（如：'019888'）
        start_date : str
            开始日期（格式：'YYYY-MM-DD'）
        end_date : str
            结束日期（格式：'YYYY-MM-DD'）
        frequency : str
            数据频率：'daily'（日度）或 'monthly'（月度）
            
        Returns
        -------
        pd.DataFrame
            包含日期、单位净值、累计净值、收益率的DataFrame
        """
        # 尝试使用efinance获取数据
        try:
            import efinance as ef
            
            # 获取基金净值数据
            df = ef.fund.get_fund_history(fund_code)
            
            if df is None or len(df) == 0:
                raise ValueError(f"无法获取基金 {fund_code} 的数据")
            
            # 处理日期列
            df['日期'] = pd.to_datetime(df['日期'])
            df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
            df = df.sort_values('日期')
            
            # 重命名列
            df = df.rename(columns={
                '日期': 'date',
                '单位净值': 'nav',
                '累计净值': 'acc_nav',
            })
            
            # 计算日收益率
            df['daily_return'] = df['nav'].pct_change()
            
        except Exception as e:
            print(f"[efinance] 获取基金数据失败: {e}")
            print("使用模拟数据...")
            df = self._generate_mock_fund_data(start_date, end_date, frequency)
        
        # 根据频率调整数据
        if frequency == 'monthly':
            df = self._resample_to_monthly(df)
        
        return df
    
    def _resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将日度数据重采样为月度数据
        
        Parameters
        ----------
        df : pd.DataFrame
            日度数据
            
        Returns
        -------
        pd.DataFrame
            月度数据
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 取每月最后一个交易日的数据
        monthly_df = df.resample('M').last()
        
        # 计算月度收益率
        monthly_df['monthly_return'] = monthly_df['nav'].pct_change()
        
        monthly_df = monthly_df.reset_index()
        monthly_df['date'] = monthly_df['date'].dt.to_period('M').dt.to_timestamp()
        
        return monthly_df
    
    def _generate_mock_fund_data(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'monthly'
    ) -> pd.DataFrame:
        """
        生成模拟基金数据（用于测试）
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        frequency : str
            数据频率
            
        Returns
        -------
        pd.DataFrame
            模拟基金数据
        """
        # 生成日期序列
        if frequency == 'monthly':
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 生成模拟净值（带趋势的随机游走）
        np.random.seed(42)
        n = len(dates)
        
        # 基础收益率（年化8%，月度约0.65%）
        base_return = 0.0065
        
        # 生成收益率序列（带自相关）
        returns = np.random.normal(base_return, 0.04, n)
        
        # 计算累计净值
        nav = 1.0
        navs = [nav]
        for ret in returns[1:]:
            nav *= (1 + ret)
            navs.append(nav)
        
        df = pd.DataFrame({
            'date': dates,
            'nav': navs,
            'acc_nav': navs,
        })
        
        if frequency == 'monthly':
            df['monthly_return'] = df['nav'].pct_change()
        else:
            df['daily_return'] = df['nav'].pct_change()
        
        return df


class FactorDataLoader:
    """
    因子数据加载器
    
    用于获取市场因子、无风险利率等数据
    """
    
    def __init__(self):
        self.data_cache = {}
    
    def get_risk_free_rate(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'monthly'
    ) -> pd.Series:
        """
        获取无风险利率（中债国债到期收益率1年期）
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        frequency : str
            数据频率
            
        Returns
        -------
        pd.Series
            无风险利率序列（日度或月度）
        """
        try:
            import akshare as ak
            
            # 获取中债国债收益率
            df = ak.bond_china_yield(start_date=start_date, end_date=end_date)
            
            # 提取1年期国债收益率
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            
            # 假设列名为'国债收益率1年'
            if '国债收益率1年' in df.columns:
                rf_series = df['国债收益率1年'] / 100  # 转换为小数
            else:
                # 使用第一列作为无风险利率
                rf_series = df.iloc[:, 0] / 100
            
        except Exception as e:
            print(f"[akshare] 获取无风险利率失败: {e}")
            print("使用固定无风险利率2%...")
            
            # 生成模拟数据
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            rf_series = pd.Series(0.02, index=dates)
        
        # 转换为月度数据
        if frequency == 'monthly':
            rf_series = rf_series.resample('M').last()
        
        return rf_series
    
    def get_market_return(
        self,
        start_date: str,
        end_date: str,
        frequency: str = 'monthly',
        index_code: str = '000906'  # 中证800
    ) -> pd.Series:
        """
        获取市场指数收益率
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        frequency : str
            数据频率
        index_code : str
            指数代码（默认中证800）
            
        Returns
        -------
        pd.Series
            市场指数收益率序列
        """
        try:
            import akshare as ak
            
            # 获取指数历史数据
            df = ak.index_zh_a_hist(symbol=index_code, period="daily", 
                                    start_date=start_date, end_date=end_date)
            
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df.sort_index()
            
            # 计算日收益率
            df['daily_return'] = df['收盘'].pct_change()
            
            if frequency == 'monthly':
                # 重采样为月度收益率
                monthly_returns = df['daily_return'].resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                )
                return monthly_returns
            else:
                return df['daily_return']
                
        except Exception as e:
            print(f"[akshare] 获取市场指数数据失败: {e}")
            print("使用模拟数据...")
            
            # 生成模拟市场收益率
            if frequency == 'monthly':
                dates = pd.date_range(start=start_date, end=end_date, freq='M')
            else:
                dates = pd.date_range(start=start_date, end=end_date, freq='B')
            
            np.random.seed(123)
            # 市场年化收益率约10%，波动率约15%
            monthly_return = 0.10 / 12
            monthly_vol = 0.15 / np.sqrt(12)
            
            returns = np.random.normal(monthly_return, monthly_vol, len(dates))
            
            return pd.Series(returns, index=dates)
    
    def get_stock_data_for_factor_building(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取构建Fama-French因子所需的股票数据
        
        包括：市值、账面市值比、盈利能力、投资水平等
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        pd.DataFrame
            股票数据，包含构建因子所需的字段
        """
        try:
            import akshare as ak
            
            # 获取A股全市场股票列表
            stock_list = ak.stock_zh_a_spot_em()
            
            # 限制股票数量（避免数据量过大）
            stock_codes = stock_list['代码'].head(500).tolist()
            
            all_data = []
            
            for code in stock_codes[:50]:  # 限制数量以加快速度
                try:
                    # 获取个股历史数据
                    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                           start_date=start_date, end_date=end_date)
                    
                    if df is not None and len(df) > 0:
                        df['股票代码'] = code
                        all_data.append(df)
                except:
                    continue
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return combined_df
            else:
                raise ValueError("无法获取股票数据")
                
        except Exception as e:
            print(f"[akshare] 获取股票数据失败: {e}")
            print("使用模拟数据...")
            return self._generate_mock_stock_data(start_date, end_date)
    
    def _generate_mock_stock_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        生成模拟股票数据（用于测试）
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        pd.DataFrame
            模拟股票数据
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        n_stocks = 100
        
        np.random.seed(456)
        
        data = []
        for i in range(n_stocks):
            stock_code = f"{600000 + i:06d}"
            
            # 生成市值（对数正态分布）
            market_cap = np.exp(np.random.normal(20, 1.5))
            
            # 生成账面市值比
            bm_ratio = np.random.normal(0.5, 0.3)
            
            # 生成盈利能力（ROE）
            roe = np.random.normal(0.10, 0.05)
            
            # 生成投资水平（资产增长率）
            asset_growth = np.random.normal(0.15, 0.10)
            
            # 生成收益率
            returns = np.random.normal(0.005, 0.08, len(dates))
            
            for j, date in enumerate(dates):
                data.append({
                    'date': date,
                    'stock_code': stock_code,
                    'market_cap': market_cap * (1 + np.random.normal(0, 0.02)),
                    'bm_ratio': bm_ratio,
                    'roe': roe,
                    'asset_growth': asset_growth,
                    'return': returns[j],
                })
        
        return pd.DataFrame(data)


if __name__ == '__main__':
    # 测试
    print("Testing FundDataLoader...")
    loader = FundDataLoader()
    
    fund_data = loader.get_fund_nav('019888', '2022-01-01', '2024-12-31', 'monthly')
    print(f"Fund data shape: {fund_data.shape}")
    print(fund_data.head())
    
    print("\nTesting FactorDataLoader...")
    factor_loader = FactorDataLoader()
    
    rf = factor_loader.get_risk_free_rate('2022-01-01', '2024-12-31', 'monthly')
    print(f"Risk-free rate shape: {rf.shape}")
    print(rf.head())
