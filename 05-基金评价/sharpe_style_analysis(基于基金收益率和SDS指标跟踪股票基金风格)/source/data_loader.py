# -*- coding: utf-8 -*-
"""
数据获取模块 - 威廉·夏普风格分析
支持获取基金净值、风格指数收益率数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class StyleDataLoader:
    """
    风格分析数据加载器
    
    支持的数据源优先级:
    1. efinance - 基金净值、指数行情
    2. akshare - A股指数、基金数据
    3. baostock - 历史行情数据
    """
    
    # 研报中提到的主要风格指数映射
    STYLE_INDICES = {
        # 中证指数
        '沪深300': '000300.SH',
        '中证500': '000905.SH',
        '中证800': '000906.SH',
        '沪深300成长': '000918.SH',
        '沪深300价值': '000919.SH',
        '中证500成长': '000920.SH',
        '中证500价值': '000921.SH',
        # 规模指数
        '大盘': '000044.SH',  # 上证超级大盘
        '中盘': '000045.SH',  # 上证中盘
        '小盘': '000046.SH',  # 上证小盘
        # 风格指数
        '成长': '000901.SH',  # 中证100成长
        '价值': '000902.SH',  # 中证100价值
    }
    
    def __init__(self):
        self.data_cache = {}
        
    def get_fund_nav(self, fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取基金净值数据
        
        Parameters:
        -----------
        fund_code : str
            基金代码，如 '021181'
        start_date : str
            开始日期，格式 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYYMMDD'
            
        Returns:
        --------
        pd.DataFrame
            包含日期、单位净值、累计净值、日收益率的DataFrame
        """
        # 尝试 efinance
        try:
            import efinance as ef
            df = ef.fund.get_fund_history(fund_code, start_date, end_date)
            if df is not None and len(df) > 0:
                df = self._process_fund_nav(df)
                print(f"[OK] efinance 获取基金 {fund_code} 净值数据: {len(df)} 条")
                return df
        except Exception as e:
            print(f"[WARN] efinance 获取基金净值失败: {e}")
        
        # 降级到 akshare
        try:
            import akshare as ak
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            if df is not None and len(df) > 0:
                df['日期'] = pd.to_datetime(df['净值日期'])
                df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
                df['nav'] = df['单位净值'].astype(float)
                df = df.sort_values('日期').reset_index(drop=True)
                df['daily_return'] = df['nav'].pct_change()
                df = df[['日期', 'nav', 'daily_return']].dropna()
                print(f"[OK] akshare 获取基金 {fund_code} 净值数据: {len(df)} 条")
                return df
        except Exception as e:
            print(f"[WARN] akshare 获取基金净值失败: {e}")
        
        raise ValueError(f"无法获取基金 {fund_code} 的净值数据")
    
    def _process_fund_nav(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理基金净值数据"""
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        
        # 计算日收益率
        if '单位净值' in df.columns:
            df['nav'] = df['单位净值'].astype(float)
        elif '净值' in df.columns:
            df['nav'] = df['净值'].astype(float)
        
        df['daily_return'] = df['nav'].pct_change()
        df = df[['日期', 'nav', 'daily_return']].dropna()
        return df
    
    def get_index_returns(self, index_codes: List[str], start_date: str, 
                          end_date: str) -> pd.DataFrame:
        """
        获取多个风格指数的日收益率数据
        
        Parameters:
        -----------
        index_codes : List[str]
            指数代码列表，如 ['000300', '000905']
        start_date : str
            开始日期 'YYYYMMDD'
        end_date : str
            结束日期 'YYYYMMDD'
            
        Returns:
        --------
        pd.DataFrame
            列名为指数代码，值为日收益率的DataFrame
        """
        all_returns = {}
        
        for code in index_codes:
            try:
                returns = self._get_single_index(code, start_date, end_date)
                if returns is not None and len(returns) > 0:
                    all_returns[code] = returns
            except Exception as e:
                print(f"[WARN] 获取指数 {code} 数据失败: {e}")
        
        if not all_returns:
            raise ValueError("无法获取任何指数数据")
        
        # 合并所有指数收益率
        df = pd.DataFrame(all_returns)
        df.index.name = '日期'
        df = df.dropna(how='all')
        
        print(f"[OK] 获取 {len(all_returns)} 个风格指数数据，共 {len(df)} 个交易日")
        return df
    
    def _get_single_index(self, index_code: str, start_date: str, 
                          end_date: str) -> pd.Series:
        """获取单个指数数据"""
        # 尝试 akshare
        try:
            import akshare as ak
            # 处理指数代码格式
            if '.' not in index_code:
                if index_code.startswith('0') or index_code.startswith('3'):
                    index_code_full = f"sh{index_code}"
                else:
                    index_code_full = f"sz{index_code}"
            else:
                index_code_full = index_code.replace('.SH', '').replace('.SZ', '')
                if index_code.endswith('.SH'):
                    index_code_full = f"sh{index_code_full}"
                else:
                    index_code_full = f"sz{index_code_full}"
            
            df = ak.index_zh_a_hist(symbol=index_code_full, period="daily",
                                    start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values('日期')
                df['return'] = df['收盘'].pct_change()
                returns = df.set_index('日期')['return'].dropna()
                return returns
        except Exception as e:
            pass
        
        # 尝试 baostock
        try:
            import baostock as bs
            lg = bs.login()
            if lg.error_code == '0':
                rs = bs.query_history_k_data_plus(
                    index_code,
                    "date,close",
                    start_date=start_date, end_date=end_date,
                    frequency="d"
                )
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                bs.logout()
                
                if data_list:
                    df = pd.DataFrame(data_list, columns=['date', 'close'])
                    df['date'] = pd.to_datetime(df['date'])
                    df['close'] = df['close'].astype(float)
                    df = df.sort_values('date')
                    df['return'] = df['close'].pct_change()
                    returns = df.set_index('date')['return'].dropna()
                    return returns
        except Exception as e:
            pass
        
        return None
    
    def get_style_index_mapping(self) -> Dict[str, str]:
        """获取风格指数映射表"""
        return self.STYLE_INDICES.copy()
    
    def create_mock_data(self, fund_code: str = '021181', 
                         periods: int = 252,
                         style_indices: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建模拟数据用于测试和演示
        
        Parameters:
        -----------
        fund_code : str
            模拟基金代码
        periods : int
            模拟交易日数量（默认252个交易日≈1年）
        style_indices : List[str]
            风格指数代码列表
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (基金净值DataFrame, 风格指数收益率DataFrame)
        """
        if style_indices is None:
            style_indices = ['000300.SH', '000905.SH', '000906.SH', 
                            '000918.SH', '000919.SH']
        
        # 生成日期序列
        end_date = datetime.now()
        dates = pd.bdate_range(end=end_date, periods=periods)
        
        # 模拟基金净值（随机游走）
        np.random.seed(42)
        fund_returns = np.random.normal(0.0003, 0.015, periods)
        fund_nav = 1.0 * np.exp(np.cumsum(fund_returns))
        
        fund_df = pd.DataFrame({
            '日期': dates,
            'nav': fund_nav,
            'daily_return': fund_returns
        })
        
        # 模拟风格指数收益率
        index_returns = {}
        for idx in style_indices:
            # 不同指数有不同的收益特征
            base_return = 0.0002
            volatility = 0.012 + np.random.rand() * 0.005
            corr_with_fund = 0.6 + np.random.rand() * 0.3
            
            noise = np.random.normal(0, volatility, periods)
            idx_returns = corr_with_fund * fund_returns + (1 - corr_with_fund) * noise
            index_returns[idx] = idx_returns
        
        index_df = pd.DataFrame(index_returns, index=dates)
        index_df.index.name = '日期'
        
        print(f"[OK] 创建模拟数据: 基金 {periods} 个交易日, {len(style_indices)} 个风格指数")
        return fund_df, index_df
