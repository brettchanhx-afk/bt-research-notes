"""
因子构建模块

实现Fama-French五因子的构建：
- R_M: 市场因子
- SMB: 市值因子（Small Minus Big）
- HML: 价值因子（High Minus Low）
- RMW: 盈利水平因子（Robust Minus Weak）
- CMA: 投资水平因子（Conservative Minus Aggressive）
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FamaFrenchFactorBuilder:
    """
    Fama-French五因子构建器
    
    基于A股全市场股票数据构建五因子
    """
    
    def __init__(self):
        self.factors = None
        self.stock_data = None
    
    def build_five_factors(
        self,
        start_date: str,
        end_date: str,
        data_loader=None
    ) -> pd.DataFrame:
        """
        构建Fama-French五因子
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        data_loader : FactorDataLoader, optional
            数据加载器
            
        Returns
        -------
        pd.DataFrame
            五因子数据框，包含R_M、SMB、HML、RMW、CMA
        """
        print("[Step 1] 构建Fama-French五因子...")
        
        # 获取股票数据
        if data_loader is None:
            from data_loader import FactorDataLoader
            data_loader = FactorDataLoader()
        
        self.stock_data = data_loader.get_stock_data_for_factor_building(
            start_date, end_date
        )
        
        # 构建各因子
        smb = self._build_smb_factor()
        hml = self._build_hml_factor()
        rmw = self._build_rmw_factor()
        cma = self._build_cma_factor()
        
        # 获取市场因子
        market_return = data_loader.get_market_return(start_date, end_date, 'monthly')
        rf = data_loader.get_risk_free_rate(start_date, end_date, 'monthly')
        
        # 计算市场超额收益
        r_m = market_return - rf.values
        
        # 合并所有因子
        factors_df = pd.DataFrame({
            'R_M': r_m,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'CMA': cma,
        })
        
        # 对齐日期
        factors_df = factors_df.dropna()
        
        self.factors = factors_df
        
        print(f"  因子数据区间: {factors_df.index[0]} ~ {factors_df.index[-1]}")
        print(f"  因子数据量: {len(factors_df)} 期")
        
        return factors_df
    
    def _build_smb_factor(self) -> pd.Series:
        """
        构建市值因子 SMB (Small Minus Big)
        
        小市值股票组合收益率 - 大市值股票组合收益率
        
        Returns
        -------
        pd.Series
            SMB因子时间序列
        """
        print("  构建SMB因子（市值因子）...")
        
        df = self.stock_data.copy()
        
        # 按日期分组
        smb_series = []
        dates = []
        
        for date, group in df.groupby('date'):
            if len(group) < 10:
                continue
            
            # 按市值分组（中位数划分）
            median_cap = group['market_cap'].median()
            small_stocks = group[group['market_cap'] <= median_cap]
            big_stocks = group[group['market_cap'] > median_cap]
            
            # 计算组合收益率
            small_return = small_stocks['return'].mean()
            big_return = big_stocks['return'].mean()
            
            # SMB = 小市值 - 大市值
            smb = small_return - big_return
            
            smb_series.append(smb)
            dates.append(date)
        
        return pd.Series(smb_series, index=pd.to_datetime(dates))
    
    def _build_hml_factor(self) -> pd.Series:
        """
        构建价值因子 HML (High Minus Low)
        
        高账面市值比组合收益率 - 低账面市值比组合收益率
        
        Returns
        -------
        pd.Series
            HML因子时间序列
        """
        print("  构建HML因子（价值因子）...")
        
        df = self.stock_data.copy()
        
        hml_series = []
        dates = []
        
        for date, group in df.groupby('date'):
            if len(group) < 10:
                continue
            
            # 按账面市值比分组（30%-70%分位数）
            low_threshold = group['bm_ratio'].quantile(0.3)
            high_threshold = group['bm_ratio'].quantile(0.7)
            
            low_stocks = group[group['bm_ratio'] <= low_threshold]
            high_stocks = group[group['bm_ratio'] >= high_threshold]
            
            # 计算组合收益率
            low_return = low_stocks['return'].mean()
            high_return = high_stocks['return'].mean()
            
            # HML = 高B/M - 低B/M
            hml = high_return - low_return
            
            hml_series.append(hml)
            dates.append(date)
        
        return pd.Series(hml_series, index=pd.to_datetime(dates))
    
    def _build_rmw_factor(self) -> pd.Series:
        """
        构建盈利水平因子 RMW (Robust Minus Weak)
        
        高盈利组合收益率 - 低盈利组合收益率
        
        Returns
        -------
        pd.Series
            RMW因子时间序列
        """
        print("  构建RMW因子（盈利因子）...")
        
        df = self.stock_data.copy()
        
        rmw_series = []
        dates = []
        
        for date, group in df.groupby('date'):
            if len(group) < 10:
                continue
            
            # 按ROE分组（30%-70%分位数）
            weak_threshold = group['roe'].quantile(0.3)
            robust_threshold = group['roe'].quantile(0.7)
            
            weak_stocks = group[group['roe'] <= weak_threshold]
            robust_stocks = group[group['roe'] >= robust_threshold]
            
            # 计算组合收益率
            weak_return = weak_stocks['return'].mean()
            robust_return = robust_stocks['return'].mean()
            
            # RMW = 高盈利 - 低盈利
            rmw = robust_return - weak_return
            
            rmw_series.append(rmw)
            dates.append(date)
        
        return pd.Series(rmw_series, index=pd.to_datetime(dates))
    
    def _build_cma_factor(self) -> pd.Series:
        """
        构建投资水平因子 CMA (Conservative Minus Aggressive)
        
        低投资水平组合收益率 - 高投资水平组合收益率
        
        Returns
        -------
        pd.Series
            CMA因子时间序列
        """
        print("  构建CMA因子（投资因子）...")
        
        df = self.stock_data.copy()
        
        cma_series = []
        dates = []
        
        for date, group in df.groupby('date'):
            if len(group) < 10:
                continue
            
            # 按资产增长率分组（30%-70%分位数）
            cons_threshold = group['asset_growth'].quantile(0.3)
            agg_threshold = group['asset_growth'].quantile(0.7)
            
            # 注意：低投资水平 = 保守，高投资水平 = 激进
            cons_stocks = group[group['asset_growth'] <= cons_threshold]
            agg_stocks = group[group['asset_growth'] >= agg_threshold]
            
            # 计算组合收益率
            cons_return = cons_stocks['return'].mean()
            agg_return = agg_stocks['return'].mean()
            
            # CMA = 保守 - 激进
            cma = cons_return - agg_return
            
            cma_series.append(cma)
            dates.append(date)
        
        return pd.Series(cma_series, index=pd.to_datetime(dates))
    
    def get_factor_statistics(self) -> pd.DataFrame:
        """
        获取因子统计信息
        
        Returns
        -------
        pd.DataFrame
            因子统计信息
        """
        if self.factors is None:
            raise ValueError("请先调用build_five_factors()构建因子")
        
        stats = []
        for factor in ['R_M', 'SMB', 'HML', 'RMW', 'CMA']:
            if factor in self.factors.columns:
                series = self.factors[factor]
                stats.append({
                    '因子': factor,
                    '均值': series.mean(),
                    '标准差': series.std(),
                    '最小值': series.min(),
                    '最大值': series.max(),
                    '年化收益率': series.mean() * 12,
                    '年化波动率': series.std() * np.sqrt(12),
                    '夏普比率': series.mean() / series.std() * np.sqrt(12) if series.std() > 0 else 0,
                })
        
        return pd.DataFrame(stats)
    
    def get_factor_correlation(self) -> pd.DataFrame:
        """
        获取因子相关性矩阵
        
        Returns
        -------
        pd.DataFrame
            因子相关性矩阵
        """
        if self.factors is None:
            raise ValueError("请先调用build_five_factors()构建因子")
        
        return self.factors.corr()


if __name__ == '__main__':
    # 测试
    print("Testing FamaFrenchFactorBuilder...")
    
    builder = FamaFrenchFactorBuilder()
    factors = builder.build_five_factors('2022-01-01', '2024-12-31')
    
    print("\n因子数据预览:")
    print(factors.head(10))
    
    print("\n因子统计信息:")
    stats = builder.get_factor_statistics()
    print(stats)
    
    print("\n因子相关性矩阵:")
    corr = builder.get_factor_correlation()
    print(corr)
