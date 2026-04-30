# -*- coding: utf-8 -*-
"""
基金评分模块

五维评分体系：
1. 年化收益率
2. 逆境战胜市场胜率
3. H-M模型择时
4. 基金份额
5. 下行风险
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 五维评分模型
# ============================================================
class FundScorer:
    """基金评分模型"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Parameters
        ----------
        weights : dict
            因子权重，默认使用研报推荐权重
        """
        self.weights = weights or {
            '年化收益率': 0.20,
            '逆境战胜市场胜率': 0.25,
            'H-M模型择时': 0.20,
            '基金份额': 0.15,
            '下行风险': 0.20,  # 负向因子
        }
        
        # 因子方向
        self.directions = {
            '年化收益率': 1,              # 正向
            '逆境战胜市场胜率': 1,        # 正向
            'H-M模型择时': 1,            # 正向
            '基金份额': 1,               # 正向（小规模效应，但实际是越小越好，这里取反向）
            '下行风险': -1,              # 负向（越小越好）
        }
    
    def calculate_scores(
        self,
        factor_df: pd.DataFrame,
        sector: str = '全市场'
    ) -> pd.DataFrame:
        """
        计算基金评分
        
        Parameters
        ----------
        factor_df : pd.DataFrame
            因子数据，index=fund_code, columns=factor_name
        sector : str
            板块名称
            
        Returns
        -------
        pd.DataFrame
            评分结果
        """
        factor_names = list(self.weights.keys())
        
        # 检查因子是否存在
        missing_factors = [f for f in factor_names if f not in factor_df.columns]
        if missing_factors:
            print(f'[WARNING] 缺失因子: {missing_factors}')
            factor_names = [f for f in factor_names if f in factor_df.columns]
        
        if len(factor_names) == 0:
            return pd.DataFrame()
        
        # 计算各因子排名得分（0-100）
        score_df = pd.DataFrame(index=factor_df.index)
        
        for factor in factor_names:
            values = factor_df[factor].dropna()
            
            if len(values) < 5:
                continue
            
            # 排名
            if self.directions.get(factor, 1) > 0:
                # 正向因子：值越大，得分越高
                ranks = values.rank(pct=True) * 100
            else:
                # 负向因子：值越小，得分越高
                ranks = (1 - values.rank(pct=True)) * 100
            
            score_df[f'{factor}_score'] = ranks
        
        # 计算加权总分
        total_score = pd.Series(0.0, index=factor_df.index)
        
        for factor in factor_names:
            if f'{factor}_score' in score_df.columns:
                weight = self.weights.get(factor, 0)
                total_score += score_df[f'{factor}_score'].fillna(50) * weight
        
        score_df['综合得分'] = total_score
        
        # 添加原始因子值
        for factor in factor_names:
            if factor in factor_df.columns:
                score_df[factor] = factor_df[factor]
        
        # 按综合得分排序
        score_df = score_df.sort_values('综合得分', ascending=False)
        
        return score_df
    
    def get_top_funds(
        self,
        score_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        获取评分最高的基金
        
        Parameters
        ----------
        score_df : pd.DataFrame
            评分结果
        top_n : int
            返回数量
            
        Returns
        -------
        pd.DataFrame
            Top N基金
        """
        return score_df.head(top_n)
    
    def generate_radar_data(
        self,
        score_df: pd.DataFrame,
        fund_code: str
    ) -> Dict[str, float]:
        """
        生成雷达图数据
        
        Parameters
        ----------
        score_df : pd.DataFrame
            评分结果
        fund_code : str
            基金代码
            
        Returns
        -------
        dict
            雷达图数据
        """
        if fund_code not in score_df.index:
            return {}
        
        fund_scores = score_df.loc[fund_code]
        
        radar_data = {}
        for factor in self.weights.keys():
            score_col = f'{factor}_score'
            if score_col in fund_scores.index:
                radar_data[factor] = fund_scores[score_col]
        
        return radar_data


# ============================================================
# 多基金对比
# ============================================================
def compare_funds(
    score_df: pd.DataFrame,
    fund_codes: List[str]
) -> pd.DataFrame:
    """
    多基金对比
    
    Parameters
    ----------
    score_df : pd.DataFrame
        评分结果
    fund_codes : List[str]
        基金代码列表
        
    Returns
    -------
    pd.DataFrame
        对比结果
    """
    comparison = score_df.loc[score_df.index.isin(fund_codes)]
    
    return comparison


# ============================================================
# 板块评分
# ============================================================
def score_by_sector(
    factor_data: Dict[str, pd.DataFrame],
    weights: Dict[str, float] = None
) -> Dict[str, pd.DataFrame]:
    """
    分板块评分
    
    Parameters
    ----------
    factor_data : dict
        {sector_name: factor_dataframe}
    weights : dict
        因子权重
        
    Returns
    -------
    dict
        {sector_name: score_dataframe}
    """
    scorer = FundScorer(weights)
    
    results = {}
    
    for sector, factor_df in factor_data.items():
        print(f'  评分板块: {sector}')
        score_df = scorer.calculate_scores(factor_df, sector)
        
        if len(score_df) > 0:
            results[sector] = score_df
    
    return results
