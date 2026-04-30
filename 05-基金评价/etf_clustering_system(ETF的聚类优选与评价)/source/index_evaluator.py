# -*- coding: utf-8 -*-
"""
指数评价模块：多维评价指标计算
基于民生证券研报《ETF的聚类优选与热点趋势策略构建》

评价维度：
1. 估值贡献影响
2. 集中度（前十权重股占比、前五行业涨跌影响）
3. 盈利能力（ROE_TTM）
4. 成长性（营收同比）
5. 夏普比率（长短期）
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')


class IndexEvaluator:
    """
    ETF跟踪指数多维评价器
    
    在聚类基础上，根据以下维度对同类指数进行评价和优选：
    - 估值贡献影响
    - 集中度
    - 盈利能力（ROE_TTM）
    - 成长性（营收同比）
    - 长短期夏普比率
    
    Parameters
    ----------
    cluster_result : pd.DataFrame
        聚类结果，包含index_code和cluster列
    evaluation_config : dict
        评价配置
    """
    
    def __init__(
        self,
        cluster_result: pd.DataFrame,
        evaluation_config: Optional[Dict] = None
    ):
        self.cluster_result = cluster_result.copy()
        self.config = evaluation_config or self._default_config()
        self.evaluation_scores_ = None
        self.filtered_indices_ = None
        self.sharpe_scores_ = None
        
    def _default_config(self) -> Dict:
        """默认评价配置"""
        return {
            'valuation_contribution_weight': 0.25,
            'concentration_weight': 0.25,
            'profitability_weight': 0.25,
            'growth_weight': 0.25,
            'sharpe_top_percent': 0.5,
            'eliminate_bottom_percent': 0.2,
        }
    
    def calculate_valuation_contribution(
        self,
        index_returns: pd.DataFrame,
        stock_pb_changes: pd.DataFrame
    ) -> pd.Series:
        """
        计算估值贡献影响
        
        估值贡献 = 成分股权重 × 成分股PB变化
        估值贡献越低，说明成分股被选入后估值回落程度越高
        
        Parameters
        ----------
        index_returns : pd.DataFrame
            指数收益数据，index为日期，columns为指数代码
        stock_pb_changes : pd.DataFrame
            成分股PB变化数据
        
        Returns
        -------
        pd.Series
            各指数的估值贡献
        """
        # 简化计算：使用指数收益与市场收益的偏离度作为估值贡献指标
        # 偏离越大，估值贡献越高（可能存在估值回调风险）
        
        market_return = index_returns.mean(axis=1)
        valuation_contrib = {}
        
        for col in index_returns.columns:
            # 计算与市场收益的差异
            excess_return = index_returns[col] - market_return
            # 估值贡献指标：使用超额收益的标准差
            valuation_contrib[col] = excess_return.std()
        
        return pd.Series(valuation_contrib)
    
    def calculate_concentration(
        self,
        constituents_dict: Dict[str, pd.DataFrame],
        stock_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算集中度指标
        
        包括：
        1. 前十权重股占比
        2. 前五行业涨跌影响程度
        
        Parameters
        ----------
        constituents_dict : dict
            成分股字典
        stock_returns : pd.DataFrame
            成分股收益数据
        
        Returns
        -------
        pd.DataFrame
            集中度指标
        """
        concentration_metrics = {}
        
        for idx_code, constituents in constituents_dict.items():
            if constituents is None or len(constituents) == 0:
                concentration_metrics[idx_code] = {
                    'top10_weight_ratio': np.nan,
                    'top5_industry_impact': np.nan
                }
                continue
            
            # 前十权重股占比
            constituents_sorted = constituents.nlargest(10, 'weight')
            top10_weight = constituents_sorted['weight'].sum()
            
            # 简化：使用前十大权重股的收益波动作为行业影响代理
            top10_codes = constituents_sorted['con_code'].tolist()
            if len(top10_codes) > 0:
                available_codes = [c for c in top10_codes if c in stock_returns.columns]
                if available_codes:
                    top10_returns = stock_returns[available_codes]
                    top5_impact = top10_returns.std().mean()
                else:
                    top5_impact = np.nan
            else:
                top5_impact = np.nan
            
            concentration_metrics[idx_code] = {
                'top10_weight_ratio': top10_weight,
                'top5_industry_impact': top5_impact
            }
        
        return pd.DataFrame(concentration_metrics).T
    
    def calculate_profitability(
        self,
        index_code: str,
        financial_data: pd.DataFrame
    ) -> float:
        """
        计算盈利能力（ROE_TTM）
        
        Parameters
        ----------
        index_code : str
            指数代码
        financial_data : pd.DataFrame
            财务数据
        
        Returns
        -------
        float
            ROE_TTM
        """
        if financial_data is None or len(financial_data) == 0:
            return np.nan
        
        # 使用最新一期ROE
        roe_ttm = financial_data['roe_ttm'].iloc[-1] if 'roe_ttm' in financial_data.columns else np.nan
        return roe_ttm
    
    def calculate_growth(
        self,
        index_code: str,
        financial_data: pd.DataFrame
    ) -> float:
        """
        计算成长性（营收同比）
        
        Parameters
        ----------
        index_code : str
            指数代码
        financial_data : pd.DataFrame
            财务数据
        
        Returns
        -------
        float
            营收同比
        """
        if financial_data is None or len(financial_data) == 0:
            return np.nan
        
        # 使用最新一期营收同比
        revenue_yoy = financial_data['revenue_yoy'].iloc[-1] if 'revenue_yoy' in financial_data.columns else np.nan
        return revenue_yoy
    
    def calculate_all_financial_metrics(
        self,
        constituents_dict: Dict[str, pd.DataFrame],
        financial_data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        计算所有指数的财务指标
        
        Parameters
        ----------
        constituents_dict : dict
            成分股字典
        financial_data_dict : dict
            财务数据字典
        
        Returns
        -------
        pd.DataFrame
            财务指标
        """
        metrics = []
        
        for idx_code in constituents_dict.keys():
            constituents = constituents_dict.get(idx_code)
            
            if constituents is None or len(constituents) == 0:
                metrics.append({
                    'index_code': idx_code,
                    'roe_ttm': np.nan,
                    'revenue_yoy': np.nan,
                    'valuation_contrib': np.nan,
                    'concentration': np.nan
                })
                continue
            
            # 加权计算指数财务指标
            weights = constituents['weight'].values
            roe_values = []
            rev_values = []
            
            for _, row in constituents.iterrows():
                stock_code = row['con_code']
                weight = row['weight']
                
                fin_data = financial_data_dict.get(stock_code)
                if fin_data is not None and len(fin_data) > 0:
                    roe = fin_data['roe_ttm'].iloc[-1] if 'roe_ttm' in fin_data.columns else np.nan
                    rev = fin_data['revenue_yoy'].iloc[-1] if 'revenue_yoy' in fin_data.columns else np.nan
                    
                    if not np.isnan(roe):
                        roe_values.append(roe * weight)
                    if not np.isnan(rev):
                        rev_values.append(rev * weight)
            
            avg_roe = np.mean(roe_values) if roe_values else np.nan
            avg_rev = np.mean(rev_values) if rev_values else np.nan
            
            metrics.append({
                'index_code': idx_code,
                'roe_ttm': avg_roe,
                'revenue_yoy': avg_rev,
                'valuation_contrib': np.nan,  # 需要收益数据计算
                'concentration': constituents.nlargest(10, 'weight')['weight'].sum() if len(constituents) > 0 else np.nan
            })
        
        return pd.DataFrame(metrics)
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.03
    ) -> float:
        """
        计算年化夏普比率
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        periods_per_year : int
            年化周期数（日频=252）
        risk_free_rate : float
            年化无风险利率
        
        Returns
        -------
        float
            年化夏普比率
        """
        if len(returns) == 0 or returns.std() == 0:
            return np.nan
        
        annual_return = returns.mean() * periods_per_year
        annual_std = returns.std() * np.sqrt(periods_per_year)
        
        sharpe = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else np.nan
        return sharpe
    
    def calculate_index_sharpe(
        self,
        index_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        计算所有指数的夏普比率
        
        Parameters
        ----------
        index_returns : pd.DataFrame
            指数收益数据
        benchmark_returns : pd.Series, optional
            基准收益数据
        
        Returns
        -------
        pd.DataFrame
            夏普比率
        """
        sharpe_data = []
        
        for col in index_returns.columns:
            returns = index_returns[col].dropna()
            
            # 全期夏普
            sharpe_all = self.calculate_sharpe_ratio(returns)
            
            # 近一年夏普
            if len(returns) > 252:
                returns_1y = returns.iloc[-252:]
            else:
                returns_1y = returns
            
            sharpe_1y = self.calculate_sharpe_ratio(returns_1y)
            
            sharpe_data.append({
                'index_code': col,
                'sharpe_all': sharpe_all,
                'sharpe_1y': sharpe_1y
            })
        
        return pd.DataFrame(sharpe_data)
    
    def score_financial_metrics(
        self,
        financial_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        对财务指标进行同类打分
        
        使用百分位排名打分
        
        Parameters
        ----------
        financial_metrics : pd.DataFrame
            财务指标
        
        Returns
        -------
        pd.DataFrame
            带分数的财务指标
        """
        result = financial_metrics.copy()
        
        # 对各指标打分（百分位）
        score_cols = ['roe_ttm', 'revenue_yoy', 'valuation_contrib', 'concentration']
        
        for col in score_cols:
            if col in result.columns and result[col].notna().sum() > 0:
                # 估值贡献和集中度越低越好
                if col in ['valuation_contrib', 'concentration']:
                    result[f'{col}_score'] = result[col].rank(pct=True, ascending=True)
                else:
                    result[f'{col}_score'] = result[col].rank(pct=True, ascending=False)
            else:
                result[f'{col}_score'] = np.nan
        
        # 综合得分（等权）
        score_cols_available = [c for c in result.columns if c.endswith('_score')]
        if score_cols_available:
            result['financial_score'] = result[score_cols_available].mean(axis=1)
        
        return result
    
    def evaluate_indices(
        self,
        financial_metrics: pd.DataFrame,
        sharpe_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        综合评价指数
        
        Parameters
        ----------
        financial_metrics : pd.DataFrame
            财务指标
        sharpe_data : pd.DataFrame
            夏普比率数据
        
        Returns
        -------
        pd.DataFrame
            评价结果
        """
        # 合并数据
        evaluation = financial_metrics.merge(
            sharpe_data,
            on='index_code',
            how='outer'
        )
        
        # 财务指标打分
        evaluation = self.score_financial_metrics(evaluation)
        
        # 合并聚类结果
        evaluation = evaluation.merge(
            self.cluster_result,
            on='index_code',
            how='left'
        )
        
        self.evaluation_scores_ = evaluation
        return evaluation
    
    def select_top_indices(
        self,
        evaluation: pd.DataFrame,
        sharpe_top_percent: float = 0.5
    ) -> pd.DataFrame:
        """
        在同类中筛选优质指数
        
        步骤：
        1. 根据财务综合得分剔除后20%
        2. 在剩余指数中选择夏普比率前50%
        
        Parameters
        ----------
        evaluation : pd.DataFrame
            评价结果
        sharpe_top_percent : float
            夏普筛选比例
        
        Returns
        -------
        pd.DataFrame
            筛选后的指数
        """
        selected_indices = []
        
        for cluster_id in evaluation['cluster'].unique():
            cluster_data = evaluation[evaluation['cluster'] == cluster_id].copy()
            
            # 步骤1：剔除财务得分后20%
            if 'financial_score' in cluster_data.columns:
                threshold = cluster_data['financial_score'].quantile(0.2)
                cluster_data = cluster_data[cluster_data['financial_score'] >= threshold]
            
            if len(cluster_data) == 0:
                continue
            
            # 步骤2：选择夏普比率前50%
            # 综合长短期夏普
            if 'sharpe_all' in cluster_data.columns and 'sharpe_1y' in cluster_data.columns:
                cluster_data['sharpe_combined'] = (
                    cluster_data['sharpe_all'].fillna(0) * 0.5 +
                    cluster_data['sharpe_1y'].fillna(0) * 0.5
                )
            elif 'sharpe_all' in cluster_data.columns:
                cluster_data['sharpe_combined'] = cluster_data['sharpe_all']
            
            n_select = max(1, int(len(cluster_data) * sharpe_top_percent))
            top_indices = cluster_data.nlargest(n_select, 'sharpe_combined')
            
            selected_indices.append(top_indices)
        
        if selected_indices:
            result = pd.concat(selected_indices, ignore_index=True)
            self.filtered_indices_ = result
            return result
        else:
            return pd.DataFrame()


def evaluate_and_select_indices(
    cluster_result: pd.DataFrame,
    index_returns: pd.DataFrame,
    constituents_dict: Dict[str, pd.DataFrame],
    financial_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    指数评价与筛选主函数
    
    Parameters
    ----------
    cluster_result : pd.DataFrame
        聚类结果
    index_returns : pd.DataFrame
        指数收益数据
    constituents_dict : dict
        成分股字典
    financial_data_dict : dict, optional
        财务数据字典
    config : dict, optional
        配置
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (评价结果, 筛选结果)
    """
    evaluator = IndexEvaluator(cluster_result, config)
    
    # 计算财务指标
    if financial_data_dict:
        financial_metrics = evaluator.calculate_all_financial_metrics(
            constituents_dict,
            financial_data_dict
        )
    else:
        # 创建空财务数据
        financial_metrics = pd.DataFrame({
            'index_code': list(constituents_dict.keys()),
            'roe_ttm': np.nan,
            'revenue_yoy': np.nan,
            'valuation_contrib': np.nan,
            'concentration': np.nan
        })
    
    # 计算夏普比率
    sharpe_data = evaluator.calculate_index_sharpe(index_returns)
    
    # 综合评价
    evaluation = evaluator.evaluate_indices(financial_metrics, sharpe_data)
    
    # 筛选优质指数
    config = config or evaluator.config
    selected = evaluator.select_top_indices(
        evaluation,
        sharpe_top_percent=config.get('sharpe_top_percent', 0.5)
    )
    
    return evaluation, selected


# ==================== 测试函数 ====================
if __name__ == '__main__':
    print("测试指数评价模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟聚类结果
    cluster_result = pd.DataFrame({
        'index_code': [f'00000{i}.SH' for i in range(20)],
        'cluster': np.random.randint(0, 4, 20)
    })
    
    # 模拟指数收益
    dates = pd.date_range('2023-01-01', periods=500, freq='B')
    index_codes = cluster_result['index_code'].tolist()
    returns_data = np.random.randn(500, len(index_codes)) * 0.02
    index_returns = pd.DataFrame(returns_data, index=dates, columns=index_codes)
    index_returns = (1 + index_returns).cumprod()
    
    # 模拟成分股
    constituents_dict = {
        idx: pd.DataFrame({
            'con_code': [f'60000{j}.SH' for j in range(30)],
            'con_name': [f'股票{j}' for j in range(30)],
            'weight': np.random.dirichlet(np.ones(30))
        }) for idx in index_codes
    }
    
    # 评价
    print("\n执行指数评价...")
    evaluation, selected = evaluate_and_select_indices(
        cluster_result,
        index_returns,
        constituents_dict
    )
    
    print(f"\n评价结果（前10）:")
    print(evaluation.head(10))
    
    print(f"\n筛选后指数数量: {len(selected)}")
    if len(selected) > 0:
        print(selected[['index_code', 'cluster', 'financial_score', 'sharpe_all', 'sharpe_1y']])
