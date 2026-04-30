# -*- coding: utf-8 -*-
"""
ETF评价与筛选模块：跟踪同一指数的ETF产品筛选
基于民生证券研报《ETF的聚类优选与热点趋势策略构建》

评价指标：
- 费率得分（管理费+托管费）× 40%
- 近一月日均成交额得分 × 20%
- 规模得分 × 20%
- 近一月跟踪误差得分 × 10%
- 近一月信息比率得分 × 10%
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')


class ETFEvaluator:
    """
    ETF产品评价器
    
    对跟踪同一指数的多只ETF产品进行综合评价和筛选：
    - 费率得分 × 40%
    - 流动性得分（日均成交额）× 20%
    - 规模得分 × 20%
    - 跟踪误差得分 × 10%
    - 信息比率得分 × 10%
    
    Parameters
    ----------
    etf_config : dict
        ETF评价配置
    """
    
    def __init__(self, etf_config: Optional[Dict] = None):
        self.config = etf_config or self._default_config()
        self.etf_scores_ = None
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'fee_score_weight': 0.40,
            'liquidity_score_weight': 0.20,
            'scale_score_weight': 0.20,
            'tracking_error_weight': 0.10,
            'info_ratio_weight': 0.10,
            'max_etf_per_index': 2,
            'min_etf_threshold': 5,
            'min_scale': 0.5,
            'min_liquidity': 0.1,
        }
    
    def calculate_fee_score(
        self,
        etf_df: pd.DataFrame
    ) -> pd.Series:
        """
        计算费率得分
        
        费率越低，得分越高
        
        Parameters
        ----------
        etf_df : pd.DataFrame
            ETF数据
        
        Returns
        -------
        pd.Series
            费率得分
        """
        if 'mgmt_fee' in etf_df.columns and 'custody_fee' in etf_df.columns:
            total_fee = etf_df['mgmt_fee'] + etf_df['custody_fee']
            # 费率越低得分越高（百分位反向）
            fee_score = 1 - total_fee.rank(pct=True)
        else:
            # 使用默认费率
            fee_score = pd.Series(0.5, index=etf_df.index)
        
        return fee_score
    
    def calculate_liquidity_score(
        self,
        etf_df: pd.DataFrame,
        daily_volume: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        计算流动性得分
        
        近一月日均成交额越高，得分越高
        
        Parameters
        ----------
        etf_df : pd.DataFrame
            ETF数据
        daily_volume : pd.Series, optional
            日均成交额数据
        
        Returns
        -------
        pd.Series
            流动性得分
        """
        if daily_volume is not None and len(daily_volume) > 0:
            liquidity = daily_volume
        elif 'avg_daily_volume' in etf_df.columns:
            liquidity = etf_df['avg_daily_volume']
        elif 'scale' in etf_df.columns:
            # 估算：规模越大流动性可能越好
            liquidity = etf_df['scale'] * np.random.uniform(0.01, 0.1, len(etf_df))
        else:
            liquidity = pd.Series(1.0, index=etf_df.index)
        
        # 百分位得分
        liquidity_score = liquidity.rank(pct=True)
        
        return liquidity_score
    
    def calculate_scale_score(
        self,
        etf_df: pd.DataFrame
    ) -> pd.Series:
        """
        计算规模得分
        
        规模适中得分高（太大或太小都不好）
        
        Parameters
        ----------
        etf_df : pd.DataFrame
            ETF数据
        
        Returns
        -------
        pd.Series
            规模得分
        """
        if 'scale' not in etf_df.columns:
            return pd.Series(0.5, index=etf_df.index)
        
        scale = etf_df['scale']
        
        # 规模适中得分高（最优规模区间10-100亿）
        # 使用二次函数模拟
        optimal_scale = 50  # 最优规模（亿元）
        scale_deviation = np.abs(np.log(scale + 1) - np.log(optimal_scale + 1))
        scale_score = 1 / (1 + scale_deviation)
        
        return scale_score
    
    def calculate_tracking_error_score(
        self,
        etf_returns: pd.DataFrame,
        index_returns: pd.Series,
        periods: int = 20
    ) -> pd.Series:
        """
        计算跟踪误差得分
        
        跟踪误差越小，得分越高
        
        Parameters
        ----------
        etf_returns : pd.DataFrame
            ETF收益数据
        index_returns : pd.Series
            指数收益数据
        periods : int
            计算周期（交易日）
        
        Returns
        -------
        pd.Series
            跟踪误差得分
        """
        tracking_errors = {}
        
        for col in etf_returns.columns:
            etf_ret = etf_returns[col].dropna()
            index_ret = index_ret = index_returns.reindex(etf_ret.index)
            
            if len(etf_ret) < periods:
                tracking_errors[col] = np.nan
                continue
            
            # 计算跟踪误差（ETF收益与指数收益的差异标准差）
            diff = etf_ret - index_ret
            tracking_error = diff.tail(periods).std() * np.sqrt(252)
            tracking_errors[col] = tracking_error
        
        tracking_error_series = pd.Series(tracking_errors)
        
        # 跟踪误差越小得分越高
        te_score = 1 - tracking_error_series.rank(pct=True)
        
        return te_score
    
    def calculate_info_ratio_score(
        self,
        etf_returns: pd.DataFrame,
        index_returns: pd.Series,
        periods: int = 20
    ) -> pd.Series:
        """
        计算信息比率得分
        
        信息比率 = 超额收益 / 跟踪误差
        信息比率越高，得分越高
        
        Parameters
        ----------
        etf_returns : pd.DataFrame
            ETF收益数据
        index_returns : pd.Series
            指数收益数据
        periods : int
            计算周期
        
        Returns
        -------
        pd.Series
            信息比率得分
        """
        info_ratios = {}
        
        for col in etf_returns.columns:
            etf_ret = etf_returns[col].dropna()
            index_ret = index_returns.reindex(etf_ret.index)
            
            if len(etf_ret) < periods:
                info_ratios[col] = np.nan
                continue
            
            # 计算超额收益
            excess_return = (etf_ret - index_ret).tail(periods).mean() * 252
            
            # 计算跟踪误差
            diff = etf_ret - index_ret
            tracking_error = diff.tail(periods).std() * np.sqrt(252)
            
            # 信息比率
            if tracking_error > 0:
                info_ratio = excess_return / tracking_error
            else:
                info_ratio = 0
            
            info_ratios[col] = info_ratio
        
        info_ratio_series = pd.Series(info_ratios)
        
        # 信息比率越高得分越高
        ir_score = info_ratio_series.rank(pct=True)
        
        return ir_score
    
    def calculate_comprehensive_score(
        self,
        etf_df: pd.DataFrame,
        returns_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        计算综合得分
        
        综合得分 = 费率得分×40% + 流动性得分×20% + 规模得分×20%
                 + 跟踪误差得分×10% + 信息比率得分×10%
        
        Parameters
        ----------
        etf_df : pd.DataFrame
            ETF数据
        returns_data : dict, optional
            收益数据字典
        
        Returns
        -------
        pd.DataFrame
            带得分的ETF数据
        """
        result = etf_df.copy()
        
        # 计算各项得分
        result['fee_score'] = self.calculate_fee_score(etf_df)
        result['liquidity_score'] = self.calculate_liquidity_score(etf_df)
        result['scale_score'] = self.calculate_scale_score(etf_df)
        
        # 如果有收益数据，计算跟踪误差和信息比率
        if returns_data and 'etf_returns' in returns_data and 'index_returns' in returns_data:
            etf_returns = returns_data['etf_returns']
            index_returns = returns_data['index_returns']
            
            result['tracking_error_score'] = self.calculate_tracking_error_score(
                etf_returns, index_returns
            )
            result['info_ratio_score'] = self.calculate_info_ratio_score(
                etf_returns, index_returns
            )
        else:
            # 默认得分
            result['tracking_error_score'] = 0.5
            result['info_ratio_score'] = 0.5
        
        # 计算综合得分
        weights = {
            'fee_score': self.config['fee_score_weight'],
            'liquidity_score': self.config['liquidity_score_weight'],
            'scale_score': self.config['scale_score_weight'],
            'tracking_error_score': self.config['tracking_error_weight'],
            'info_ratio_score': self.config['info_ratio_weight'],
        }
        
        score_cols = []
        weight_sum = 0
        for col, weight in weights.items():
            if col in result.columns:
                result[col] = result[col].fillna(0.5)
                score_cols.append(col)
                weight_sum += weight
        
        # 归一化权重
        if score_cols:
            result['comprehensive_score'] = sum(
                result[col] * (weights.get(col, 1/len(score_cols)) / weight_sum)
                for col in score_cols
            )
        
        self.etf_scores_ = result
        return result
    
    def select_best_etfs(
        self,
        etf_scores: pd.DataFrame,
        index_col: str = 'index_code'
    ) -> pd.DataFrame:
        """
        筛选最佳ETF
        
        对于跟踪同一指数的多只ETF，选择综合得分最高的前1-2只
        
        Parameters
        ----------
        etf_scores : pd.DataFrame
            带得分的ETF数据
        index_col : str
            指数代码列名
        
        Returns
        -------
        pd.DataFrame
            筛选后的ETF
        """
        if 'comprehensive_score' not in etf_scores.columns:
            raise ValueError("请先调用calculate_comprehensive_score方法")
        
        selected_etfs = []
        
        for index_code in etf_scores[index_col].unique():
            index_etfs = etf_scores[etf_scores[index_col] == index_code].copy()
            
            # 按综合得分排序
            index_etfs = index_etfs.sort_values('comprehensive_score', ascending=False)
            
            # 确定保留数量
            if len(index_etfs) > self.config['min_etf_threshold']:
                n_select = self.config['max_etf_per_index']
            else:
                n_select = min(2, len(index_etfs))
            
            # 过滤最低标准
            if 'scale' in index_etfs.columns:
                index_etfs = index_etfs[index_etfs['scale'] >= self.config['min_scale']]
            
            # 选择前N只
            top_etfs = index_etfs.head(n_select)
            selected_etfs.append(top_etfs)
        
        if selected_etfs:
            result = pd.concat(selected_etfs, ignore_index=True)
            return result
        else:
            return pd.DataFrame()


def evaluate_and_select_etfs(
    etf_df: pd.DataFrame,
    etf_returns: Optional[pd.DataFrame] = None,
    index_returns: Optional[pd.Series] = None,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ETF评价与筛选主函数
    
    Parameters
    ----------
    etf_df : pd.DataFrame
        ETF基础数据
    etf_returns : pd.DataFrame, optional
        ETF收益数据
    index_returns : pd.Series, optional
        指数收益数据
    config : dict, optional
        配置
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (评价结果, 筛选结果)
    """
    evaluator = ETFEvaluator(config)
    
    # 计算综合得分
    returns_data = None
    if etf_returns is not None and index_returns is not None:
        returns_data = {
            'etf_returns': etf_returns,
            'index_returns': index_returns
        }
    
    evaluation = evaluator.calculate_comprehensive_score(etf_df, returns_data)
    
    # 筛选最佳ETF
    selected = evaluator.select_best_etfs(evaluation)
    
    return evaluation, selected


def generate_mock_etf_metrics(etf_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成模拟ETF指标数据
    
    Parameters
    ----------
    etf_df : pd.DataFrame
        ETF基础数据
    
    Returns
    -------
    pd.DataFrame
        带模拟指标的ETF数据
    """
    np.random.seed(42)
    result = etf_df.copy()
    
    # 确保etf_type存在
    if 'etf_type' not in result.columns:
        # 基于指数代码推断ETF类型
        def infer_etf_type(index_code):
            if pd.isna(index_code):
                return '规模风格'
            index_str = str(index_code).upper()
            if '3999' in index_str or '9999' in index_str:
                return '行业'
            elif 'SH' in index_str and any(x in index_str for x in ['50', '300', '500', '800']):
                return '规模风格'
            elif 'BOND' in index_str or 'zj' in index_str.lower():
                return '债券'
            elif 'FUT' in index_str or 'IMCI' in index_str:
                return '商品'
            elif 'HK' in index_str or 'OVERSEA' in index_str:
                return '跨境'
            return '规模风格'
        
        result['etf_type'] = result.get('index_code', '000300.SH').apply(infer_etf_type)
    
    # 模拟费率
    if 'mgmt_fee' not in result.columns:
        fee_rates = {
            '行业': 0.005,
            '规模风格': 0.005,
            '宽基': 0.005,
            '债券': 0.003,
            '商品': 0.006,
            '跨境': 0.006,
        }
        result['mgmt_fee'] = result['etf_type'].map(fee_rates).fillna(0.005)
    
    if 'custody_fee' not in result.columns:
        result['custody_fee'] = 0.001
    
    # 模拟规模
    if 'scale' not in result.columns:
        result['scale'] = np.random.uniform(1, 500, len(result))
    
    # 模拟日均成交额
    if 'avg_daily_volume' not in result.columns:
        result['avg_daily_volume'] = result['scale'] * np.random.uniform(0.01, 0.1)
    
    return result


# ==================== 测试函数 ====================
if __name__ == '__main__':
    print("测试ETF评价模块...")
    
    # 创建模拟ETF数据
    np.random.seed(42)
    
    etf_data = []
    index_codes = ['000300.SH', '000016.SH', '399966.SZ', '000688.SH']
    
    for idx in index_codes:
        n_etfs = np.random.randint(2, 5)
        for i in range(n_etfs):
            etf_data.append({
                'fund_code': f'{idx}_{i}',
                'fund_name': f'ETF_{idx}_{i}',
                'index_code': idx,
                'etf_type': '规模风格',
                'scale': np.random.uniform(1, 200),
                'mgmt_fee': np.random.uniform(0.003, 0.006),
                'custody_fee': 0.001,
                'avg_daily_volume': np.random.uniform(0.1, 10),
            })
    
    etf_df = pd.DataFrame(etf_data)
    
    # 评价
    print("\n执行ETF评价...")
    etf_df = generate_mock_etf_metrics(etf_df)
    
    evaluation, selected = evaluate_and_select_etfs(etf_df)
    
    print(f"\n评价结果（前10）:")
    print(evaluation[['fund_code', 'fund_name', 'index_code', 'fee_score', 
                      'liquidity_score', 'scale_score', 'comprehensive_score']].head(10))
    
    print(f"\n筛选后ETF数量: {len(selected)}")
    if len(selected) > 0:
        print("\n筛选结果:")
        print(selected[['fund_code', 'fund_name', 'index_code', 'comprehensive_score']])
