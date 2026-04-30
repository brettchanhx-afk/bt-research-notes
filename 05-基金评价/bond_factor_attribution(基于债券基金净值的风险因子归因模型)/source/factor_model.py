# -*- coding: utf-8 -*-
"""
多因子回归模型模块

功能：
  - 因子回归分析
  - 滚动回归
  - 因子暴露计算
  - 收益归因分解
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import statsmodels.api as sm
from scipy import stats


# ============================================================
# 1. 因子回归模型
# ============================================================
class FactorRegressionModel:
    """多因子回归模型。
    
    R_fund = α + Σ β_i × F_i + ε
    
    其中：
      - R_fund: 基金收益率
      - α: Alpha（超额收益）
      - β_i: 因子i的暴露系数
      - F_i: 因子i的收益
      - ε: 残差
    """
    
    def __init__(self):
        self.model_ = None
        self.results_ = None
        self.params_ = None
        self.summary_ = None
    
    def fit(
        self,
        fund_returns: pd.Series,
        factors: pd.DataFrame,
        factor_names: List[str] = None,
        add_constant: bool = True
    ) -> dict:
        """拟合因子回归模型。
        
        Parameters
        ----------
        fund_returns : pd.Series
            基金收益率序列
        factors : pd.DataFrame
            因子数据
        factor_names : List[str]
            因子名称列表
        add_constant : bool
            是否添加常数项
        
        Returns
        -------
        dict
            回归结果
        """
        # 对齐日期
        common_dates = fund_returns.index.intersection(factors.index)
        
        if len(common_dates) < 10:
            print('  [ERROR] 数据不足，无法回归')
            return {}
        
        # 准备数据
        y = fund_returns.loc[common_dates].dropna()
        
        if factor_names is None:
            factor_names = [col for col in factors.columns if not col.endswith('_std')]
        
        X = factors.loc[common_dates, factor_names].dropna()
        
        # 再次对齐
        common = y.index.intersection(X.index)
        y = y.loc[common]
        X = X.loc[common]
        
        if len(y) < 10:
            print('  [ERROR] 对齐后数据不足')
            return {}
        
        # 添加常数项
        if add_constant:
            X = sm.add_constant(X)
        
        # OLS回归
        self.model_ = sm.OLS(y, X)
        self.results_ = self.model_.fit()
        
        # 提取结果
        self.params_ = self.results_.params
        self.summary_ = self.results_.summary()
        
        # 构建结果字典
        results = {
            'alpha': self.params_.get('const', 0),
            'factor_exposures': {},
            'factor_tstats': {},
            'factor_pvalues': {},
            'r_squared': self.results_.rsquared,
            'adj_r_squared': self.results_.rsquared_adj,
            'f_statistic': self.results_.fvalue,
            'f_pvalue': self.results_.f_pvalue,
            'n_obs': len(y),
            'residual_std': np.std(self.results_.resid),
        }
        
        # 因子暴露
        for name in factor_names:
            if name in self.params_.index:
                results['factor_exposures'][name] = self.params_[name]
                results['factor_tstats'][name] = self.results_.tvalues[name]
                results['factor_pvalues'][name] = self.results_.pvalues[name]
        
        return results
    
    def predict(
        self,
        factors: pd.DataFrame,
        factor_names: List[str] = None
    ) -> pd.Series:
        """预测基金收益率。
        
        Parameters
        ----------
        factors : pd.DataFrame
            因子数据
        factor_names : List[str]
            因子名称列表
        
        Returns
        -------
        pd.Series
            预测收益率
        """
        if self.results_ is None:
            return pd.Series()
        
        if factor_names is None:
            factor_names = [col for col in factors.columns if not col.endswith('_std')]
        
        X = factors[factor_names]
        X = sm.add_constant(X)
        
        return self.results_.predict(X)
    
    def decompose_returns(
        self,
        fund_returns: pd.Series,
        factors: pd.DataFrame,
        factor_names: List[str] = None
    ) -> pd.DataFrame:
        """分解收益来源。
        
        Parameters
        ----------
        fund_returns : pd.Series
            基金收益率
        factors : pd.DataFrame
            因子数据
        factor_names : List[str]
            因子名称列表
        
        Returns
        -------
        pd.DataFrame
            收益分解
        """
        if self.params_ is None:
            return pd.DataFrame()
        
        if factor_names is None:
            factor_names = [col for col in factors.columns if not col.endswith('_std')]
        
        # 对齐日期
        common = fund_returns.index.intersection(factors.index)
        
        decomposition = pd.DataFrame(index=common)
        decomposition['total_return'] = fund_returns.loc[common]
        
        # Alpha贡献
        alpha = self.params_.get('const', 0)
        decomposition['alpha_contrib'] = alpha
        
        # 各因子贡献
        total_factor_contrib = 0
        for name in factor_names:
            if name in self.params_.index and name in factors.columns:
                beta = self.params_[name]
                factor_return = factors.loc[common, name]
                decomposition[f'{name}_contrib'] = beta * factor_return
                total_factor_contrib += decomposition[f'{name}_contrib']
        
        # 残差
        decomposition['residual'] = decomposition['total_return'] - alpha - total_factor_contrib
        
        return decomposition


# ============================================================
# 2. 滚动回归
# ============================================================
def rolling_regression(
    fund_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 60,
    step: int = 20,
    factor_names: List[str] = None
) -> pd.DataFrame:
    """滚动窗口回归。
    
    Parameters
    ----------
    fund_returns : pd.Series
        基金收益率
    factors : pd.DataFrame
        因子数据
    window : int
        回归窗口
    step : int
        滚动步长
    factor_names : List[str]
        因子名称列表
    
    Returns
    -------
    pd.DataFrame
        滚动回归结果
    """
    # 对齐日期
    common = fund_returns.index.intersection(factors.index)
    
    if len(common) < window:
        return pd.DataFrame()
    
    y = fund_returns.loc[common]
    X = factors.loc[common]
    
    if factor_names is None:
        factor_names = [col for col in X.columns if not col.endswith('_std')]
    
    results = []
    dates = []
    
    for i in range(window, len(common), step):
        end_date = common[i]
        start_date = common[i - window]
        
        # 窗口数据
        y_window = y.loc[start_date:end_date]
        X_window = X.loc[start_date:end_date, factor_names]
        
        # 回归
        model = FactorRegressionModel()
        result = model.fit(y_window, X_window, factor_names)
        
        if result:
            result['date'] = end_date
            result['window_start'] = start_date
            result['window_end'] = end_date
            results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('date')
    
    return df_results


# ============================================================
# 3. 因子暴露分析
# ============================================================
def analyze_factor_exposure(
    regression_results: pd.DataFrame
) -> dict:
    """分析因子暴露情况。
    
    Parameters
    ----------
    regression_results : pd.DataFrame
        回归结果（来自rolling_regression）
    
    Returns
    -------
    dict
        因子暴露分析
    """
    if len(regression_results) == 0:
        return {}
    
    analysis = {}
    
    # Alpha分析
    if 'alpha' in regression_results.columns:
        analysis['alpha_mean'] = regression_results['alpha'].mean()
        analysis['alpha_std'] = regression_results['alpha'].std()
        analysis['alpha_tstat'] = analysis['alpha_mean'] / (analysis['alpha_std'] / np.sqrt(len(regression_results)))
    
    # R²分析
    if 'r_squared' in regression_results.columns:
        analysis['r_squared_mean'] = regression_results['r_squared'].mean()
        analysis['r_squared_std'] = regression_results['r_squared'].std()
    
    # 因子暴露分析
    factor_exposures = regression_results['factor_exposures']
    
    if isinstance(factor_exposures.iloc[0], dict):
        all_factors = set()
        for exp in factor_exposures:
            all_factors.update(exp.keys())
        
        for factor in all_factors:
            values = [exp.get(factor, np.nan) for exp in factor_exposures]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                analysis[f'{factor}_mean'] = np.mean(values)
                analysis[f'{factor}_std'] = np.std(values)
                analysis[f'{factor}_tstat'] = analysis[f'{factor}_mean'] / (analysis[f'{factor}_std'] / np.sqrt(len(values)))
    
    return analysis


# ============================================================
# 4. 因子贡献度计算
# ============================================================
def calculate_factor_contribution(
    fund_returns: pd.Series,
    factors: pd.DataFrame,
    regression_results: dict
) -> pd.DataFrame:
    """计算各因子对总收益的贡献度。
    
    Parameters
    ----------
    fund_returns : pd.Series
        基金收益率
    factors : pd.DataFrame
        因子数据
    regression_results : dict
        回归结果
    
    Returns
    -------
    pd.DataFrame
        因子贡献度
    """
    if not regression_results or 'factor_exposures' not in regression_results:
        return pd.DataFrame()
    
    # 总收益
    total_return = fund_returns.sum()
    
    contributions = []
    
    # Alpha贡献
    alpha = regression_results.get('alpha', 0)
    alpha_contrib = alpha * len(fund_returns)
    contributions.append({
        'factor': 'Alpha',
        'exposure': alpha,
        'contribution': alpha_contrib,
        'contribution_pct': alpha_contrib / total_return * 100 if total_return != 0 else 0,
    })
    
    # 因子贡献
    factor_exposures = regression_results['factor_exposures']
    
    for factor_name, beta in factor_exposures.items():
        if factor_name in factors.columns:
            factor_return = factors[factor_name].sum()
            contrib = beta * factor_return
            contributions.append({
                'factor': factor_name,
                'exposure': beta,
                'contribution': contrib,
                'contribution_pct': contrib / total_return * 100 if total_return != 0 else 0,
            })
    
    df = pd.DataFrame(contributions)
    
    # 按贡献度排序
    df = df.sort_values('contribution', key=abs, ascending=False)
    
    return df
