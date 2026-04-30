"""
回测归因模块

实现Fama-French五因子回归分析和归因
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils import calculate_metrics, format_results, print_regression_summary


class FamaFrenchAttribution:
    """
    Fama-French多因子归因分析器
    
    对基金收益进行五因子回归分析，分解超额收益来源
    """
    
    def __init__(self):
        self.results = None
        self.regression_model = None
    
    def run_regression(
        self,
        fund_data: pd.DataFrame,
        factors: pd.DataFrame,
        rf_series: Optional[pd.Series] = None
    ) -> Dict:
        """
        运行Fama-French五因子回归
        
        回归方程：
        R = α + b×R_M + s×SMB + h×HML + r×RMW + c×CMA + ε
        
        其中R是基金相对于无风险的超额收益
        
        Parameters
        ----------
        fund_data : pd.DataFrame
            基金数据，包含收益率列
        factors : pd.DataFrame
            五因子数据
        rf_series : pd.Series, optional
            无风险利率序列
            
        Returns
        -------
        Dict
            回归结果字典
        """
        print("\n[Step 2] 运行Fama-French五因子回归...")
        
        # 确定基金收益率列名
        if 'monthly_return' in fund_data.columns:
            fund_return_col = 'monthly_return'
        elif 'daily_return' in fund_data.columns:
            fund_return_col = 'daily_return'
        else:
            raise ValueError("基金数据中未找到收益率列")
        
        # 准备数据
        fund_returns = fund_data.set_index('date')[fund_return_col]
        
        # 如果提供了无风险利率，计算超额收益
        if rf_series is not None:
            fund_excess_returns = fund_returns - rf_series.reindex(fund_returns.index).fillna(0.02/12)
        else:
            # 假设无风险利率为2%年化
            fund_excess_returns = fund_returns - 0.02/12
        
        # 对齐日期 - 将两个Series都用年月周期(月末)对齐，取共同部分
        fund_period = fund_excess_returns.index.to_period('M')
        factor_period = factors.index.to_period('M')
        
        # 找到共同的年月周期
        common_periods = fund_period[fund_period.isin(factor_period)]
        common_yyyymm_str = common_periods.astype(str).tolist()
        
        if len(common_yyyymm_str) < 12:
            raise ValueError(f"共同观测数太少（{len(common_yyyymm_str)}期），无法进行回归")
        
        # 用年月字符串从各自的Series中提取数据（避免日期不一致问题）
        fund_for_regression = fund_excess_returns.copy()
        fund_for_regression.index = fund_period.astype(str)
        factor_for_regression = factors[['R_M', 'SMB', 'HML', 'RMW', 'CMA']].copy()
        factor_for_regression.index = factor_period.astype(str)
        
        y = fund_for_regression.loc[common_yyyymm_str]
        X = factor_for_regression.loc[common_yyyymm_str]
        
        # 删除缺失值
        mask = ~(y.isna() | X.isna().any(axis=1))
        y_clean = y[mask]
        X_clean = X[mask]
        
        print(f"  回归样本量: {len(y_clean)} 期")
        
        # 添加常数项
        X_with_const = sm.add_constant(X_clean)
        
        # 运行OLS回归
        model = sm.OLS(y_clean, X_with_const)
        results = model.fit()
        
        self.regression_model = results
        
        # 提取结果
        self.results = {
            'params': {
                'R_M': results.params['R_M'],
                'SMB': results.params['SMB'],
                'HML': results.params['HML'],
                'RMW': results.params['RMW'],
                'CMA': results.params['CMA'],
            },
            't_values': {
                'R_M': results.tvalues['R_M'],
                'SMB': results.tvalues['SMB'],
                'HML': results.tvalues['HML'],
                'RMW': results.tvalues['RMW'],
                'CMA': results.tvalues['CMA'],
            },
            'p_values': {
                'R_M': results.pvalues['R_M'],
                'SMB': results.pvalues['SMB'],
                'HML': results.pvalues['HML'],
                'RMW': results.pvalues['RMW'],
                'CMA': results.pvalues['CMA'],
            },
            'alpha': results.params['const'],
            'alpha_t': results.tvalues['const'],
            'alpha_p': results.pvalues['const'],
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'n_obs': int(results.nobs),
            'residuals': results.resid,
            'fitted_values': results.fittedvalues,
            'std_errors': {
                'R_M': results.bse['R_M'],
                'SMB': results.bse['SMB'],
                'HML': results.bse['HML'],
                'RMW': results.bse['RMW'],
                'CMA': results.bse['CMA'],
                'const': results.bse['const'],
            },
            'conf_int': results.conf_int(),
        }
        
        # 打印结果
        print_regression_summary(self.results)
        
        return self.results
    
    def calculate_factor_contribution(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        计算各因子对基金收益的贡献
        
        贡献 = 因子暴露 × 因子平均收益率
        
        Parameters
        ----------
        factors : pd.DataFrame
            因子收益率数据
            
        Returns
        -------
        pd.DataFrame
            各因子贡献
        """
        if self.results is None:
            raise ValueError("请先运行run_regression()")
        
        print("\n[Step 3] 计算因子贡献分解...")
        
        contributions = []
        
        factor_labels = {
            'R_M': '市场因子',
            'SMB': '市值因子',
            'HML': '价值因子',
            'RMW': '盈利因子',
            'CMA': '投资因子',
        }
        
        total_factor_contrib = 0
        
        for factor in ['R_M', 'SMB', 'HML', 'RMW', 'CMA']:
            if factor in self.results['params'] and factor in factors.columns:
                exposure = self.results['params'][factor]
                avg_return = factors[factor].mean()
                contribution = exposure * avg_return
                total_factor_contrib += contribution
                
                contributions.append({
                    '因子': factor_labels.get(factor, factor),
                    '暴露系数': exposure,
                    '因子平均收益': avg_return,
                    '贡献': contribution,
                    '贡献占比': 0,  # 稍后计算
                })
        
        # Alpha贡献
        alpha_contrib = self.results['alpha']
        contributions.append({
            '因子': 'Alpha',
            '暴露系数': 1.0,
            '因子平均收益': alpha_contrib,
            '贡献': alpha_contrib,
            '贡献占比': 0,
        })
        
        # 计算贡献占比
        total_contrib = total_factor_contrib + alpha_contrib
        for contrib in contributions:
            if total_contrib != 0:
                contrib['贡献占比'] = contrib['贡献'] / total_contrib * 100
        
        contrib_df = pd.DataFrame(contributions)
        
        # 打印贡献分解
        print("\n因子贡献分解:")
        print("-" * 60)
        print(f"{'因子':<12} {'暴露':>10} {'因子收益':>12} {'贡献':>12} {'占比':>8}")
        print("-" * 60)
        
        for _, row in contrib_df.iterrows():
            print(f"{row['因子']:<12} {row['暴露系数']:>10.4f} {row['因子平均收益']:>12.4f} "
                  f"{row['贡献']:>12.4f} {row['贡献占比']:>7.1f}%")
        
        print("-" * 60)
        print(f"{'合计':<12} {'':>10} {'':>12} {total_contrib:>12.4f} {'':>8}")
        
        return contrib_df
    
    def run_rolling_regression(
        self,
        fund_data: pd.DataFrame,
        factors: pd.DataFrame,
        window: int = 12,
        min_periods: int = 6
    ) -> pd.DataFrame:
        """
        运行滚动窗口回归
        
        用于观察因子暴露随时间的变化
        
        Parameters
        ----------
        fund_data : pd.DataFrame
            基金数据
        factors : pd.DataFrame
            因子数据
        window : int
            滚动窗口大小（月）
        min_periods : int
            最小观测数
            
        Returns
        -------
        pd.DataFrame
            滚动回归系数
        """
        print(f"\n[Step 4] 运行滚动回归（窗口={window}个月）...")
        
        # 准备数据
        if 'monthly_return' in fund_data.columns:
            fund_return_col = 'monthly_return'
        else:
            fund_return_col = 'daily_return'
        
        fund_returns = fund_data.set_index('date')[fund_return_col]
        
        # 对齐日期 - 使用年月周期避免月初/月末不匹配
        fund_period = fund_returns.index.to_period('M')
        factor_period = factors.index.to_period('M')
        
        # 用年月字符串作为共同索引
        fund_for_regression = fund_returns.copy()
        fund_for_regression.index = fund_period.astype(str)
        factor_for_regression = factors[['R_M', 'SMB', 'HML', 'RMW', 'CMA']].copy()
        factor_for_regression.index = factor_period.astype(str)
        
        # 找到共同的年月
        common_yyyymm = fund_for_regression.index[fund_for_regression.index.isin(factor_for_regression.index)]
        y = fund_for_regression.loc[common_yyyymm]
        X = factor_for_regression.loc[common_yyyymm]
        
        # 滚动回归
        rolling_results = []
        dates = []
        
        for i in range(window, len(y) + 1):
            y_window = y.iloc[i-window:i]
            X_window = X.iloc[i-window:i]
            
            # 删除缺失值
            mask = ~(y_window.isna() | X_window.isna().any(axis=1))
            y_clean = y_window[mask]
            X_clean = X_window[mask]
            
            if len(y_clean) < min_periods:
                continue
            
            # 回归
            X_with_const = sm.add_constant(X_clean)
            model = sm.OLS(y_clean, X_with_const)
            result = model.fit()
            
            rolling_results.append({
                'Alpha': result.params['const'],
                'R_M': result.params['R_M'],
                'SMB': result.params['SMB'],
                'HML': result.params['HML'],
                'RMW': result.params['RMW'],
                'CMA': result.params['CMA'],
                'R_squared': result.rsquared,
            })
            dates.append(y.index[i-1])
        
        rolling_df = pd.DataFrame(rolling_results, index=dates)
        
        print(f"  滚动回归期数: {len(rolling_df)}")
        print("\n滚动回归统计:")
        print(rolling_df.describe())
        
        return rolling_df
    
    def get_attribution_summary(self) -> Dict:
        """
        获取归因分析摘要
        
        Returns
        -------
        Dict
            归因分析摘要
        """
        if self.results is None:
            raise ValueError("请先运行run_regression()")
        
        # 判断各因子显著性
        significant_factors = []
        for factor in ['R_M', 'SMB', 'HML', 'RMW', 'CMA']:
            if self.results['p_values'][factor] < 0.05:
                significant_factors.append(factor)
        
        # 判断Alpha显著性
        alpha_significant = self.results['alpha_p'] < 0.05
        
        # 模型解释力度
        r_squared = self.results['r_squared']
        
        summary = {
            '模型解释力度': r_squared,
            'Alpha年化': self.results['alpha'] * 12,
            'Alpha显著': alpha_significant,
            '显著因子数': len(significant_factors),
            '显著因子': significant_factors,
            '市场Beta': self.results['params']['R_M'],
        }
        
        return summary


if __name__ == '__main__':
    # 测试
    print("Testing FamaFrenchAttribution...")
    
    from data_loader import FundDataLoader, FactorDataLoader
    from factor import FamaFrenchFactorBuilder
    
    # 加载数据
    fund_loader = FundDataLoader()
    fund_data = fund_loader.get_fund_nav('019888', '2022-01-01', '2024-12-31', 'monthly')
    
    # 构建因子
    factor_builder = FamaFrenchFactorBuilder()
    factors = factor_builder.build_five_factors('2022-01-01', '2024-12-31')
    
    # 归因分析
    attribution = FamaFrenchAttribution()
    results = attribution.run_regression(fund_data, factors)
    
    # 因子贡献
    contrib = attribution.calculate_factor_contribution(factors)
    
    # 滚动回归
    rolling = attribution.run_rolling_regression(fund_data, factors, window=12)