# -*- coding: utf-8 -*-
"""
因子计算模块 - 威廉·夏普风格分析模型

核心算法（来自华泰研报）：
min Σ(R - Σβ_j * x_j)^2
s.t. Σβ_j = 1, 0 ≤ β_j ≤ 1

其中：
- R: 基金收益率序列
- x_j: 第j个风格指数收益率序列  
- β_j: 基金在第j个风格指数上的暴露
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SharpeStyleModel:
    """
    威廉·夏普风格分析模型
    
    基于收益率序列，通过带约束的二次规划求解风格暴露
    """
    
    def __init__(self, style_indices: List[str]):
        """
        Parameters:
        -----------
        style_indices : List[str]
            风格指数代码列表，如 ['000300.SH', '000905.SH', '000918.SH', '000919.SH']
        """
        self.style_indices = style_indices
        self.exposures = None
        self.r_squared = None
        self.residual_std = None
        
    def fit(self, fund_returns: pd.Series, index_returns: pd.DataFrame) -> Dict:
        """
        拟合风格分析模型
        
        Parameters:
        -----------
        fund_returns : pd.Series
            基金日收益率序列，index为日期
        index_returns : pd.DataFrame
            风格指数日收益率，列为指数代码，index为日期
            
        Returns:
        --------
        Dict
            {
                'exposures': 风格暴露系数,
                'r_squared': R平方,
                'residual_std': 残差标准差,
                'tracking_error': 跟踪误差,
                'fitted_returns': 拟合收益率,
                'residuals': 残差
            }
        """
        # 对齐日期
        common_dates = fund_returns.index.intersection(index_returns.index)
        fund_r = fund_returns.loc[common_dates].dropna()
        index_r = index_returns.loc[common_dates].dropna()
        
        # 确保所有风格指数都有数据
        available_indices = [col for col in self.style_indices if col in index_r.columns]
        if len(available_indices) < 2:
            raise ValueError(f"可用风格指数不足，至少需要2个，当前只有{len(available_indices)}个")
        
        index_r = index_r[available_indices]
        
        # 对齐后再次检查
        common_dates = fund_r.index.intersection(index_r.index)
        fund_r = fund_r.loc[common_dates].values
        index_r = index_r.loc[common_dates].values
        
        n_samples = len(fund_r)
        n_indices = len(available_indices)
        
        # 定义目标函数：残差平方和
        def objective(beta):
            fitted = index_r @ beta
            residuals = fund_r - fitted
            return np.sum(residuals ** 2)
        
        # 约束条件
        constraints = {'type': 'eq', 'fun': lambda beta: np.sum(beta) - 1.0}
        
        # 边界条件：0 <= beta <= 1
        bounds = [(0, 1) for _ in range(n_indices)]
        
        # 初始值：等权重
        x0 = np.ones(n_indices) / n_indices
        
        # 优化求解
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            print(f"[WARN] 优化未收敛: {result.message}")
        
        # 计算结果
        beta_optimal = result.x
        fitted_returns = index_r @ beta_optimal
        residuals = fund_r - fitted_returns
        
        # R平方
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((fund_r - np.mean(fund_r)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # 残差标准差（年化）
        residual_std = np.std(residuals) * np.sqrt(252)
        
        # 跟踪误差（年化）
        tracking_error = np.std(residuals) * np.sqrt(252)
        
        # 保存结果
        self.exposures = pd.Series(beta_optimal, index=available_indices)
        self.r_squared = r_squared
        self.residual_std = residual_std
        
        return {
            'exposures': self.exposures,
            'r_squared': r_squared,
            'residual_std': residual_std,
            'tracking_error': tracking_error,
            'fitted_returns': pd.Series(fitted_returns, index=common_dates),
            'residuals': pd.Series(residuals, index=common_dates),
            'available_indices': available_indices
        }
    
    def get_style_label(self, exposure_threshold: float = 0.3) -> str:
        """
        根据暴露系数判断基金风格标签
        
        Parameters:
        -----------
        exposure_threshold : float
            判定为主要风格的暴露阈值
            
        Returns:
        --------
        str
            风格标签，如 "大盘成长", "均衡", "小盘价值" 等
        """
        if self.exposures is None:
            raise ValueError("请先调用 fit() 方法拟合模型")
        
        # 找出主要暴露的风格
        primary_styles = self.exposures[self.exposures >= exposure_threshold]
        
        if len(primary_styles) == 0:
            # 没有明显风格，取最大暴露
            max_style = self.exposures.idxmax()
            return f"偏{self._get_style_name(max_style)}"
        
        if len(primary_styles) >= 3:
            return "均衡配置"
        
        # 组合风格标签
        style_names = [self._get_style_name(s) for s in primary_styles.index]
        return "".join(style_names)
    
    def _get_style_name(self, index_code: str) -> str:
        """将指数代码转换为风格名称"""
        mapping = {
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '000906.SH': '中证800',
            '000918.SH': '成长',
            '000919.SH': '价值',
            '000920.SH': '成长',
            '000921.SH': '价值',
            '000044.SH': '大盘',
            '000045.SH': '中盘',
            '000046.SH': '小盘',
            '000901.SH': '成长',
            '000902.SH': '价值',
        }
        return mapping.get(index_code, index_code)


def compute_sds(exposures_list: List[pd.Series]) -> float:
    """
    计算风格漂移指标 SDS (Style Drift Score)
    
    基于Idzorek方法，衡量基金风格稳定性
    公式：SDS = √[Var(β_i1) + Var(β_i2) + ... + Var(β_in)]
    
    Parameters:
    -----------
    exposures_list : List[pd.Series]
        多个子区间的风格暴露系数列表
        
    Returns:
    --------
    float
        SDS指标值，越大表示风格越不稳定
    """
    if len(exposures_list) < 2:
        return 0.0
    
    # 对齐所有暴露系数的索引
    all_indices = set()
    for exp in exposures_list:
        all_indices.update(exp.index)
    all_indices = sorted(list(all_indices))
    
    # 构建暴露矩阵
    exposure_matrix = []
    for exp in exposures_list:
        aligned = pd.Series(index=all_indices, dtype=float)
        for idx in all_indices:
            aligned[idx] = exp.get(idx, 0.0)
        exposure_matrix.append(aligned.values)
    
    exposure_matrix = np.array(exposure_matrix)
    
    # 计算每个风格指数的方差
    variances = np.var(exposure_matrix, axis=0, ddof=1)
    
    # SDS = 方差之和的平方根
    sds = np.sqrt(np.sum(variances))
    
    return sds


class RollingStyleAnalyzer:
    """
    滚动窗口风格分析器
    
    用于分析基金风格随时间的变化
    """
    
    def __init__(self, window: int = 63, step: int = 21):
        """
        Parameters:
        -----------
        window : int
            滚动窗口大小（交易日），默认63天≈3个月
        step : int
            滚动步长（交易日），默认21天≈1个月
        """
        self.window = window
        self.step = step
        self.results = []
        
    def analyze(self, fund_returns: pd.Series, index_returns: pd.DataFrame,
                style_indices: List[str]) -> pd.DataFrame:
        """
        执行滚动窗口风格分析
        
        Parameters:
        -----------
        fund_returns : pd.Series
            基金收益率序列
        index_returns : pd.DataFrame
            风格指数收益率
        style_indices : List[str]
            风格指数代码列表
            
        Returns:
        --------
        pd.DataFrame
            每个窗口的风格分析结果
        """
        results = []
        dates = fund_returns.index
        
        for start_idx in range(0, len(dates) - self.window + 1, self.step):
            end_idx = start_idx + self.window
            window_dates = dates[start_idx:end_idx]
            
            fund_window = fund_returns.loc[window_dates]
            index_window = index_returns.loc[window_dates]
            
            try:
                model = SharpeStyleModel(style_indices)
                result = model.fit(fund_window, index_window)
                
                result_row = {
                    'start_date': window_dates[0],
                    'end_date': window_dates[-1],
                    'r_squared': result['r_squared'],
                    'tracking_error': result['tracking_error'],
                }
                
                # 添加各风格暴露
                for idx, exp in result['exposures'].items():
                    result_row[f'exp_{idx}'] = exp
                
                results.append(result_row)
                
            except Exception as e:
                print(f"[WARN] 窗口 {window_dates[0]} - {window_dates[-1]} 分析失败: {e}")
                continue
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def detect_drift(self, threshold: float = 0.15) -> List[Dict]:
        """
        检测风格漂移事件
        
        Parameters:
        -----------
        threshold : float
            风格漂移判定阈值（暴露变化幅度）
            
        Returns:
        --------
        List[Dict]
            风格漂移事件列表
        """
        if len(self.results) < 2:
            return []
        
        drift_events = []
        exposure_cols = [c for c in self.results.columns if c.startswith('exp_')]
        
        for i in range(1, len(self.results)):
            prev = self.results.iloc[i-1]
            curr = self.results.iloc[i]
            
            # 计算各风格暴露变化
            max_change = 0
            changed_style = None
            
            for col in exposure_cols:
                change = abs(curr[col] - prev[col])
                if change > max_change:
                    max_change = change
                    changed_style = col.replace('exp_', '')
            
            if max_change > threshold:
                drift_events.append({
                    'date': curr['end_date'],
                    'changed_style': changed_style,
                    'change_magnitude': max_change,
                    'prev_exposure': prev[f'exp_{changed_style}'],
                    'curr_exposure': curr[f'exp_{changed_style}']
                })
        
        return drift_events
