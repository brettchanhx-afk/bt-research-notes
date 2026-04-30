# -*- coding: utf-8 -*-
"""
backtest.py - 债券基金风格回测模块

功能：
1. 风格稳定性检验（滚动窗口回测）
2. 风格漂移检测
3. 绩效归因分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StyleBacktest:
    """债券基金风格回测分析器"""
    
    def __init__(self, estimator):
        """
        Parameters:
        -----------
        estimator : BondStyleEstimator
            风格估计器实例
        """
        self.estimator = estimator
        self.rolling_results = None
        
    def run_rolling_backtest(self, X: pd.DataFrame, y: pd.Series,
                            window: int = 60, step: int = 20) -> pd.DataFrame:
        """
        运行滚动窗口回测
        
        Parameters:
        -----------
        X : pd.DataFrame
            指数收益率
        y : pd.Series
            基金收益率
        window : int
            滚动窗口（交易日，约3个月=60天）
        step : int
            滚动步长（交易日，约1个月=20天）
            
        Returns:
        --------
        pd.DataFrame
            滚动回测结果
        """
        logger.info(f"开始滚动回测: 窗口={window}, 步长={step}")
        
        results = []
        n_periods = (len(y) - window) // step + 1
        
        for i in range(window, len(y), step):
            start_idx = i - window
            end_idx = i
            
            X_window = X.iloc[start_idx:end_idx]
            y_window = y.iloc[start_idx:end_idx]
            end_date = y.index[end_idx - 1]
            
            try:
                # 拟合模型
                fit_result = self.estimator.fit(X_window, y_window)
                
                # 估计久期和信用
                dur_result = self.estimator.estimate_duration()
                cred_result = self.estimator.estimate_credit()
                
                results.append({
                    "end_date": end_date,
                    "duration": dur_result["estimated_duration"],
                    "credit": cred_result["estimated_credit"],
                    "r2": fit_result["r2"],
                    "residual_std": fit_result["residual_std"],
                    "n_samples": fit_result["n_samples"],
                    "duration_style": dur_result["style_label"],
                    "credit_style": cred_result["style_label"]
                })
                
            except Exception as e:
                logger.warning(f"窗口 {end_date} 回测失败: {e}")
        
        self.rolling_results = pd.DataFrame(results)
        logger.info(f"回测完成: 共 {len(results)} 个窗口")
        
        return self.rolling_results
    
    def calculate_style_stability(self) -> Dict:
        """
        计算风格稳定性指标
        
        Returns:
        --------
        Dict
            稳定性指标
        """
        if self.rolling_results is None:
            raise ValueError("请先运行滚动回测")
        
        df = self.rolling_results
        
        # 久期稳定性
        duration_mean = df["duration"].mean()
        duration_std = df["duration"].std()
        duration_cv = duration_std / duration_mean if duration_mean != 0 else np.inf
        
        # 信用稳定性
        credit_mean = df["credit"].mean()
        credit_std = df["credit"].std()
        credit_cv = credit_std / credit_mean if credit_mean != 0 else np.inf
        
        # R² 稳定性
        r2_mean = df["r2"].mean()
        r2_std = df["r2"].std()
        
        # 风格一致性（风格标签变化次数）
        duration_changes = (df["duration_style"] != df["duration_style"].shift(1)).sum() - 1
        credit_changes = (df["credit_style"] != df["credit_style"].shift(1)).sum() - 1
        
        return {
            "duration": {
                "mean": duration_mean,
                "std": duration_std,
                "cv": duration_cv,
                "changes": duration_changes
            },
            "credit": {
                "mean": credit_mean,
                "std": credit_std,
                "cv": credit_cv,
                "changes": credit_changes
            },
            "r2": {
                "mean": r2_mean,
                "std": r2_std
            },
            "style_consistency": {
                "duration_stable": duration_changes <= len(df) * 0.2,
                "credit_stable": credit_changes <= len(df) * 0.2
            }
        }
    
    def detect_style_drift(self, threshold: float = 2.0) -> List[Dict]:
        """
        检测风格漂移点
        
        当风格指标变化超过threshold倍标准差时，认为发生风格漂移
        
        Parameters:
        -----------
        threshold : float
            漂移检测阈值（标准差倍数）
            
        Returns:
        --------
        List[Dict]
            漂移点列表
        """
        if self.rolling_results is None:
            raise ValueError("请先运行滚动回测")
        
        df = self.rolling_results
        drifts = []
        
        # 久期漂移检测
        duration_changes = df["duration"].diff().abs()
        duration_threshold = df["duration"].std() * threshold
        
        for idx in duration_changes[duration_changes > duration_threshold].index:
            drifts.append({
                "date": df.loc[idx, "end_date"],
                "type": "久期漂移",
                "change": duration_changes.loc[idx],
                "from_value": df.loc[idx-1, "duration"] if idx > 0 else None,
                "to_value": df.loc[idx, "duration"]
            })
        
        # 信用漂移检测
        credit_changes = df["credit"].diff().abs()
        credit_threshold = df["credit"].std() * threshold
        
        for idx in credit_changes[credit_changes > credit_threshold].index:
            drifts.append({
                "date": df.loc[idx, "end_date"],
                "type": "信用漂移",
                "change": credit_changes.loc[idx],
                "from_value": df.loc[idx-1, "credit"] if idx > 0 else None,
                "to_value": df.loc[idx, "credit"]
            })
        
        return drifts
    
    def performance_attribution(self, fund_returns: pd.Series, 
                               index_returns: pd.DataFrame) -> Dict:
        """
        绩效归因分析
        
        将基金收益归因于各债券指数因子的暴露
        
        Parameters:
        -----------
        fund_returns : pd.Series
            基金收益率
        index_returns : pd.DataFrame
            指数收益率
            
        Returns:
        --------
        Dict
            归因结果
        """
        # 确保数据对齐
        common_dates = fund_returns.index.intersection(index_returns.index)
        fund_ret = fund_returns.loc[common_dates]
        index_ret = index_returns.loc[common_dates]
        
        # 计算各指数的平均收益率（作为因子收益）
        factor_returns = index_ret.mean()
        
        # 获取回归系数作为因子暴露
        coef = self.estimator.coef_
        
        # 归因：收益 = Σ(暴露 × 因子收益)
        attribution = coef * factor_returns
        
        # 分类归因
        index_info = self.estimator.index_info
        
        # 按久期分类归因
        duration_attribution = {}
        for dur_range, label in [("短久期", (0, 3.5)), ("中久期", (3.5, 6)), ("长久期", (6, 20))]:
            mask = (index_info["duration"] >= label[0]) & (index_info["duration"] < label[1])
            codes = index_info[mask].index.intersection(attribution.index)
            duration_attribution[dur_range] = attribution.loc[codes].sum()
        
        # 按信用分类归因
        credit_attribution = {}
        for credit_range, label in [("高信用", (16, 20)), ("中信用", (14, 16)), ("低信用", (0, 14))]:
            mask = (index_info["credit_score"] >= label[0]) & (index_info["credit_score"] < label[1])
            codes = index_info[mask].index.intersection(attribution.index)
            credit_attribution[credit_range] = attribution.loc[codes].sum()
        
        return {
            "total_attribution": attribution.sum(),
            "factor_attribution": attribution.to_dict(),
            "duration_attribution": duration_attribution,
            "credit_attribution": credit_attribution,
            "alpha": fund_ret.mean() - attribution.sum()  # 超额收益
        }


def calculate_performance_metrics(returns: pd.Series, 
                                  risk_free_rate: float = 0.02) -> Dict:
    """
    计算绩效指标
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列
    risk_free_rate : float
        无风险利率（年化）
        
    Returns:
    --------
    Dict
        绩效指标
    """
    # 年化收益率
    annual_return = returns.mean() * 252
    
    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0
    
    # 最大回撤
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar
    }


if __name__ == "__main__":
    print("Style Backtest Module - Test")
