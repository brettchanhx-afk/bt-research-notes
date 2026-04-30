# -*- coding: utf-8 -*-
"""
factor.py - 债券基金风格估计核心模块

复现华泰证券研报核心算法：
1. 久期配置风格估计：基于净值回归估计基金久期
2. 信用配置风格估计：基于净值回归估计基金信用评分

核心公式（研报公式1、2）：
- (1) R = α + β₁×R₁ + ... + βₙ×Rₙ  （收益率回归）
- (2) D = α + β₁×D₁ + ... + βₙ×Dₙ  （久期估计）
- (2) C = α + β₁×C₁ + ... + βₙ×Cₙ  （信用评分估计）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BondStyleEstimator:
    """
    债券基金风格估计器
    
    基于净值回归方法估计基金的久期配置风格和信用配置风格
    """
    
    def __init__(self, index_info: pd.DataFrame, constraint_sum: bool = True):
        """
        Parameters:
        -----------
        index_info : pd.DataFrame
            指数信息表，columns: [code, name, duration, credit_score, type]
        constraint_sum : bool
            是否约束回归系数之和为1（研报方法）
        """
        self.index_info = index_info.set_index("code")
        self.constraint_sum = constraint_sum
        self.model = None
        self.coef_ = None
        self.intercept_ = None
        self.r2_ = None
        
    def prepare_data(self, fund_returns: pd.Series, 
                     index_returns: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备回归数据
        
        Parameters:
        -----------
        fund_returns : pd.Series
            基金日收益率序列，index为日期
        index_returns : Dict[str, pd.Series]
            各指数日收益率序列，{index_code: returns_series}
            
        Returns:
        --------
        X : pd.DataFrame
            指数收益率矩阵（自变量）
        y : pd.Series
            基金收益率（因变量）
        """
        # 合并所有数据
        data = pd.DataFrame({"fund": fund_returns})
        
        for code, returns in index_returns.items():
            data[code] = returns
        
        # 删除缺失值
        data = data.dropna()
        
        if len(data) == 0:
            raise ValueError("无有效数据用于回归")
        
        y = data["fund"]
        X = data.drop(columns=["fund"])
        
        return X, y
    
    def fit(self, X: pd.DataFrame, y: pd.Series, method: str = "ols") -> Dict:
        """
        拟合回归模型（研报公式1）
        
        R = α + β₁×R₁ + ... + βₙ×Rₙ
        
        Parameters:
        -----------
        X : pd.DataFrame
            指数收益率矩阵
        y : pd.Series
            基金收益率
        method : str
            回归方法："ols"普通最小二乘，"ridge"岭回归
            
        Returns:
        --------
        Dict
            回归结果
        """
        if method == "ols":
            model = LinearRegression(fit_intercept=True)
        elif method == "ridge":
            model = Ridge(alpha=0.1, fit_intercept=True)
        else:
            raise ValueError(f"未知的回归方法: {method}")
        
        model.fit(X, y)
        
        # 保存结果
        self.model = model
        self.coef_ = pd.Series(model.coef_, index=X.columns)
        self.intercept_ = model.intercept_
        
        # 计算R²
        y_pred = model.predict(X)
        self.r2_ = r2_score(y, y_pred)
        
        # 如果约束系数和为1，进行归一化处理
        if self.constraint_sum and self.coef_.sum() != 0:
            coef_sum = self.coef_.sum()
            self.coef_ = self.coef_ / coef_sum
            self.intercept_ = 0  # 归一化后截距设为0
        
        # 计算残差标准差
        residuals = y - y_pred
        residual_std = np.std(residuals)
        
        return {
            "coef": self.coef_,
            "intercept": self.intercept_,
            "r2": self.r2_,
            "residual_std": residual_std,
            "n_samples": len(y)
        }
    
    def estimate_duration(self) -> Dict:
        """
        估计基金久期（研报公式2）
        
        D = α + β₁×D₁ + ... + βₙ×Dₙ
        
        Returns:
        --------
        Dict
            {
                "estimated_duration": 估计久期,
                "duration_contrib": 各指数久期贡献,
                "r2": 回归R²
            }
        """
        if self.coef_ is None:
            raise ValueError("请先调用fit()方法拟合模型")
        
        # 获取各指数的久期
        index_durations = self.index_info.loc[self.coef_.index, "duration"]
        
        # 计算久期估计值
        duration_contrib = self.coef_ * index_durations
        estimated_duration = duration_contrib.sum() + self.intercept_
        
        # 久期风格分类
        if estimated_duration < 3.5:
            style_label = "短久期"
        elif estimated_duration < 6:
            style_label = "中久期"
        else:
            style_label = "长久期"
        
        return {
            "estimated_duration": estimated_duration,
            "duration_contrib": duration_contrib,
            "intercept": self.intercept_,
            "r2": self.r2_,
            "style_label": style_label,
            "coef": self.coef_
        }
    
    def estimate_credit(self) -> Dict:
        """
        估计基金信用评分（研报公式2变体）
        
        C = α + β₁×C₁ + ... + βₙ×Cₙ
        
        Returns:
        --------
        Dict
            {
                "estimated_credit": 估计信用评分,
                "credit_contrib": 各指数信用贡献,
                "r2": 回归R²
            }
        """
        if self.coef_ is None:
            raise ValueError("请先调用fit()方法拟合模型")
        
        # 获取各指数的信用评分
        index_credits = self.index_info.loc[self.coef_.index, "credit_score"]
        
        # 计算信用评分估计值
        credit_contrib = self.coef_ * index_credits
        estimated_credit = credit_contrib.sum() + self.intercept_
        
        # 信用风格分类（基于研报信用评分标准）
        # AAA=16.5, AA+=15, AA=14, AA-=12.5
        if estimated_credit >= 16:
            style_label = "高信用(AAA级)"
        elif estimated_credit >= 14.5:
            style_label = "中高信用(AA+级)"
        elif estimated_credit >= 13:
            style_label = "中信用(AA级)"
        else:
            style_label = "低信用(AA-及以下)"
        
        return {
            "estimated_credit": estimated_credit,
            "credit_contrib": credit_contrib,
            "intercept": self.intercept_,
            "r2": self.r2_,
            "style_label": style_label,
            "coef": self.coef_
        }
    
    def get_style_box(self) -> Dict:
        """
        获取风格箱定位（久期 × 信用）
        
        Returns:
        --------
        Dict
            风格箱定位结果
        """
        duration_result = self.estimate_duration()
        credit_result = self.estimate_credit()
        
        # 3×3 风格箱定位
        dur = duration_result["estimated_duration"]
        cred = credit_result["estimated_credit"]
        
        # 久期维度：短(<3.5) / 中(3.5-6) / 长(>6)
        if dur < 3.5:
            dur_pos = "短"
        elif dur < 6:
            dur_pos = "中"
        else:
            dur_pos = "长"
        
        # 信用维度：高(>=16) / 中(14-16) / 低(<14)
        if cred >= 16:
            cred_pos = "高"
        elif cred >= 14:
            cred_pos = "中"
        else:
            cred_pos = "低"
        
        style_box = f"{dur_pos}久期{cred_pos}信用"
        
        return {
            "duration": dur,
            "credit": cred,
            "duration_label": duration_result["style_label"],
            "credit_label": credit_result["style_label"],
            "style_box": style_box,
            "r2": self.r2_,
            "coef": self.coef_
        }
    
    def rolling_estimate(self, X: pd.DataFrame, y: pd.Series, 
                         window: int = 60, step: int = 20) -> pd.DataFrame:
        """
        滚动窗口估计风格漂移
        
        Parameters:
        -----------
        X : pd.DataFrame
            指数收益率
        y : pd.Series
            基金收益率
        window : int
            滚动窗口大小（交易日）
        step : int
            滚动步长
            
        Returns:
        --------
        pd.DataFrame
            滚动估计结果
        """
        results = []
        
        for i in range(window, len(y), step):
            X_window = X.iloc[i-window:i]
            y_window = y.iloc[i-window:i]
            end_date = y.index[i-1]
            
            try:
                self.fit(X_window, y_window)
                dur_result = self.estimate_duration()
                cred_result = self.estimate_credit()
                
                results.append({
                    "date": end_date,
                    "duration": dur_result["estimated_duration"],
                    "credit": cred_result["estimated_credit"],
                    "r2": self.r2_,
                    "duration_label": dur_result["style_label"],
                    "credit_label": cred_result["style_label"]
                })
            except Exception as e:
                logger.warning(f"窗口 {end_date} 估计失败: {e}")
        
        return pd.DataFrame(results)


def calculate_macaulay_duration(cash_flows: List[float], 
                                times: List[float], 
                                ytm: float) -> float:
    """
    计算麦考利久期（研报定义）
    
    Macaulay's duration = Σ(t × wₜ)
    其中 wₜ = CFₜ/(1+y)ᵗ / P
    
    Parameters:
    -----------
    cash_flows : List[float]
        各期现金流
    times : List[float]
        各期时间（年）
    ytm : float
        到期收益率
        
    Returns:
    --------
    float
        麦考利久期
    """
    pv_factors = [(1 + ytm) ** t for t in times]
    pv_cash_flows = [cf / pv for cf, pv in zip(cash_flows, pv_factors)]
    
    price = sum(pv_cash_flows)
    weights = [pv / price for pv in pv_cash_flows]
    
    macaulay_duration = sum(t * w for t, w in zip(times, weights))
    
    return macaulay_duration


def calculate_modified_duration(macaulay_duration: float, ytm: float) -> float:
    """
    计算修正久期（研报定义）
    
    Modified duration = Macaulay's duration / (1 + y)
    
    Parameters:
    -----------
    macaulay_duration : float
        麦考利久期
    ytm : float
        到期收益率
        
    Returns:
    --------
    float
        修正久期
    """
    return macaulay_duration / (1 + ytm)


# 信用评分映射表（人民银行2006年规范）
CREDIT_RATING_SCORE = {
    "AAA": 16.5,
    "AA+": 15.0,
    "AA": 14.0,
    "AA-": 12.5,
    "A+": 11.0,
    "A": 10.0,
    "A-": 9.0,
    "BBB+": 8.0,
    "BBB": 7.0,
    "BBB-": 6.0,
    "BB": 5.0,
    "B": 4.0,
    "CCC": 3.0,
    "CC": 2.0,
    "C": 1.0,
}


def get_credit_score(rating: str) -> float:
    """
    获取信用评分
    
    Parameters:
    -----------
    rating : str
        信用评级，如 "AAA", "AA+"
        
    Returns:
    --------
    float
        信用评分
    """
    return CREDIT_RATING_SCORE.get(rating.upper(), 10.0)


if __name__ == "__main__":
    # 测试代码
    print("Bond Style Estimator - Test")
    
    # 测试麦考利久期计算
    cash_flows = [5, 5, 5, 105]  # 3年期债券，票面利率5%
    times = [1, 2, 3, 3]
    ytm = 0.05
    
    mac_dur = calculate_macaulay_duration(cash_flows, times, ytm)
    mod_dur = calculate_modified_duration(mac_dur, ytm)
    
    print(f"麦考利久期: {mac_dur:.2f}")
    print(f"修正久期: {mod_dur:.2f}")
