# -*- coding: utf-8 -*-
"""
回测模块 - 风格漂移检测与风格持续性分析

基于研报中的SDS指标和名义风格对比方法
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StyleDriftDetector:
    """
    风格漂移检测器
    
    功能：
    1. 对比名义风格与实际风格
    2. 计算SDS风格漂移指标
    3. 识别风格漂移事件
    """
    
    def __init__(self, nominal_style: str = None):
        """
        Parameters:
        -----------
        nominal_style : str
            基金招募说明书宣称的名义风格，如 "大盘成长", "均衡" 等
        """
        self.nominal_style = nominal_style
        self.sub_period_results = []
        self.sds_score = None
        
    def analyze_sub_periods(self, fund_returns: pd.Series, 
                           index_returns: pd.DataFrame,
                           style_indices: List[str],
                           n_periods: int = 4) -> pd.DataFrame:
        """
        将研究期间划分为多个子区间，分别进行风格分析
        
        Parameters:
        -----------
        fund_returns : pd.Series
            基金完整收益率序列
        index_returns : pd.DataFrame
            风格指数收益率
        style_indices : List[str]
            风格指数代码列表
        n_periods : int
            子区间数量，默认4个（如年度分析分为4个季度）
            
        Returns:
        --------
        pd.DataFrame
            各子区间的风格分析结果
        """
        from .factor import SharpeStyleModel
        
        total_days = len(fund_returns)
        period_length = total_days // n_periods
        
        results = []
        dates = fund_returns.index
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else total_days
            
            period_dates = dates[start_idx:end_idx]
            fund_period = fund_returns.loc[period_dates]
            index_period = index_returns.loc[period_dates]
            
            try:
                model = SharpeStyleModel(style_indices)
                result = model.fit(fund_period, index_period)
                
                result_row = {
                    'period': i + 1,
                    'start_date': period_dates[0].strftime('%Y-%m-%d'),
                    'end_date': period_dates[-1].strftime('%Y-%m-%d'),
                    'n_days': len(period_dates),
                    'r_squared': result['r_squared'],
                    'tracking_error': result['tracking_error'],
                    'style_label': model.get_style_label()
                }
                
                # 添加各风格暴露
                for idx, exp in result['exposures'].items():
                    result_row[f'exp_{idx}'] = round(exp, 4)
                
                results.append(result_row)
                self.sub_period_results.append(result['exposures'])
                
            except Exception as e:
                print(f"[WARN] 子区间 {i+1} 分析失败: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def compute_sds(self) -> float:
        """
        计算SDS风格漂移指标
        
        基于Idzorek方法：
        SDS = √[Var(β_1) + Var(β_2) + ... + Var(β_n)]
        
        Returns:
        --------
        float
            SDS指标值
        """
        from .factor import compute_sds
        
        if len(self.sub_period_results) < 2:
            print("[WARN] 子区间数量不足，无法计算SDS")
            return None
        
        self.sds_score = compute_sds(self.sub_period_results)
        return self.sds_score
    
    def check_style_drift(self, sub_period_df: pd.DataFrame) -> Dict:
        """
        检查是否发生风格漂移
        
        判断逻辑：
        1. 对比名义风格与实际风格
        2. 检查各子区间风格是否一致
        
        Parameters:
        -----------
        sub_period_df : pd.DataFrame
            子区间分析结果
            
        Returns:
        --------
        Dict
            {
                'has_drift': 是否发生漂移,
                'drift_periods': 发生漂移的区间,
                'consistency_score': 风格一致性评分,
                'analysis': 分析说明
            }
        """
        if len(sub_period_df) == 0:
            return {'has_drift': False, 'analysis': '无分析数据'}
        
        # 获取各期风格标签
        style_labels = sub_period_df['style_label'].tolist()
        
        # 检查风格一致性
        unique_styles = list(set(style_labels))
        consistency = len(unique_styles) == 1
        
        # 计算风格一致性评分（基于暴露系数的变异系数）
        exposure_cols = [c for c in sub_period_df.columns if c.startswith('exp_')]
        if exposure_cols:
            cv_scores = []
            for col in exposure_cols:
                values = sub_period_df[col].values
                if np.mean(values) > 0.01:  # 只考虑有显著暴露的风格
                    cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    cv_scores.append(cv)
            
            consistency_score = 1 - np.mean(cv_scores) if cv_scores else 0
        else:
            consistency_score = 0
        
        # 判断是否漂移
        has_drift = not consistency or (self.sds_score and self.sds_score > 0.2)
        
        # 找出风格变化的区间
        drift_periods = []
        for i in range(1, len(style_labels)):
            if style_labels[i] != style_labels[i-1]:
                drift_periods.append({
                    'from_period': i,
                    'from_style': style_labels[i-1],
                    'to_style': style_labels[i]
                })
        
        # 构建分析说明
        analysis_parts = []
        if has_drift:
            analysis_parts.append("检测到风格漂移")
            if drift_periods:
                analysis_parts.append(f"在{len(drift_periods)}个区间发生风格切换")
        else:
            analysis_parts.append("风格保持稳定")
        
        if self.nominal_style:
            actual_style = style_labels[-1] if style_labels else "未知"
            if self.nominal_style not in actual_style and actual_style not in self.nominal_style:
                analysis_parts.append(f"实际风格({actual_style})与名义风格({self.nominal_style})存在偏差")
        
        return {
            'has_drift': has_drift,
            'drift_periods': drift_periods,
            'consistency_score': round(consistency_score, 4),
            'sds_score': round(self.sds_score, 4) if self.sds_score else None,
            'style_history': style_labels,
            'analysis': "；".join(analysis_parts)
        }
    
    def generate_report(self, fund_code: str, fund_name: str = "") -> str:
        """
        生成风格漂移检测报告
        
        Parameters:
        -----------
        fund_code : str
            基金代码
        fund_name : str
            基金名称
            
        Returns:
        --------
        str
            格式化的检测报告
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"基金风格漂移分析报告")
        lines.append("=" * 60)
        lines.append(f"基金代码: {fund_code}")
        if fund_name:
            lines.append(f"基金名称: {fund_name}")
        if self.nominal_style:
            lines.append(f"名义风格: {self.nominal_style}")
        lines.append("")
        
        if self.sds_score is not None:
            lines.append(f"SDS风格漂移指标: {self.sds_score:.4f}")
            if self.sds_score < 0.1:
                lines.append("  → 风格高度稳定")
            elif self.sds_score < 0.2:
                lines.append("  → 风格相对稳定")
            elif self.sds_score < 0.3:
                lines.append("  → 风格存在一定波动")
            else:
                lines.append("  → 风格漂移风险较高")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class StyleBacktest:
    """
    风格分析回测框架
    
    用于评估风格分析模型的有效性
    """
    
    def __init__(self):
        self.results = {}
        
    def run_backtest(self, fund_returns: pd.Series, index_returns: pd.DataFrame,
                     style_indices: List[str], 
                     train_size: int = 126,
                     test_size: int = 21) -> pd.DataFrame:
        """
        运行滚动回测
        
        使用历史数据训练风格模型，预测未来收益
        
        Parameters:
        -----------
        fund_returns : pd.Series
            基金收益率序列
        index_returns : pd.DataFrame
            风格指数收益率
        style_indices : List[str]
            风格指数代码列表
        train_size : int
            训练窗口大小（交易日），默认126天≈6个月
        test_size : int
            测试窗口大小（交易日），默认21天≈1个月
            
        Returns:
        --------
        pd.DataFrame
            回测结果
        """
        from .factor import SharpeStyleModel
        
        results = []
        dates = fund_returns.index
        
        for start_idx in range(0, len(dates) - train_size - test_size + 1, test_size):
            train_start = start_idx
            train_end = start_idx + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            # 训练数据
            train_dates = dates[train_start:train_end]
            fund_train = fund_returns.loc[train_dates]
            index_train = index_returns.loc[train_dates]
            
            # 测试数据
            test_dates = dates[test_start:test_end]
            fund_test = fund_returns.loc[test_dates]
            index_test = index_returns.loc[test_dates]
            
            try:
                # 训练模型
                model = SharpeStyleModel(style_indices)
                train_result = model.fit(fund_train, index_train)
                
                # 预测测试期收益
                exposures = train_result['exposures'].values
                predicted_returns = index_test[train_result['available_indices']].values @ exposures
                actual_returns = fund_test.values
                
                # 计算预测效果
                correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1]
                mse = np.mean((predicted_returns - actual_returns) ** 2)
                
                results.append({
                    'test_start': test_dates[0],
                    'test_end': test_dates[-1],
                    'train_r2': train_result['r_squared'],
                    'pred_correlation': correlation,
                    'pred_mse': mse,
                    'actual_return': np.sum(actual_returns),
                    'predicted_return': np.sum(predicted_returns)
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def evaluate_model(self, backtest_df: pd.DataFrame) -> Dict:
        """
        评估模型预测效果
        
        Parameters:
        -----------
        backtest_df : pd.DataFrame
            回测结果DataFrame
            
        Returns:
        --------
        Dict
            评估指标
        """
        if len(backtest_df) == 0:
            return {'error': '无回测数据'}
        
        return {
            'n_periods': len(backtest_df),
            'avg_train_r2': backtest_df['train_r2'].mean(),
            'avg_pred_correlation': backtest_df['pred_correlation'].mean(),
            'avg_pred_mse': backtest_df['pred_mse'].mean(),
            'direction_accuracy': (
                (backtest_df['actual_return'] * backtest_df['predicted_return'] > 0).mean()
            )
        }
