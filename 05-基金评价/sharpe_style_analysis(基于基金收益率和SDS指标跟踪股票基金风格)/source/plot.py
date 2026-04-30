# -*- coding: utf-8 -*-
"""
可视化模块 - 威廉·夏普风格分析图表

包含：
1. 风格暴露雷达图/条形图
2. 风格漂移时间序列图
3. 实际vs拟合收益对比
4. 风格九宫格定位图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class StyleVisualizer:
    """
    风格分析可视化工具
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def plot_style_exposure(self, exposures: pd.Series, title: str = "风格暴露分析",
                           save_path: str = None) -> plt.Figure:
        """
        绘制风格暴露条形图
        
        Parameters:
        -----------
        exposures : pd.Series
            风格暴露系数，index为风格名称
        title : str
            图表标题
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 排序并过滤小值
        exposures = exposures.sort_values(ascending=True)
        exposures = exposures[exposures > 0.01]  # 只显示显著暴露
        
        colors = ['#e74c3c' if x > 0.5 else '#3498db' if x > 0.2 else '#95a5a6' 
                  for x in exposures.values]
        
        bars = ax.barh(range(len(exposures)), exposures.values, color=colors)
        ax.set_yticks(range(len(exposures)))
        ax.set_yticklabels(exposures.index)
        ax.set_xlabel('暴露系数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, exposures.values)):
            ax.text(val + 0.01, i, f'{val:.2%}', va='center', fontsize=10)
        
        # 添加网格线
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='主要风格阈值')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def plot_style_timeline(self, rolling_results: pd.DataFrame, 
                           style_indices: List[str],
                           title: str = "风格暴露时序变化",
                           save_path: str = None) -> plt.Figure:
        """
        绘制风格暴露时间序列图
        
        Parameters:
        -----------
        rolling_results : pd.DataFrame
            滚动窗口分析结果
        style_indices : List[str]
            风格指数代码列表
        title : str
            图表标题
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 提取暴露列
        exposure_cols = [c for c in rolling_results.columns if c.startswith('exp_')]
        
        # 绘制堆叠面积图
        dates = pd.to_datetime(rolling_results['end_date'])
        exposure_data = rolling_results[exposure_cols].values.T
        
        # 简化标签
        labels = [c.replace('exp_', '').replace('.SH', '') for c in exposure_cols]
        
        ax.stackplot(dates, exposure_data, labels=labels, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel('暴露系数', fontsize=12)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def plot_return_comparison(self, actual_returns: pd.Series,
                               fitted_returns: pd.Series,
                               title: str = "实际收益 vs 风格拟合收益",
                               save_path: str = None) -> plt.Figure:
        """
        绘制实际收益与拟合收益对比图
        
        Parameters:
        -----------
        actual_returns : pd.Series
            实际收益率
        fitted_returns : pd.Series
            拟合收益率
        title : str
            图表标题
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
            matplotlib图表对象
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 对齐日期
        common_dates = actual_returns.index.intersection(fitted_returns.index)
        actual = actual_returns.loc[common_dates]
        fitted = fitted_returns.loc[common_dates]
        
        # 上图：日收益率对比
        ax1 = axes[0]
        ax1.plot(common_dates, actual.cumsum(), label='实际收益', linewidth=1.5)
        ax1.plot(common_dates, fitted.cumsum(), label='风格拟合', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('累计收益', fontsize=11)
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        
        # 下图：残差
        ax2 = axes[1]
        residuals = actual - fitted
        ax2.fill_between(common_dates, residuals, alpha=0.5, color='gray')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel('残差', fontsize=11)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.set_title('残差序列', fontsize=12)
        ax2.grid(alpha=0.3)
        
        # 计算R²并显示
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        ax1.text(0.02, 0.95, f'R² = {r_squared:.4f}', 
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def plot_style_grid(self, size_score: float, vg_score: float,
                       fund_name: str = "",
                       save_path: str = None) -> plt.Figure:
        """
        绘制风格九宫格定位图（结合晨星风格箱）
        
        Parameters:
        -----------
        size_score : float
            规模得分 (0-300)
        vg_score : float
            价值-成长得分 (0-300)
        fund_name : str
            基金名称
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制九宫格背景
        # 规模分界线：100, 200
        # 价值-成长分界线：100, 200
        
        # 填充九宫格
        colors_grid = [
            ['#ffcccc', '#ffe6cc', '#ffffcc'],  # 大盘：价值、平衡、成长
            ['#e6ccff', '#f0f0f0', '#ccffcc'],  # 中盘：价值、平衡、成长
            ['#ccccff', '#ccffff', '#ccffe6'],  # 小盘：价值、平衡、成长
        ]
        
        for i in range(3):
            for j in range(3):
                rect = plt.Rectangle((i*100, j*100), 100, 100, 
                                     facecolor=colors_grid[2-j][i], 
                                     edgecolor='gray', linewidth=1)
                ax.add_patch(rect)
        
        # 添加标签
        size_labels = ['小盘', '中盘', '大盘']
        vg_labels = ['价值型', '平衡型', '成长型']
        
        for i, label in enumerate(vg_labels):
            ax.text(i*100 + 50, -20, label, ha='center', fontsize=11, fontweight='bold')
        
        for i, label in enumerate(size_labels):
            ax.text(-30, i*100 + 50, label, ha='center', fontsize=11, fontweight='bold', rotation=90)
        
        # 标记基金位置
        ax.scatter(vg_score, size_score, s=300, c='red', marker='*', 
                  edgecolors='darkred', linewidths=2, zorder=5)
        
        # 添加基金名称标签
        if fund_name:
            ax.annotate(fund_name, (vg_score, size_score),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # 设置坐标轴
        ax.set_xlim(-50, 350)
        ax.set_ylim(-50, 350)
        ax.set_xlabel('价值-成长得分', fontsize=12)
        ax.set_ylabel('规模得分', fontsize=12)
        ax.set_title('基金风格九宫格定位', fontsize=14, fontweight='bold')
        
        # 添加网格线
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=200, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=100, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=200, color='gray', linestyle='--', linewidth=1)
        
        # 添加得分标注
        ax.text(320, size_score, f'Y={size_score:.1f}', fontsize=10, va='center')
        ax.text(vg_score, 320, f'X={vg_score:.1f}', fontsize=10, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def plot_sds_analysis(self, sub_period_df: pd.DataFrame,
                         sds_score: float,
                         save_path: str = None) -> plt.Figure:
        """
        绘制SDS风格漂移分析图
        
        Parameters:
        -----------
        sub_period_df : pd.DataFrame
            子区间分析结果
        sds_score : float
            SDS指标值
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
            matplotlib图表对象
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 提取暴露列
        exposure_cols = [c for c in sub_period_df.columns if c.startswith('exp_')]
        
        # 上图：各期暴露变化
        ax1 = axes[0]
        x = range(len(sub_period_df))
        
        for col in exposure_cols:
            style_name = col.replace('exp_', '').replace('.SH', '')
            ax1.plot(x, sub_period_df[col], marker='o', label=style_name, linewidth=2)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"P{i+1}" for i in x])
        ax1.set_ylabel('暴露系数', fontsize=11)
        ax1.set_title('各子区间风格暴露变化', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 下图：R²变化
        ax2 = axes[1]
        ax2.bar(x, sub_period_df['r_squared'], color='steelblue', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"P{i+1}" for i in x])
        ax2.set_ylabel('R²', fontsize=11)
        ax2.set_xlabel('子区间', fontsize=11)
        ax2.set_title('模型拟合优度 (R²)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加SDS标注
        fig.text(0.5, 0.02, f'SDS风格漂移指标: {sds_score:.4f}', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def create_summary_report(self, fund_code: str, fund_name: str,
                             style_result: Dict,
                             drift_result: Dict,
                             save_dir: str = None) -> List[str]:
        """
        生成完整的可视化报告（保存所有图表）
        
        Parameters:
        -----------
        fund_code : str
            基金代码
        fund_name : str
            基金名称
        style_result : Dict
            风格分析结果
        drift_result : Dict
            风格漂移检测结果
        save_dir : str
            保存目录
            
        Returns:
        --------
        List[str]
            保存的文件路径列表
        """
        if save_dir is None:
            save_dir = f"output/{fund_code}"
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        # 1. 风格暴露图
        if 'exposures' in style_result:
            fig1 = self.plot_style_exposure(
                style_result['exposures'],
                title=f"{fund_name} ({fund_code}) 风格暴露分析",
                save_path=f"{save_dir}/style_exposure.png"
            )
            saved_files.append(f"{save_dir}/style_exposure.png")
            plt.close(fig1)
        
        # 2. 收益对比图
        if 'fitted_returns' in style_result and 'residuals' in style_result:
            # 这里需要从外部传入actual_returns
            pass
        
        print(f"[OK] 报告已生成，共 {len(saved_files)} 个图表")
        return saved_files
