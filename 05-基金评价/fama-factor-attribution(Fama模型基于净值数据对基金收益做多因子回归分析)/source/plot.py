"""
可视化模块

提供Fama-French归因分析的可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（适配Windows/macOS/Linux）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


class FamaFrenchPlotter:
    """
    Fama-French归因分析可视化器
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        初始化
        
        Parameters
        ----------
        style : str
            matplotlib样式
        """
        self.style = style
        plt.style.use(style)
    
    def plot_factor_exposure(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        绘制因子暴露系数图
        
        Parameters
        ----------
        results : Dict
            回归结果
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        factor_names = ['R_M', 'SMB', 'HML', 'RMW', 'CMA']
        factor_labels = ['市场因子\n(R_M)', '市值因子\n(SMB)', '价值因子\n(HML)', 
                        '盈利因子\n(RMW)', '投资因子\n(CMA)']
        
        coefficients = [results['params'].get(f, 0) for f in factor_names]
        errors = [results['std_errors'].get(f, 0) for f in factor_names]
        colors = ['#2E86AB' if c >= 0 else '#E94F37' for c in coefficients]
        
        bars = ax.bar(factor_labels, coefficients, yerr=errors, color=colors, 
                      alpha=0.8, edgecolor='black', capsize=5)
        
        # 添加显著性标记
        for i, factor in enumerate(factor_names):
            p_val = results['p_values'].get(factor, 1)
            if p_val < 0.01:
                sig_mark = '***'
            elif p_val < 0.05:
                sig_mark = '**'
            elif p_val < 0.1:
                sig_mark = '*'
            else:
                sig_mark = ''
            
            if sig_mark:
                ax.text(i, coefficients[i] + (0.01 if coefficients[i] >=0 else -0.02), 
                        sig_mark, ha='center', fontsize=12, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title('Fama-French五因子暴露系数', fontsize=14, fontweight='bold')
        ax.set_ylabel('因子暴露系数', fontsize=12)
        ax.set_xlabel('因子名称', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_return_decomposition(
        self,
        contribution_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        绘制收益分解饼图
        
        Parameters
        ----------
        contribution_df : pd.DataFrame
            因子贡献数据
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用绝对值绘图（饼图不支持负值）
        labels = contribution_df['因子'].tolist()
        sizes_abs = [abs(s) for s in contribution_df['贡献'].tolist()]
        
        # 添加颜色映射，正负用不同颜色标记
        colors = []
        for s in contribution_df['贡献']:
            if s >= 0:
                colors.append('#66b3ff')
            else:
                colors.append('#ff9999')
        
        contrib_abs = [abs(s) for s in contribution_df['贡献'].tolist()]
        wedges, texts, autotexts = ax.pie(contrib_abs, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90, 
                                          textprops={'fontsize': 11})
        
        ax.set_title(f'基金收益因子贡献分解\n(正: 盈利贡献 / 负: 亏损贡献)', fontsize=14, fontweight='bold')
        
        # 添加图例
        # 图例显示原始值（带正负号）和占比
        total_abs = sum(abs(s) for s in contribution_df['贡献'])
        ax.legend(wedges, [f'{l}: {s:+.4f} ({abs(s)/total_abs*100:.1f}%)' 
                           for l, s in zip(labels, contribution_df['贡献'].tolist())],
                  title="因子明细", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rolling_exposure(
        self,
        rolling_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        绘制滚动窗口因子暴露变化图
        
        Parameters
        ----------
        rolling_df : pd.DataFrame
            滚动回归结果
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        factor_names = ['R_M', 'SMB', 'HML', 'RMW', 'CMA', 'Alpha']
        factor_labels = ['市场因子(R_M)', '市值因子(SMB)', '价值因子(HML)', 
                        '盈利因子(RMW)', '投资因子(CMA)', 'Alpha']
        
        for i, (factor, label) in enumerate(zip(factor_names, factor_labels)):
            if factor in rolling_df.columns:
                ax = axes[i]
                rolling_df[factor].plot(ax=ax, color='#2E86AB', linewidth=2)
                ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
                ax.set_title(f'{label}滚动暴露', fontsize=12)
                ax.set_ylabel('暴露系数', fontsize=10)
                ax.grid(linestyle='--', alpha=0.7)
        
        # 隐藏最后一个空子图
        if len(factor_names) < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle('滚动窗口因子暴露变化（12个月窗口）', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_factor_correlation(
        self,
        factors: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        绘制因子相关性热图
        
        Parameters
        ----------
        factors : pd.DataFrame
            五因子数据
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = factors.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax, fmt='.2f')
        
        ax.set_title('Fama-French五因子相关性矩阵', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cumulative_returns(
        self,
        fund_returns: pd.Series,
        fitted_returns: pd.Series,
        save_path: Optional[str] = None
    ):
        """
        绘制基金实际收益与模型拟合收益对比
        
        Parameters
        ----------
        fund_returns : pd.Series
            基金实际收益率
        fitted_returns : pd.Series
            模型拟合收益率
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 对齐日期
        common_idx = fund_returns.index.intersection(fitted_returns.index)
        fund_returns_aligned = fund_returns.loc[common_idx]
        fitted_returns_aligned = fitted_returns.loc[common_idx]
        
        # 计算累计收益
        fund_cum = (1 + fund_returns_aligned).cumprod()
        fitted_cum = (1 + fitted_returns_aligned).cumprod()
        
        fund_cum.plot(ax=ax, label='基金实际收益', color='#2E86AB', linewidth=2)
        fitted_cum.plot(ax=ax, label='模型拟合收益', color='#E94F37', linewidth=2, linestyle='--')
        
        ax.set_title('基金实际收益 vs 模型拟合收益', fontsize=14, fontweight='bold')
        ax.set_ylabel('累计净值', fontsize=12)
        ax.set_xlabel('日期', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()