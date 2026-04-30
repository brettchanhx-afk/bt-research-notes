# -*- coding: utf-8 -*-
"""
plot.py - 债券基金风格分析可视化模块

图表类型：
1. 风格时序图（久期、信用评分变化）
2. 风格箱定位图
3. 回归系数热力图
4. R²时序图
5. 风格漂移检测图
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional
import warnings

# 设置中文字体（必须在plt.style.use之后调用）
def _setup_chinese_font():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120

plt.style.use('seaborn-v0_8-whitegrid')
_setup_chinese_font()


class BondStylePlotter:
    """债券基金风格可视化器"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        
    def plot_style_evolution(self, rolling_results: pd.DataFrame, 
                            save_path: str = None) -> plt.Figure:
        """
        绘制风格演变时序图
        
        Parameters:
        -----------
        rolling_results : pd.DataFrame
            滚动回测结果，columns: [end_date, duration, credit, r2]
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        dates = pd.to_datetime(rolling_results["end_date"])
        
        # 久期时序
        ax1 = axes[0]
        ax1.plot(dates, rolling_results["duration"], 'b-', linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=3.5, color='gray', linestyle='--', alpha=0.5, label='短久期/中久期分界')
        ax1.axhline(y=6, color='gray', linestyle='--', alpha=0.5, label='中久期/长久期分界')
        ax1.fill_between(dates, 0, 3.5, alpha=0.2, color='green', label='短久期')
        ax1.fill_between(dates, 3.5, 6, alpha=0.2, color='yellow', label='中久期')
        ax1.fill_between(dates, 6, 10, alpha=0.2, color='red', label='长久期')
        ax1.set_ylabel('久期（年）', fontsize=12)
        ax1.set_title('基金久期配置风格演变', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelbottom=True)  # 显示X轴刻度标签
        
        # 信用评分时序
        ax2 = axes[1]
        ax2.plot(dates, rolling_results["credit"], 'r-', linewidth=2, marker='s', markersize=4)
        ax2.axhline(y=16, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=14, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(dates, 16, 18, alpha=0.2, color='darkgreen', label='高信用(AAA)')
        ax2.fill_between(dates, 14, 16, alpha=0.2, color='green', label='中高信用(AA+)')
        ax2.fill_between(dates, 12, 14, alpha=0.2, color='orange', label='中信用(AA)')
        ax2.fill_between(dates, 0, 12, alpha=0.2, color='red', label='低信用')
        ax2.set_ylabel('信用评分', fontsize=12)
        ax2.set_title('基金信用配置风格演变', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.set_ylim(10, 18)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelbottom=True)  # 显示X轴刻度标签
        
        # R²时序
        ax3 = axes[2]
        ax3.plot(dates, rolling_results["r2"], 'g-', linewidth=2, marker='^', markersize=4)
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='R²=0.7')
        ax3.fill_between(dates, 0.7, 1, alpha=0.2, color='green', label='拟合优度良好')
        ax3.fill_between(dates, 0, 0.7, alpha=0.2, color='red', label='拟合优度一般')
        ax3.set_ylabel('R²', fontsize=12)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_title('回归模型拟合优度', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def plot_style_box(self, duration: float, credit: float, 
                      fund_name: str = "", save_path: str = None) -> plt.Figure:
        """
        绘制风格箱定位图
        
        Parameters:
        -----------
        duration : float
            估计久期
        credit : float
            估计信用评分
        fund_name : str
            基金名称
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制3×3风格箱
        # 久期维度：短(<3.5) / 中(3.5-6) / 长(>6)
        # 信用维度：高(>=16) / 中(14-16) / 低(<14)
        
        colors = {
            "短久期高信用": "#2E7D32",
            "短久期中信用": "#81C784",
            "短久期低信用": "#FFEB3B",
            "中久期高信用": "#1976D2",
            "中久期中信用": "#64B5F6",
            "中久期低信用": "#FF9800",
            "长久期高信用": "#C62828",
            "长久期中信用": "#E57373",
            "长久期低信用": "#F44336"
        }
        
        # 绘制网格
        for i, dur_label in enumerate(["长久期", "中久期", "短久期"]):
            for j, cred_label in enumerate(["低信用", "中信用", "高信用"]):
                x, y = j * 4 + 2, i * 3 + 1.5
                label = f"{dur_label}{cred_label}"
                color = colors.get(label, "lightgray")
                rect = mpatches.Rectangle((j*4, i*3), 4, 3, 
                                         linewidth=1, edgecolor='black', 
                                         facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                ax.text(x, y, label.replace("久期", "\n"), 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 绘制基金位置
        # 映射坐标
        if duration < 3.5:
            y_pos = 1.5  # 短久期
        elif duration < 6:
            y_pos = 4.5  # 中久期
        else:
            y_pos = 7.5  # 长久期
        
        if credit < 14:
            x_pos = 2  # 低信用
        elif credit < 16:
            x_pos = 6  # 中信用
        else:
            x_pos = 10  # 高信用
        
        ax.scatter(x_pos, y_pos, s=500, c='red', marker='*', 
                  edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(f'{fund_name}\n久期: {duration:.2f}\n信用: {credit:.2f}', 
                   xy=(x_pos, y_pos), xytext=(x_pos+1.5, y_pos+1),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 9)
        ax.set_xlabel('信用配置风格', fontsize=12)
        ax.set_ylabel('久期配置风格', fontsize=12)
        ax.set_title(f'{fund_name} 债券基金风格箱定位', fontsize=14, fontweight='bold')
        ax.set_xticks([2, 6, 10])
        ax.set_xticklabels(['低信用\n(<14)', '中信用\n(14-16)', '高信用\n(>=16)'])
        ax.set_yticks([1.5, 4.5, 7.5])
        ax.set_yticklabels(['短久期\n(<3.5年)', '中久期\n(3.5-6年)', '长久期\n(>6年)'])
        ax.grid(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def plot_factor_exposure(self, coef: pd.Series, index_info: pd.DataFrame,
                            save_path: str = None) -> plt.Figure:
        """
        绘制因子暴露（回归系数）图
        
        Parameters:
        -----------
        coef : pd.Series
            回归系数
        index_info : pd.DataFrame
            指数信息
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 合并系数和指数信息
        # 重置index_info的index为代码列（如果有的话）
        if 'code' in index_info.columns:
            index_info_copy = index_info.set_index('code')
        else:
            index_info_copy = index_info.copy()
        
        # 找到匹配的代码
        common_codes = [c for c in coef.index if c in index_info_copy.index]
        if len(common_codes) == 0:
            # 如果没有匹配，使用所有系数，生成简化图
            data = pd.DataFrame({
                'name': coef.index,
                'coef': coef.values,
                'type': '未知'
            })
        else:
            data = index_info_copy.loc[common_codes].copy()
            data['coef'] = coef.loc[common_codes].values
        
        data = data.sort_values('coef', ascending=True)
        
        # 按类型着色
        colors = data["type"].map({
            "国债": "green",
            "金融债": "blue",
            "高信用": "darkgreen",
            "企业债AAA": "purple",
            "企业债AA+": "orange",
            "企业债AA": "red",
            "信用债": "brown"
        }).fillna("gray")
        
        bars = ax.barh(data["name"], data["coef"], color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, val in zip(bars, data["coef"]):
            ax.text(val + 0.01 if val >= 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=9)
        
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('回归系数（因子暴露）', fontsize=12)
        ax.set_title('债券指数因子暴露', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color='green', label='国债'),
            mpatches.Patch(color='blue', label='金融债'),
            mpatches.Patch(color='purple', label='企业债AAA'),
            mpatches.Patch(color='orange', label='企业债AA+'),
            mpatches.Patch(color='red', label='企业债AA')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig
    
    def plot_style_drift(self, rolling_results: pd.DataFrame,
                        drifts: List[Dict], save_path: str = None) -> plt.Figure:
        """
        绘制风格漂移检测图
        
        Parameters:
        -----------
        rolling_results : pd.DataFrame
            滚动回测结果
        drifts : List[Dict]
            漂移点列表
        save_path : str
            保存路径
            
        Returns:
        --------
        plt.Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        dates = pd.to_datetime(rolling_results["end_date"])
        
        # 久期漂移
        ax1 = axes[0]
        ax1.plot(dates, rolling_results["duration"], 'b-', linewidth=2, label='久期')
        
        # 标记漂移点
        for drift in drifts:
            if drift["type"] == "久期漂移":
                ax1.axvline(x=drift["date"], color='red', linestyle='--', alpha=0.5)
                ax1.scatter([drift["date"]], [drift["to_value"]], 
                          color='red', s=100, zorder=5, marker='v')
        
        ax1.set_ylabel('久期（年）', fontsize=12)
        ax1.set_title('久期风格漂移检测', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelbottom=True)  # 显示X轴刻度标签
        
        # 信用漂移
        ax2 = axes[1]
        ax2.plot(dates, rolling_results["credit"], 'r-', linewidth=2, label='信用评分')
        
        # 标记漂移点
        for drift in drifts:
            if drift["type"] == "信用漂移":
                ax2.axvline(x=drift["date"], color='red', linestyle='--', alpha=0.5)
                ax2.scatter([drift["date"]], [drift["to_value"]], 
                          color='red', s=100, zorder=5, marker='v')
        
        ax2.set_ylabel('信用评分', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_title('信用风格漂移检测', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 图表已保存: {save_path}")
        
        return fig


if __name__ == "__main__":
    print("Bond Style Plotter - Test")
