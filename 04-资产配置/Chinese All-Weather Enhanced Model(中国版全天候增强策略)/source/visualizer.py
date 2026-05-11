
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    可视化模块，用于绘制策略回测结果
    """
    
    COLORS = {
        'asset_rp': '#1f77b4',
        'allweather': '#ff7f0e',
        'enhanced': '#2ca02c'
    }
    
    LABELS = {
        'asset_rp': '传统资产风险平价',
        'allweather': '全天候基准策略',
        'enhanced': '全天候增强策略'
    }
    
    @staticmethod
    def plot_portfolio_value(results, title='策略净值对比', save_path=None):
        """
        绘制策略净值曲线
        """
        plt.figure(figsize=(14, 7))
        
        for name, result in results.items():
            values = result['portfolio_value']
            plt.plot(values.index, values, label=Visualizer.LABELS[name],
                    color=Visualizer.COLORS[name], linewidth=2)
        
        plt.title(title, fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_drawdown(results, title='策略回撤对比', save_path=None):
        """
        绘制回撤曲线
        """
        plt.figure(figsize=(14, 7))
        
        for name, result in results.items():
            values = result['portfolio_value']
            rolling_max = values.expanding().max()
            drawdown = (values - rolling_max) / rolling_max
            plt.plot(drawdown.index, drawdown, label=Visualizer.LABELS[name],
                    color=Visualizer.COLORS[name], linewidth=2)
        
        plt.title(title, fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('回撤', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_weights_evolution(weights_record, title='仓位演变', save_path=None):
        """
        绘制仓位演变图
        """
        plt.figure(figsize=(14, 7))
        
        asset_names = {
            '510300.SH': '沪深300ETF',
            '512100.SH': '中证1000ETF',
            '512890.SH': '红利低波ETF',
            '511260.SH': '十年国债ETF',
            '511090.SH': '三十年国债ETF',
            '159980.SZ': '有色ETF',
            '159981.SZ': '能化ETF',
            '159985.SZ': '豆粕ETF',
            '518880.SH': '黄金ETF'
        }
        
        renamed_weights = weights_record.rename(columns=asset_names)
        renamed_weights.plot.area(stacked=True, ax=plt.gca(), cmap='tab20')
        
        plt.title(title, fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('权重', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_asset_category_weights(weights_record, title='大类资产配置', save_path=None):
        """
        绘制大类资产配置图
        """
        categories = {
            '510300.SH': '股票',
            '512100.SH': '股票',
            '512890.SH': '股票',
            '511260.SH': '债券',
            '511090.SH': '债券',
            '159980.SZ': '商品',
            '159981.SZ': '商品',
            '159985.SZ': '商品',
            '518880.SH': '黄金'
        }
        
        category_weights = pd.DataFrame(index=weights_record.index)
        for category in ['股票', '债券', '商品', '黄金']:
            assets = [k for k, v in categories.items() if v == category]
            valid_assets = [a for a in assets if a in weights_record.columns]
            if len(valid_assets) > 0:
                category_weights[category] = weights_record[valid_assets].sum(axis=1)
        
        plt.figure(figsize=(14, 7))
        category_weights.plot.area(stacked=True, ax=plt.gca(), cmap='Set2')
        
        plt.title(title, fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('权重', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_performance_comparison(comparison_df, title='策略绩效对比', save_path=None):
        """
        绘制绩效对比图
        """
        metrics = ['年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '月度胜率']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        comparison_df = comparison_df.rename(index=Visualizer.LABELS)
        
        for i, metric in enumerate(metrics):
            colors = [Visualizer.COLORS[name] for name in ['asset_rp', 'allweather', 'enhanced']]
            comparison_df[metric].plot(kind='bar', ax=axes[i], color=colors)
            axes[i].set_title(metric, fontsize=14)
            axes[i].tick_params(axis='x', rotation=0)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            if metric in ['最大回撤']:
                axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_yearly_performance(portfolio_returns, title='年度收益', save_path=None):
        """
        绘制年度收益图
        """
        yearly = portfolio_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        yearly.index = yearly.index.year
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if x < 0 else 'green' for x in yearly]
        yearly.plot(kind='bar', color=colors, ax=plt.gca())
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title(title, fontsize=16)
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('收益率', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_quadrant_performance(quadrant_performance, title='四象限绩效', save_path=None):
        """
        绘制四象限绩效图
        """
        quadrant_names = {
            'growth_above': '增长超预期',
            'growth_below': '增长不及预期',
            'inflation_above': '通胀超预期',
            'inflation_below': '通胀不及预期'
        }
        
        qp = quadrant_performance.rename(index=quadrant_names)
        
        metrics = ['年化收益', '年化波动', '夏普比率']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, metric in enumerate(metrics):
            qp[metric].plot(kind='bar', ax=axes[i], colormap='Set2')
            axes[i].set_title(metric, fontsize=14)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def display_performance_table(comparison_df):
        """
        显示绩效对比表格
        """
        df = comparison_df.rename(index=Visualizer.LABELS)
        
        df_styled = df.style.format({
            '累计收益': '{:.2%}',
            '年化收益': '{:.2%}',
            '年化波动': '{:.2%}',
            '夏普比率': '{:.2f}',
            '最大回撤': '{:.2%}',
            '卡玛比率': '{:.2f}',
            '月度胜率': '{:.2%}'
        }).background_gradient(cmap='RdYlGn', subset=['年化收益', '夏普比率', '卡玛比率', '月度胜率'])
        
        return df_styled
