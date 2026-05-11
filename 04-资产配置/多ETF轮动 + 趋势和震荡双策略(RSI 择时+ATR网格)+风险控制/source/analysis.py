import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import os
from source.config import OUTPUT_DIR


class PerformanceAnalyzer:
    def __init__(self, backtest_results):
        self.equity = backtest_results['equity']
        self.trades = backtest_results['trades']
        self.initial_cash = backtest_results['initial_cash']
        self.total_injected = backtest_results['total_injected']
        self.final_value = backtest_results['final_value']
        
        self.calculate_returns()
    
    def calculate_returns(self):
        self.equity['returns'] = self.equity['total_value'].pct_change().fillna(0)
        self.equity['cum_returns'] = (1 + self.equity['returns']).cumprod() - 1
        
        first_value = self.equity['total_value'].iloc[0]
        self.equity['simple_cum_returns'] = self.equity['total_value'] / first_value - 1
    
    def calculate_metrics(self):
        if len(self.equity) == 0:
            return {}
        
        total_return = self.equity['simple_cum_returns'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(self.equity)) - 1
        
        daily_returns = self.equity['returns']
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        cumulative = self.equity['total_value']
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        profit_factor = self._calculate_profit_factor()
        
        trading_days = len(self.equity)
        num_trades = len(self.trades) if not self.trades.empty else 0
        trades_per_year = num_trades / (trading_days / 252) if trading_days > 0 else 0
        
        metrics = {
            '总收益率': round(total_return * 100, 2),
            '年化收益率': round(annual_return * 100, 2),
            '波动率': round(volatility * 100, 2),
            '夏普比率': round(sharpe_ratio, 2),
            '最大回撤': round(max_drawdown * 100, 2),
            '日胜率': round(win_rate * 100, 2),
            '盈亏比': round(profit_factor, 2),
            '总交易次数': num_trades,
            '年均交易次数': round(trades_per_year, 1),
            '初始资金': self.initial_cash,
            '累计投入': self.total_injected,
            '期末资金': round(self.final_value, 2)
        }
        
        return metrics
    
    def _calculate_profit_factor(self):
        if self.trades.empty:
            return 0
        
        trades_copy = self.trades.copy()
        trades_copy['profit'] = -trades_copy['value'] - trades_copy['commission']
        
        gross_profit = trades_copy[trades_copy['profit'] > 0]['profit'].sum()
        gross_loss = -trades_copy[trades_copy['profit'] < 0]['profit'].sum()
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def plot_equity_curve(self, save_path=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.equity.index, self.equity['total_value'], label='净值曲线', linewidth=2)
        ax.set_xlabel('日期')
        ax.set_ylabel('资产价值（元）')
        ax.set_title('策略净值曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_drawdown(self, save_path=None):
        cumulative = self.equity['total_value']
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red', label='回撤')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax.set_xlabel('日期')
        ax.set_ylabel('回撤（%）')
        ax.set_title('回撤曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_monthly_returns(self, save_path=None):
        monthly = self.equity['total_value'].resample('ME').last()
        monthly_returns = monthly.pct_change().dropna()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['red' if x < 0 else 'green' for x in monthly_returns]
        ax.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
        ax.set_xlabel('月份')
        ax.set_ylabel('月收益率')
        ax.set_title('月度收益率')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        
        xticks = [i for i in range(len(monthly_returns)) if i % 6 == 0]
        xlabels = [monthly_returns.index[i].strftime('%Y-%m') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_annual_returns(self, save_path=None):
        annual = self.equity['total_value'].resample('YE').last()
        annual_returns = annual.pct_change().dropna()
        
        if len(annual_returns) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if x < 0 else 'green' for x in annual_returns]
        ax.bar(range(len(annual_returns)), annual_returns.values, color=colors, alpha=0.7)
        ax.set_xlabel('年份')
        ax.set_ylabel('年收益率')
        ax.set_title('年度收益率')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        
        xlabels = [d.strftime('%Y') for d in annual_returns.index]
        ax.set_xticks(range(len(annual_returns)))
        ax.set_xticklabels(xlabels)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, output_dir=None):
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        metrics = self.calculate_metrics()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=' * 50 + '\n')
            f.write('策略回测报告\n')
            f.write('=' * 50 + '\n\n')
            
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
        
        self.plot_equity_curve(os.path.join(output_dir, f'equity_curve_{timestamp}.png'))
        self.plot_drawdown(os.path.join(output_dir, f'drawdown_{timestamp}.png'))
        
        try:
            self.plot_monthly_returns(os.path.join(output_dir, f'monthly_returns_{timestamp}.png'))
            self.plot_annual_returns(os.path.join(output_dir, f'annual_returns_{timestamp}.png'))
        except:
            pass
        
        print(f'报告已保存至: {output_dir}')
        
        return metrics, report_path
