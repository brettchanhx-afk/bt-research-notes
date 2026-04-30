"""
backtest.py - 回测逻辑模块
实现Brinson归因的回测框架，支持滚动归因分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from factor import SinglePeriodBrinson, MultiPeriodBrinson, BrinsonAttribution
from data_loader import FundDataLoader, DataProcessor


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    rebalance_freq: str = 'Q'  # 'M'月度, 'Q'季度, 'Y'年度
    lookback_periods: int = 4  # 回看期数
    include_interaction: bool = True
    benchmark_code: str = '000300'  # 沪深300


class BrinsonBacktest:
    """
    Brinson归因回测引擎
    
    支持:
    - 滚动归因分析
    - 多期累计归因
    - 行业轮动分析
    - 归因稳定性检验
    """
    
    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎
        
        Parameters:
            config: 回测配置
        """
        self.config = config
        self.data_loader = FundDataLoader()
        self.results = []
        self.portfolio_nav = None
        self.benchmark_nav = None
    
    def run_backtest(
        self,
        fund_code: str,
        portfolio_weights_func: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        运行回测
        
        Parameters:
            fund_code: 基金代码
            portfolio_weights_func: 自定义组合权重函数（可选）
        
        Returns:
            pd.DataFrame: 回测结果
        """
        print(f"开始Brinson归因回测: {fund_code}")
        print(f"回测区间: {self.config.start_date} 至 {self.config.end_date}")
        print(f"调仓频率: {self.config.rebalance_freq}")
        
        # 1. 获取数据
        print("\n[1/4] 获取基金持仓数据...")
        holdings = self.data_loader.get_fund_holdings(
            fund_code, 
            self.config.start_date, 
            self.config.end_date
        )
        
        print("[2/4] 获取行业收益数据...")
        sector_returns = self.data_loader.get_sector_index_returns(
            self.config.start_date,
            self.config.end_date,
            freq=self.config.rebalance_freq
        )
        
        print("[3/4] 获取基准收益数据...")
        benchmark_returns = self.data_loader.get_benchmark_returns(
            self.config.benchmark_code,
            self.config.start_date,
            self.config.end_date,
            freq=self.config.rebalance_freq
        )
        
        # 2. 处理数据
        print("[4/4] 处理数据并计算归因...")
        self.results = self._calculate_rolling_attribution(
            holdings, sector_returns, benchmark_returns,
            portfolio_weights_func
        )
        
        print(f"\n回测完成! 共计算 {len(self.results)} 期归因结果")
        
        return pd.DataFrame(self.results)
    
    def _calculate_rolling_attribution(
        self,
        holdings: pd.DataFrame,
        sector_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        portfolio_weights_func: Optional[Callable] = None
    ) -> List[Dict]:
        """
        计算滚动归因
        
        Parameters:
            holdings: 持仓数据
            sector_returns: 行业收益数据
            benchmark_returns: 基准收益数据
            portfolio_weights_func: 自定义权重函数
        
        Returns:
            List[Dict]: 各期归因结果
        """
        results = []
        
        # 获取调仓日期
        rebalance_dates = self._get_rebalance_dates()
        
        for date in rebalance_dates:
            try:
                # 获取当期数据
                period_holdings = holdings[holdings['date'] == date]
                
                if period_holdings.empty:
                    continue
                
                # 构建组合权重和收益
                if portfolio_weights_func:
                    wp, rp = portfolio_weights_func(period_holdings, sector_returns, date)
                else:
                    wp, rp = self._build_portfolio_weights_and_returns(
                        period_holdings, sector_returns, date
                    )
                
                # 构建基准权重和收益
                wb, rb = self._build_benchmark_weights_and_returns(
                    sector_returns, benchmark_returns, date
                )
                
                # 计算单期归因
                attr = SinglePeriodBrinson.calculate_attribution(
                    wp, rp, wb, rb,
                    include_interaction=self.config.include_interaction
                )
                
                # 计算各行业贡献
                sector_contrib = SinglePeriodBrinson.calculate_sector_contribution(
                    wp, rp, wb, rb,
                    include_interaction=self.config.include_interaction
                )
                
                result = {
                    'date': date,
                    **attr.to_dict(),
                    'sector_contribution': sector_contrib
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"计算 {date} 归因失败: {e}")
                continue
        
        return results
    
    def _get_rebalance_dates(self) -> List[str]:
        """获取调仓日期列表"""
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.rebalance_freq
        )
        return [d.strftime('%Y-%m-%d') for d in dates]
    
    def _build_portfolio_weights_and_returns(
        self,
        holdings: pd.DataFrame,
        sector_returns: pd.DataFrame,
        date: str
    ) -> Tuple[pd.Series, pd.Series]:
        """
        构建组合权重和收益
        
        Parameters:
            holdings: 当期持仓
            sector_returns: 行业收益
            date: 日期
        
        Returns:
            Tuple[pd.Series, pd.Series]: (weights, returns)
        """
        # 按行业聚合权重
        sector_weights = holdings.groupby('sector')['weight'].sum()
        
        # 获取行业收益
        if date in sector_returns.index:
            sector_rets = sector_returns.loc[date]
        else:
            # 使用最近可用日期
            available_dates = sector_returns.index[sector_returns.index <= date]
            if len(available_dates) > 0:
                sector_rets = sector_returns.loc[available_dates[-1]]
            else:
                sector_rets = pd.Series(0, index=sector_weights.index)
        
        # 对齐索引
        all_sectors = sector_weights.index.union(sector_rets.index)
        wp = sector_weights.reindex(all_sectors, fill_value=0)
        rp = sector_rets.reindex(all_sectors, fill_value=0)
        
        # 归一化权重
        wp = wp / wp.sum()
        
        return wp, rp
    
    def _build_benchmark_weights_and_returns(
        self,
        sector_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        date: str
    ) -> Tuple[pd.Series, pd.Series]:
        """
        构建基准权重和收益
        
        Parameters:
            sector_returns: 行业收益
            benchmark_returns: 基准收益
            date: 日期
        
        Returns:
            Tuple[pd.Series, pd.Series]: (weights, returns)
        """
        # 使用行业等权或市值加权作为基准
        # 这里简化处理，使用等权
        sectors = sector_returns.columns
        wb = pd.Series(1.0 / len(sectors), index=sectors)
        
        # 获取行业收益
        if date in sector_returns.index:
            rb = sector_returns.loc[date]
        else:
            available_dates = sector_returns.index[sector_returns.index <= date]
            if len(available_dates) > 0:
                rb = sector_returns.loc[available_dates[-1]]
            else:
                rb = pd.Series(0, index=sectors)
        
        return wb, rb
    
    def get_multi_period_attribution(self) -> BrinsonAttribution:
        """
        获取多期累计归因结果
        
        Returns:
            BrinsonAttribution: 多期累计归因
        """
        if not self.results:
            raise ValueError("请先运行回测")
        
        single_period_attrs = [r['result'] for r in self.results if 'result' in r]
        
        if not single_period_attrs:
            # 从字典重建
            attrs = []
            for r in self.results:
                attr = BrinsonAttribution(
                    total=r.get('total', 0),
                    allocation=r.get('allocation', 0),
                    selection=r.get('selection', 0),
                    interaction=r.get('interaction', 0),
                    q1=r.get('q1', 0),
                    q2=r.get('q2', 0),
                    q3=r.get('q3', 0),
                    q4=r.get('q4', 0)
                )
                attrs.append(attr)
            single_period_attrs = attrs
        
        return MultiPeriodBrinson.calculate_multi_period_attribution(single_period_attrs)
    
    def get_sector_contributions(self) -> pd.DataFrame:
        """
        获取所有期的行业贡献
        
        Returns:
            pd.DataFrame: 行业贡献数据
        """
        all_contributions = []
        
        for result in self.results:
            if 'sector_contribution' in result:
                contrib = result['sector_contribution'].copy()
                contrib['date'] = result['date']
                all_contributions.append(contrib)
        
        if all_contributions:
            return pd.concat(all_contributions, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        计算绩效指标
        
        Returns:
            Dict[str, float]: 绩效指标
        """
        if not self.results:
            raise ValueError("请先运行回测")
        
        df = pd.DataFrame(self.results)
        
        # 计算累计收益
        df['cum_total'] = (1 + df['total']).cumprod() - 1
        df['cum_allocation'] = (1 + df['allocation']).cumprod() - 1
        df['cum_selection'] = (1 + df['selection']).cumprod() - 1
        
        metrics = {
            'total_periods': len(df),
            'positive_allocation_periods': (df['allocation'] > 0).sum(),
            'positive_selection_periods': (df['selection'] > 0).sum(),
            'avg_allocation': df['allocation'].mean(),
            'avg_selection': df['selection'].mean(),
            'avg_interaction': df['interaction'].mean(),
            'allocation_win_rate': (df['allocation'] > 0).mean(),
            'selection_win_rate': (df['selection'] > 0).mean(),
            'total_cumulative_return': df['cum_total'].iloc[-1],
            'allocation_cumulative': df['cum_allocation'].iloc[-1],
            'selection_cumulative': df['cum_selection'].iloc[-1],
        }
        
        return metrics
    
    def analyze_attribution_stability(self) -> pd.DataFrame:
        """
        分析归因稳定性
        
        Returns:
            pd.DataFrame: 稳定性分析结果
        """
        if not self.results:
            raise ValueError("请先运行回测")
        
        df = pd.DataFrame(self.results)
        
        # 滚动统计
        window = min(self.config.lookback_periods, len(df) // 2)
        
        stability = pd.DataFrame({
            'date': df['date'],
            'allocation_mean': df['allocation'].rolling(window).mean(),
            'allocation_std': df['allocation'].rolling(window).std(),
            'selection_mean': df['selection'].rolling(window).mean(),
            'selection_std': df['selection'].rolling(window).std(),
            'interaction_mean': df['interaction'].rolling(window).mean(),
            'interaction_std': df['interaction'].rolling(window).std(),
        })
        
        # 计算信息比率（简化版）
        stability['allocation_ir'] = stability['allocation_mean'] / stability['allocation_std']
        stability['selection_ir'] = stability['selection_mean'] / stability['selection_std']
        
        return stability


class BrinsonAnalysisReport:
    """Brinson分析报告生成器"""
    
    def __init__(self, backtest: BrinsonBacktest):
        """
        初始化报告生成器
        
        Parameters:
            backtest: 回测引擎实例
        """
        self.backtest = backtest
    
    def generate_report(self) -> str:
        """
        生成文字报告
        
        Returns:
            str: 报告文本
        """
        metrics = self.backtest.calculate_performance_metrics()
        multi_attr = self.backtest.get_multi_period_attribution()
        
        report = []
        report.append("=" * 60)
        report.append("Brinson绩效归因分析报告")
        report.append("=" * 60)
        report.append(f"回测区间: {self.backtest.config.start_date} 至 {self.backtest.config.end_date}")
        report.append(f"调仓频率: {self.backtest.config.rebalance_freq}")
        report.append(f"计算期数: {metrics['total_periods']}")
        report.append("")
        
        report.append("【多期累计归因结果】")
        report.append(f"总超额收益:     {multi_attr.total*100:.2f}%")
        report.append(f"类别配置收益:   {multi_attr.allocation*100:.2f}%")
        report.append(f"个券选择收益:   {multi_attr.selection*100:.2f}%")
        report.append(f"交互作用收益:   {multi_attr.interaction*100:.2f}%")
        report.append("")
        
        report.append("【绩效统计】")
        report.append(f"配置收益胜率:   {metrics['allocation_win_rate']*100:.1f}%")
        report.append(f"选择收益胜率:   {metrics['selection_win_rate']*100:.1f}%")
        report.append(f"平均配置收益:   {metrics['avg_allocation']*100:.2f}%")
        report.append(f"平均选择收益:   {metrics['avg_selection']*100:.2f}%")
        report.append("")
        
        report.append("【累计收益】")
        report.append(f"总超额累计:     {metrics['total_cumulative_return']*100:.2f}%")
        report.append(f"配置累计:       {metrics['allocation_cumulative']*100:.2f}%")
        report.append(f"选择累计:       {metrics['selection_cumulative']*100:.2f}%")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """
        保存报告到文件
        
        Parameters:
            filepath: 文件路径
        """
        report = self.generate_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存至: {filepath}")


if __name__ == "__main__":
    # 测试回测模块
    print("测试Brinson回测模块...")
    
    # 创建回测配置
    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        rebalance_freq='Q',
        include_interaction=True
    )
    
    # 创建回测引擎
    backtest = BrinsonBacktest(config)
    
    # 运行回测
    results = backtest.run_backtest('000001')
    print("\n回测结果:")
    print(results)
    
    # 计算多期归因
    multi_attr = backtest.get_multi_period_attribution()
    print("\n多期累计归因:")
    print(f"总超额: {multi_attr.total*100:.2f}%")
    print(f"配置: {multi_attr.allocation*100:.2f}%")
    print(f"选择: {multi_attr.selection*100:.2f}%")
    
    # 生成报告
    report = BrinsonAnalysisReport(backtest)
    print("\n" + report.generate_report())
    
    print("\n回测模块测试完成!")
