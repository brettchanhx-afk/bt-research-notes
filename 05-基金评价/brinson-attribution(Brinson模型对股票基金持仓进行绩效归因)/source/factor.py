"""
factor.py - Brinson模型因子计算模块
实现单期和多期Brinson绩效归因模型
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BrinsonAttribution:
    """Brinson归因结果数据类"""
    total: float              # 总超额收益
    allocation: float         # 类别配置收益
    selection: float          # 个券选择收益
    interaction: float        # 交互作用收益
    q1: float                 # 基准组合收益
    q2: float                 # 类别配置组合收益
    q3: float                 # 股票选择组合收益
    q4: float                 # 实际组合收益
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'total': self.total,
            'allocation': self.allocation,
            'selection': self.selection,
            'interaction': self.interaction,
            'q1': self.q1,
            'q2': self.q2,
            'q3': self.q3,
            'q4': self.q4
        }


class SinglePeriodBrinson:
    """
    单期Brinson模型
    
    四象限矩阵:
    Q1 (基准组合): ∑wb·rb
    Q2 (类别配置组合): ∑wp·rb
    Q3 (股票选择组合): ∑wb·rp
    Q4 (实际投资组合): ∑wp·rp
    """
    
    @staticmethod
    def calculate_four_quadrants(
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float, float, float]:
        """
        计算Brinson四象限
        
        Parameters:
            portfolio_weights: 实际组合权重 (wp)
            portfolio_returns: 实际组合各行业收益率 (rp)
            benchmark_weights: 基准组合权重 (wb)
            benchmark_returns: 基准组合各行业收益率 (rb)
        
        Returns:
            Tuple[float, float, float, float]: (Q1, Q2, Q3, Q4)
        """
        # 确保索引对齐
        sectors = portfolio_weights.index.intersection(benchmark_weights.index)
        
        wp = portfolio_weights.reindex(sectors, fill_value=0)
        rp = portfolio_returns.reindex(sectors, fill_value=0)
        wb = benchmark_weights.reindex(sectors, fill_value=0)
        rb = benchmark_returns.reindex(sectors, fill_value=0)
        
        # 四象限计算
        q1 = (wb * rb).sum()  # 基准组合
        q2 = (wp * rb).sum()  # 类别配置组合
        q3 = (wb * rp).sum()  # 股票选择组合
        q4 = (wp * rp).sum()  # 实际组合
        
        return q1, q2, q3, q4
    
    @staticmethod
    def calculate_attribution(
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
        include_interaction: bool = True
    ) -> BrinsonAttribution:
        """
        计算Brinson归因
        
        Parameters:
            portfolio_weights: 实际组合权重
            portfolio_returns: 实际组合各行业收益率
            benchmark_weights: 基准组合权重
            benchmark_returns: 基准组合各行业收益率
            include_interaction: 是否单独计算交互作用（True: 三因素分解，False: 两因素分解）
        
        Returns:
            BrinsonAttribution: 归因结果
        """
        q1, q2, q3, q4 = SinglePeriodBrinson.calculate_four_quadrants(
            portfolio_weights, portfolio_returns,
            benchmark_weights, benchmark_returns
        )
        
        # 收益分解
        total = q4 - q1
        allocation = q2 - q1
        selection = q3 - q1
        interaction = total - allocation - selection
        
        if not include_interaction:
            # 两因素分解：将交互作用归入个券选择
            selection = q4 - q2
            interaction = 0
        
        return BrinsonAttribution(
            total=total,
            allocation=allocation,
            selection=selection,
            interaction=interaction,
            q1=q1, q2=q2, q3=q3, q4=q4
        )
    
    @staticmethod
    def calculate_sector_contribution(
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
        include_interaction: bool = True
    ) -> pd.DataFrame:
        """
        计算各行业的归因贡献
        
        Parameters:
            portfolio_weights: 实际组合权重
            portfolio_returns: 实际组合各行业收益率
            benchmark_weights: 基准组合权重
            benchmark_returns: 基准组合各行业收益率
            include_interaction: 是否单独计算交互作用
        
        Returns:
            pd.DataFrame: 各行业归因贡献
        """
        # 确保索引对齐
        sectors = portfolio_weights.index.intersection(benchmark_weights.index)
        
        wp = portfolio_weights.reindex(sectors, fill_value=0)
        rp = portfolio_returns.reindex(sectors, fill_value=0)
        wb = benchmark_weights.reindex(sectors, fill_value=0)
        rb = benchmark_returns.reindex(sectors, fill_value=0)
        
        # 各行业贡献
        allocation_contrib = (wp - wb) * rb
        selection_contrib = (rp - rb) * wb
        interaction_contrib = (rp - rb) * (wp - wb)
        
        if not include_interaction:
            # 两因素分解
            selection_contrib = (rp - rb) * wp
            interaction_contrib = pd.Series(0, index=sectors)
        
        total_contrib = allocation_contrib + selection_contrib + interaction_contrib
        
        result = pd.DataFrame({
            'sector': sectors,
            'portfolio_weight': wp.values,
            'benchmark_weight': wb.values,
            'portfolio_return': rp.values,
            'benchmark_return': rb.values,
            'allocation': allocation_contrib.values,
            'selection': selection_contrib.values,
            'interaction': interaction_contrib.values,
            'total': total_contrib.values
        })
        
        return result


class MultiPeriodBrinson:
    """
    多期Brinson模型
    
    使用几何链接法计算多期累计收益
    """
    
    @staticmethod
    def geometric_link(returns: List[float]) -> float:
        """
        几何链接计算累计收益
        
        Parameters:
            returns: 各期收益率列表
        
        Returns:
            float: 累计收益率
        """
        if not returns:
            return 0.0
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)
        return cumulative - 1
    
    @staticmethod
    def calculate_multi_period_attribution(
        single_period_results: List[BrinsonAttribution]
    ) -> BrinsonAttribution:
        """
        计算多期累计归因
        
        Parameters:
            single_period_results: 各期单期归因结果列表
        
        Returns:
            BrinsonAttribution: 多期累计归因结果
        """
        # 提取各期收益
        total_returns = [r.total for r in single_period_results]
        allocation_returns = [r.allocation for r in single_period_results]
        selection_returns = [r.selection for r in single_period_results]
        interaction_returns = [r.interaction for r in single_period_results]
        
        q1_returns = [r.q1 for r in single_period_results]
        q2_returns = [r.q2 for r in single_period_results]
        q3_returns = [r.q3 for r in single_period_results]
        q4_returns = [r.q4 for r in single_period_results]
        
        # 几何链接计算累计收益
        return BrinsonAttribution(
            total=MultiPeriodBrinson.geometric_link(total_returns),
            allocation=MultiPeriodBrinson.geometric_link(allocation_returns),
            selection=MultiPeriodBrinson.geometric_link(selection_returns),
            interaction=MultiPeriodBrinson.geometric_link(interaction_returns),
            q1=MultiPeriodBrinson.geometric_link(q1_returns),
            q2=MultiPeriodBrinson.geometric_link(q2_returns),
            q3=MultiPeriodBrinson.geometric_link(q3_returns),
            q4=MultiPeriodBrinson.geometric_link(q4_returns)
        )
    
    @staticmethod
    def rolling_attribution(
        portfolio_weights_df: pd.DataFrame,
        portfolio_returns_df: pd.DataFrame,
        benchmark_weights_df: pd.DataFrame,
        benchmark_returns_df: pd.DataFrame,
        date_col: str = 'date',
        include_interaction: bool = True
    ) -> pd.DataFrame:
        """
        滚动计算多期归因
        
        Parameters:
            portfolio_weights_df: 实际组合权重DataFrame (date, sector1, sector2, ...)
            portfolio_returns_df: 实际组合收益DataFrame (date, sector1, sector2, ...)
            benchmark_weights_df: 基准组合权重DataFrame
            benchmark_returns_df: 基准组合收益DataFrame
            date_col: 日期列名
            include_interaction: 是否单独计算交互作用
        
        Returns:
            pd.DataFrame: 各期归因结果
        """
        results = []
        
        dates = portfolio_weights_df[date_col].unique()
        
        for date in dates:
            # 获取当期数据
            pw = portfolio_weights_df[portfolio_weights_df[date_col] == date].set_index(date_col)
            pr = portfolio_returns_df[portfolio_returns_df[date_col] == date].set_index(date_col)
            bw = benchmark_weights_df[benchmark_weights_df[date_col] == date].set_index(date_col)
            br = benchmark_returns_df[benchmark_returns_df[date_col] == date].set_index(date_col)
            
            # 转换为Series（假设每行是一个行业）
            if len(pw) > 0:
                pw_series = pw.iloc[0]
                pr_series = pr.iloc[0]
                bw_series = bw.iloc[0]
                br_series = br.iloc[0]
                
                # 计算单期归因
                attr = SinglePeriodBrinson.calculate_attribution(
                    pw_series, pr_series, bw_series, br_series, include_interaction
                )
                
                results.append({
                    'date': date,
                    **attr.to_dict()
                })
        
        return pd.DataFrame(results)


class BrinsonAttributionAnalyzer:
    """
    Brinson归因分析器
    整合单期和多期归因功能
    """
    
    def __init__(self, include_interaction: bool = True):
        """
        初始化
        
        Parameters:
            include_interaction: 是否单独计算交互作用
        """
        self.include_interaction = include_interaction
        self.single_period_results = []
    
    def add_period(
        self,
        date: str,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series
    ):
        """
        添加单期数据
        
        Parameters:
            date: 日期
            portfolio_weights: 实际组合权重
            portfolio_returns: 实际组合各行业收益率
            benchmark_weights: 基准组合权重
            benchmark_returns: 基准组合各行业收益率
        """
        attr = SinglePeriodBrinson.calculate_attribution(
            portfolio_weights, portfolio_returns,
            benchmark_weights, benchmark_returns,
            self.include_interaction
        )
        
        self.single_period_results.append({
            'date': date,
            'result': attr
        })
    
    def get_single_period_results(self) -> pd.DataFrame:
        """获取单期归因结果"""
        data = []
        for item in self.single_period_results:
            row = {'date': item['date']}
            row.update(item['result'].to_dict())
            data.append(row)
        return pd.DataFrame(data)
    
    def get_multi_period_result(self) -> BrinsonAttribution:
        """获取多期累计归因结果"""
        if not self.single_period_results:
            raise ValueError("没有单期数据，无法计算多期归因")
        
        results = [item['result'] for item in self.single_period_results]
        return MultiPeriodBrinson.calculate_multi_period_attribution(results)
    
    def get_sector_contributions(self) -> Dict[str, pd.DataFrame]:
        """
        获取各行业贡献（需要重写以存储每期权重和收益数据）
        
        Returns:
            Dict[str, pd.DataFrame]: 各期行业贡献
        """
        # 这里简化处理，实际应该存储每期详细数据
        pass


if __name__ == "__main__":
    # 测试Brinson模型
    print("测试Brinson模型...")
    
    # 构造测试数据
    sectors = ['金融', '科技', '消费', '医药', '能源']
    
    # 实际组合
    wp = pd.Series([0.25, 0.30, 0.20, 0.15, 0.10], index=sectors)
    rp = pd.Series([0.05, 0.10, 0.03, 0.08, -0.02], index=sectors)
    
    # 基准组合
    wb = pd.Series([0.20, 0.25, 0.25, 0.15, 0.15], index=sectors)
    rb = pd.Series([0.04, 0.08, 0.05, 0.06, -0.01], index=sectors)
    
    # 单期归因
    print("\n单期归因结果（三因素）:")
    attr = SinglePeriodBrinson.calculate_attribution(wp, rp, wb, rb, include_interaction=True)
    print(f"总超额收益: {attr.total:.4f}")
    print(f"类别配置: {attr.allocation:.4f}")
    print(f"个券选择: {attr.selection:.4f}")
    print(f"交互作用: {attr.interaction:.4f}")
    
    print("\n单期归因结果（两因素）:")
    attr2 = SinglePeriodBrinson.calculate_attribution(wp, rp, wb, rb, include_interaction=False)
    print(f"总超额收益: {attr2.total:.4f}")
    print(f"类别配置: {attr2.allocation:.4f}")
    print(f"个券选择: {attr2.selection:.4f}")
    
    # 各行业贡献
    print("\n各行业贡献:")
    contrib = SinglePeriodBrinson.calculate_sector_contribution(wp, rp, wb, rb)
    print(contrib)
    
    print("\nBrinson模型测试完成!")
