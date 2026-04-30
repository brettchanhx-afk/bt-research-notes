# -*- coding: utf-8 -*-
"""
Campisi 归因模型核心模块

基于华泰金工研报实现债券基金业绩归因分析。

核心公式：
    R = y × dt + (-MD) × dy_treasury + (-MD) × dy_credit
    
三部分收益贡献：
    1. 票息效应 = Σ(w_i × y_i × dt) / Σ(w_i × R_i)
    2. 国债利率变化效应 = Σ(w_i × (-MD_i) × dy_treasury,i) / Σ(w_i × R_i)
    3. 信用利差变化效应 = Σ(w_i × (-MD_i) × dy_credit,i) / Σ(w_i × R_i)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

# 导入其他模块
from .bond_analytics import decompose_bond_return
from .yield_curve import YieldCurve, calculate_yield_change


# ============================================================
# Campisi 归因分析器
# ============================================================
class CampisiAttribution:
    """Campisi债券基金归因分析器。
    
    将债券基金收益率分解为：
      - 票息效应（Coupon Effect）
      - 国债利率变化效应（Treasury Effect）
      - 信用利差变化效应（Credit Effect）
    """
    
    def __init__(self):
        self.results_ = None
        self.summary_ = None
    
    def analyze(
        self,
        holdings: pd.DataFrame,
        bond_info: pd.DataFrame,
        treasury_curve_start: YieldCurve,
        treasury_curve_end: YieldCurve,
        holding_period_days: int = 90
    ) -> pd.DataFrame:
        """执行Campisi归因分析。
        
        Parameters
        ----------
        holdings : pd.DataFrame
            基金持仓数据，包含：
            - bond_code: 债券代码
            - weight: 持仓权重（%）
            - amount: 持仓金额
        bond_info : pd.DataFrame
            债券信息，包含：
            - bond_code: 债券代码
            - modified_duration: 修正久期
            - ytm: 到期收益率（%）
            - credit_rating: 信用评级
            - maturity_date: 到期日
        treasury_curve_start : YieldCurve
            期初国债收益率曲线
        treasury_curve_end : YieldCurve
            期末国债收益率曲线
        holding_period_days : int
            持有期天数
        
        Returns
        -------
        pd.DataFrame
            归因结果，包含每只债券的收益分解
        """
        # 合并持仓和债券信息
        merged = holdings.merge(bond_info, on='bond_code', how='left')
        
        if len(merged) == 0:
            print('[ERROR] 持仓数据与债券信息无法匹配')
            return pd.DataFrame()
        
        # 持有期（年）
        holding_period_years = holding_period_days / 365
        
        # 计算每只债券的收益分解
        results = []
        
        for _, row in merged.iterrows():
            bond_code = row['bond_code']
            weight = row['weight'] / 100  # 转为小数
            md = row.get('modified_duration', 0)
            ytm = row.get('ytm', 0) / 100  # 转为小数
            rating = row.get('credit_rating', 'A')
            maturity_date = row.get('maturity_date', '')
            
            # 计算剩余期限
            if pd.notna(maturity_date) and maturity_date != '':
                try:
                    mat_dt = pd.to_datetime(maturity_date)
                    years_to_mat = max((mat_dt - pd.Timestamp.now()).days / 365, 0.1)
                except Exception:
                    years_to_mat = 5.0  # 默认5年
            else:
                years_to_mat = 5.0
            
            # 计算国债利率变化
            dy_treasury = calculate_yield_change(
                treasury_curve_start, treasury_curve_end, years_to_mat
            ) / 100  # 转为小数
            
            # 计算信用利差变化（简化：使用评级利差）
            dy_credit = self._estimate_credit_spread_change(rating, holding_period_years)
            
            # Campisi分解
            decomposition = decompose_bond_return(
                ytm_start=ytm,
                ytm_end=ytm + dy_treasury + dy_credit,  # 期末YTM
                treasury_yield_change=dy_treasury,
                credit_spread_change=dy_credit,
                modified_duration=md,
                holding_period_years=holding_period_years
            )
            
            results.append({
                'bond_code': bond_code,
                'weight': weight,
                'modified_duration': md,
                'ytm': ytm,
                'credit_rating': rating,
                'years_to_maturity': years_to_mat,
                'dy_treasury': dy_treasury,
                'dy_credit': dy_credit,
                'total_return': decomposition['total_return'],
                'coupon_effect': decomposition['coupon_effect'],
                'treasury_effect': decomposition['treasury_effect'],
                'credit_effect': decomposition['credit_effect'],
            })
        
        self.results_ = pd.DataFrame(results)
        
        # 计算基金层面的归因
        self._calculate_fund_attribution()
        
        return self.results_
    
    def _estimate_credit_spread_change(self, rating: str, period_years: float) -> float:
        """估算信用利差变化。
        
        简化处理：基于历史波动率估算。
        
        Parameters
        ----------
        rating : str
            信用评级
        period_years : float
            时间长度（年）
        
        Returns
        -------
        float
            信用利差变化（小数）
        """
        # 不同评级的年化波动率（bp）
        volatility_map = {
            'AAA': 20,
            'AA': 30,
            'A': 50,
            'BBB': 80,
            'BB': 120,
            'B': 180,
        }
        
        annual_vol = volatility_map.get(rating, 50)
        
        # 时间调整
        period_vol = annual_vol * np.sqrt(period_years)
        
        # 假设均值为0，返回波动率范围
        return 0.0  # 简化：假设利差不变
    
    def _calculate_fund_attribution(self):
        """计算基金层面的归因贡献。"""
        if self.results_ is None or len(self.results_) == 0:
            return
        
        df = self.results_
        
        # 加权收益
        total_return = (df['weight'] * df['total_return']).sum()
        coupon_contrib = (df['weight'] * df['coupon_effect']).sum()
        treasury_contrib = (df['weight'] * df['treasury_effect']).sum()
        credit_contrib = (df['weight'] * df['credit_effect']).sum()
        
        # 归因贡献比例
        if abs(total_return) > 1e-8:
            coupon_pct = coupon_contrib / total_return * 100
            treasury_pct = treasury_contrib / total_return * 100
            credit_pct = credit_contrib / total_return * 100
        else:
            coupon_pct = treasury_pct = credit_pct = 0.0
        
        self.summary_ = {
            'total_return': total_return,
            'coupon_contrib': coupon_contrib,
            'treasury_contrib': treasury_contrib,
            'credit_contrib': credit_contrib,
            'coupon_pct': coupon_pct,
            'treasury_pct': treasury_pct,
            'credit_pct': credit_pct,
            'n_bonds': len(df),
            'avg_duration': df['modified_duration'].mean(),
            'avg_ytm': df['ytm'].mean(),
        }
    
    def get_summary(self) -> Dict:
        """获取归因摘要。"""
        return self.summary_ or {}
    
    def get_top_contributors(self, effect: str = 'coupon', n: int = 10) -> pd.DataFrame:
        """获取某效应贡献最大的债券。
        
        Parameters
        ----------
        effect : str
            效应类型：'coupon', 'treasury', 'credit'
        n : int
            返回数量
        
        Returns
        -------
        pd.DataFrame
            贡献最大的债券
        """
        if self.results_ is None:
            return pd.DataFrame()
        
        col_map = {
            'coupon': 'coupon_effect',
            'treasury': 'treasury_effect',
            'credit': 'credit_effect',
        }
        
        effect_col = col_map.get(effect, 'coupon_effect')
        
        # 计算加权贡献
        df = self.results_.copy()
        df['contribution'] = df['weight'] * df[effect_col]
        
        return df.nlargest(n, 'contribution')[
            ['bond_code', 'weight', 'modified_duration', effect_col, 'contribution']
        ]


# ============================================================
# 滚动归因分析
# ============================================================
def rolling_attribution(
    fund_code: str,
    start_date: str,
    end_date: str,
    window_days: int = 90,
    step_days: int = 30
) -> pd.DataFrame:
    """滚动窗口归因分析。
    
    Parameters
    ----------
    fund_code : str
        基金代码
    start_date, end_date : str
        分析区间
    window_days : int
        滚动窗口天数
    step_days : int
        滚动步长天数
    
    Returns
    -------
    pd.DataFrame
        时间序列归因结果
    """
    from .data_loader import (
        get_fund_bond_holdings,
        get_bond_info,
        get_treasury_yield_curve,
        get_fund_nav_history
    )
    
    # 生成时间点
    dates = pd.date_range(start=start_date, end=end_date, freq=f'{step_days}D')
    
    results = []
    analyzer = CampisiAttribution()
    
    for i, end_dt in enumerate(dates[1:], 1):
        start_dt = dates[i-1]
        
        try:
            # 获取持仓
            holdings = get_fund_bond_holdings(fund_code, date=end_dt.strftime('%Y%m%d'))
            if len(holdings) == 0:
                continue
            
            # 获取债券信息
            bond_info = get_bond_info(holdings['bond_code'].tolist())
            if len(bond_info) == 0:
                continue
            
            # 获取收益率曲线
            curve_start = get_treasury_yield_curve(start_dt.strftime('%Y%m%d'))
            curve_end = get_treasury_yield_curve(end_dt.strftime('%Y%m%d'))
            
            if len(curve_start) == 0 or len(curve_end) == 0:
                continue
            
            # 转换为YieldCurve对象
            yc_start = YieldCurve(curve_start['term'].values, curve_start['yield_rate'].values)
            yc_end = YieldCurve(curve_end['term'].values, curve_end['yield_rate'].values)
            
            # 执行归因
            result = analyzer.analyze(
                holdings, bond_info, yc_start, yc_end,
                holding_period_days=(end_dt - start_dt).days
            )
            
            summary = analyzer.get_summary()
            summary['date'] = end_dt
            summary['period'] = f"{start_dt.strftime('%Y-%m-%d')}~{end_dt.strftime('%Y-%m-%d')}"
            results.append(summary)
            
        except Exception as e:
            print(f"  [ERROR] {end_dt} 归因失败: {e}")
    
    if results:
        return pd.DataFrame(results)
    
    return pd.DataFrame()
