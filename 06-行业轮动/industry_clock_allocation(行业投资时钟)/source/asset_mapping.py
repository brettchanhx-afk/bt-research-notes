"""
宏观-资产映射模块
构建投资时钟：增长-通胀时钟、信用-货币时钟
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class AssetMapping:
    def __init__(self):
        self.growth_inflation_clock = {}
        self.credit_monetary_clock = {}
        self.asset_returns = {}
        self.industry_returns = {}

    def calculate_factor_regimes(self, factor_series, threshold=None):
        """
        将因子划分为上行/下行区间
        """
        if factor_series.empty:
            return pd.Series()

        factor = factor_series.copy()
        factor_diff = factor.diff()

        regimes = pd.Series(index=factor.index, dtype=str)
        regimes[:] = 'neutral'

        if threshold is None:
            threshold = factor_diff.median()

        current_regime = 'neutral'
        for i in range(1, len(factor)):
            if factor_diff.iloc[i] > 0:
                regimes.iloc[i] = 'up'
                current_regime = 'up'
            elif factor_diff.iloc[i] < 0:
                regimes.iloc[i] = 'down'
                current_regime = 'down'
            else:
                regimes.iloc[i] = regimes.iloc[i-1] if i > 0 else 'neutral'

        return regimes

    def calculate_regime_returns(self, asset_returns, regimes):
        """
        计算不同因子状态下的资产收益率
        """
        common_idx = asset_returns.index.intersection(regimes.index)
        if len(common_idx) == 0:
            return {}

        asset_ret = asset_returns.loc[common_idx]
        factor_reg = regimes.loc[common_idx]

        result = {}
        for regime in factor_reg.unique():
            mask = factor_reg == regime
            if mask.sum() > 0:
                regime_returns = asset_ret.loc[mask]
                result[regime] = {
                    'mean': regime_returns.mean(),
                    'std': regime_returns.std(),
                    'count': mask.sum()
                }

        return result

    def build_macro_asset_mapping(self, factor_dict, asset_returns_df):
        """
        构建宏观因子与资产收益率的映射关系
        """
        mapping_results = {}

        for factor_name, factor_series in factor_dict.items():
            if factor_series.empty or asset_returns_df.empty:
                continue

            regimes = self.calculate_factor_regimes(factor_series)

            mapping_results[factor_name] = {}

            for asset_name in asset_returns_df.columns:
                asset_ret = asset_returns_df[asset_name]
                regime_returns = self.calculate_regime_returns(asset_ret, regimes)

                mapping_results[factor_name][asset_name] = regime_returns

        return mapping_results

    def build_growth_inflation_clock(self, growth_factor, inflation_factor, asset_returns_df):
        """
        构建增长-通胀投资时钟
        四个象限：复苏、过热、滞胀、衰退
        """
        if growth_factor.empty or inflation_factor.empty:
            return {}

        growth_regimes = self.calculate_factor_regimes(growth_factor)
        inflation_regimes = self.calculate_factor_regimes(inflation_factor)

        common_idx = growth_regimes.index.intersection(inflation_regimes.index)
        if len(common_idx) == 0:
            return {}

        clock = {}

        for date in common_idx:
            growth_state = growth_regimes.loc[date]
            inflation_state = inflation_regimes.loc[date]

            if growth_state == 'up' and inflation_state == 'up':
                clock[date] = '过热'
            elif growth_state == 'up' and inflation_state == 'down':
                clock[date] = '复苏'
            elif growth_state == 'down' and inflation_state == 'up':
                clock[date] = '滞胀'
            elif growth_state == 'down' and inflation_state == 'down':
                clock[date] = '衰退'
            else:
                clock[date] = 'neutral'

        self.growth_inflation_clock = clock

        regime_returns = {}
        for regime in ['复苏', '过热', '滞胀', '衰退']:
            mask = pd.Series([clock[d] == regime for d in common_idx], index=common_idx)
            if mask.sum() > 0:
                regime_asset_ret = asset_returns_df.loc[common_idx[mask.values]].mean()
                regime_returns[regime] = regime_asset_ret.to_dict()

        return {
            'clock': clock,
            'regime_returns': regime_returns
        }

    def build_credit_monetary_clock(self, credit_factor, monetary_factor, asset_returns_df):
        """
        构建信用-货币投资时钟
        """
        if credit_factor.empty or monetary_factor.empty:
            return {}

        credit_regimes = self.calculate_factor_regimes(credit_factor)
        monetary_regimes = self.calculate_factor_regimes(monetary_factor)

        common_idx = credit_regimes.index.intersection(monetary_regimes.index)
        if len(common_idx) == 0:
            return {}

        clock = {}

        for date in common_idx:
            credit_state = credit_regimes.loc[date]
            monetary_state = monetary_regimes.loc[date]

            if monetary_state == 'down' and credit_state == 'up':
                clock[date] = '宽货币+宽信用'
            elif monetary_state == 'down' and credit_state == 'down':
                clock[date] = '宽货币+紧信用'
            elif monetary_state == 'up' and credit_state == 'up':
                clock[date] = '紧货币+宽信用'
            elif monetary_state == 'up' and credit_state == 'down':
                clock[date] = '紧货币+紧信用'
            else:
                clock[date] = 'neutral'

        self.credit_monetary_clock = clock

        regime_returns = {}
        for regime in ['宽货币+宽信用', '宽货币+紧信用', '紧货币+宽信用', '紧货币+紧信用']:
            mask = pd.Series([clock[d] == regime for d in common_idx], index=common_idx)
            if mask.sum() > 0:
                regime_asset_ret = asset_returns_df.loc[common_idx[mask.values]].mean()
                regime_returns[regime] = regime_asset_ret.to_dict()

        return {
            'clock': clock,
            'regime_returns': regime_returns
        }

    def get_optimal_allocation(self, factor_views, mapping_results):
        """
        根据宏观因子观点获取最优配置
        """
        allocations = {}

        for factor_name, view in factor_views.items():
            if factor_name not in mapping_results:
                continue

            factor_mapping = mapping_results[factor_name]

            if view == 'up':
                best_asset = max(factor_mapping.items(),
                               key=lambda x: x[1].get('up', {}).get('mean', -999))
                allocations[best_asset[0]] = 1.0
            elif view == 'down':
                best_asset = min(factor_mapping.items(),
                               key=lambda x: x[1].get('down', {}).get('mean', 999))
                allocations[best_asset[0]] = 1.0

        return allocations

    def calculate_asset_score(self, asset_name, factor_views, mapping_results):
        """
        计算资产得分（基于各因子的加权）
        """
        total_score = 0
        weight_sum = 0

        for factor_name, view in factor_views.items():
            if factor_name not in mapping_results:
                continue

            factor_mapping = mapping_results[factor_name]

            if asset_name in factor_mapping:
                up_return = factor_mapping[asset_name].get('up', {}).get('mean', 0)
                down_return = factor_mapping[asset_name].get('down', {}).get('mean', 0)

                if view == 1:
                    score = up_return
                elif view == -1:
                    score = down_return
                else:
                    score = (up_return + down_return) / 2

                weight = 1.0
                total_score += score * weight
                weight_sum += weight

        if weight_sum > 0:
            return total_score / weight_sum
        return 0

    def build_industry_mapping(self, factor_dict, industry_returns_df):
        """
        构建宏观因子与行业收益率的映射关系
        """
        mapping_results = {}

        for factor_name, factor_series in factor_dict.items():
            if factor_series.empty or industry_returns_df.empty:
                continue

            regimes = self.calculate_factor_regimes(factor_series)

            mapping_results[factor_name] = {}

            for industry_name in industry_returns_df.columns:
                industry_ret = industry_returns_df[industry_name]
                regime_returns = self.calculate_regime_returns(industry_ret, regimes)

                mapping_results[factor_name][industry_name] = regime_returns

        return mapping_results

    def get_optimal_industries(self, factor_views, mapping_results, top_n=5):
        """
        根据宏观观点获取最优行业
        """
        industry_scores = {}

        for industry_name in mapping_results.get(list(factor_views.keys())[0], {}).keys():
            total_score = 0
            valid_count = 0

            for factor_name, view in factor_views.items():
                if factor_name in mapping_results:
                    if industry_name in mapping_results[factor_name]:
                        factor_mapping = mapping_results[factor_name][industry_name]

                        if view == 1:
                            score = factor_mapping.get('up', {}).get('mean', 0)
                        elif view == -1:
                            score = factor_mapping.get('down', {}).get('mean', 0)
                        else:
                            score = 0

                        total_score += score
                        valid_count += 1

            if valid_count > 0:
                industry_scores[industry_name] = total_score / valid_count

        if not industry_scores:
            return []

        sorted_industries = sorted(industry_scores.items(), key=lambda x: x[1], reverse=True)

        return [ind for ind, score in sorted_industries[:top_n]]

    def get_clock_phase(self, date):
        """
        获取指定日期的投资时钟状态
        """
        if date in self.growth_inflation_clock:
            return {
                'growth_inflation': self.growth_inflation_clock.get(date, 'neutral'),
                'credit_monetary': self.credit_monetary_clock.get(date, 'neutral')
            }
        return None

    def get_regime_statistics(self, clock_type='growth_inflation'):
        """
        获取各状态区间的统计数据
        """
        if clock_type == 'growth_inflation':
            clock = self.growth_inflation_clock
        else:
            clock = self.credit_monetary_clock

        if not clock:
            return {}

        regime_stats = {}
        for date, regime in clock.items():
            if regime not in regime_stats:
                regime_stats[regime] = {'count': 0, 'dates': []}
            regime_stats[regime]['count'] += 1
            regime_stats[regime]['dates'].append(date)

        return regime_stats
