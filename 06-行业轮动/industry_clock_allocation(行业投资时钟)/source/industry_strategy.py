"""
行业轮动策略模块
基于投资时钟的行业配置
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class IndustryStrategy:
    def __init__(self, top_n=5, momentum_window=20):
        self.top_n = top_n
        self.momentum_window = momentum_window
        self.selected_industries = []
        self.industry_scores = {}

    def calculate_industry_scores(self, industry_returns_df, factor_views, industry_mapping):
        """
        计算各行业得分
        """
        if industry_returns_df.empty:
            return {}

        scores = {}

        for industry_name in industry_returns_df.columns:
            total_score = 0
            valid_factors = 0

            for factor_name, view in factor_views.items():
                if factor_name in industry_mapping:
                    if industry_name in industry_mapping[factor_name]:
                        mapping = industry_mapping[factor_name][industry_name]

                        if view == 1:
                            score = mapping.get('up', {}).get('mean', 0)
                        elif view == -1:
                            score = mapping.get('down', {}).get('mean', 0)
                        else:
                            score = 0

                        total_score += score
                        valid_factors += 1

            if valid_factors > 0:
                scores[industry_name] = total_score / valid_factors
            else:
                scores[industry_name] = 0

        self.industry_scores = scores
        return scores

    def select_industries_by_score(self, scores, top_n=None):
        """
        根据得分选择行业
        """
        if top_n is None:
            top_n = self.top_n

        if not scores:
            return []

        sorted_industries = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        selected = []
        for industry, score in sorted_industries[:top_n]:
            selected.append(industry)

        same_score_threshold = sorted_industries[top_n - 1][1] if len(sorted_industries) >= top_n else 0
        for industry, score in sorted_industries[top_n:]:
            if score == same_score_threshold:
                selected.append(industry)
            else:
                break

        self.selected_industries = selected
        return selected

    def apply_momentum_filter(self, selected_industries, industry_returns_df, momentum_window=None):
        """
        动量过滤：选择近期表现较好的行业
        """
        if momentum_window is None:
            momentum_window = self.momentum_window

        if industry_returns_df.empty or not selected_industries:
            return selected_industries

        recent_returns = industry_returns_df[selected_industries].tail(momentum_window)
        momentum_returns = (1 + recent_returns).prod() - 1

        momentum_sorted = momentum_returns.sort_values(ascending=False)

        final_selection = momentum_sorted.head(self.top_n).index.tolist()

        return final_selection

    def build_clock_industry_mapping(self):
        """
        构建投资时钟-行业映射关系
        基于研报结论
        """
        mapping = {
            'growth_up': ['周期', '金融', '可选消费'],
            'growth_down': ['防御消费', '公共事业'],
            'inflation_up': ['上游资源', '中游材料', '部分消费'],
            'inflation_down': ['TMT', '大金融'],
            'credit_up': ['TMT', '大金融'],
            'credit_down': ['防御板块'],
            'monetary_up': ['公共产业'],
            'monetary_down': ['债券敏感型']
        }

        return mapping

    def get_industries_by_regime(self, regime, industry_mapping):
        """
        根据宏观状态获取受益行业
        """
        industry_mapping = self.build_clock_industry_mapping()

        regime_industries = {
            '复苏': ['可选消费', '金融', '周期'],
            '过热': ['上游资源', '中游材料', '可选消费'],
            '滞胀': ['防御消费', '公共事业'],
            '衰退': ['金融', '公共事业'],
            '宽货币+宽信用': ['TMT', '大金融'],
            '宽货币+紧信用': ['债券敏感型'],
            '紧货币+宽信用': ['金融', '周期'],
            '紧货币+紧信用': ['防御板块']
        }

        return regime_industries.get(regime, [])

    def select_industries(self, industry_returns_df, factor_views, industry_mapping,
                         use_momentum=True, momentum_window=None):
        """
        选择行业
        """
        scores = self.calculate_industry_scores(industry_returns_df, factor_views, industry_mapping)

        selected = self.select_industries_by_score(scores)

        if use_momentum:
            selected = self.apply_momentum_filter(selected, industry_returns_df, momentum_window)

        return selected

    def calculate_equal_weights(self, selected_industries):
        """
        计算等权配置
        """
        if not selected_industries:
            return {}

        weight = 1.0 / len(selected_industries)
        return {ind: weight for ind in selected_industries}

    def calculate_industry_return(self, weights, returns_series):
        """
        计算行业组合收益
        """
        if not weights or returns_series.empty:
            return 0

        total_return = 0
        for industry, weight in weights.items():
            if industry in returns_series.index:
                total_return += weight * returns_series[industry]

        return total_return

    def get_industry_performance_stats(self, industry_returns_df, benchmark_returns=None):
        """
        获取行业表现统计
        """
        if industry_returns_df.empty:
            return {}

        stats = {}

        for industry in industry_returns_df.columns:
            returns = industry_returns_df[industry]

            cum_return = (1 + returns).prod() - 1
            annual_return = returns.mean() * 12
            annual_vol = returns.std() * np.sqrt(12)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            stats[industry] = {
                'cumulative_return': cum_return,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe
            }

        return stats

    def compare_with_benchmark(self, portfolio_returns, benchmark_returns):
        """
        与基准比较
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}

        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) == 0:
            return {}

        excess_returns = portfolio_returns.loc[common_idx] - benchmark_returns.loc[common_idx]

        tracking_error = excess_returns.std() * np.sqrt(12)
        information_ratio = excess_returns.mean() * 12 / tracking_error if tracking_error > 0 else 0

        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'excess_return': excess_returns.mean() * 12
        }

    def get_strategy_summary(self, returns_series, selected_history, benchmark_returns=None):
        """
        获取策略摘要
        """
        if returns_series.empty:
            return {}

        cumulative_return = (1 + returns_series).prod() - 1
        annual_return = returns_series.mean() * 12
        annual_volatility = returns_series.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        running_max = (1 + returns_series).cumprod().cummax()
        drawdown = (1 + returns_series).cumprod() / running_max - 1
        max_drawdown = drawdown.min()

        winning_months = (returns_series > 0).sum()
        total_months = len(returns_series)
        win_rate = winning_months / total_months if total_months > 0 else 0

        summary = {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

        if benchmark_returns is not None:
            benchmark_stats = self.compare_with_benchmark(returns_series, benchmark_returns)
            summary.update(benchmark_stats)

        return summary

    def get_industry_allocation_history(self):
        """获取历史行业配置"""
        return self.selected_industries

    def save_allocation(self, filepath):
        """保存配置记录"""
        if self.selected_industries:
            df = pd.DataFrame({
                'date': [s.get('date') for s in self.selected_industries],
                'industries': [s.get('industries') for s in self.selected_industries]
            })
            df.to_csv(filepath, index=False)
