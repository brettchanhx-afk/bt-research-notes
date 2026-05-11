"""
评估模块 - 评估行业拆分前后的效果
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ReturnHomogeneityEvaluator:
    """
    收益率同质性评估
    """

    def __init__(self):
        """
        初始化评估器
        """
        pass

    def calculate_intra_industry_corr(self, returns_df, industry_stocks):
        """
        计算行业内平均相关系数

        Parameters:
        -----------
        returns_df : pd.DataFrame
            个股收益率矩阵
        industry_stocks : list
            行业内股票代码列表

        Returns:
        --------
        float
            行业内平均相关系数
        """
        available_stocks = [s for s in industry_stocks if s in returns_df.columns]
        if len(available_stocks) < 2:
            return 0

        stock_returns = returns_df[available_stocks].dropna()
        if stock_returns.shape[0] < 2:
            return 0

        corr_matrix = stock_returns.corr()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        mean_corr = upper_triangle.stack().mean()
        return mean_corr if not np.isnan(mean_corr) else 0

    def calculate_inter_industry_corr(self, returns_df, industry_stocks, other_stocks):
        """
        计算行业间平均相关系数

        Parameters:
        -----------
        returns_df : pd.DataFrame
            个股收益率矩阵
        industry_stocks : list
            行业内股票代码列表
        other_stocks : list
            其他股票代码列表

        Returns:
        --------
        float
            行业间平均相关系数
        """
        available_industry = [s for s in industry_stocks if s in returns_df.columns]
        available_other = [s for s in other_stocks if s in returns_df.columns]

        if len(available_industry) < 1 or len(available_other) < 1:
            return 0

        industry_returns = returns_df[available_industry].mean(axis=1)
        other_returns = returns_df[available_other].mean(axis=1)

        common_dates = industry_returns.dropna().index.intersection(other_returns.dropna().index)
        if len(common_dates) < 30:
            return 0

        corr = industry_returns.loc[common_dates].corr(other_returns.loc[common_dates])
        return corr if not np.isnan(corr) else 0

    def evaluate_split(self, returns_df, original_industry, split_subindustries,
                      original_stocks, subindustry_stocks_dict):
        """
        评估拆分效果

        Parameters:
        -----------
        returns_df : pd.DataFrame
            个股收益率矩阵
        original_industry : str
            原行业名称
        split_subindustries : list
            拆分后的子行业列表
        original_stocks : list
            原行业全部股票列表
        subindustry_stocks_dict : dict
            子行业到股票列表的映射

        Returns:
        --------
        dict
            评估结果
        """
        all_other_stocks = [s for s in returns_df.columns if s not in original_stocks]

        original_intra = self.calculate_intra_industry_corr(returns_df, original_stocks)
        original_inter = self.calculate_inter_industry_corr(
            returns_df, original_stocks, all_other_stocks
        )
        original_diff = original_intra - original_inter

        subindustry_results = {}
        for subindustry in split_subindustries:
            sub_stocks = subindustry_stocks_dict.get(subindustry, [])
            if sub_stocks:
                sub_intra = self.calculate_intra_industry_corr(returns_df, sub_stocks)
                sub_inter = self.calculate_inter_industry_corr(
                    returns_df, sub_stocks, all_other_stocks
                )
                sub_diff = sub_intra - sub_inter
                subindustry_results[subindustry] = {
                    'intra_industry_corr': sub_intra,
                    'inter_industry_corr': sub_inter,
                    'diff': sub_diff
                }

        return {
            'original': {
                'industry': original_industry,
                'intra_industry_corr': original_intra,
                'inter_industry_corr': original_inter,
                'diff': original_diff
            },
            'split': subindustry_results
        }


class FundamentalHomogeneityEvaluator:
    """
    基本面同质性评估
    """

    def __init__(self):
        """
        初始化评估器
        """
        pass

    def calculate_joint_variance(self, fundamental_df, groups):
        """
        计算联合方差

        Parameters:
        -----------
        fundamental_df : pd.DataFrame
            财务指标矩阵
        groups : dict
            分组信息 {组名: [股票代码列表]}

        Returns:
        --------
        float
            联合方差
        """
        total_weighted_variance = 0
        total_weight = 0

        for group_name, stocks in groups.items():
            available_stocks = [s for s in stocks if s in fundamental_df.index]
            if len(available_stocks) < 2:
                continue

            group_data = fundamental_df.loc[available_stocks].dropna()
            if len(group_data) < 2:
                continue

            variance = group_data.var()
            n = len(available_stocks)

            total_weighted_variance += (n - 1) * variance
            total_weight += (n - 1)

        if total_weight == 0:
            return 0

        joint_var = total_weighted_variance / total_weight
        return joint_var

    def f_test(self, variance1, variance2, n1, n2):
        """
        F检验

        Parameters:
        -----------
        variance1 : float
            方差1
        variance2 : float
            方差2
        n1 : int
            样本数1
        n2 : int
            样本数2

        Returns:
        --------
        float
            F统计量
        """
        if variance2 == 0:
            return float('inf') if variance1 > 0 else 0

        return variance1 / variance2

    def evaluate_split(self, fundamental_df, original_stocks, subindustry_stocks_dict,
                      n_industries_a, n_industries_b):
        """
        评估拆分效果

        Parameters:
        -----------
        fundamental_df : pd.DataFrame
            财务指标矩阵
        original_stocks : list
            原行业全部股票列表
        subindustry_stocks_dict : dict
            子行业到股票列表的映射
        n_industries_a : int
            新方案行业数
        n_industries_b : int
            旧方案行业数

        Returns:
        --------
        dict
            评估结果
        """
        original_groups = {'original_industry': original_stocks}
        original_variance = self.calculate_joint_variance(fundamental_df, original_groups)

        split_groups = subindustry_stocks_dict
        split_variance = self.calculate_joint_variance(fundamental_df, split_groups)

        f_stat = self.f_test(
            split_variance, original_variance,
            n_industries_a, n_industries_b
        )

        return {
            'original_variance': original_variance,
            'split_variance': split_variance,
            'f_statistic': f_stat,
            'conclusion': 'split_better' if f_stat < 1 else 'original_better'
        }


class ComprehensiveEvaluator:
    """
    综合评估类
    """

    def __init__(self):
        """
        初始化评估器
        """
        self.return_evaluator = ReturnHomogeneityEvaluator()
        self.fundamental_evaluator = FundamentalHomogeneityEvaluator()

    def evaluate_industry_split(self, returns_df, fundamental_df,
                               original_industry, split_subindustries,
                               original_stocks, subindustry_stocks_dict,
                               n_industries_new, n_industries_old):
        """
        综合评估行业拆分

        Parameters:
        -----------
        returns_df : pd.DataFrame
            个股收益率矩阵
        fundamental_df : pd.DataFrame
            财务指标矩阵
        original_industry : str
            原行业名称
        split_subindustries : list
            拆分后的子行业列表
        original_stocks : list
            原行业全部股票列表
        subindustry_stocks_dict : dict
            子行业到股票列表的映射
        n_industries_new : int
            新方案行业数
        n_industries_old : int
            旧方案行业数

        Returns:
        --------
        dict
            综合评估结果
        """
        return_result = self.return_evaluator.evaluate_split(
            returns_df, original_industry, split_subindustries,
            original_stocks, subindustry_stocks_dict
        )

        fundamental_result = self.fundamental_evaluator.evaluate_split(
            fundamental_df, original_stocks, subindustry_stocks_dict,
            n_industries_new, n_industries_old
        )

        return {
            'return_evaluation': return_result,
            'fundamental_evaluation': fundamental_result
        }

    def generate_evaluation_report(self, evaluation_result):
        """
        生成评估报告

        Parameters:
        -----------
        evaluation_result : dict
            评估结果

        Returns:
        --------
        str
            评估报告文本
        """
        report = []
        report.append("=" * 60)
        report.append("行业拆分效果评估报告")
        report.append("=" * 60)

        if 'return_evaluation' in evaluation_result:
            report.append("\n【收益率同质性评估】")
            ret_eval = evaluation_result['return_evaluation']

            if 'original' in ret_eval:
                orig = ret_eval['original']
                report.append(f"\n原始行业: {orig['industry']}")
                report.append(f"  行业内平均相关系数: {orig['intra_industry_corr']:.4f}")
                report.append(f"  行业间平均相关系数: {orig['inter_industry_corr']:.4f}")
                report.append(f"  差异: {orig['diff']:.4f}")

            if 'split' in ret_eval:
                report.append("\n拆分后子行业:")
                for subindustry, metrics in ret_eval['split'].items():
                    report.append(f"\n  {subindustry}:")
                    report.append(f"    行业内平均相关系数: {metrics['intra_industry_corr']:.4f}")
                    report.append(f"    行业间平均相关系数: {metrics['inter_industry_corr']:.4f}")
                    report.append(f"    差异: {metrics['diff']:.4f}")

        if 'fundamental_evaluation' in evaluation_result:
            report.append("\n\n【基本面同质性评估】")
            fund_eval = evaluation_result['fundamental_evaluation']
            report.append(f"原始方案联合方差: {fund_eval['original_variance']:.6f}")
            report.append(f"拆分方案联合方差: {fund_eval['split_variance']:.6f}")
            report.append(f"F统计量: {fund_eval['f_statistic']:.4f}")
            conclusion = "拆分方案更优" if fund_eval['conclusion'] == 'split_better' else "原始方案更优"
            report.append(f"结论: {conclusion}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


if __name__ == "__main__":
    print("测试评估模块...")
    evaluator = ComprehensiveEvaluator()
    print("综合评估器初始化完成")
