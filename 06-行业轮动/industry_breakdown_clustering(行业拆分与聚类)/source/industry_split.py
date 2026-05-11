"""
行业拆分模块 - 计算行业内个股分化度，确定拆分方案
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class IndustryReturnDivergence:
    """
    收益率分化度计算类
    """

    def __init__(self, n_simulations=1000, min_days=750):
        """
        Parameters:
        -----------
        n_simulations : int
            蒙特卡洛模拟次数
        min_days : int
            每次模拟最小天数
        """
        self.n_simulations = n_simulations
        self.min_days = min_days

    def calculate_long_short_return(self, returns_df):
        """
        计算多空累计收益分化度

        将行业内个股按当日收益率分为5层，计算多空累计收益差

        Parameters:
        -----------
        returns_df : pd.DataFrame
            个股收益率矩阵

        Returns:
        --------
        float
            多空累计收益
        """
        long_short_cum = []
        for date in returns_df.index:
            daily_returns = returns_df.loc[date].dropna()
            if len(daily_returns) >= 5:
                sorted_returns = daily_returns.sort_values()
                n = len(sorted_returns)
                layer_size = max(1, n // 5)
                top_layer = sorted_returns.iloc[-layer_size:]
                bottom_layer = sorted_returns.iloc[:layer_size]
                long_return = top_layer.median()
                short_return = bottom_layer.median()
                long_short_cum.append(long_return - short_return)

        if long_short_cum:
            return np.sum(long_short_cum)
        return 0

    def calculate_regression_r2(self, stock_returns, industry_returns):
        """
        计算回归拟合优度分化度

        Parameters:
        -----------
        stock_returns : pd.Series
            个股收益率序列
        industry_returns : pd.Series
            行业指数收益率序列

        Returns:
        --------
        float
            1 - R^2均值
        """
        common_dates = stock_returns.dropna().index.intersection(industry_returns.dropna().index)
        if len(common_dates) < 30:
            return 1.0

        X = industry_returns.loc[common_dates].values.reshape(-1, 1)
        y = stock_returns.loc[common_dates].values

        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)

        return 1 - r2

    def calculate_correlation_divergence(self, returns_df):
        """
        计算平均相关系数分化度

        Parameters:
        -----------
        returns_df : pd.DataFrame
            个股收益率矩阵

        Returns:
        --------
        float
            1 - 平均相关系数
        """
        corr_matrix = returns_df.corr()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        mean_corr = upper_triangle.stack().mean()
        return 1 - mean_corr if not np.isnan(mean_corr) else 1.0

    def monte_carlo_simulation(self, returns_df, industry_returns):
        """
        蒙特卡洛模拟计算分化度排名

        Parameters:
        -----------
        returns_df : pd.DataFrame
            个股收益率矩阵
        industry_returns : pd.Series
            行业指数收益率序列

        Returns:
        --------
        dict
            三种分化度的排名
        """
        n_days = len(returns_df)
        if n_days < self.min_days:
            return {'long_short': 1, 'regression': 1, 'correlation': 1}

        results = {
            'long_short': [],
            'regression': [],
            'correlation': []
        }

        for _ in range(self.n_simulations):
            max_start = n_days - self.min_days
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            length = np.random.randint(self.min_days, n_days - start_idx + 1)
            end_idx = start_idx + length

            period_returns = returns_df.iloc[start_idx:end_idx]
            period_industry = industry_returns.iloc[start_idx:end_idx]

            ls_return = self.calculate_long_short_return(period_returns)
            results['long_short'].append(ls_return)

            r2_values = []
            for stock in period_returns.columns:
                stock_ret = period_returns[stock].dropna()
                if len(stock_ret) > 30:
                    r2 = self.calculate_regression_r2(stock_ret, period_industry)
                    r2_values.append(r2)
            if r2_values:
                results['regression'].append(np.mean(r2_values))

            corr_div = self.calculate_correlation_divergence(period_returns)
            results['correlation'].append(corr_div)

        return {
            'long_short': np.mean(results['long_short']),
            'regression': np.mean(results['regression']),
            'correlation': np.mean(results['correlation'])
        }

    def rank_industries(self, industry_data_dict):
        """
        对所有行业进行分化度排名

        Parameters:
        -----------
        industry_data_dict : dict
            行业名称到(个股收益率矩阵, 行业收益率序列)的映射

        Returns:
        --------
        pd.DataFrame
            各行业分化度及排名
        """
        rankings = []

        for industry, (stock_returns, industry_returns) in industry_data_dict.items():
            divergence = self.monte_carlo_simulation(stock_returns, industry_returns)
            rankings.append({
                'industry': industry,
                'long_short_div': divergence['long_short'],
                'regression_div': divergence['regression'],
                'correlation_div': divergence['correlation']
            })

        df = pd.DataFrame(rankings)

        df['ls_rank'] = df['long_short_div'].rank(ascending=False)
        df['reg_rank'] = df['regression_div'].rank(ascending=False)
        df['corr_rank'] = df['correlation_div'].rank(ascending=False)
        df['avg_rank'] = (df['ls_rank'] + df['reg_rank'] + df['corr_rank']) / 3

        return df.sort_values('avg_rank', ascending=False)


class IndustryFundamentalDivergence:
    """
    基本面分化度计算类
    """

    def __init__(self):
        """
        初始化基本面分化度计算器
        """
        self.fundamental_metrics = [
            'pe_ttm', 'pb', 'roe', 'roa', 'roic',
            'netprofit_margin', 'grossprofit_margin',
            'debt_to_assets', 'asset_turnover', 'inventory_turnover'
        ]

    def calculate_metric_divergence(self, metric_series):
        """
        计算单个财务指标的行业分化度

        Parameters:
        -----------
        metric_series : pd.Series
            行业内各股的财务指标值

        Returns:
        --------
        float
            指标分化度（用变异系数CV表征）
        """
        valid_data = metric_series.dropna()
        if len(valid_data) < 2:
            return 0
        mean = valid_data.mean()
        std = valid_data.std()
        if mean == 0:
            return 0
        cv = std / abs(mean)
        return cv

    def calculate_fundamental_divergence(self, fundamental_df):
        """
        计算基本面综合分化度

        Parameters:
        -----------
        fundamental_df : pd.DataFrame
            行业内个股的财务指标矩阵

        Returns:
        --------
        dict
            各维度分化度
        """
        divergence_results = {}

        if 'pe_ttm' in fundamental_df.columns or 'pe' in fundamental_df.columns:
            col = 'pe_ttm' if 'pe_ttm' in fundamental_df.columns else 'pe'
            divergence_results['pe_div'] = self.calculate_metric_divergence(fundamental_df[col])

        if 'pb' in fundamental_df.columns:
            divergence_results['pb_div'] = self.calculate_metric_divergence(fundamental_df['pb'])

        if 'roe' in fundamental_df.columns:
            divergence_results['roe_div'] = self.calculate_metric_divergence(fundamental_df['roe'])

        if 'roa' in fundamental_df.columns:
            divergence_results['roa_div'] = self.calculate_metric_divergence(fundamental_df['roa'])

        if 'roic' in fundamental_df.columns:
            divergence_results['roic_div'] = self.calculate_metric_divergence(fundamental_df['roic'])

        if 'netprofit_margin' in fundamental_df.columns:
            divergence_results['netprofit_div'] = self.calculate_metric_divergence(fundamental_df['netprofit_margin'])

        if 'grossprofit_margin' in fundamental_df.columns:
            divergence_results['grossprofit_div'] = self.calculate_metric_divergence(fundamental_df['grossprofit_margin'])

        if 'debt_to_assets' in fundamental_df.columns:
            divergence_results['debt_div'] = self.calculate_metric_divergence(fundamental_df['debt_to_assets'])

        if 'asset_turnover' in fundamental_df.columns:
            divergence_results['turnover_div'] = self.calculate_metric_divergence(fundamental_df['asset_turnover'])

        if 'inventory_turnover' in fundamental_df.columns:
            divergence_results['inventory_div'] = self.calculate_metric_divergence(fundamental_df['inventory_turnover'])

        return divergence_results

    def rank_industries_fundamental(self, industry_fundamental_dict):
        """
        对所有行业进行基本面分化度排名

        Parameters:
        -----------
        industry_fundamental_dict : dict
            行业名称到财务数据的映射

        Returns:
        --------
        pd.DataFrame
            各行业基本面分化度及排名
        """
        rankings = []

        for industry, fundamental_df in industry_fundamental_dict.items():
            div = self.calculate_fundamental_divergence(fundamental_df)
            rankings.append({
                'industry': industry,
                **div
            })

        df = pd.DataFrame(rankings)

        value_cols = [col for col in df.columns if col.endswith('_div')]
        if value_cols:
            df['avg_div'] = df[value_cols].mean(axis=1)
            df['div_rank'] = df['avg_div'].rank(ascending=False)

        return df.sort_values('div_rank')


class IndustrySplitter:
    """
    行业拆分决策类
    """

    SPLIT_RULES = {
        '食品饮料': ['酒类', '饮料', '食品'],
        '非银行金融': ['证券', '保险', '多元金融']
    }

    @staticmethod
    def should_split(return_div_rank, fundamental_div_rank, industry_name,
                     market_cap_ratio=None, lifecycle_stage=None):
        """
        决策是否需要拆分行业

        Parameters:
        -----------
        return_div_rank : int
            收益分化度排名
        fundamental_div_rank : int
            基本面分化度排名
        industry_name : str
            行业名称
        market_cap_ratio : float
            行业市值占比
        lifecycle_stage : str
            生命周期阶段

        Returns:
        --------
        bool
            是否需要拆分
        """
        threshold = 15

        if return_div_rank <= threshold and fundamental_div_rank <= threshold:
            return True

        if industry_name in ['食品饮料', '非银行金融']:
            return True

        return False

    @staticmethod
    def get_split_subindustries(industry_name):
        """
        获取拆分后的子行业

        Parameters:
        -----------
        industry_name : str
            行业名称

        Returns:
        --------
        list
            子行业列表
        """
        return IndustrySplitter.SPLIT_RULES.get(industry_name, [industry_name])


if __name__ == "__main__":
    print("测试行业拆分模块...")
    divergence_calc = IndustryReturnDivergence(n_simulations=10)
    print(f"收益率分化度计算器初始化完成，模拟次数: {divergence_calc.n_simulations}")
