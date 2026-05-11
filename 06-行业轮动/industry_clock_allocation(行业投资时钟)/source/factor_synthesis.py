"""
因子合成模块
包括：OECD法、主成分分析法、扩散指数法
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class FactorSynthesis:
    def __init__(self):
        self.factors = {}

    def oecd_method(self, indicator_df, benchmark_series=None, normalize=True):
        """
        OECD法合成因子
        步骤：
        1. 计算标准化偏差SD
        2. 计算标准化序列SC
        3. 求和得到S
        4. 按基准指标幅度调整
        """
        if indicator_df.empty:
            return pd.Series()

        df = indicator_df.copy()

        if normalize:
            for col in df.columns:
                if col == 'trade_date':
                    continue
                mean_val = df[col].mean()
                sd_val = (df[col] - mean_val).abs().mean()

                if sd_val > 0:
                    df[col] = (df[col] - mean_val) / sd_val

        indicator_cols = [col for col in df.columns if col != 'trade_date']

        s_values = df[indicator_cols].sum(axis=1)

        if benchmark_series is not None and len(benchmark_series) > 0:
            common_idx = s_values.index.intersection(benchmark_series.index)
            if len(common_idx) > 12:
                bench = benchmark_series.loc[common_idx]
                s_sub = s_values.loc[common_idx]

                k = bench.abs().mean() / s_sub.abs().mean() if s_sub.abs().mean() > 0 else 1
                d = bench.mean() - s_sub.mean() * k

                s_values = k * s_values + d

        return s_values

    def pca_method(self, indicator_df, n_components=1, benchmark_series=None):
        """
        主成分分析法合成因子
        提取第一主成分作为合成因子
        """
        if indicator_df.empty:
            return pd.Series()

        df = indicator_df.copy()
        indicator_cols = [col for col in df.columns if col != 'trade_date']

        if len(indicator_cols) < 2:
            return df[indicator_cols[0]] if indicator_cols else pd.Series()

        data = df[indicator_cols].dropna()

        if len(data) < n_components:
            return pd.Series()

        for col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                data[col] = (data[col] - mean_val) / std_val

        try:
            pca = PCA(n_components=n_components)
            pca.fit(data)

            pc_scores = pca.transform(data)

            factor_values = pc_scores[:, 0]

            factor_series = pd.Series(factor_values, index=data.index)

            if benchmark_series is not None and len(benchmark_series) > 0:
                common_idx = factor_series.index.intersection(benchmark_series.index)
                if len(common_idx) > 12:
                    bench = benchmark_series.loc[common_idx]
                    factor_sub = factor_series.loc[common_idx]

                    k = bench.abs().mean() / factor_sub.abs().mean() if factor_sub.abs().mean() > 0 else 1
                    d = bench.mean() - factor_sub.mean() * k

                    factor_series = k * factor_series + d

            return factor_series

        except Exception as e:
            print(f"PCA合成失败: {e}")
            return pd.Series()

    def diffusion_index_method(self, indicator_df, threshold=50):
        """
        扩散指数法
        计算扩张状态指标占比
        """
        if indicator_df.empty:
            return pd.Series()

        df = indicator_df.copy()
        indicator_cols = [col for col in df.columns if col != 'trade_date']

        if len(indicator_cols) < 2:
            return pd.Series(index=df.index)

        diff_indicator = df[indicator_cols].diff()

        expansion = (diff_indicator > 0).sum(axis=1)

        di = (expansion / len(indicator_cols)) * 100

        di_ma5 = di.rolling(window=5, min_periods=1).mean()

        return di_ma5

    def synthesize_growth_factor(self, leading_indicators, benchmark=None):
        """
        合成增长因子
        """
        if not leading_indicators:
            return pd.Series()

        if isinstance(leading_indicators, dict):
            indicator_df = pd.DataFrame(leading_indicators)
        else:
            indicator_df = leading_indicators.copy()

        if 'trade_date' in indicator_df.columns:
            dates = indicator_df['trade_date']
            indicator_df = indicator_df.drop('trade_date', axis=1)
            indicator_df.index = dates

        benchmark_series = None
        if benchmark is not None:
            benchmark_series = benchmark

        factor = self.oecd_method(indicator_df, benchmark_series)

        self.factors['growth'] = factor
        return factor

    def synthesize_inflation_factor(self, leading_indicators, benchmark=None):
        """
        合成通胀因子
        """
        if not leading_indicators:
            return pd.Series()

        if isinstance(leading_indicators, dict):
            indicator_df = pd.DataFrame(leading_indicators)
        else:
            indicator_df = leading_indicators.copy()

        if 'trade_date' in indicator_df.columns:
            dates = indicator_df['trade_date']
            indicator_df = indicator_df.drop('trade_date', axis=1)
            indicator_df.index = dates

        benchmark_series = None
        if benchmark is not None:
            benchmark_series = benchmark

        factor = self.oecd_method(indicator_df, benchmark_series)

        self.factors['inflation'] = factor
        return factor

    def synthesize_credit_factor(self, credit_indicators):
        """
        合成信用因子
        直接选取M1、M2、社融等指标合成
        """
        if not credit_indicators:
            return pd.Series()

        if isinstance(credit_indicators, dict):
            indicator_df = pd.DataFrame(credit_indicators)
        else:
            indicator_df = credit_indicators.copy()

        if 'trade_date' in indicator_df.columns:
            dates = indicator_df['trade_date']
            indicator_df = indicator_df.drop('trade_date', axis=1)
            indicator_df.index = dates

        factor = self.oecd_method(indicator_df)

        self.factors['credit'] = factor
        return factor

    def synthesize_monetary_factor(self, bond_yield_series):
        """
        合成货币因子
        使用一年期国债收益率HP滤波结果
        """
        if bond_yield_series.empty:
            return pd.Series()

        factor = bond_yield_series.copy()

        self.factors['monetary'] = factor
        return factor

    def rolling_synthesis(self, indicator_df, window=36, method='oecd', benchmark=None):
        """
        滚动合成因子
        用于回测时避免引入未来信息
        """
        if indicator_df.empty:
            return pd.Series()

        dates = indicator_df.index
        n = len(dates)

        if n < window:
            return self.oecd_method(indicator_df, benchmark) if method == 'oecd' else self.pca_method(indicator_df, benchmark_series=benchmark)

        rolling_factors = []

        for i in range(window, n + 1):
            window_data = indicator_df.iloc[:i]

            if method == 'oecd':
                factor = self.oecd_method(window_data, benchmark)
            elif method == 'pca':
                factor = self.pca_method(window_data, benchmark_series=benchmark)
            else:
                factor = self.diffusion_index_method(window_data)

            if len(factor) > 0:
                rolling_factors.append({
                    'date': dates[i - 1],
                    'factor': factor.iloc[-1] if len(factor) > 0 else np.nan
                })

        if rolling_factors:
            result_df = pd.DataFrame(rolling_factors)
            return result_df.set_index('date')['factor']
        else:
            return pd.Series()

    def combine_factors(self, factor_dict, method='equal'):
        """
        组合多个因子
        """
        if not factor_dict:
            return pd.Series()

        factor_series = []
        for name, factor in factor_dict.items():
            if len(factor) > 0:
                factor_df = pd.DataFrame({'date': factor.index, name: factor.values})
                factor_series.append(factor_df)

        if not factor_series:
            return pd.Series()

        combined = factor_series[0]
        for df in factor_series[1:]:
            combined = combined.merge(df, on='date', how='outer')

        combined = combined.sort_values('date')
        combined = combined.set_index('date')

        if method == 'equal':
            combined_mean = combined.mean(axis=1)
        elif method == 'pca':
            pca = PCA(n_components=1)
            data = combined.dropna()
            if len(data) > 1:
                pca.fit(data)
                combined_mean = pd.Series(pca.transform(data)[:, 0], index=data.index)
            else:
                combined_mean = combined.mean(axis=1)
        else:
            combined_mean = combined.mean(axis=1)

        return combined_mean

    def get_all_factors(self):
        """获取所有合成的因子"""
        return self.factors

    def save_factors(self, filepath):
        """保存因子数据"""
        if not self.factors:
            return

        factor_df = pd.DataFrame(self.factors)
        factor_df.index.name = 'date'
        factor_df.to_csv(filepath)

    def load_factors(self, filepath):
        """加载因子数据"""
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        for col in df.columns:
            self.factors[col] = df[col]
        return df
