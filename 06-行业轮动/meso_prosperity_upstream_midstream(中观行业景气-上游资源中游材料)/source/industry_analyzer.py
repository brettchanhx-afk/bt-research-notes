"""
行业景气度指数构建模块
整合数据获取、Nowcasting模型和评价指标

参考研报: 华泰证券-中观景气度之上游资源中游材料 (2021-10-14)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .data_fetcher import IndustryDataLoader, init_tushare
from .nowcasting_model import NowcastingModel, SentimentIndexBuilder
from .industry_indicators import IndustryIndicatorLibrary, INDUSTRY_INDICATORS
from .evaluation import evaluate_sentiment_index, compare_with_benchmark
from .preprocessing import IndicatorPreprocessor, align_frequencies


class IndustrySentimentAnalyzer:
    """行业景气度分析师"""

    def __init__(self, industry_name: str, start_date: str = '20100101',
                 end_date: str = '20211231'):
        """
        Parameters:
        -----------
        industry_name : str
            行业名称
        start_date : str
            开始日期
        end_date : str
            结束日期
        """
        self.industry_name = industry_name
        self.start_date = start_date
        self.end_date = end_date

        self.library = IndustryIndicatorLibrary(industry_name)
        self.data_loader = IndustryDataLoader(industry_name)
        self.preprocessor = IndicatorPreprocessor()

        self.roe_data = None
        self.indicators_data = {}
        self.realtime_index = None
        self.global_index = None
        self.evaluation_results = {}

    def load_roe_data(self) -> pd.Series:
        """加载行业ROE_TTM数据"""
        print(f"正在加载{self.industry_name}行业ROE数据...")
        df = self.data_loader.load_roe_data(self.start_date, self.end_date)

        if len(df) > 0:
            self.roe_data = df.set_index('trade_date')['roe_ttm']
            self.roe_data = self.roe_data.sort_index()
            print(f"  加载了 {len(self.roe_data)} 条ROE数据")
        else:
            print(f"  警告: 无法获取{self.industry_name}的ROE数据")
            self.roe_data = pd.Series()

        return self.roe_data

    def load_indicator_data(self, indicator_name: str) -> Optional[pd.Series]:
        """
        加载单个指标数据

        Parameters:
        -----------
        indicator_name : str
            指标名称

        Returns:
        --------
        pd.Series or None
        """
        from .data_fetcher import (get_macro_ppi, get_market_indicator,
                                  get_commodity_price, get_futures_data,
                                  get_index_daily)

        indicator_map = {
            'PMI': lambda: get_market_indicator('PMI', self.start_date, self.end_date),
            'PPI': lambda: get_market_indicator('PPI', self.start_date, self.end_date),
            '美元指数': lambda: get_index_daily('DXY.UTF', self.start_date, self.end_date),
        }

        if indicator_name in indicator_map:
            df = indicator_map[indicator_name]()
            if len(df) > 0 and 'close' in df.columns:
                return df.set_index('trade_date')['close']

        return None

    def load_all_indicators(self) -> Dict[str, pd.Series]:
        """加载所有代理指标数据"""
        print(f"正在加载{self.industry_name}行业的代理指标数据...")

        all_indicators = self.data_loader.load_indicator_data(
            self.start_date, self.end_date
        )

        self.indicators_data = all_indicators

        print(f"  成功加载 {len(self.indicators_data)} 个指标")

        return self.indicators_data

    def build_sentiment_indices(self) -> Tuple[pd.Series, pd.Series]:
        """
        构建景气度指数

        Returns:
        --------
        realtime_index : pd.Series
            实时景气度指数
        global_index : pd.Series
            全局景气度指数
        """
        if len(self.indicators_data) == 0:
            print("错误: 没有可用的指标数据")
            return pd.Series(), pd.Series()

        print(f"\n正在为{self.industry_name}构建景气度指数...")

        aligned_data = align_frequencies(self.indicators_data, target_freq='M')

        for name, series in aligned_data.items():
            aligned_data[name] = self.preprocessor.process_indicator(
                series,
                remove_trend=True,
                handle_outliers=True,
                fill_missing=True,
                standardize_result=True
            )

        common_dates = None
        for series in aligned_data.values():
            if common_dates is None:
                common_dates = set(series.index)
            else:
                common_dates = common_dates.intersection(set(series.index))

        if not common_dates or len(common_dates) < 12:
            print("错误: 公共日期点不足")
            return pd.Series(), pd.Series()

        common_dates = sorted(common_dates)

        matrix_data = {}
        for name, series in aligned_data.items():
            series_aligned = series.loc[common_dates]
            matrix_data[name] = series_aligned.values

        X = np.column_stack([matrix_data[name] for name in aligned_data.keys()])
        dates = common_dates

        model = NowcastingModel(n_components=1, p=2)
        mask = (~np.isnan(X)).astype(int)
        X = np.nan_to_num(X, nan=0)

        model.fit(X, mask)

        factors = model.get_factors()

        if len(factors) != len(dates):
            min_len = min(len(factors), len(dates))
            factors = factors[:min_len]
            dates = dates[:min_len]

        self.global_index = pd.Series(factors, index=pd.DatetimeIndex(dates))
        self.global_index = self.global_index.sort_index()

        self.realtime_index = self.global_index

        print(f"  全局景气度指数量: {len(self.global_index)}")
        print(f"  实时景气度指数量: {len(self.realtime_index)}")

        return self.realtime_index, self.global_index

    def evaluate_indices(self) -> Dict[str, Dict]:
        """
        评估景气度指数

        Returns:
        --------
        Dict with evaluation results
        """
        if self.roe_data is None or len(self.roe_data) == 0:
            print("错误: 没有ROE数据可用于评估")
            return {}

        if self.global_index is None or len(self.global_index) == 0:
            print("错误: 没有景气度指数可用于评估")
            return {}

        print(f"\n正在评估{self.industry_name}的景气度指数...")

        common_idx = self.global_index.index.intersection(self.roe_data.index)

        if len(common_idx) < 10:
            print(f"  警告: 公共日期点不足 ({len(common_idx)})")
            return {}

        index_aligned = self.global_index.loc[common_idx]
        roe_aligned = self.roe_data.loc[common_idx]

        global_eval = evaluate_sentiment_index(index_aligned, roe_aligned)

        if self.realtime_index is not None:
            common_idx_rt = self.realtime_index.index.intersection(self.roe_data.index)
            if len(common_idx_rt) >= 10:
                rt_index = self.realtime_index.loc[common_idx_rt]
                rt_roe = self.roe_data.loc[common_idx_rt]
                realtime_eval = evaluate_sentiment_index(rt_index, rt_roe)
            else:
                realtime_eval = {}
        else:
            realtime_eval = {}

        self.evaluation_results = {
            'global': global_eval,
            'realtime': realtime_eval
        }

        print(f"  全局指数 ROE复现度: {global_eval.get('roe_reproduction', 0):.4f}")
        print(f"  全局指数 最新方向准确率: {global_eval.get('latest_direction_accuracy', 0):.4f}")
        print(f"  全局指数 预测方向准确率: {global_eval.get('prediction_direction_accuracy', 0):.4f}")

        if realtime_eval:
            print(f"  实时指数 ROE复现度: {realtime_eval.get('roe_reproduction', 0):.4f}")
            print(f"  实时指数 最新方向准确率: {realtime_eval.get('latest_direction_accuracy', 0):.4f}")
            print(f"  实时指数 预测方向准确率: {realtime_eval.get('prediction_direction_accuracy', 0):.4f}")

        return self.evaluation_results

    def get_loadings(self) -> pd.DataFrame:
        """
        获取因子载荷

        Returns:
        --------
        pd.DataFrame
        """
        return self.library.get_indicator_loadings()

    def generate_report(self) -> str:
        """
        生成分析报告

        Returns:
        --------
        str
        """
        report = []
        report.append("=" * 80)
        report.append(f"{self.industry_name}行业景气度分析报告")
        report.append("=" * 80)
        report.append("")

        report.append(f"分析日期范围: {self.start_date} - {self.end_date}")
        report.append("")

        report.append("1. 数据概况")
        report.append("-" * 40)
        if self.roe_data is not None:
            report.append(f"  ROE_TTM数据量: {len(self.roe_data)}")
        report.append(f"  代理指标数量: {len(self.indicators_data)}")
        report.append("")

        if self.evaluation_results:
            report.append("2. 评估结果")
            report.append("-" * 40)

            if 'global' in self.evaluation_results:
                eval_data = self.evaluation_results['global']
                report.append("  全局景气度指数:")
                report.append(f"    ROE复现度: {eval_data.get('roe_reproduction', 0):.4f}")
                report.append(f"    相关系数: {eval_data.get('correlation', 0):.4f}")
                report.append(f"    最新一期方向准确率: {eval_data.get('latest_direction_accuracy', 0):.4f}")
                report.append(f"    下期预测方向准确率: {eval_data.get('prediction_direction_accuracy', 0):.4f}")
                report.append("")

            if 'realtime' in self.evaluation_results:
                eval_data = self.evaluation_results['realtime']
                report.append("  实时景气度指数:")
                report.append(f"    ROE复现度: {eval_data.get('roe_reproduction', 0):.4f}")
                report.append(f"    最新一期方向准确率: {eval_data.get('latest_direction_accuracy', 0):.4f}")
                report.append(f"    下期预测方向准确率: {eval_data.get('prediction_direction_accuracy', 0):.4f}")
                report.append("")

        report.append("3. 主要代理指标（载荷最高的5个）")
        report.append("-" * 40)
        top_indicators = self.library.get_top_indicators(5)
        for indicator, loading in top_indicators:
            sign = '+' if loading > 0 else '-'
            report.append(f"  {sign} {indicator}: {abs(loading):.2f}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)


class MultiIndustrySentimentAnalyzer:
    """多行业景气度分析师"""

    def __init__(self, industries: Optional[List[str]] = None,
                 start_date: str = '20100101',
                 end_date: str = '20211231'):
        """
        Parameters:
        -----------
        industries : List[str], optional
            行业列表，如果为None则分析所有6个行业
        start_date : str
            开始日期
        end_date : str
            结束日期
        """
        if industries is None:
            self.industries = list(INDUSTRY_INDICATORS.keys())
        else:
            self.industries = industries

        self.start_date = start_date
        self.end_date = end_date

        self.analyzers = {}
        self.summary_results = {}

    def run_analysis(self) -> Dict[str, Dict]:
        """
        运行所有行业的分析

        Returns:
        --------
        Dict with results for each industry
        """
        print("=" * 80)
        print("多行业景气度分析")
        print("=" * 80)
        print(f"分析行业: {', '.join(self.industries)}")
        print(f"日期范围: {self.start_date} - {self.end_date}")
        print("=" * 80)

        init_tushare()

        for industry in self.industries:
            print(f"\n{'=' * 80}")
            print(f"正在分析: {industry}")
            print('=' * 80)

            analyzer = IndustrySentimentAnalyzer(
                industry, self.start_date, self.end_date
            )

            analyzer.load_roe_data()
            analyzer.load_all_indicators()
            analyzer.build_sentiment_indices()
            analyzer.evaluate_indices()

            self.analyzers[industry] = analyzer
            self.summary_results[industry] = analyzer.evaluation_results

        return self.summary_results

    def get_summary_table(self) -> pd.DataFrame:
        """
        获取汇总表格

        Returns:
        --------
        pd.DataFrame
        """
        data = []

        for industry in self.industries:
            if industry not in self.summary_results:
                continue

            results = self.summary_results[industry]

            row = {'行业': industry}

            if 'global' in results:
                row['全局ROE复现度'] = results['global'].get('roe_reproduction', 0)
                row['全局方向准确率'] = results['global'].get('latest_direction_accuracy', 0)
            else:
                row['全局ROE复现度'] = 0
                row['全局方向准确率'] = 0

            if 'realtime' in results:
                row['实时ROE复现度'] = results['realtime'].get('roe_reproduction', 0)
                row['实时方向准确率'] = results['realtime'].get('latest_direction_accuracy', 0)
            else:
                row['实时ROE复现度'] = 0
                row['实时方向准确率'] = 0

            expected = INDUSTRY_INDICATORS[industry]['roe_reproduction']
            row['预期全局ROE复现度'] = expected['global']
            row['差异'] = row['全局ROE复现度'] - row['预期全局ROE复现度']

            data.append(row)

        return pd.DataFrame(data)

    def save_results(self, output_dir: str = 'output'):
        """
        保存分析结果

        Parameters:
        -----------
        output_dir : str
            输出目录
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        summary_df = self.get_summary_table()
        summary_df.to_csv(f'{output_dir}/industry_sentiment_summary.csv', index=False)
        print(f"汇总结果已保存到: {output_dir}/industry_sentiment_summary.csv")

        for industry, analyzer in self.analyzers.items():
            safe_name = industry.replace('/', '_')
            report = analyzer.generate_report()

            with open(f'{output_dir}/{safe_name}_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"分析报告已保存到: {output_dir}/{safe_name}_report.txt")


if __name__ == '__main__':
    print("测试行业景气度分析模块...")

    init_tushare()

    print("\n测试单个行业分析:")
    analyzer = IndustrySentimentAnalyzer('石油石化', '20150101', '20211231')

    analyzer.load_roe_data()
    analyzer.load_all_indicators()

    if len(analyzer.indicators_data) > 0:
        analyzer.build_sentiment_indices()
        analyzer.evaluate_indices()
        print(analyzer.generate_report())
    else:
        print("警告: 未能加载指标数据")

    print("\n测试多行业分析:")
    multi_analyzer = MultiIndustrySentimentAnalyzer(
        industries=['石油石化', '煤炭'],
        start_date='20150101',
        end_date='20211231'
    )
    results = multi_analyzer.run_analysis()
