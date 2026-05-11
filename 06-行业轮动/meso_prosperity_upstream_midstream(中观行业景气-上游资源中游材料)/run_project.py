"""
运行项目主脚本
从CSV文件加载数据，构建行业景气度指数，输出结果到output文件夹
"""

import sys
sys.path.insert(0, 'd:/Documents/trae_projects/meso_prosperity_upstream_midstream/source')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import (
    load_roe_from_csv,
    load_indicators_from_csv,
    IndustryDataLoader,
    load_all_roe_from_csv
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from preprocessing import IndicatorPreprocessor, align_frequencies
from evaluation import evaluate_sentiment_index
from industry_indicators import INDUSTRY_INDICATORS

OUTPUT_DIR = 'd:/Documents/trae_projects/meso_prosperity_upstream_midstream/output'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_sentiment_index(industry_name, indicators_dict, roe_series, n_components=1):
    """构建单个行业的景气度指数"""
    if len(indicators_dict) == 0:
        print(f"  警告: {industry_name}没有可用指标")
        return None, None

    aligned_data = align_frequencies(indicators_dict, target_freq='M')

    preprocessor = IndicatorPreprocessor()
    processed_indicators = {}

    for name, series in aligned_data.items():
        try:
            processed = preprocessor.process_indicator(
                series,
                remove_trend=True,
                handle_outliers=True,
                fill_missing=True,
                standardize_result=True
            )
            processed_indicators[name] = processed
        except Exception as e:
            print(f"  处理指标 {name} 失败: {e}")
            continue

    if len(processed_indicators) == 0:
        print(f"  警告: {industry_name}没有可用的处理后指标")
        return None, None

    def normalize_date(dt):
        return pd.Timestamp(dt.year, dt.month, 1)

    indicator_dates = {}
    for name, series in processed_indicators.items():
        normalized_index = series.index.map(normalize_date)
        series.index = normalized_index
        indicator_dates[name] = set(series.index)

    all_dates = set()
    for dates in indicator_dates.values():
        all_dates = all_dates.union(dates)

    date_coverage = {}
    for date in all_dates:
        count = sum(1 for dates in indicator_dates.values() if date in dates)
        date_coverage[date] = count

    min_indicators = max(3, int(len(processed_indicators) * 0.5))
    common_dates = sorted([d for d, c in date_coverage.items() if c >= min_indicators])

    if len(common_dates) < 6:
        print(f"  警告: {industry_name}公共日期点不足 (仅{len(common_dates)}个, 要求至少{min_indicators}个指标)")
        return None, None

    matrix_data = {}
    for name, series in processed_indicators.items():
        series_aligned = series.reindex(common_dates)
        matrix_data[name] = series_aligned.values

    X = np.column_stack([matrix_data[name] for name in processed_indicators.keys()])
    mask = (~np.isnan(X)).astype(int)
    X_filled = X.copy()
    for j in range(X.shape[1]):
        col_mask = mask[:, j] == 1
        if col_mask.any():
            col_mean = X[col_mask, j].mean()
            X_filled[~col_mask, j] = col_mean

    dates = common_dates

    try:
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(X_filled)

        if n_components == 1:
            estimated_factor = factors.flatten()
        else:
            estimated_factor = factors[:, 0]

        sentiment_index = pd.Series(estimated_factor, index=pd.DatetimeIndex(dates))
        sentiment_index.index.name = 'trade_date'

        class SimpleModel:
            def __init__(self, factors, loadings, pca):
                self.factors_ = factors
                self.loadings_ = loadings
                self.pca = pca
            def get_factors(self):
                return self.factors_
            def get_loadings(self):
                return self.loadings_

        model = SimpleModel(factors, pca.components_.T, pca)

        return sentiment_index, model
    except Exception as e:
        print(f"  模型拟合失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    print("=" * 80)
    print("中观景气度之上游资源/中游材料 - 项目运行")
    print("=" * 80)

    ensure_dir(OUTPUT_DIR)

    industries = ['石油石化', '煤炭', '有色金属', '钢铁', '基础化工', '建材']
    results = {}

    print("\n=== 阶段1: 加载ROE数据 ===")
    all_roe = load_all_roe_from_csv()
    print(f"ROE数据: {len(all_roe)} 行, {len(all_roe.columns)} 列")
    print(f"日期范围: {all_roe['date'].min()} 至 {all_roe['date'].max()}")

    print("\n=== 阶段2: 构建各行业景气度指数 ===")

    for industry in industries:
        print(f"\n--- 处理 {industry} ---")

        loader = IndustryDataLoader(industry)

        roe_df = loader.load_roe_data()
        print(f"  ROE数据: {len(roe_df)} 条")

        indicators = loader.load_indicator_data()
        print(f"  代理指标: {len(indicators)} 个")

        if len(roe_df) == 0 or len(indicators) == 0:
            print(f"  跳过 {industry}: 数据不足")
            continue

        roe_series = roe_df.set_index('trade_date')['roe_ttm']
        roe_series.index = roe_series.index.map(lambda dt: pd.Timestamp(dt.year, dt.month, 1))

        sentiment_index, model = build_sentiment_index(
            industry, indicators, roe_series
        )

        if sentiment_index is not None:
            metrics = evaluate_sentiment_index(sentiment_index, roe_series)

            results[industry] = {
                'roe_data': roe_df,
                'indicators': indicators,
                'sentiment_index': sentiment_index,
                'model': model,
                'metrics': metrics
            }

            print(f"  景气度指数长度: {len(sentiment_index)}")
            print(f"  ROE复现度 R²: {metrics.get('roe_reproduction', 'N/A'):.4f}" if isinstance(metrics.get('roe_reproduction'), float) else f"  ROE复现度: {metrics.get('roe_reproduction', 'N/A')}")
            print(f"  相关系数: {metrics.get('correlation', 'N/A'):.4f}" if isinstance(metrics.get('correlation'), float) else f"  相关系数: {metrics.get('correlation', 'N/A')}")

    print("\n=== 阶段3: 保存结果 ===")

    for industry, data in results.items():
        industry_dir = os.path.join(OUTPUT_DIR, industry)
        ensure_dir(industry_dir)

        data['roe_data'].to_csv(
            os.path.join(industry_dir, 'roe_data.csv'),
            index=False
        )

        sentiment_df = pd.DataFrame({'trade_date': data['sentiment_index'].index,
                                     'sentiment_index': data['sentiment_index'].values})
        sentiment_df.to_csv(
            os.path.join(industry_dir, 'sentiment_index.csv'),
            index=False
        )

        metrics_df = pd.DataFrame([data['metrics']])
        metrics_df.to_csv(
            os.path.join(industry_dir, 'metrics.csv'),
            index=False
        )

        print(f"  {industry}: 已保存到 {industry_dir}")

    print("\n=== 阶段4: 生成可视化 ===")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, industry in enumerate(industries):
        if industry in results:
            ax = axes[i]
            data = results[industry]

            sentiment = data['sentiment_index']
            roe = data['roe_data'].set_index('trade_date')['roe_ttm']

            roe.index = pd.to_datetime(roe.index)
            sentiment.index = pd.to_datetime(sentiment.index)

            def normalize_to_month(dt):
                return pd.Timestamp(dt.year, dt.month, 1)

            roe.index = roe.index.map(normalize_to_month)
            sentiment.index = sentiment.index.map(normalize_to_month)

            common_idx = sentiment.index.intersection(roe.index)

            ax_twin = ax.twinx()
            ax.plot(sentiment.loc[common_idx].index, sentiment.loc[common_idx].values,
                   'b-', label='景气度指数', linewidth=1.5)
            ax_twin.plot(roe.loc[common_idx].index, roe.loc[common_idx].values,
                        'r--', label='ROE_TTM', linewidth=1.5, alpha=0.7)

            ax.set_title(f'{industry}', fontsize=12, fontweight='bold')
            ax.set_ylabel('景气度指数', color='b')
            ax_twin.set_ylabel('ROE_TTM', color='r')
            ax.tick_params(axis='y', labelcolor='b')
            ax_twin.tick_params(axis='y', labelcolor='r')

            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'industry_comparison.png'), dpi=150)
    print(f"\n  可视化已保存到 {os.path.join(OUTPUT_DIR, 'industry_comparison.png')}")

    summary_file = os.path.join(OUTPUT_DIR, 'summary.csv')
    summary_data = []
    for industry, data in results.items():
        metrics = data['metrics']
        summary_data.append({
            '行业': industry,
            'ROE复现度': metrics.get('roe_reproduction', 'N/A'),
            '相关系数': metrics.get('correlation', 'N/A'),
            'p值': metrics.get('p_value', 'N/A'),
            '数据点数': len(data['sentiment_index'])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"  汇总表已保存到 {summary_file}")

    print("\n" + "=" * 80)
    print("项目运行完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()
