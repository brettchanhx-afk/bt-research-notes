"""
主程序入口
运行完整的Nowcasting行业景气度分析流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.config import (
    TARGET_INDUSTRY, TARGET_INDUSTRY_CODE,
    START_DATE, END_DATE, OUTPUT_DIR,
    LATENT_FACTOR_NUM, MAX_INDICATORS
)
from source.data_fetcher import SteelIndustryDataFetcher, DataCache, MultiSourceDataFetcher, LocalDataLoader
from source.sentiment_index import SteelIndustrySentimentIndex, SentimentIndexComparison
from source.backtest import IndustryTimingBacktest, GodViewBacktest, TimingComparison
from source.utils import check_stationarity


def main():
    """
    主函数：运行完整的Nowcasting行业景气度分析
    """
    print("=" * 60)
    print("Nowcasting行业景气度分析系统")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n1. 数据获取阶段")
    print("-" * 40)

    indicator_data, steel_index, roe_data = load_local_data()

    if indicator_data is None or indicator_data.empty:
        print("本地数据加载失败，使用模拟数据")
        data_fetcher = SteelIndustryDataFetcher()
        print(f"\n数据源可用性:")
        print(f"  - efinance: {data_fetcher.efinance_available}")
        print(f"  - akshare: {data_fetcher.akshare_available}")
        print(f"  - baostock: {data_fetcher.baostock_available}")
        print(f"  - tushare: {data_fetcher.tushare_available}")
        indicator_data = fetch_data_from_api(data_fetcher)

    if indicator_data is None or indicator_data.empty:
        print("API数据获取也失败，使用模拟数据")
        indicator_data = generate_simulated_data()

    print(f"\n使用指标数据: {indicator_data.shape}")
    print(f"指标列表: {list(indicator_data.columns[:10])}...")

    print("\n2. 景气度指数构建阶段")
    print("-" * 40)

    sentiment_builder = SteelIndustrySentimentIndex(
        n_factors=LATENT_FACTOR_NUM,
        n_indicators=min(MAX_INDICATORS, len(indicator_data.columns))
    )

    try:
        sentiment_index = sentiment_builder.build_sentiment_index(
            indicator_data,
            benchmark=roe_data['ROE_TTM'] if roe_data is not None and not roe_data.empty else None,
            use_selector=True
        )

        print(f"景气度指数构建完成: {len(sentiment_index)} 个数据点")

        plt.figure(figsize=(14, 6))
        plt.plot(sentiment_index.index, sentiment_index.values, 'b-', linewidth=1.5)
        plt.title(f'{TARGET_INDUSTRY} 行业景气度指数 (Nowcasting)')
        plt.xlabel('日期')
        plt.ylabel('景气度指数')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_index.png'), dpi=150)
        plt.close()
        print(f"景气度指数图已保存: {os.path.join(OUTPUT_DIR, 'sentiment_index.png')}")

    except Exception as e:
        print(f"景气度指数构建失败: {e}")
        return None

    print("\n3. 指标统计分析阶段")
    print("-" * 40)

    if sentiment_builder.indicator_stats is not None:
        print("\n指标统计信息:")
        print(sentiment_builder.indicator_stats[['indicator', 'mean', 'stationarity_pvalue', 'n_obs']].head(10).to_string())

        if hasattr(sentiment_builder.nowcasting_model, 'indicator_weights') and sentiment_builder.nowcasting_model.indicator_weights is not None:
            print("\n最重要的10个指标:")
            top_weights = sentiment_builder.nowcasting_model.indicator_weights.nlargest(10)
            print(top_weights.to_string())

            plt.figure(figsize=(12, 6))
            top_weights.plot(kind='bar', color='steelblue')
            plt.title('指标权重 (Top 10)')
            plt.xlabel('指标')
            plt.ylabel('权重')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'indicator_weights.png'), dpi=150)
            plt.close()
            print(f"指标权重图已保存: {os.path.join(OUTPUT_DIR, 'indicator_weights.png')}")

    print("\n4. 回测分析阶段")
    print("-" * 40)

    if sentiment_index is not None and len(sentiment_index) > 20:
        try:
            if steel_index is not None and not steel_index.empty:
                price_series = steel_index['close'].copy()
                price_series.name = 'price'
                print(f"使用真实钢铁行业指数数据: {len(price_series)} 条")
            else:
                price_series = generate_simulated_price(sentiment_index)
                print("使用模拟价格数据")

            backtester = IndustryTimingBacktest(initial_capital=1000000.0)
            result = backtester.run_backtest(
                sentiment_index,
                price_series,
                sentiment_threshold=0.0
            )

            print(f"\n回测结果:")
            print(f"  总收益: {result.total_return:.2%}")
            print(f"  年化收益: {result.annual_return:.2%}")
            print(f"  夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  最大回撤: {result.max_drawdown:.2%}")
            print(f"  胜率: {result.win_rate:.2%}")
            print(f"  交易次数: {result.n_trades}")

            plt.figure(figsize=(14, 8))

            plt.subplot(2, 1, 1)
            plt.plot(result.cumulative_returns.index, result.cumulative_returns.values, 'b-', linewidth=1.5)
            plt.title('策略累积收益')
            plt.xlabel('日期')
            plt.ylabel('累积收益')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.plot(result.excess_returns.index, result.excess_returns.values, 'g-', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.title('超额收益')
            plt.xlabel('日期')
            plt.ylabel('超额收益')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_results.png'), dpi=150)
            plt.close()
            print(f"回测结果图已保存: {os.path.join(OUTPUT_DIR, 'backtest_results.png')}")

            if roe_data is not None and not roe_data.empty:
                print("\n5. ROE_TTM对比分析")
                print("-" * 40)

                comparison = SentimentIndexComparison()
                comparison.add_index('景气度指数', sentiment_index)
                comparison.add_index('ROE_TTM', roe_data['ROE_TTM'])

                result_df = comparison.compare_with_benchmark(roe_data['ROE_TTM'])
                if result_df is not None and not result_df.empty:
                    print("\n指数对比结果:")
                    print(result_df.to_string())

                    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

                    sentiment_index.plot(ax=axes[0], title='行业景气度指数 vs ROE_TTM', label='景气度指数', color='blue')
                    axes[0].set_ylabel('景气度指数')
                    axes[0].legend(loc='upper left')
                    axes[0].grid(True, alpha=0.3)

                    roe_data['ROE_TTM'].plot(ax=axes[1], label='ROE_TTM', color='orange')
                    axes[1].set_ylabel('ROE_TTM')
                    axes[1].legend(loc='upper left')
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, 'roe_comparison.png'), dpi=150)
                    plt.close()
                    print(f"ROE_TTM对比图已保存: {os.path.join(OUTPUT_DIR, 'roe_comparison.png')}")

        except Exception as e:
            print(f"回测分析失败: {e}")

    print("\n6. 结果保存阶段")
    print("-" * 40)

    if sentiment_index is not None:
        sentiment_path = os.path.join(OUTPUT_DIR, 'sentiment_index.csv')
        sentiment_index.to_csv(sentiment_path)
        print(f"景气度指数已保存: {sentiment_path}")

        if indicator_data is not None:
            indicator_path = os.path.join(OUTPUT_DIR, 'indicator_data.csv')
            indicator_data.to_csv(indicator_path)
            print(f"指标数据已保存: {indicator_path}")

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    return {
        'sentiment_index': sentiment_index,
        'indicator_stats': sentiment_builder.indicator_stats if hasattr(sentiment_builder, 'indicator_stats') else None,
        'roe_data': roe_data
    }


def load_local_data():
    """
    从本地文件加载数据

    Returns:
    --------
    tuple: (indicator_data, steel_index, roe_data)
    """
    print("\n尝试从本地文件加载数据...")

    indicator_data = LocalDataLoader.load_steel_indicators()
    steel_index = LocalDataLoader.load_steel_index()
    roe_data = LocalDataLoader.load_steel_roe()

    if indicator_data is not None:
        print(f"  - 指标数据: {indicator_data.shape}")
    if steel_index is not None:
        print(f"  - 钢铁行业指数: {steel_index.shape}")
    if roe_data is not None:
        print(f"  - ROE_TTM: {roe_data.shape}")

    return indicator_data, steel_index, roe_data


def fetch_data_from_api(data_fetcher: SteelIndustryDataFetcher) -> pd.DataFrame:
    """
    从API获取数据

    Parameters:
    -----------
    data_fetcher : SteelIndustryDataFetcher
        数据获取器

    Returns:
    --------
    pd.DataFrame
        指标数据
    """
    try:
        indicators = data_fetcher.get_steel_indicators(START_DATE, END_DATE)
        if indicators is not None and not indicators.empty:
            print(f"成功获取API数据: {indicators.shape}")
            LocalDataLoader.save_fetched_data(indicators, 'api_fetched_indicators.csv')
            return indicators
    except Exception as e:
        print(f"获取API数据失败: {e}")

    return None


def generate_simulated_data() -> pd.DataFrame:
    """
    生成模拟数据（用于演示）

    Returns:
    --------
    pd.DataFrame
        模拟指标数据
    """
    np.random.seed(42)

    dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='M')
    n = len(dates)

    data = {}

    latent_factor = np.cumsum(np.random.randn(n) * 0.5)
    latent_factor = (latent_factor - latent_factor.mean()) / latent_factor.std()

    for i in range(31):
        loading = np.random.uniform(0.3, 1.0)
        noise = np.random.randn(n) * 0.3
        indicator = loading * latent_factor + noise
        data[f'indicator_{i+1}'] = indicator

    result = pd.DataFrame(data, index=dates)

    print("使用模拟数据进行演示")
    return result


def generate_simulated_price(sentiment_index: pd.Series) -> pd.Series:
    """
    生成模拟价格序列

    Parameters:
    -----------
    sentiment_index : pd.Series
        景气度指数

    Returns:
    --------
    pd.Series
        价格序列
    """
    np.random.seed(123)

    n = len(sentiment_index)
    returns = sentiment_index.pct_change().fillna(0) * 2 + np.random.randn(n) * 0.02

    price = 100 * (1 + returns).cumprod()

    return price


if __name__ == '__main__':
    results = main()
