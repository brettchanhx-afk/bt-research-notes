"""
评价指标模块
计算ROE复现度和景气度变化方向预测准确率

参考研报: 华泰证券-中观景气度之上游资源中游材料 (2021-10-14)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats


def calculate_r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    计算R²（决定系数）

    Parameters:
    -----------
    y_true : pd.Series
        真实值
    y_pred : pd.Series
        预测值

    Returns:
    --------
    r2 : float
        R²值
    """
    common_idx = y_true.index.intersection(y_pred.index)

    if len(common_idx) < 5:
        return 0.0

    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]

    ss_res = np.sum((y_true_aligned - y_pred_aligned) ** 2)
    ss_tot = np.sum((y_true_aligned - y_true_aligned.mean()) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def calculate_correlation(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """
    计算相关系数及其p值

    Parameters:
    -----------
    y_true : pd.Series
        真实值
    y_pred : pd.Series
        预测值

    Returns:
    --------
    corr : float
        相关系数
    p_value : float
        p值
    """
    common_idx = y_true.index.intersection(y_pred.index)

    if len(common_idx) < 5:
        return 0.0, 1.0

    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]

    corr, p_value = stats.pearsonr(y_true_aligned, y_pred_aligned)

    return corr, p_value


def calculate_roe_reproduction(index: pd.Series, roe: pd.Series) -> Dict[str, float]:
    """
    计算ROE复现度

    衡量景气度指数对行业ROE_TTM的解释程度

    Parameters:
    -----------
    index : pd.Series
        景气度指数
    roe : pd.Series
        ROE_TTM

    Returns:
    --------
    Dict with 'r2', 'correlation', 'p_value'
    """
    common_idx = index.index.intersection(roe.index)

    if len(common_idx) < 5:
        return {'r2': 0.0, 'correlation': 0.0, 'p_value': 1.0}

    index_aligned = index.loc[common_idx]
    roe_aligned = roe.loc[common_idx]

    r2 = calculate_r2_score(roe_aligned, index_aligned)
    corr, p_value = calculate_correlation(index_aligned, roe_aligned)

    return {
        'r2': r2,
        'correlation': corr,
        'p_value': p_value
    }


def calculate_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    计算滚动均值

    Parameters:
    -----------
    series : pd.Series
        原始序列
    window : int
        滚动窗口大小

    Returns:
    --------
    pd.Series
    """
    return series.rolling(window=window).mean()


def calculate_direction_signal(series: pd.Series, period: int = 12) -> pd.Series:
    """
    计算方向信号

    基于同比变化判断方向（上涨为1，下跌为0）

    Parameters:
    -----------
    series : pd.Series
        时间序列
    period : int
        同比周期

    Returns:
    --------
    pd.Series
        方向信号，1表示上涨，0表示下跌
    """
    if len(series) < period + 1:
        return pd.Series(index=series.index, dtype=int)

    change = series.diff(period)

    direction = (change > 0).astype(int)

    return direction


def calculate_latest_direction_accuracy(index: pd.Series, roe: pd.Series,
                                       window: int = 3) -> float:
    """
    计算最新一期方向命中准确率

    侧重因子动量

    Parameters:
    -----------
    index : pd.Series
        景气度指数
    roe : pd.Series
        ROE_TTM
    window : int
        滚动窗口大小

    Returns:
    --------
    accuracy : float
        准确率（0-1）
    """
    index_rolled = calculate_rolling_mean(index, window)
    roe_rolled = calculate_rolling_mean(roe, window)

    index_direction = calculate_direction_signal(index_rolled, period=12)
    roe_direction = calculate_direction_signal(roe_rolled, period=4)

    common_idx = index_direction.index.intersection(roe_direction.index)

    if len(common_idx) < 5:
        return 0.0

    index_dir = index_direction.loc[common_idx].dropna()
    roe_dir = roe_direction.loc[common_idx].dropna()

    common_final = index_dir.index.intersection(roe_dir.index)

    if len(common_final) == 0:
        return 0.0

    index_dir_final = index_dir.loc[common_final]
    roe_dir_final = roe_dir.loc[common_final]

    valid_mask = ~(index_dir_final.isna() | roe_dir_final.isna())
    index_dir_final = index_dir_final[valid_mask]
    roe_dir_final = roe_dir_final[valid_mask]

    if len(index_dir_final) == 0:
        return 0.0

    correct = (index_dir_final == roe_dir_final).sum()
    total = len(index_dir_final)

    return correct / total


def calculate_prediction_direction_accuracy(index: pd.Series, roe: pd.Series,
                                            window: int = 3) -> float:
    """
    计算下期预测方向命中准确率

    侧重短期预测

    Parameters:
    -----------
    index : pd.Series
        景气度指数
    roe : pd.Series
        ROE_TTM
    window : int
        滚动窗口大小

    Returns:
    --------
    accuracy : float
        准确率（0-1）
    """
    index_rolled = calculate_rolling_mean(index, window)
    roe_rolled = calculate_rolling_mean(roe, window)

    index_pred = index.shift(-1)
    index_pred_rolled = calculate_rolling_mean(index_pred, window)

    index_pred_direction = calculate_direction_signal(index_pred_rolled, period=12)
    roe_direction = calculate_direction_signal(roe_rolled, period=4)

    common_idx = index_pred_direction.index.intersection(roe_direction.index)

    if len(common_idx) < 5:
        return 0.0

    index_pred_dir = index_pred_direction.loc[common_idx].dropna()
    roe_dir = roe_direction.loc[common_idx].dropna()

    common_final = index_pred_dir.index.intersection(roe_dir.index)

    if len(common_final) == 0:
        return 0.0

    index_pred_dir_final = index_pred_dir.loc[common_final]
    roe_dir_final = roe_dir.loc[common_final]

    valid_mask = ~(index_pred_dir_final.isna() | roe_dir_final.isna())
    index_pred_dir_final = index_pred_dir_final[valid_mask]
    roe_dir_final = roe_dir_final[valid_mask]

    if len(index_pred_dir_final) == 0:
        return 0.0

    correct = (index_pred_dir_final == roe_dir_final).sum()
    total = len(index_pred_dir_final)

    return correct / total


def evaluate_sentiment_index(index: pd.Series, roe: pd.Series,
                             window: int = 3) -> Dict[str, float]:
    """
    综合评价景气度指数

    Parameters:
    -----------
    index : pd.Series
        景气度指数
    roe : pd.Series
        ROE_TTM
    window : int
        滚动窗口大小

    Returns:
    --------
    Dict with evaluation metrics
    """
    roe_repro = calculate_roe_reproduction(index, roe)

    latest_dir_acc = calculate_latest_direction_accuracy(index, roe, window)
    pred_dir_acc = calculate_prediction_direction_accuracy(index, roe, window)

    return {
        'roe_reproduction': roe_repro['r2'],
        'correlation': roe_repro['correlation'],
        'p_value': roe_repro['p_value'],
        'latest_direction_accuracy': latest_dir_acc,
        'prediction_direction_accuracy': pred_dir_acc
    }


def compare_with_benchmark(evaluated_metrics: Dict[str, float],
                          benchmark_metrics: Dict[str, float]) -> pd.DataFrame:
    """
    将评估结果与基准对比

    Parameters:
    -----------
    evaluated_metrics : Dict
        评估得到的指标
    benchmark_metrics : Dict
        基准指标（如研报中的预期值）

    Returns:
    --------
    pd.DataFrame
    """
    metrics_list = [
        ('ROE复现度', 'roe_reproduction'),
        ('最新一期方向准确率', 'latest_direction_accuracy'),
        ('下期预测方向准确率', 'prediction_direction_accuracy')
    ]

    data = []
    for name, key in metrics_list:
        eval_val = evaluated_metrics.get(key, 0)
        bench_val = benchmark_metrics.get(key, 0)
        diff = eval_val - bench_val

        data.append({
            '指标': name,
            '评估值': eval_val,
            '基准值': bench_val,
            '差异': diff,
            '达成率': eval_val / bench_val if bench_val > 0 else 0
        })

    return pd.DataFrame(data)


class SentimentIndexEvaluator:
    """景气度指数评估器"""

    def __init__(self, industry_name: str):
        self.industry_name = industry_name
        self.evaluations = {}

    def evaluate(self, index: pd.Series, roe: pd.Series,
                index_type: str = 'global') -> Dict[str, float]:
        """
        评估景气度指数

        Parameters:
        -----------
        index : pd.Series
            景气度指数
        roe : pd.Series
            ROE_TTM
        index_type : str
            指数类型，'realtime'或'global'

        Returns:
        --------
        Dict with evaluation metrics
        """
        metrics = evaluate_sentiment_index(index, roe)

        key = f"{self.industry_name}_{index_type}"
        self.evaluations[key] = metrics

        return metrics

    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        获取评估汇总

        Returns:
        --------
        pd.DataFrame
        """
        data = []

        for key, metrics in self.evaluations.items():
            industry, index_type = key.rsplit('_', 1)
            data.append({
                '行业': industry,
                '指数类型': index_type,
                'ROE复现度': metrics['roe_reproduction'],
                '相关系数': metrics['correlation'],
                '最新一期方向准确率': metrics['latest_direction_accuracy'],
                '下期预测方向准确率': metrics['prediction_direction_accuracy']
            })

        return pd.DataFrame(data)


if __name__ == '__main__':
    print("测试评价指标模块...")

    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='M')

    true_roe = pd.Series(np.random.randn(120).cumsum() + 10, index=dates)

    sentiment_index = true_roe + np.random.randn(120) * 0.5

    print("\n1. 测试ROE复现度计算:")
    roe_repro = calculate_roe_reproduction(sentiment_index, true_roe)
    print(f"  R²: {roe_repro['r2']:.4f}")
    print(f"  相关系数: {roe_repro['correlation']:.4f}")
    print(f"  p值: {roe_repro['p_value']:.4e}")

    print("\n2. 测试方向信号计算:")
    direction = calculate_direction_signal(sentiment_index, period=12)
    print(f"  方向信号（前10个）: {direction.head(10).tolist()}")

    print("\n3. 测试综合评估:")
    metrics = evaluate_sentiment_index(sentiment_index, true_roe)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n4. 测试评估器:")
    evaluator = SentimentIndexEvaluator("测试行业")
    eval_metrics = evaluator.evaluate(sentiment_index, true_roe, 'global')
    print(f"  评估结果: {eval_metrics}")
