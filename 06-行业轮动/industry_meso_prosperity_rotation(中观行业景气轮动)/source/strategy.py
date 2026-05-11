import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FactorResult:
    industry: str
    factor_name: str
    factor_value: float
    factor_mode: str


@dataclass
class Portfolio:
    date: pd.Timestamp
    industries: List[str]
    weights: List[float]


class MomentumSignal:
    @staticmethod
    def calculate_orig(prosperity_index: pd.Series) -> float:
        if len(prosperity_index) < 1:
            return 0
        return prosperity_index.iloc[-1]

    @staticmethod
    def calculate_mom(prosperity_index: pd.Series) -> float:
        if len(prosperity_index) < 2:
            return 0
        return prosperity_index.iloc[-1] - prosperity_index.iloc[-2]

    @staticmethod
    def calculate_moma3(prosperity_index: pd.Series) -> float:
        if len(prosperity_index) < 4:
            return 0
        return prosperity_index.iloc[-1] - prosperity_index.iloc[-4]

    @staticmethod
    def calculate_qoq(prosperity_index: pd.Series) -> float:
        if len(prosperity_index) < 2:
            return 0
        return (prosperity_index.iloc[-1] - prosperity_index.iloc[-2]) / (abs(prosperity_index.iloc[-2]) + 1e-6)


class IndustryRotationStrategy:
    def __init__(
        self,
        top_n: int = 4,
        min_industries: int = 1
    ):
        self.top_n = top_n
        self.min_industries = min_industries
        self.momentum = MomentumSignal()

    def calculate_prosperity_score(
        self,
        prosperity_dict: Dict[str, pd.Series],
        factor_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        if factor_weights is None:
            factor_weights = {
                'orig': 1.0,
                'mom': 1.0,
                'moma3': 0.5,
                'qoq': 0.5
            }

        scores = []

        for industry, prosperity in prosperity_dict.items():
            if len(prosperity) < 4:
                continue

            orig = self.momentum.calculate_orig(prosperity)
            mom = self.momentum.calculate_mom(prosperity)
            moma3 = self.momentum.calculate_moma3(prosperity)
            qoq = self.momentum.calculate_qoq(prosperity)

            score = (
                factor_weights.get('orig', 1) * orig +
                factor_weights.get('mom', 1) * mom +
                factor_weights.get('moma3', 0.5) * moma3 +
                factor_weights.get('qoq', 0.5) * qoq
            )

            scores.append({
                'industry': industry,
                'orig': orig,
                'mom': mom,
                'moma3': moma3,
                'qoq': qoq,
                'total_score': score
            })

        return pd.DataFrame(scores)

    def select_industries(
        self,
        scores_df: pd.DataFrame,
        top_n: Optional[int] = None
    ) -> List[str]:
        if top_n is None:
            top_n = self.top_n

        if scores_df.empty:
            return []

        sorted_df = scores_df.sort_values('total_score', ascending=False)
        return sorted_df.head(top_n)['industry'].tolist()

    def generate_signals(
        self,
        prosperity_dict: Dict[str, pd.Series],
        factor_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        return self.calculate_prosperity_score(prosperity_dict, factor_weights)


class MultiFactorStrategy:
    def __init__(
        self,
        top_n: int = 4,
        equal_weight_first_type: bool = True,
        equal_weight_second_type: bool = True
    ):
        self.top_n = top_n
        self.equal_weight_first_type = equal_weight_first_type
        self.equal_weight_second_type = equal_weight_second_type
        self.momentum = MomentumSignal()

    def select_factors(self) -> Dict[str, List[str]]:
        first_type_factors = ['净利_orig', 'ROE_orig', '营收_orig']
        second_type_factors = ['净利_mom', '营收_qoq']
        return {
            'first_type': first_type_factors,
            'second_type': second_type_factors
        }

    def calculate_industry_score(
        self,
        industry_prosperity: Dict[str, pd.Series],
        financial_dimension: str = '净利'
    ) -> pd.DataFrame:
        scores = []

        for industry_name, prosperity_series in industry_prosperity.items():
            if len(prosperity_series) < 4:
                continue

            orig = self.momentum.calculate_orig(prosperity_series)
            mom = self.momentum.calculate_mom(prosperity_series)
            qoq = self.momentum.calculate_qoq(prosperity_series)

            scores.append({
                'industry': industry_name,
                f'{financial_dimension}_orig': orig,
                f'{financial_dimension}_mom': mom,
                f'{financial_dimension}_qoq': qoq
            })

        return pd.DataFrame(scores)

    def calculate_combined_score(
        self,
        score_df: pd.DataFrame,
        factor_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        if factor_weights is None:
            factor_weights = {
                '净利_orig': 1/3,
                'ROE_orig': 1/3,
                '营收_orig': 1/3,
                '净利_mom': 1/2,
                '营收_qoq': 1/2
            }

        result = score_df.copy()

        if 'first_type_score' not in result.columns:
            first_type_cols = [c for c in ['净利_orig', 'ROE_orig', '营收_orig'] if c in result.columns]
            if first_type_cols:
                result['first_type_score'] = result[first_type_cols].mean(axis=1)

        if 'second_type_score' not in result.columns:
            second_type_cols = [c for c in ['净利_mom', '营收_qoq'] if c in result.columns]
            if second_type_cols:
                result['second_type_score'] = result[second_type_cols].mean(axis=1)

        result['combined_score'] = (
            result.get('first_type_score', 0) +
            result.get('second_type_score', 0)
        )

        return result

    def rank_industries(
        self,
        score_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = score_df.copy()
        df['rank'] = df['combined_score'].rank(ascending=False)
        return df.sort_values('rank')


class SectorTimingStrategy:
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.momentum = MomentumSignal()

    def generate_timing_signal(
        self,
        industry_prosperity: Dict[str, pd.Series],
        cycle_industries: List[str]
    ) -> Tuple[int, List[str]]:
        bullish_count = 0
        bullish_industries = []

        for industry in cycle_industries:
            if industry not in industry_prosperity:
                continue

            prosperity = industry_prosperity[industry]
            if len(prosperity) < 2:
                continue

            mom = self.momentum.calculate_mom(prosperity)

            if mom > 0:
                bullish_count += 1
                bullish_industries.append(industry)

        return bullish_count, bullish_industries

    def get_position_size(
        self,
        bullish_count: int,
        max_position: float = 0.125
    ) -> float:
        if bullish_count >= self.threshold:
            return 1.0
        elif bullish_count > 0:
            return bullish_count * max_position
        else:
            return 0.0


def calculate_portfolio_returns(
    selected_industries: List[str],
    industry_returns: Dict[str, pd.Series],
    weights: Optional[List[float]] = None
) -> pd.Series:
    if not selected_industries:
        return pd.Series()

    if weights is None:
        weights = [1.0 / len(selected_industries)] * len(selected_industries)
    elif len(weights) != len(selected_industries):
        weights = [1.0 / len(selected_industries)] * len(selected_industries)

    returns = None
    for industry, weight in zip(selected_industries, weights):
        if industry in industry_returns:
            if returns is None:
                returns = industry_returns[industry] * weight
            else:
                returns = returns + industry_returns[industry] * weight

    return returns if returns is not None else pd.Series()


def calculate_equal_weight_returns(
    industry_returns: Dict[str, pd.Series]
) -> pd.Series:
    if not industry_returns:
        return pd.Series()

    returns_df = pd.DataFrame(industry_returns)
    return returns_df.mean(axis=1)


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def calculate_excess_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> pd.Series:
    return portfolio_returns - benchmark_returns


class BacktestPortfolio:
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.history = []

    def rebalance(
        self,
        date: pd.Timestamp,
        selected_industries: List[str],
        weights: Optional[List[float]] = None
    ):
        self.positions = {
            industry: weight for industry, weight in zip(selected_industries, weights or [1/len(selected_industries)] * len(selected_industries))
        }
        self.history.append({
            'date': date,
            'action': 'rebalance',
            'industries': selected_industries,
            'weights': weights
        })

    def update(self, date: pd.Timestamp, returns: pd.Series):
        if not self.positions:
            return

        daily_return = 0
        for industry, weight in self.positions.items():
            if industry in returns.index:
                daily_return += returns[industry] * weight

        self.current_capital *= (1 + daily_return)

        self.history.append({
            'date': date,
            'action': 'update',
            'return': daily_return,
            'capital': self.current_capital
        })

    def get_positions(self) -> Dict[str, float]:
        return self.positions.copy()

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def main():
    print("Testing Industry Rotation Strategy...")

    dates = pd.date_range('2019-01-01', '2022-06-30', freq='M')
    n = len(dates)

    industries = ['石油石化', '煤炭', '有色金属', '钢铁', '基础化工', '建材']

    prosperity_dict = {}
    for industry in industries:
        base = np.random.randn(n).cumsum()[-1] + 10
        prosperity_dict[industry] = pd.Series(
            np.random.randn(n) * 2 + base,
            index=dates,
            name=industry
        )

    strategy = IndustryRotationStrategy(top_n=3)
    scores_df = strategy.calculate_prosperity_score(prosperity_dict)
    print(f"\nProsperity Scores:\n{scores_df}")

    selected = strategy.select_industries(scores_df)
    print(f"\nSelected Industries: {selected}")

    multi_strategy = MultiFactorStrategy(top_n=3)
    combined_scores = multi_strategy.calculate_combined_score(scores_df)
    ranked = multi_strategy.rank_industries(combined_scores)
    print(f"\nRanked Industries:\n{ranked}")


if __name__ == "__main__":
    main()
