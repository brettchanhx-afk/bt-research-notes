import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

from .config import MACRO_SCORING_RULES, FACTOR_CONFIG

warnings.filterwarnings("ignore")


class MacroScoring:
    def __init__(
        self,
        score_levels: Optional[List[int]] = None,
        coefficient_mapping: Optional[Dict[int, float]] = None,
    ):
        if score_levels is None:
            score_levels = MACRO_SCORING_RULES["score_levels"]
        if coefficient_mapping is None:
            coefficient_mapping = MACRO_SCORING_RULES["coefficient_mapping"]

        self.score_levels = score_levels
        self.coefficient_mapping = coefficient_mapping

    def generate_factor_deviation_from_scores(
        self,
        baseline_exposure: pd.Series,
        factor_volatility: pd.Series,
        factor_scores: Dict[str, int],
    ) -> pd.Series:
        factor_deviation = pd.Series(dtype=float)

        for factor_name, score in factor_scores.items():
            if factor_name not in baseline_exposure.index:
                continue

            if factor_name not in factor_volatility.index:
                continue

            baseline = baseline_exposure[factor_name]
            volatility = factor_volatility[factor_name]

            coefficient = self.coefficient_mapping.get(score, 0.0)

            if abs(score) == 2:
                multiplier = 1.0
            elif abs(score) == 1:
                multiplier = 0.5
            else:
                multiplier = 0.0

            deviation = coefficient * multiplier * volatility

            factor_deviation[factor_name] = deviation

        return factor_deviation

    def calculate_target_factor_exposure(
        self,
        baseline_exposure: pd.Series,
        factor_deviation: pd.Series,
    ) -> pd.Series:
        target_exposure = baseline_exposure + factor_deviation
        return target_exposure

    def calculate_factor_volatility(
        self,
        factor_returns: pd.DataFrame,
        window: int = 60,
    ) -> pd.Series:
        factor_volatility = factor_returns.rolling(window).std().iloc[-1]
        return factor_volatility

    def generate_macro_view_scores(
        self,
        growth_view: str = "neutral",
        inflation_view: str = "neutral",
        interest_rate_view: str = "neutral",
        credit_view: str = "neutral",
        exchange_rate_view: str = "neutral",
        liquidity_view: str = "neutral",
    ) -> Dict[str, int]:
        view_mapping = {
            "strong_bullish": 2,
            "bullish": 1,
            "neutral": 0,
            "bearish": -1,
            "strong_bearish": -2,
        }

        factor_scores = {
            "Growth": view_mapping.get(growth_view, 0),
            "Inflation": view_mapping.get(inflation_view, 0),
            "IntRate": view_mapping.get(interest_rate_view, 0),
            "Credit": view_mapping.get(credit_view, 0),
            "ExchRate": view_mapping.get(exchange_rate_view, 0),
            "Liquidity": view_mapping.get(liquidity_view, 0),
        }

        return factor_scores

    def convert_scores_to_deviation_table(
        self,
        dates: pd.DatetimeIndex,
        factor_volatility: pd.Series,
        score_history: Optional[List[Dict[str, int]]] = None,
    ) -> pd.DataFrame:
        if score_history is None:
            score_history = [
                self.generate_macro_view_scores() for _ in range(len(dates))
            ]

        deviation_dict = {}

        for i, date in enumerate(dates):
            if i >= len(score_history):
                break

            scores = score_history[i]

            deviation = pd.Series(dtype=float)
            for factor_name, score in scores.items():
                if factor_name in factor_volatility.index:
                    volatility = factor_volatility[factor_name]
                    coefficient = self.coefficient_mapping.get(score, 0.0)

                    if abs(score) == 2:
                        multiplier = 1.0
                    elif abs(score) == 1:
                        multiplier = 0.5
                    else:
                        multiplier = 0.0

                    deviation[factor_name] = coefficient * multiplier * volatility
                else:
                    deviation[factor_name] = 0.0

            deviation_dict[date] = deviation

        if deviation_dict:
            deviation_df = pd.DataFrame(deviation_dict).T
            return deviation_df
        else:
            return pd.DataFrame()

    def apply_macro_views_to_portfolio(
        self,
        baseline_weights: pd.Series,
        factor_exposure: pd.DataFrame,
        factor_scores: Dict[str, int],
        factor_volatility: pd.Series,
    ) -> pd.Series:
        baseline_exposure = self.calculate_factor_exposure_of_portfolio(
            baseline_weights, factor_exposure
        )

        factor_deviation = self.generate_factor_deviation_from_scores(
            baseline_exposure, factor_volatility, factor_scores
        )

        target_exposure = self.calculate_target_factor_exposure(
            baseline_exposure, factor_deviation
        )

        adjustment_weights = self._calculate_weight_adjustment(
            baseline_weights, factor_exposure, target_exposure
        )

        return adjustment_weights

    def calculate_factor_exposure_of_portfolio(
        self,
        weights: pd.Series,
        factor_exposure: pd.DataFrame,
    ) -> pd.Series:
        aligned_weights = weights.reindex(factor_exposure.index).fillna(0)
        portfolio_exposure = factor_exposure.mul(aligned_weights, axis=0).sum(axis=0)
        return portfolio_exposure

    def _calculate_weight_adjustment(
        self,
        baseline_weights: pd.Series,
        factor_exposure: pd.DataFrame,
        target_exposure: pd.Series,
    ) -> pd.Series:
        current_exposure = self.calculate_factor_exposure_of_portfolio(
            baseline_weights, factor_exposure
        )

        exposure_diff = target_exposure - current_exposure

        factor_coupling = factor_exposure.T @ factor_exposure
        factor_coupling_inv = np.linalg.pinv(factor_coupling.values)

        weight_adjustment = factor_coupling_inv @ exposure_diff.values

        adjusted_weights = baseline_weights + weight_adjustment * 0.1

        adjusted_weights = adjusted_weights.clip(lower=0.0)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        return adjusted_weights

    def get_default_factor_deviation(
        self,
        factor_volatility: pd.Series,
        deviation_scale: float = 0.01,
    ) -> pd.Series:
        default_deviation = pd.Series(
            {
                "Growth": deviation_scale,
                "Inflation": deviation_scale,
                "IntRate": deviation_scale,
                "Credit": deviation_scale,
                "ExchRate": deviation_scale,
                "Liquidity": deviation_scale * 0.5,
            }
        )

        aligned_volatility = factor_volatility.reindex(default_deviation.index).fillna(1.0)
        scaled_deviation = default_deviation * aligned_volatility

        return scaled_deviation
