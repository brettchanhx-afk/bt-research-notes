import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CompositeIndicatorBuilder:
    def __init__(self):
        self.selected_indicators = []
        self.correlation_matrix = None
        self.performance_results = {}

    def calculate_correlation(self, indicator_data, indicator_list=None):
        if indicator_data is None or len(indicator_data) == 0:
            return pd.DataFrame()

        if indicator_list is None:
            indicator_list = [col for col in indicator_data.columns
                           if col not in ['industry_code', 'trade_date']]

        signal_cols = [col for col in indicator_data.columns if col in indicator_list]

        if len(signal_cols) < 2:
            return pd.DataFrame()

        corr_matrix = indicator_data[signal_cols].corr()
        self.correlation_matrix = corr_matrix
        return corr_matrix

    def select_indicators_by_correlation(self, indicator_data, indicator_list,
                                        performance_dict=None, correlation_threshold=0.5):
        if performance_dict is None:
            performance_dict = self.performance_results

        if indicator_list is None:
            indicator_list = [col for col in indicator_data.columns
                           if col not in ['industry_code', 'trade_date']]

        selected = []
        remaining = indicator_list.copy()

        remaining_sorted = sorted(
            remaining,
            key=lambda x: performance_dict.get(x, {}).get('annual_excess_return', 0),
            reverse=True
        )

        for ind_col in remaining_sorted:
            is_low_corr = True

            for sel_col in selected:
                if self.correlation_matrix is not None and ind_col in self.correlation_matrix.columns and sel_col in self.correlation_matrix.columns:
                    corr_val = self.correlation_matrix.loc[ind_col, sel_col]
                    if abs(corr_val) > correlation_threshold:
                        is_low_corr = False
                        break

            if is_low_corr:
                selected.append(ind_col)
                if ind_col not in self.selected_indicators:
                    self.selected_indicators.append(ind_col)

        return selected

    def build_composite_indicator(self, indicator_data, selected_indicators=None, method='equal_weight'):
        if indicator_data is None or len(indicator_data) == 0:
            return pd.DataFrame()

        if selected_indicators is None:
            selected_indicators = [col for col in indicator_data.columns
                                 if col not in ['industry_code', 'trade_date']]

        available_cols = [col for col in selected_indicators if col in indicator_data.columns]

        if len(available_cols) == 0:
            return pd.DataFrame()

        result = indicator_data[['industry_code', 'trade_date']].copy()

        if method == 'equal_weight':
            result['composite_score'] = indicator_data[available_cols].mean(axis=1)
        elif method == 'weighted':
            weights = self._calculate_weights(selected_indicators)
            weight_sum = sum(weights.values())
            result['composite_score'] = sum(
                indicator_data[col] * (weights.get(ind, 1.0) / weight_sum)
                for col, ind in zip(available_cols, selected_indicators)
                if ind in weights and col in indicator_data.columns
            )
        elif method == 'rank_average':
            rank_df = indicator_data[available_cols].rank(axis=1, pct=True)
            result['composite_score'] = rank_df.mean(axis=1)
        elif method == 'signal_count':
            result['composite_score'] = indicator_data[available_cols].sum(axis=1)

        return result

    def _calculate_weights(self, indicator_list):
        weights = {}

        if not self.performance_results:
            return {ind: 1.0 for ind in indicator_list}

        total_excess = sum(
            max(self.performance_results.get(ind, {}).get('annual_excess_return', 1), 0.1)
            for ind in indicator_list
        )

        for ind in indicator_list:
            excess = max(self.performance_results.get(ind, {}).get('annual_excess_return', 1), 0.1)
            weights[ind] = excess

        return weights

    def incremental_selection(self, indicator_data, all_indicators, performance_dict=None, n_max=20):
        if performance_dict is None:
            performance_dict = self.performance_results

        sorted_indicators = sorted(
            all_indicators,
            key=lambda x: performance_dict.get(x, {}).get('annual_excess_return', 0),
            reverse=True
        )

        if len(sorted_indicators) == 0:
            return []

        results = []
        current_selected = []

        for ind in sorted_indicators[:min(n_max, len(sorted_indicators))]:
            test_selected = current_selected + [ind]

            test_composite = self.build_composite_indicator(
                indicator_data,
                test_selected,
                method='signal_count'
            )

            if len(test_composite) > 0:
                performance = {'n_indicators': len(test_selected), 'indicators': test_selected.copy()}
                results.append(performance)

            current_selected = test_selected

        return results

    def get_optimal_n_indicators(self, incremental_results):
        if len(incremental_results) == 0:
            return []

        best_result = max(incremental_results, key=lambda x: len(x.get('indicators', [])))
        return best_result.get('indicators', [])


class ProsperitySignal:
    def __init__(self):
        self.composite_builder = CompositeIndicatorBuilder()

    def generate_signals(self, composite_data, date, top_n=5, bottom_n=5):
        if composite_data is None or len(composite_data) == 0:
            return pd.DataFrame()

        date_data = composite_data[composite_data['trade_date'] == date].copy()

        if len(date_data) == 0:
            return pd.DataFrame()

        date_data = date_data.sort_values('composite_score', ascending=False)

        date_data['signal'] = 0

        if 'industry_code' in date_data.columns:
            date_data = date_data.reset_index(drop=True)

            if top_n > 0 and top_n < len(date_data):
                date_data.loc[:top_n-1, 'signal'] = 1

            if bottom_n > 0 and bottom_n < len(date_data):
                date_data.loc[-(bottom_n):, 'signal'] = -1

        return date_data

    def generate_all_signals(self, composite_data, dates, top_n=5, bottom_n=5):
        all_signals = []
        for date in dates:
            signals = self.generate_signals(composite_data, date, top_n, bottom_n)
            if len(signals) > 0:
                signals['rebalance_date'] = date
                all_signals.append(signals)

        if not all_signals:
            return pd.DataFrame()

        return pd.concat(all_signals, ignore_index=True)


class IndustryProsperityCalculator:
    def __init__(self):
        self.prosperity_data = None

    def calculate_prosperity_index(self, financial_data, consensus_data=None, industry_returns=None):
        if financial_data is None or len(financial_data) == 0:
            return pd.DataFrame()

        from indicators import ProsperityIndicator, ConsensusIndicator

        prosperity_ind = ProsperityIndicator()
        financial_indicators = prosperity_ind.build_industry_indicators(financial_data)

        if consensus_data is not None and len(consensus_data) > 0:
            consensus_ind = ConsensusIndicator()
            consensus_indicators = consensus_ind.build_consensus_indicators(consensus_data)

            if len(financial_indicators) > 0 and len(consensus_indicators) > 0:
                merged = financial_indicators.merge(
                    consensus_indicators[['industry_code', 'trade_date', 'consensus_score']],
                    on=['industry_code', 'trade_date'],
                    how='outer'
                )
            else:
                merged = financial_indicators
        else:
            merged = financial_indicators

        if len(merged) > 0:
            indicator_cols = [col for col in merged.columns
                            if col not in ['industry_code', 'trade_date', 'consensus_score']]

            merged['prosperity_score'] = merged[indicator_cols].sum(axis=1) if indicator_cols else 0

            if 'consensus_score' in merged.columns:
                merged['prosperity_score'] = merged['prosperity_score'] + merged['consensus_score'].fillna(0)

        self.prosperity_data = merged
        return merged

    def get_prosperity_ranking(self, date):
        if self.prosperity_data is None or len(self.prosperity_data) == 0:
            return pd.DataFrame()

        date_data = self.prosperity_data[self.prosperity_data['trade_date'] == date].copy()

        if len(date_data) == 0:
            return pd.DataFrame()

        date_data = date_data.sort_values('prosperity_score', ascending=False)
        date_data['prosperity_rank'] = range(1, len(date_data) + 1)

        return date_data

    def calculate_prosperity_cycle(self):
        if self.prosperity_data is None or len(self.prosperity_data) == 0:
            return pd.DataFrame()

        self.prosperity_data['n_prosperity_industries'] = (
            self.prosperity_data[self.prosperity_data.columns[2:]] > 0
        ).sum(axis=1)

        return self.prosperity_data[['trade_date', 'n_prosperity_industries']]


if __name__ == "__main__":
    print("Testing CompositeIndicatorBuilder...")

    builder = CompositeIndicatorBuilder()
    print("CompositeIndicatorBuilder initialized")

    signal_generator = ProsperitySignal()
    print("ProsperitySignal generator initialized")

    calculator = IndustryProsperityCalculator()
    print("IndustryProsperityCalculator initialized")

    print("Composite indicator module test completed!")
