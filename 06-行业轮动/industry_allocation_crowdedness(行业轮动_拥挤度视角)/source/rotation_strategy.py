import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class IndustryRotationStrategy:
    def __init__(self, industry_data: Dict[str, pd.DataFrame], crowdedness_signals: pd.DataFrame):
        self.industry_data = industry_data
        self.crowdedness_signals = crowdedness_signals
        self.benchmark_returns = None

    def calculate_benchmark_returns(self) -> pd.Series:
        all_returns = []
        for code, data in self.industry_data.items():
            ret = data['close'].pct_change()
            all_returns.append(ret)
        if all_returns:
            benchmark = pd.concat(all_returns, axis=1).mean(axis=1)
            self.benchmark_returns = benchmark.fillna(0)
        return self.benchmark_returns

    def strategy_one_monthly_short(self, rebalance_freq='M') -> Tuple[pd.Series, pd.DataFrame]:
        if self.benchmark_returns is None:
            self.calculate_benchmark_returns()
        monthly_signals = self.crowdedness_signals.resample(rebalance_freq).last()
        strategy_returns = pd.Series(index=monthly_signals.index, dtype=float)
        positions = pd.DataFrame(index=monthly_signals.index, columns=self.crowdedness_signals.columns)
        for date in monthly_signals.index:
            if date not in monthly_signals.index:
                continue
            crowded_industries = monthly_signals.loc[date][monthly_signals.loc[date] == True].index.tolist()
            if not crowded_industries:
                positions.loc[date] = 0
                continue
            next_month_idx = monthly_signals.index.get_loc(date) + 1
            if next_month_idx < len(monthly_signals.index):
                next_date = monthly_signals.index[next_month_idx]
                month_returns = {}
                for ind in crowded_industries:
                    if ind in self.industry_data:
                        ind_data = self.industry_data[ind]
                        try:
                            start_idx = ind_data.index.get_loc(date) if date in ind_data.index else None
                            end_idx = ind_data.index.get_loc(next_date) if next_date in ind_data.index else None
                            if start_idx is not None and end_idx is not None and end_idx > start_idx:
                                month_ret = ind_data.iloc[start_idx:end_idx]['close'].pct_change().sum()
                                month_returns[ind] = month_ret
                        except:
                            continue
                if month_returns:
                    avg_ret = np.mean(list(month_returns.values()))
                    strategy_returns.loc[date] = avg_ret
                    for ind in crowded_industries:
                        positions.loc[date, ind] = -1 / len(crowded_industries) if ind in month_returns else 0
                else:
                    positions.loc[date] = 0
            else:
                positions.loc[date] = 0
        return strategy_returns.fillna(0), positions

    def strategy_two_daily_risk_monitor(self) -> Tuple[pd.Series, pd.Series]:
        if self.benchmark_returns is None:
            self.calculate_benchmark_returns()
        daily_signals = self.crowdedness_signals.copy()
        all_dates = sorted(daily_signals.index)
        portfolio_value = [1.0]
        current_positions = {}
        last_rebalance_date = None
        for i, date in enumerate(all_dates[:-1]):
            if last_rebalance_date is None or (date.month != last_rebalance_date.month and date.day <= 5):
                non_crowded = daily_signals.loc[date][daily_signals.loc[date] == False].index.tolist()
                non_crowded = [ind for ind in non_crowded if ind in self.industry_data]
                if non_crowded:
                    current_positions = {ind: 1.0 / len(non_crowded) for ind in non_crowded}
                    last_rebalance_date = date
            crowded_today = daily_signals.loc[date][daily_signals.loc[date] == True].index.tolist()
            for ind in crowded_today:
                if ind in current_positions:
                    del current_positions[ind]
            next_date = all_dates[i + 1]
            day_return = 0
            if current_positions:
                total_weight = sum(current_positions.values())
                for ind, weight in current_positions.items():
                    if ind in self.industry_data:
                        try:
                            if date in self.industry_data[ind].index and next_date in self.industry_data[ind].index:
                                start_idx = self.industry_data[ind].index.get_loc(date)
                                end_idx = self.industry_data[ind].index.get_loc(next_date)
                                if end_idx > start_idx:
                                    ind_ret = self.industry_data[ind].iloc[start_idx:end_idx]['close'].pct_change().sum()
                                    day_return += (weight / total_weight) * ind_ret
                        except:
                            continue
            portfolio_value.append(portfolio_value[-1] * (1 + day_return))
        portfolio_series = pd.Series(portfolio_value[1:], index=all_dates[1:])
        strategy_returns = portfolio_series.pct_change().fillna(0)
        return strategy_returns, portfolio_series

    def strategy_three_market_timing(self, crowded_threshold=10) -> Tuple[pd.Series, pd.Series]:
        if self.benchmark_returns is None:
            self.calculate_benchmark_returns()
        daily_signals = self.crowdedness_signals.copy()
        crowded_count = daily_signals.sum(axis=1)
        all_dates = sorted(daily_signals.index)
        portfolio_value = [1.0]
        in_market = True
        exit_date = None
        for i, date in enumerate(all_dates[:-1]):
            if exit_date is not None and date < exit_date:
                portfolio_value.append(portfolio_value[-1])
                continue
            current_crowded = crowded_count.loc[date] if date in crowded_count.index else 0
            next_date = all_dates[i + 1]
            if current_crowded > crowded_threshold and in_market:
                in_market = False
                exit_date = all_dates[min(i + 20, len(all_dates) - 1)]
                portfolio_value.append(portfolio_value[-1])
            elif in_market:
                benchmark_ret = 0
                if self.benchmark_returns is not None and date in self.benchmark_returns.index:
                    benchmark_ret = self.benchmark_returns.loc[date]
                portfolio_value.append(portfolio_value[-1] * (1 + benchmark_ret))
            else:
                portfolio_value.append(portfolio_value[-1])
        portfolio_series = pd.Series(portfolio_value[1:], index=all_dates[1:])
        strategy_returns = portfolio_series.pct_change().fillna(0)
        return strategy_returns, portfolio_series

class ProsperityCrowdednessStrategy:
    def __init__(self, industry_data: Dict[str, pd.DataFrame], crowdedness_signals: pd.DataFrame):
        self.industry_data = industry_data
        self.crowdedness_signals = crowdedness_signals
        self.prosperity_indicator = None

    def calculate_simple_prosperity(self) -> pd.DataFrame:
        prosperity = {}
        for code, data in self.industry_data.items():
            close = data['close']
            ret_20d = close.pct_change(20)
            ret_60d = close.pct_change(60)
            volume = data['volume']
            vol_change = volume.pct_change(20)
            prosperity[code] = pd.DataFrame({
                'ret_20d': ret_20d,
                'ret_60d': ret_60d,
                'vol_change': vol_change,
                'prosperity_score': ret_20d * 0.4 + ret_60d * 0.4 + vol_change * 0.2
            }, index=data.index)
        return pd.DataFrame({k: v['prosperity_score'] for k, v in prosperity.items()})

    def strategy_monthly_prosperity_crowdedness(self, top_n=5, rebalance_freq='M') -> Tuple[pd.Series, pd.Series, Dict]:
        if self.prosperity_indicator is None:
            self.prosperity_indicator = self.calculate_simple_prosperity()
        monthly_prosperity = self.prosperity_indicator.resample(rebalance_freq).last()
        monthly_crowded = self.crowdedness_signals.resample(rebalance_freq).last()
        all_dates = sorted(monthly_prosperity.dropna().index)
        portfolio_value = [1.0]
        signals_history = {}
        for i, date in enumerate(all_dates[:-1]):
            pros_score = monthly_prosperity.loc[date].dropna()
            crowded_status = monthly_crowded.loc[date] if date in monthly_crowded.index else pd.Series(0, index=pros_score.index)
            high_prosperity = pros_score.nlargest(20).index
            high_pros_low_crowded = [ind for ind in high_prosperity if crowded_status.get(ind, False) == False][:top_n]
            signals_history[date] = {
                'selected': high_pros_low_crowded,
                'prosperity': pros_score[high_pros_low_crowded].to_dict() if high_pros_low_crowded else {}
            }
            next_date = all_dates[i + 1]
            month_return = 0
            if high_pros_low_crowded:
                for ind in high_pros_low_crowded:
                    if ind in self.industry_data:
                        try:
                            start_idx = self.industry_data[ind].index.get_loc(date) if date in self.industry_data[ind].index else None
                            end_idx = self.industry_data[ind].index.get_loc(next_date) if next_date in self.industry_data[ind].index else None
                            if start_idx is not None and end_idx is not None and end_idx > start_idx:
                                ind_ret = self.industry_data[ind].iloc[start_idx:end_idx]['close'].pct_change().sum()
                                month_return += ind_ret / len(high_pros_low_crowded)
                        except:
                            continue
            portfolio_value.append(portfolio_value[-1] * (1 + month_return))
        portfolio_series = pd.Series(portfolio_value[1:], index=all_dates[1:])
        strategy_returns = portfolio_series.pct_change().fillna(0)
        benchmark = self.calculate_benchmark_returns()
        benchmark_monthly = benchmark.resample(rebalance_freq).last().reindex(portfolio_series.index).fillna(0)
        excess_returns = strategy_returns - benchmark_monthly
        return strategy_returns, excess_returns, signals_history

    def calculate_benchmark_returns(self) -> pd.Series:
        all_returns = []
        for code, data in self.industry_data.items():
            ret = data['close'].pct_change()
            all_returns.append(ret)
        if all_returns:
            return pd.concat(all_returns, axis=1).mean(axis=1).fillna(0)
        return pd.Series()

def run_all_strategies(industry_data: Dict[str, pd.DataFrame], crowdedness_signals: pd.DataFrame) -> Dict:
    results = {}
    rotation = IndustryRotationStrategy(industry_data, crowdedness_signals)
    rotation.calculate_benchmark_returns()
    print("运行策略一：月度空头行业轮动...")
    try:
        strat1_returns, strat1_pos = rotation.strategy_one_monthly_short()
        results['strategy_1_monthly_short'] = {'returns': strat1_returns, 'positions': strat1_pos}
    except Exception as e:
        print(f"策略一运行失败: {e}")
    print("运行策略二：日度行业风险监控...")
    try:
        strat2_returns, strat2_portfolio = rotation.strategy_two_daily_risk_monitor()
        results['strategy_2_daily_monitor'] = {'returns': strat2_returns, 'portfolio': strat2_portfolio}
    except Exception as e:
        print(f"策略二运行失败: {e}")
    print("运行策略三：大盘择时...")
    try:
        strat3_returns, strat3_portfolio = rotation.strategy_three_market_timing()
        results['strategy_3_market_timing'] = {'returns': strat3_returns, 'portfolio': strat3_portfolio}
    except Exception as e:
        print(f"策略三运行失败: {e}")
    print("运行景气度+拥挤度复合策略...")
    try:
        composite_strategy = ProsperityCrowdednessStrategy(industry_data, crowdedness_signals)
        comp_returns, comp_excess, comp_signals = composite_strategy.strategy_monthly_prosperity_crowdedness()
        results['prosperity_crowdedness_composite'] = {'returns': comp_returns, 'excess': comp_excess, 'signals': comp_signals}
    except Exception as e:
        print(f"复合策略运行失败: {e}")
    return results

if __name__ == "__main__":
    print("行业轮动策略模块测试...")