
import pandas as pd
import numpy as np
from source.data_loader import DataLoader
from source.strategy_builder import StrategyBuilder
from source.risk_calculator import RiskCalculator


class BacktestEngine:
    """
    回测引擎，用于回测各种策略
    """
    
    def __init__(self, return_data, rebalance_freq='M', fee_rate=0.0005):
        """
        参数:
            return_data: 收益率数据
            rebalance_freq: 调仓频率 ('D'日, 'W'周, 'M'月)
            fee_rate: 单边交易成本
        """
        self.return_data = return_data
        self.rebalance_freq = rebalance_freq
        self.fee_rate = fee_rate
        self.strategy_builder = StrategyBuilder(DataLoader())
        
    def get_rebalance_dates(self):
        """
        获取调仓日期
        """
        dates = self.return_data.index
        rebalance_dates = dates.to_series().resample(self.rebalance_freq).last().dropna()
        return rebalance_dates.index
    
    def backtest_strategy(self, strategy_type='allweather', initial_capital=1.0,
                          lookback_window=252, use_semicovariance=True,
                          momentum_lookback=60):
        """
        回测策略
        
        参数:
            strategy_type: 策略类型 ('allweather', 'enhanced', 'asset_rp')
            initial_capital: 初始资金
        """
        rebalance_dates = self.get_rebalance_dates()
        
        portfolio_value = pd.Series(initial_capital, index=[self.return_data.index[0]])
        weights_record = pd.DataFrame(columns=self.return_data.columns)
        weights_record.loc[self.return_data.index[0]] = 0
        
        current_weights = pd.Series(0.0, index=self.return_data.columns)
        
        for i, rebalance_date in enumerate(rebalance_dates):
            if i == 0:
                idx = self.return_data.index.get_loc(rebalance_date)
                if idx < lookback_window:
                    continue
                history_data = self.return_data.iloc[:idx+1]
            else:
                prev_date = rebalance_dates[i-1]
                idx_prev = self.return_data.index.get_loc(prev_date)
                idx_curr = self.return_data.index.get_loc(rebalance_date)
                history_data = self.return_data.iloc[:idx_curr+1]
                
                period_returns = self.return_data.iloc[idx_prev+1:idx_curr+1]
                daily_value = portfolio_value.iloc[-1] * (1 + (period_returns * current_weights).sum(axis=1)).cumprod()
                portfolio_value = pd.concat([portfolio_value, daily_value.iloc[1:]])
            
            if strategy_type == 'allweather':
                strategy_result = self.strategy_builder.build_allweather_strategy(
                    history_data, rebalance_date, lookback_window, use_semicovariance
                )
                new_weights = strategy_result['asset_weights']
            elif strategy_type == 'enhanced':
                strategy_result = self.strategy_builder.build_enhanced_allweather_strategy(
                    history_data, rebalance_date, lookback_window, use_semicovariance,
                    momentum_lookback
                )
                new_weights = strategy_result['asset_weights']
            elif strategy_type == 'asset_rp':
                strategy_result = self.strategy_builder.build_asset_risk_parity_strategy(
                    history_data, lookback_window, use_semicovariance
                )
                new_weights = strategy_result['asset_weights']
            else:
                raise ValueError(f"未知策略类型: {strategy_type}")
            
            weight_change = abs(new_weights - current_weights).sum()
            fee = weight_change * self.fee_rate
            
            if len(portfolio_value) > 0:
                portfolio_value.iloc[-1] *= (1 - fee)
            
            current_weights = new_weights.copy()
            weights_record.loc[rebalance_date] = new_weights
        
        if len(rebalance_dates) > 0:
            last_date = rebalance_dates[-1]
            idx_last = self.return_data.index.get_loc(last_date)
            if idx_last < len(self.return_data) - 1:
                remaining_returns = self.return_data.iloc[idx_last+1:]
                remaining_value = portfolio_value.iloc[-1] * (1 + (remaining_returns * current_weights).sum(axis=1)).cumprod()
                portfolio_value = pd.concat([portfolio_value, remaining_value.iloc[1:]])
        
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns,
            'weights_record': weights_record,
            'metrics': RiskCalculator.calculate_performance_metrics(portfolio_returns, periods=252)
        }
    
    def backtest_all_strategies(self, initial_capital=1.0, lookback_window=252,
                               use_semicovariance=True, momentum_lookback=60):
        """
        回测所有策略进行对比
        """
        results = {}
        
        print("正在回测传统资产风险平价策略...")
        results['asset_rp'] = self.backtest_strategy(
            'asset_rp', initial_capital, lookback_window, use_semicovariance
        )
        
        print("正在回测全天候基准策略...")
        results['allweather'] = self.backtest_strategy(
            'allweather', initial_capital, lookback_window, use_semicovariance
        )
        
        print("正在回测全天候增强策略...")
        results['enhanced'] = self.backtest_strategy(
            'enhanced', initial_capital, lookback_window, use_semicovariance, momentum_lookback
        )
        
        return results
    
    @staticmethod
    def compare_strategies(results):
        """
        对比各策略表现
        """
        comparison = {}
        for name, result in results.items():
            comparison[name] = result['metrics']
        
        return pd.DataFrame(comparison).T
    
    @staticmethod
    def calculate_yearly_performance(portfolio_returns):
        """
        计算年度表现
        """
        yearly = portfolio_returns.resample('Y').apply(
            lambda x: (1 + x).prod() - 1
        )
        return yearly
