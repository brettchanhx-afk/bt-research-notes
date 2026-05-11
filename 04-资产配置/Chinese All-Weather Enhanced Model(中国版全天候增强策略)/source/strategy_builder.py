
import pandas as pd
import numpy as np
from source.data_loader import DataLoader
from source.risk_calculator import RiskCalculator


class StrategyBuilder:
    """
    策略构建模块，包含全天候基准策略和增强策略
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.quadrant_assets = DataLoader.QUADRANT_ASSETS
    
    def build_quadrant_portfolios(self, return_data):
        """
        构建四象限等权组合
        """
        quadrant_returns = pd.DataFrame(index=return_data.index)
        
        for quadrant, assets in self.quadrant_assets.items():
            valid_assets = [a for a in assets if a in return_data.columns]
            if len(valid_assets) > 0:
                quadrant_returns[quadrant] = return_data[valid_assets].mean(axis=1)
        
        return quadrant_returns
    
    def calculate_quadrant_weights(self, quadrant_returns, lookback_window=252, 
                                   use_semicovariance=True, lambda_param=0.94):
        """
        计算四象限风险平价权重
        """
        if len(quadrant_returns) < lookback_window:
            lookback_window = len(quadrant_returns)
        
        recent_returns = quadrant_returns.tail(lookback_window)
        cov_matrix = RiskCalculator.calculate_ewma_covariance(
            recent_returns, lambda_param, use_semicovariance
        )
        weights = RiskCalculator.calculate_risk_parity_weights(cov_matrix)
        
        return weights
    
    def calculate_asset_weights_from_quadrant(self, quadrant_weights, return_data):
        """
        根据四象限权重计算最终资产权重（象限内等权）
        """
        asset_weights = pd.Series(0.0, index=return_data.columns)
        
        for quadrant, q_weight in quadrant_weights.items():
            assets = self.quadrant_assets[quadrant]
            valid_assets = [a for a in assets if a in return_data.columns]
            if len(valid_assets) > 0:
                asset_weight = q_weight / len(valid_assets)
                for asset in valid_assets:
                    asset_weights[asset] += asset_weight
        
        return asset_weights
    
    def calculate_expected_resonance_momentum(self, quadrant_returns, lookback=60):
        """
        计算预期共振动量（简化版：使用价格动量）
        """
        momentum = {}
        
        for quadrant in quadrant_returns.columns:
            mom = RiskCalculator.calculate_path_ratio_momentum(
                quadrant_returns[quadrant], lookback
            )
            momentum[quadrant] = mom
        
        return momentum
    
    def select_quadrants_for_enhanced(self, momentum):
        """
        根据预期共振动量选择增强策略的象限
        """
        growth_quadrants = ['growth_above', 'growth_below']
        inflation_quadrants = ['inflation_above', 'inflation_below']
        
        selected_quadrants = []
        
        growth_mom = {q: momentum.get(q, 0) for q in growth_quadrants}
        selected_growth = max(growth_mom.items(), key=lambda x: x[1])[0]
        selected_quadrants.append(selected_growth)
        
        inflation_mom = {q: momentum.get(q, 0) for q in inflation_quadrants}
        selected_inflation = max(inflation_mom.items(), key=lambda x: x[1])[0]
        selected_quadrants.append(selected_inflation)
        
        return selected_quadrants
    
    def build_allweather_strategy(self, return_data, rebalance_date=None, 
                                 lookback_window=252, use_semicovariance=True):
        """
        构建全天候基准策略
        """
        quadrant_returns = self.build_quadrant_portfolios(return_data)
        quadrant_weights = self.calculate_quadrant_weights(
            quadrant_returns, lookback_window, use_semicovariance
        )
        asset_weights = self.calculate_asset_weights_from_quadrant(
            quadrant_weights, return_data
        )
        
        return {
            'quadrant_weights': quadrant_weights,
            'asset_weights': asset_weights,
            'quadrant_returns': quadrant_returns
        }
    
    def build_enhanced_allweather_strategy(self, return_data, rebalance_date=None,
                                          lookback_window=252, use_semicovariance=True,
                                          momentum_lookback=60):
        """
        构建全天候增强策略
        """
        quadrant_returns = self.build_quadrant_portfolios(return_data)
        momentum = self.calculate_expected_resonance_momentum(quadrant_returns, momentum_lookback)
        selected_quadrants = self.select_quadrants_for_enhanced(momentum)
        
        selected_quadrant_returns = quadrant_returns[selected_quadrants]
        cov_matrix = RiskCalculator.calculate_ewma_covariance(
            selected_quadrant_returns.tail(lookback_window), 
            lambda_param=0.94, 
            use_semicovariance=use_semicovariance
        )
        quadrant_weights = RiskCalculator.calculate_risk_parity_weights(cov_matrix)
        
        asset_weights = pd.Series(0.0, index=return_data.columns)
        for quadrant, q_weight in quadrant_weights.items():
            assets = self.quadrant_assets[quadrant]
            valid_assets = [a for a in assets if a in return_data.columns]
            if len(valid_assets) > 0:
                asset_weight = q_weight / len(valid_assets)
                for asset in valid_assets:
                    asset_weights[asset] += asset_weight
        
        return {
            'selected_quadrants': selected_quadrants,
            'quadrant_weights': quadrant_weights,
            'asset_weights': asset_weights,
            'quadrant_returns': quadrant_returns,
            'momentum': momentum
        }
    
    def build_asset_risk_parity_strategy(self, return_data, lookback_window=252,
                                         use_semicovariance=True):
        """
        构建传统资产风险平价策略（用于对比）
        """
        if len(return_data) < lookback_window:
            lookback_window = len(return_data)
        
        recent_returns = return_data.tail(lookback_window)
        cov_matrix = RiskCalculator.calculate_ewma_covariance(
            recent_returns, lambda_param=0.94, use_semicovariance=use_semicovariance
        )
        asset_weights = RiskCalculator.calculate_risk_parity_weights(cov_matrix)
        
        return {
            'asset_weights': asset_weights
        }
