
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskCalculator:
    """
    风险计算模块，包含EWMA半协方差、风险平价等功能
    """
    
    @staticmethod
    def calculate_ewma_covariance(returns, lambda_param=0.94, use_semicovariance=False):
        """
        计算EWMA协方差矩阵或半协方差矩阵
        
        参数:
            returns: 收益率数据
            lambda_param: EWMA衰减因子
            use_semicovariance: 是否使用半协方差（只考虑下行风险）
        """
        if use_semicovariance:
            returns = returns.copy()
            returns[returns > 0] = 0
        
        T = len(returns)
        weights = np.array([(1 - lambda_param) * (lambda_param ** (T - t - 1)) for t in range(T)])
        weights = weights / weights.sum()
        
        mean_returns = (returns * weights[:, np.newaxis]).sum(axis=0)
        demeaned = returns - mean_returns
        
        cov_matrix = np.zeros((returns.shape[1], returns.shape[1]))
        for t in range(T):
            cov_matrix += weights[t] * np.outer(demeaned.iloc[t], demeaned.iloc[t])
        
        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
    
    @staticmethod
    def calculate_risk_parity_weights(cov_matrix, method='ccd', max_iter=1000):
        """
        计算风险平价权重
        
        参数:
            cov_matrix: 协方差矩阵
            method: 优化方法 ('ccd'或'minimize')
        """
        n = len(cov_matrix)
        initial_weights = np.ones(n) / n
        
        def risk_contribution(weights, cov):
            sigma = np.sqrt(weights @ cov @ weights)
            mrc = cov @ weights
            rc = weights * mrc / sigma
            return rc
        
        def objective(weights, cov):
            rc = risk_contribution(weights, cov)
            target_rc = np.ones_like(rc) / len(rc)
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n)]
        
        result = minimize(
            objective,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter}
        )
        
        if result.success:
            weights = result.x
        else:
            weights = initial_weights
            print("风险平价优化失败，使用等权")
        
        return pd.Series(weights, index=cov_matrix.index)
    
    @staticmethod
    def calculate_performance_metrics(returns, risk_free_rate=0.0, periods=252):
        """
        计算绩效指标
        
        参数:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            periods: 年化期数（日频252，月频12）
        """
        if len(returns) == 0:
            return {}
        
        cumulative_return = (1 + returns).prod() - 1
        annual_return = (1 + cumulative_return) ** (periods / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(periods)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        win_rate = (returns > 0).sum() / len(returns)
        
        metrics = {
            '累计收益': cumulative_return,
            '年化收益': annual_return,
            '年化波动': annual_vol,
            '夏普比率': sharpe,
            '最大回撤': max_drawdown,
            '卡玛比率': calmar,
            '月度胜率': win_rate
        }
        
        return metrics
    
    @staticmethod
    def calculate_quadrant_performance(quadrant_returns, risk_free_rate=0.0):
        """
        计算四象限组合的绩效
        """
        performance = {}
        for quadrant in quadrant_returns.columns:
            performance[quadrant] = RiskCalculator.calculate_performance_metrics(
                quadrant_returns[quadrant], risk_free_rate, periods=252
            )
        return pd.DataFrame(performance).T
    
    @staticmethod
    def calculate_path_ratio_momentum(returns, lookback=20):
        """
        计算位移路径比动量（用于预期共振动量）
        
        参数:
            returns: 收益率序列
            lookback: 回看期
        """
        if len(returns) < lookback:
            return np.nan
        
        recent_returns = returns.tail(lookback)
        price = (1 + recent_returns).cumprod()
        start_price = price.iloc[0]
        end_price = price.iloc[-1]
        
        path_sum = np.sum(np.abs(price.diff().iloc[1:]))
        net_change = end_price - start_price
        
        if path_sum == 0:
            return 0
        
        momentum = net_change / path_sum
        return momentum
    
    @staticmethod
    def calculate_expected_momentum(returns, lookback=20):
        """
        计算预期动量（位移路径比动量）
        """
        momentum_series = returns.rolling(window=lookback).apply(
            lambda x: RiskCalculator.calculate_path_ratio_momentum(x, lookback),
            raw=False
        )
        return momentum_series

