import numpy as np
import pandas as pd
from scipy.optimize import minimize

class IndexFitting:
    def __init__(self, max_tracking_error=0.01, max_drawdown=0.05):
        self.max_tracking_error = max_tracking_error
        self.max_drawdown = max_drawdown
    
    def fit(self, index_returns, fund_returns):
        common_dates = index_returns.index.intersection(fund_returns.index)
        if len(common_dates) < 10:
            raise ValueError("数据不足，无法进行拟合")
        
        index_ret = index_returns.loc[common_dates].values.flatten()
        fund_ret = fund_returns.loc[common_dates].values
        
        n = fund_ret.shape[1]
        
        def tracking_error(w):
            fitted_ret = fund_ret @ w
            return np.sqrt(np.mean((fitted_ret - index_ret)**2))
        
        def objective(w):
            return tracking_error(w)
        
        def drawdown_constraint(w):
            fitted_ret = fund_ret @ w
            cum_ret = np.cumprod(1 + fitted_ret)
            running_max = np.maximum.accumulate(cum_ret)
            drawdown = 1 - cum_ret / running_max
            return self.max_drawdown - np.max(drawdown)
        
        bounds = [(0, 1) for _ in range(n)]
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': drawdown_constraint}
        ]
        
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
        
        weights = pd.Series(result.x, index=fund_returns.columns)
        tracking_err = tracking_error(result.x)
        
        return weights, tracking_err
    
    def fit_enhanced(self, index_returns, fund_returns):
        common_dates = index_returns.index.intersection(fund_returns.index)
        if len(common_dates) < 10:
            raise ValueError("数据不足，无法进行拟合")
        
        index_ret = index_returns.loc[common_dates].values.flatten()
        fund_ret = fund_returns.loc[common_dates].values
        
        n = fund_ret.shape[1]
        
        def objective(w):
            fitted_ret = fund_ret @ w
            tracking_error = np.sqrt(np.mean((fitted_ret - index_ret)**2))
            excess_return = np.mean(fitted_ret) - np.mean(index_ret)
            return tracking_error - 0.1 * excess_return
        
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
        
        weights = pd.Series(result.x, index=fund_returns.columns)
        fitted_ret = fund_ret @ result.x
        tracking_err = np.sqrt(np.mean((fitted_ret - index_ret)**2))
        excess_return = np.mean(fitted_ret) - np.mean(index_ret)
        
        return weights, tracking_err, excess_return

class FundSelector:
    def __init__(self, lookback_period=60):
        self.lookback_period = lookback_period
    
    def select_funds(self, index_code, fund_candidates, start_date, end_date):
        from data_fetcher import get_index_daily, get_fund_daily
        
        index_data = get_index_daily(index_code, start_date, end_date)
        if index_data is None:
            raise ValueError(f"无法获取指数 {index_code} 的数据")
        
        fund_data_dict = {}
        for fund in fund_candidates:
            fund_data = get_fund_daily(fund['ts_code'], start_date, end_date)
            if fund_data is not None and len(fund_data) > self.lookback_period:
                fund_data_dict[fund['ts_code']] = fund_data
        
        if len(fund_data_dict) == 0:
            raise ValueError("没有找到合适的基金数据")
        
        fund_returns = pd.DataFrame()
        for fund_code, df in fund_data_dict.items():
            fund_returns[fund_code] = df.set_index('date')['return']
        
        fund_returns = fund_returns.dropna(axis=1)
        
        index_returns = index_data.set_index('date')['return']
        
        return index_returns, fund_returns