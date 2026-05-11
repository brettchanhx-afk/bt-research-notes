import numpy as np
import pandas as pd
from scipy.optimize import minimize

class BlackLittermanModel:
    def __init__(self, confidence=0.5):
        self.confidence = confidence
    
    def fit(self, returns_df):
        sigma = returns_df.cov() * 252
        pi = returns_df.mean() * 252
        return sigma, pi
    
    def get_weights(self, returns_df, views=None):
        sigma, pi = self.fit(returns_df)
        n = len(returns_df.columns)
        
        if views is None:
            P = np.eye(n)
            Q = pi.values
        else:
            P = views['P']
            Q = views['Q']
        
        tau = self.confidence
        omega = np.diag(np.diag(P @ sigma @ P.T)) * (1 - self.confidence) / self.confidence
        
        sigma_inv = np.linalg.inv(sigma)
        bl_mean = pi + tau * sigma @ P.T @ np.linalg.inv(tau * P @ sigma @ P.T + omega) @ (Q - P @ pi)
        bl_cov = (1 + tau) * sigma
        
        weights = self._optimize(bl_mean, bl_cov)
        return pd.Series(weights, index=returns_df.columns)
    
    def _optimize(self, mean, cov):
        def objective(w):
            return -w @ mean + 0.5 * w @ cov @ w
        
        n = len(mean)
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
        return result.x

class RiskParityModel:
    def __init__(self, target_risk=0.1):
        self.target_risk = target_risk
    
    def get_weights(self, returns_df):
        cov_matrix = returns_df.cov() * 252
        n = len(returns_df.columns)
        
        def risk_contribution(w):
            sigma = np.sqrt(w @ cov_matrix @ w)
            mrc = (cov_matrix @ w) / sigma
            rc = w * mrc
            return rc
        
        def objective(w):
            rc = risk_contribution(w)
            target_rc = self.target_risk / n
            return np.sum((rc - target_rc)**2)
        
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=returns_df.columns)

class MacroFactorModel:
    def __init__(self, lambda_reg=0.1):
        self.lambda_reg = lambda_reg
    
    def get_weights(self, returns_df, macro_data=None):
        sigma = returns_df.cov() * 252
        mu = returns_df.mean() * 252
        
        if macro_data is not None:
            macro_df = pd.DataFrame(macro_data)
            macro_df = macro_df.set_index('trade_date')
            common_dates = returns_df.index.intersection(macro_df.index)
            if len(common_dates) > 0:
                X = macro_df.loc[common_dates].values
                y = returns_df.loc[common_dates].values
                beta = np.linalg.inv(X.T @ X + self.lambda_reg * np.eye(X.shape[1])) @ X.T @ y
                mu = mu + beta.mean(axis=1)
        
        def objective(w):
            return -w @ mu + 0.5 * w @ sigma @ w
        
        n = len(mu)
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=returns_df.columns)

class EqualWeightModel:
    def get_weights(self, returns_df):
        n = len(returns_df.columns)
        return pd.Series(np.ones(n)/n, index=returns_df.columns)