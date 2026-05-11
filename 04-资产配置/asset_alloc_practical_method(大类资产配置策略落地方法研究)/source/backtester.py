import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, transaction_cost=0.0005):
        self.transaction_cost = transaction_cost
    
    def run_backtest(self, price_data, weight_generator, rebalance_dates):
        dates = price_data.index
        portfolio_value = pd.Series(index=dates)
        current_weights = None
        prev_weights = None
        
        portfolio_value.iloc[0] = 1.0
        
        for i, date in enumerate(dates):
            if i == 0:
                continue
            
            if date in rebalance_dates:
                current_weights = weight_generator(date)
                if prev_weights is not None:
                    turnover = np.sum(np.abs(current_weights - prev_weights))
                    cost = turnover * self.transaction_cost
                    portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 - cost)
                prev_weights = current_weights.copy()
            
            if current_weights is None:
                portfolio_value.iloc[i] = portfolio_value.iloc[i-1]
            else:
                returns = price_data.loc[date] / price_data.loc[dates[i-1]] - 1
                portfolio_return = np.sum(current_weights * returns)
                portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)
        
        return portfolio_value
    
    def calculate_metrics(self, portfolio_value):
        daily_returns = portfolio_value.pct_change().dropna()
        
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        cum_returns = (1 + daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = 1 - cum_returns / running_max
        max_drawdown = drawdown.max()
        
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_ratio = winning_days / total_days if total_days > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_ratio': win_ratio,
            'total_days': total_days
        }
        
        return metrics
    
    def plot_results(self, portfolio_value, title='策略回测结果'):
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value, label='策略净值')
        plt.title(title)
        plt.xlabel('日期')
        plt.ylabel('净值')
        plt.legend()
        plt.grid(True)
        return plt

class PortfolioAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, portfolio_values, model_names):
        results = []
        
        for name, pv in zip(model_names, portfolio_values):
            metrics = Backtester().calculate_metrics(pv)
            metrics['model_name'] = name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, portfolio_values, model_names):
        plt.figure(figsize=(12, 6))
        for name, pv in zip(model_names, portfolio_values):
            plt.plot(pv, label=name)
        
        plt.title('各模型策略回测对比')
        plt.xlabel('日期')
        plt.ylabel('净值')
        plt.legend()
        plt.grid(True)
        return plt