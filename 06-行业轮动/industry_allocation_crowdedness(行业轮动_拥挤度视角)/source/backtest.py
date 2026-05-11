import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self, initial_capital=1000000, commission_rate=0.0003, slippage=0.0001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.portfolio_value = initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_records = []

    def reset(self):
        self.portfolio_value = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_records = []

    def calculate_commission(self, price, volume, is_buy=True):
        trade_value = price * volume
        commission = trade_value * self.commission_rate
        slippage_cost = trade_value * self.slippage if is_buy else 0
        return commission + slippage_cost

    def execute_trade(self, symbol, price, volume, date, is_buy=True):
        commission = self.calculate_commission(price, volume, is_buy)
        trade_value = price * volume
        if is_buy:
            cost = trade_value + commission
            if cost <= self.portfolio_value:
                self.portfolio_value -= cost
                if symbol in self.positions:
                    avg_price = (self.positions[symbol]['price'] * self.positions[symbol]['volume'] + price * volume) / (self.positions[symbol]['volume'] + volume)
                    self.positions[symbol] = {
                        'volume': self.positions[symbol]['volume'] + volume,
                        'price': avg_price
                    }
                else:
                    self.positions[symbol] = {'volume': volume, 'price': price}
                self.trade_history.append({
                    'date': date, 'symbol': symbol, 'action': 'buy',
                    'price': price, 'volume': volume, 'commission': commission
                })
                return True
        else:
            if symbol in self.positions and self.positions[symbol]['volume'] >= volume:
                revenue = trade_value - commission
                self.portfolio_value += revenue
                self.positions[symbol]['volume'] -= volume
                if self.positions[symbol]['volume'] == 0:
                    del self.positions[symbol]
                self.trade_history.append({
                    'date': date, 'symbol': symbol, 'action': 'sell',
                    'price': price, 'volume': volume, 'commission': commission
                })
                return True
        return False

    def update_portfolio_value(self, current_prices: Dict[str, float], date):
        total_value = self.portfolio_value
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                total_value += pos['volume'] * current_prices[symbol]
        self.daily_records.append({
            'date': date,
            'portfolio_value': total_value,
            'cash': self.portfolio_value,
            'position_value': total_value - self.portfolio_value,
            'num_positions': len(self.positions)
        })

    def get_portfolio_returns(self) -> pd.Series:
        if not self.daily_records:
            return pd.Series()
        df = pd.DataFrame(self.daily_records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        returns = df['portfolio_value'].pct_change().fillna(0)
        return returns

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                          risk_free_rate=0.03) -> Dict:
        if len(returns) == 0:
            return {}
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        cumulative_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trading_days': len(returns)
        }
        if benchmark_returns is not None:
            aligned_returns = returns.align(benchmark_returns, join='inner')
            excess_returns = aligned_returns[0] - aligned_returns[1]
            metrics['annual_excess_return'] = excess_returns.mean() * 252
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = metrics['annual_excess_return'] / metrics['tracking_error'] if metrics['tracking_error'] > 0 else 0
        return metrics

    @staticmethod
    def generate_equity_curve(returns: pd.Series, initial_capital=1000000) -> pd.Series:
        cumulative = (1 + returns).cumprod()
        equity_curve = cumulative * initial_capital
        return equity_curve

    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown

    @staticmethod
    def calculate_rolling_sharpe(returns: pd.Series, window=60, risk_free_rate=0.03) -> pd.Series:
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean - risk_free_rate / 252) / rolling_std * np.sqrt(252)
        return rolling_sharpe

def run_backtest(strategy_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                 initial_capital=1000000, strategy_name="Strategy") -> Dict:
    engine = BacktestEngine(initial_capital=initial_capital)
    returns_df = pd.DataFrame({'strategy': strategy_returns})
    if benchmark_returns is not None:
        returns_df['benchmark'] = benchmark_returns
    equity_curve = PerformanceAnalyzer.generate_equity_curve(strategy_returns, initial_capital)
    drawdown = PerformanceAnalyzer.calculate_drawdown_series(equity_curve)
    metrics = PerformanceAnalyzer.calculate_metrics(strategy_returns, benchmark_returns)
    results = {
        'metrics': metrics,
        'equity_curve': equity_curve,
        'drawdown': drawdown,
        'returns': strategy_returns
    }
    if benchmark_returns is not None:
        benchmark_equity = PerformanceAnalyzer.generate_equity_curve(benchmark_returns, initial_capital)
        results['benchmark_equity'] = benchmark_equity
    print(f"\n{'='*50}")
    print(f"Backtest Results: {strategy_name}")
    print(f"{'='*50}")
    print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Cumulative Return: {metrics.get('cumulative_return', 0):.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    if 'annual_excess_return' in metrics:
        print(f"Annual Excess Return: {metrics['annual_excess_return']:.2%}")
    return results

def compare_strategies(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    comparison = []
    for name, result in results_dict.items():
        metrics = result.get('metrics', {})
        row = {'Strategy': name}
        row.update(metrics)
        comparison.append(row)
    df = pd.DataFrame(comparison)
    numeric_cols = [col for col in df.columns if col != 'Strategy']
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    return df

def save_backtest_results(results: Dict, output_dir: str, strategy_name: str):
    os.makedirs(output_dir, exist_ok=True)
    if 'equity_curve' in results and results['equity_curve'] is not None:
        equity_path = os.path.join(output_dir, f'{strategy_name}_equity.csv')
        results['equity_curve'].to_csv(equity_path)
    if 'returns' in results and results['returns'] is not None:
        returns_path = os.path.join(output_dir, f'{strategy_name}_returns.csv')
        results['returns'].to_csv(returns_path)
    if 'drawdown' in results and results['drawdown'] is not None:
        dd_path = os.path.join(output_dir, f'{strategy_name}_drawdown.csv')
        results['drawdown'].to_csv(dd_path)
    metrics_path = os.path.join(output_dir, f'{strategy_name}_metrics.csv')
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Backtest results saved to {output_dir}")

if __name__ == "__main__":
    print("回测引擎模块测试...")
    test_returns = pd.Series(np.random.randn(100) * 0.02, index=pd.date_range('2020-01-01', periods=100, freq='D'))
    results = run_backtest(test_returns, strategy_name="测试策略")
    print(f"\n测试完成，指标: {results['metrics']}")