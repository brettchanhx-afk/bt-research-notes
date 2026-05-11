import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from datetime import datetime

def log(msg):
    print(msg, file=sys.stdout)
    sys.stdout.flush()

BACKTEST_CONFIG = {
    'start_date': '20210101',
    'end_date': '20240111',
    'rebalance_freq': 'monthly',
    'transaction_cost': 0.0005,
    'assets': ['沪深300', '新华商品指数', '债券指数', '标普500', '上证50', '纳斯达克100'],
    'max_weight': 0.30
}

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BlackLittermanModel:
    def __init__(self, confidence=0.5, max_weight=0.30):
        self.confidence = confidence
        self.max_weight = max_weight

    def get_weights(self, returns_df, views=None):
        sigma = returns_df.cov().values * 252
        pi = returns_df.mean().values * 252
        n = len(returns_df.columns)

        if views is None:
            P = np.eye(n)
            Q = pi
        else:
            P = views['P']
            Q = views['Q']
            P = P[:len(Q), :n]

        tau = self.confidence
        omega = np.diag(np.diag(P @ sigma @ P.T)) * (1 - self.confidence) / self.confidence

        bl_mean = pi + tau * sigma @ P.T @ np.linalg.inv(tau * P @ sigma @ P.T + omega) @ (Q - P @ pi)

        def objective(w):
            return -w @ bl_mean + 0.5 * w @ sigma @ w

        bounds = [(0.05, self.max_weight) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints, method='SLSQP')

        return pd.Series(result.x, index=returns_df.columns)

class RiskParityModel:
    def __init__(self, target_risk=0.1, max_weight=0.30):
        self.target_risk = target_risk
        self.max_weight = max_weight

    def get_weights(self, returns_df):
        cov_matrix = returns_df.cov().values * 252
        n = len(returns_df.columns)

        def objective(w):
            sigma = np.sqrt(w @ cov_matrix @ w)
            mrc = (cov_matrix @ w) / sigma
            rc = w * mrc
            target_rc = self.target_risk / n
            return np.sum((rc - target_rc)**2)

        bounds = [(0.05, self.max_weight) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints, method='SLSQP')

        return pd.Series(result.x, index=returns_df.columns)

class MacroFactorModel:
    def __init__(self, lambda_reg=0.1, max_weight=0.30):
        self.lambda_reg = lambda_reg
        self.max_weight = max_weight

    def get_weights(self, returns_df, macro_data=None):
        sigma = returns_df.cov().values * 252
        mu = returns_df.mean().values * 252

        def objective(w):
            return -w @ mu + 0.5 * w @ sigma @ w + self.lambda_reg * np.sum(w**2)

        n = len(mu)
        bounds = [(0.05, self.max_weight) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints, method='SLSQP')

        return pd.Series(result.x, index=returns_df.columns)

class MinimumVarianceModel:
    def __init__(self, max_weight=0.30):
        self.max_weight = max_weight

    def get_weights(self, returns_df):
        cov_matrix = returns_df.cov().values * 252
        n = len(returns_df.columns)

        def objective(w):
            return w @ cov_matrix @ w

        bounds = [(0.05, self.max_weight) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints, method='SLSQP')

        return pd.Series(result.x, index=returns_df.columns)

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

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_ratio': win_ratio
        }

def load_local_data(start_date, end_date, assets):
    log("正在读取本地数据...")
    df = pd.read_csv('data/processed_asset_data.csv', index_col=0, encoding='utf-8-sig')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.dropna(subset=[df.columns[0]])
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    available_assets = [col for col in assets if col in df.columns]
    log(f"可用资产: {available_assets}")

    df = df[available_assets]
    df = df.dropna()
    log(f"有效数据: {len(df)} 条交易日")

    return df

def generate_rebalance_dates(start_date, end_date):
    return set(pd.date_range(start=start_date, end=end_date, freq='MS'))

def get_model_weights(model_name, returns_df, max_weight):
    n = len(returns_df.columns)

    if model_name == 'BL1':
        model = BlackLittermanModel(confidence=0.5, max_weight=max_weight)
        return model.get_weights(returns_df)
    elif model_name == 'BL2':
        model = BlackLittermanModel(confidence=0.7, max_weight=max_weight)
        if n >= 2:
            views = {'P': np.array([[1, -1] + [0]*(n-2)]), 'Q': np.array([0.05])}
        else:
            views = None
        return model.get_weights(returns_df, views)
    elif model_name == 'RiskParity':
        model = RiskParityModel(target_risk=0.1, max_weight=max_weight)
        return model.get_weights(returns_df)
    elif model_name == 'MacroFactor':
        model = MacroFactorModel(lambda_reg=0.1, max_weight=max_weight)
        return model.get_weights(returns_df)
    elif model_name == 'MinVariance':
        model = MinimumVarianceModel(max_weight=max_weight)
        return model.get_weights(returns_df)
    else:
        raise ValueError(f"未知模型: {model_name}")

def run_strategy_backtest(model_name, asset_prices, rebalance_dates, transaction_cost, max_weight):
    lookback_window = 60

    def weight_generator(date):
        try:
            start_idx = max(0, asset_prices.index.get_loc(date) - lookback_window)
            lookback_data = asset_prices.iloc[start_idx:asset_prices.index.get_loc(date)]
            returns_df = lookback_data.pct_change(fill_method=None).dropna()
        except KeyError:
            n = len(asset_prices.columns)
            return pd.Series(np.ones(n)/n, index=asset_prices.columns)

        if len(returns_df) < 30:
            n = len(asset_prices.columns)
            return pd.Series(np.ones(n)/n, index=asset_prices.columns)

        return get_model_weights(model_name, returns_df, max_weight)

    backtester = Backtester(transaction_cost=transaction_cost)
    portfolio_value = backtester.run_backtest(asset_prices, weight_generator, rebalance_dates)
    metrics = backtester.calculate_metrics(portfolio_value)

    return portfolio_value, metrics

def main():
    log("=" * 60)
    log("        大类资产配置策略落地方法研究")
    log("           国泰君安量化配置团队")
    log("=" * 60)
    log(f"回测时间: {BACKTEST_CONFIG['start_date']} 至 {BACKTEST_CONFIG['end_date']}")
    log(f"调仓频率: {BACKTEST_CONFIG['rebalance_freq']}")
    log(f"交易成本: {BACKTEST_CONFIG['transaction_cost']*10000} 万分之")
    log(f"单资产权重上限: {BACKTEST_CONFIG['max_weight']*100:.0f}%")
    log("-" * 60)

    asset_prices = load_local_data(
        BACKTEST_CONFIG['start_date'],
        BACKTEST_CONFIG['end_date'],
        BACKTEST_CONFIG['assets']
    )

    log(f"\n资产配置: {list(asset_prices.columns)}")

    rebalance_dates = generate_rebalance_dates(BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'])
    log(f"调仓日期数量: {len(rebalance_dates)}")

    models = ['BL1', 'BL2', 'RiskParity', 'MacroFactor', 'MinVariance']
    model_names = ['BL模型1', 'BL模型2', '风险平价模型', '宏观因子模型', '最小方差模型']

    portfolio_values = []
    all_metrics = []
    final_weights_list = []

    log("\n" + "-" * 60)
    log("                    多资产配置回测")
    log("-" * 60)

    for model, name in zip(models, model_names):
        log(f"\n正在回测 {name}...")
        try:
            pv, metrics = run_strategy_backtest(
                model, asset_prices, rebalance_dates,
                BACKTEST_CONFIG['transaction_cost'],
                BACKTEST_CONFIG['max_weight']
            )
            portfolio_values.append(pv)
            metrics['model'] = name
            metrics['type'] = '多资产'
            all_metrics.append(metrics)

            last_date = pv.dropna().index[-1]
            start_idx = max(0, asset_prices.index.get_loc(last_date) - 60)
            lookback = asset_prices.iloc[start_idx:asset_prices.index.get_loc(last_date)]
            returns_df = lookback.pct_change(fill_method=None).dropna()
            if len(returns_df) >= 30:
                weights = get_model_weights(model, returns_df, BACKTEST_CONFIG['max_weight'])
            else:
                n = len(asset_prices.columns)
                weights = pd.Series(np.ones(n)/n, index=asset_prices.columns)
            final_weights_list.append(weights)

            log(f"  年化收益: {metrics['annualized_return']:.2%}")
            log(f"  年化波动率: {metrics['volatility']:.2%}")
            log(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
            log(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            log(f"  胜率: {metrics['win_ratio']:.2%}")
        except Exception as e:
            log(f"  回测失败: {e}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        results_path = os.path.join(OUTPUT_DIR, 'backtest_results_multi.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        log(f"\n回测结果已保存到: {results_path}")

        plt.figure(figsize=(14, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        for pv, name in zip(portfolio_values, model_names):
            plt.plot(pv, label=name)

        plt.title('各模型策略净值曲线 (2021-2024) - 多资产配置')
        plt.xlabel('日期')
        plt.ylabel('净值')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        equity_path = os.path.join(OUTPUT_DIR, 'equity_curve_multi.png')
        plt.savefig(equity_path, dpi=300, bbox_inches='tight')
        log(f"净值曲线已保存到: {equity_path}")

        log("\n" + "-" * 60)
        log("                    回测结果汇总")
        log("-" * 60)
        results_display = results_df[['model', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']].copy()
        results_display.columns = ['模型', '年化收益', '年化波动率', '夏普比率', '最大回撤']
        log(results_display.to_string(
            formatters={
                '年化收益': '{:.2%}'.format,
                '年化波动率': '{:.2%}'.format,
                '夏普比率': '{:.2f}'.format,
                '最大回撤': '{:.2%}'.format
            }
        ))

        log("\n" + "-" * 60)
        log("                    期末资产权重")
        log("-" * 60)
        weights_df = pd.DataFrame(final_weights_list, index=model_names).T
        log(weights_df.round(4).to_string())
        log("\n(各权重四舍五入至40%上限约束)")

    return portfolio_values, all_metrics

if __name__ == '__main__':
    portfolio_values, results_df = main()
    log("\n程序执行完成！")