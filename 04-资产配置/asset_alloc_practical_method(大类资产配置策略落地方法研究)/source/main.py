import pandas as pd
import numpy as np
import os
from datetime import datetime

from .config import ASSET_CONFIG, BACKTEST_CONFIG, FIT_CONFIG, OUTPUT_DIR
from .data_fetcher import get_index_daily, get_fund_daily, get_fund_list_by_type
from .allocation_models import BlackLittermanModel, RiskParityModel, MacroFactorModel
from .index_fitting import IndexFitting, FundSelector
from .backtester import Backtester, PortfolioAnalyzer

def load_all_asset_data(start_date, end_date):
    asset_data = {}
    for asset_type, config in ASSET_CONFIG.items():
        print(f"正在获取 {config['name']} ({config['index']}) 数据...")
        data = get_index_daily(config['index'], start_date, end_date)
        if data is not None:
            asset_data[asset_type] = data.set_index('date')['close']
        else:
            print(f"警告：无法获取 {config['name']} 数据")
    
    return pd.DataFrame(asset_data)

def generate_rebalance_dates(start_date, end_date, freq='monthly'):
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    return set(dates)

def get_model_weights(model_name, returns_df, macro_data=None):
    if model_name == 'BL1':
        model = BlackLittermanModel(confidence=0.5)
        return model.get_weights(returns_df)
    elif model_name == 'BL2':
        model = BlackLittermanModel(confidence=0.7)
        views = {
            'P': np.array([[1, -1, 0, 0], [0, 0, 1, -1]]),
            'Q': np.array([0.05, 0.03])
        }
        return model.get_weights(returns_df, views)
    elif model_name == 'RiskParity':
        model = RiskParityModel(target_risk=0.1)
        return model.get_weights(returns_df)
    elif model_name == 'MacroFactor':
        model = MacroFactorModel(lambda_reg=0.1)
        return model.get_weights(returns_df, macro_data)
    else:
        raise ValueError(f"未知模型: {model_name}")

def run_strategy_backtest(model_name, asset_prices, rebalance_dates, transaction_cost):
    lookback_window = 60
    
    def weight_generator(date):
        start_idx = max(0, asset_prices.index.get_loc(date) - lookback_window)
        lookback_data = asset_prices.iloc[start_idx:asset_prices.index.get_loc(date)]
        returns_df = lookback_data.pct_change().dropna()
        
        if len(returns_df) < 30:
            n = len(asset_prices.columns)
            return pd.Series(np.ones(n)/n, index=asset_prices.columns)
        
        return get_model_weights(model_name, returns_df)
    
    backtester = Backtester(transaction_cost=transaction_cost)
    portfolio_value = backtester.run_backtest(asset_prices, weight_generator, rebalance_dates)
    
    metrics = backtester.calculate_metrics(portfolio_value)
    return portfolio_value, metrics

def run_fund_fitting_backtest(model_name, asset_prices, fund_weights, rebalance_dates, transaction_cost):
    lookback_window = 60
    
    def weight_generator(date):
        start_idx = max(0, asset_prices.index.get_loc(date) - lookback_window)
        lookback_data = asset_prices.iloc[start_idx:asset_prices.index.get_loc(date)]
        returns_df = lookback_data.pct_change().dropna()
        
        if len(returns_df) < 30:
            n = len(asset_prices.columns)
            base_weights = pd.Series(np.ones(n)/n, index=asset_prices.columns)
        else:
            base_weights = get_model_weights(model_name, returns_df)
        
        fitted_weights = {}
        for asset_type, weight in base_weights.items():
            if asset_type in fund_weights:
                for fund, fund_w in fund_weights[asset_type].items():
                    fitted_weights[fund] = weight * fund_w
        
        return pd.Series(fitted_weights)
    
    fund_prices = pd.DataFrame()
    for asset_type, weights in fund_weights.items():
        for fund_code in weights.index:
            fund_data = get_fund_daily(fund_code, BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'])
            if fund_data is not None:
                fund_prices[fund_code] = fund_data.set_index('date')['close']
    
    fund_prices = fund_prices.dropna()
    
    backtester = Backtester(transaction_cost=transaction_cost)
    portfolio_value = backtester.run_backtest(fund_prices, weight_generator, rebalance_dates)
    
    metrics = backtester.calculate_metrics(portfolio_value)
    return portfolio_value, metrics

def main():
    print("=== 大类资产配置策略落地方法研究 ===")
    print(f"回测时间: {BACKTEST_CONFIG['start_date']} 至 {BACKTEST_CONFIG['end_date']}")
    print(f"调仓频率: {BACKTEST_CONFIG['rebalance_freq']}")
    print(f"交易成本: {BACKTEST_CONFIG['transaction_cost']*10000} 万分之")
    print("-" * 50)
    
    asset_prices = load_all_asset_data(BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'])
    print(f"获取到 {len(asset_prices)} 个交易日数据")
    
    rebalance_dates = generate_rebalance_dates(
        BACKTEST_CONFIG['start_date'], 
        BACKTEST_CONFIG['end_date'], 
        BACKTEST_CONFIG['rebalance_freq']
    )
    print(f"调仓日期数量: {len(rebalance_dates)}")
    
    models = ['BL1', 'BL2', 'RiskParity', 'MacroFactor']
    portfolio_values = []
    all_metrics = []
    
    print("\n=== 指数层面回测 ===")
    for model in models:
        print(f"正在回测 {model} 模型...")
        pv, metrics = run_strategy_backtest(
            model, 
            asset_prices, 
            rebalance_dates, 
            BACKTEST_CONFIG['transaction_cost']
        )
        portfolio_values.append(pv)
        metrics['model'] = model
        metrics['type'] = '指数'
        all_metrics.append(metrics)
        print(f"  年化收益: {metrics['annualized_return']:.2%}, 夏普比率: {metrics['sharpe_ratio']:.2f}, 最大回撤: {metrics['max_drawdown']:.2%}")
    
    print("\n=== 基金拟合落地回测 ===")
    fund_weights = {}
    selector = FundSelector(lookback_period=FIT_CONFIG['lookback_period'])
    fitter = IndexFitting(
        max_tracking_error=FIT_CONFIG['max_tracking_error'],
        max_drawdown=FIT_CONFIG['max_drawdown']
    )
    
    for asset_type, config in ASSET_CONFIG.items():
        print(f"正在拟合 {config['name']}...")
        try:
            fund_candidates = get_fund_list_by_type('index')
            index_returns, fund_returns = selector.select_funds(
                config['index'], 
                fund_candidates,
                BACKTEST_CONFIG['start_date'], 
                BACKTEST_CONFIG['end_date']
            )
            
            if len(fund_returns.columns) >= 3:
                weights, tracking_err = fitter.fit(index_returns, fund_returns)
                fund_weights[asset_type] = weights
                print(f"  跟踪误差: {tracking_err:.4f}")
        except Exception as e:
            print(f"  拟合失败: {e}")
    
    if fund_weights:
        for model in models:
            print(f"正在回测 {model} 基金落地方案...")
            try:
                pv, metrics = run_fund_fitting_backtest(
                    model,
                    asset_prices,
                    fund_weights,
                    rebalance_dates,
                    BACKTEST_CONFIG['transaction_cost']
                )
                metrics['model'] = model
                metrics['type'] = '基金落地'
                all_metrics.append(metrics)
                print(f"  年化收益: {metrics['annualized_return']:.2%}, 夏普比率: {metrics['sharpe_ratio']:.2f}, 最大回撤: {metrics['max_drawdown']:.2%}")
            except Exception as e:
                print(f"  回测失败: {e}")
    
    results_df = pd.DataFrame(all_metrics)
    results_path = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n回测结果已保存到: {results_path}")
    
    return portfolio_values, results_df

if __name__ == '__main__':
    portfolio_values, results_df = main()
    print(results_df)