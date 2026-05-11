import sys

def log(msg):
    print(msg, file=sys.stdout)
    sys.stdout.flush()

log("=" * 60)
log("        大类资产配置策略落地方法研究 - 调试版")
log("=" * 60)

# 配置参数
TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
TUSHARE_URL = "http://jiaoch.site"

ASSET_CONFIG = {
    'equity': {'index': '000300.SH', 'name': '沪深300'},
    'bond': {'index': '000016.SH', 'name': '上证50国债'},
    'commodity': {'index': '000998.SH', 'name': '中证商品'},
    'gold': {'index': '518880.SH', 'name': '黄金ETF'}
}

BACKTEST_CONFIG = {
    'start_date': '20210101',
    'end_date': '20240111',
    'rebalance_freq': 'monthly',
    'transaction_cost': 0.0005
}

OUTPUT_DIR = 'output'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== 数据获取模块 ==============
def get_index_daily(symbol, start_date, end_date):
    import tushare as ts
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        pro._DataApi__token = TUSHARE_TOKEN
        pro._DataApi__http_url = TUSHARE_URL
        df = pro.index_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if not df.empty:
            df = df[['trade_date', 'close']]
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)
            return df
    except Exception as e:
        log(f"tushare获取数据失败: {e}")
    return None

log("Step 1: 导入模块")
import pandas as pd
import numpy as np
from scipy.optimize import minimize
log("Step 1: 完成")

# ============== 主回测函数 ==============
log("Step 2: 加载资产数据")
asset_data = {}
for asset_type, config in ASSET_CONFIG.items():
    log(f"  获取 {config['name']} ({config['index']})...")
    data = get_index_daily(config['index'], BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'])
    if data is not None:
        asset_data[asset_type] = data.set_index('date')['close']
        log(f"  成功: {len(data)} 条")
    else:
        log(f"  失败")

if not asset_data:
    log("错误：无法获取任何资产数据")
    sys.exit(1)

asset_prices = pd.DataFrame(asset_data)
log(f"Step 2: 完成，获取到 {len(asset_prices)} 个交易日数据")

log("Step 3: 生成调仓日期")
dates = pd.date_range(start=BACKTEST_CONFIG['start_date'], end=BACKTEST_CONFIG['end_date'], freq='MS')
rebalance_dates = set(dates)
log(f"Step 3: 完成，共 {len(rebalance_dates)} 个调仓日")

log("Step 4: 定义模型权重函数")
class BlackLittermanModel:
    def __init__(self, confidence=0.5):
        self.confidence = confidence
    
    def get_weights(self, returns_df, views=None):
        sigma = returns_df.cov() * 252
        pi = returns_df.mean() * 252
        n = len(returns_df.columns)
        
        if views is None:
            P = np.eye(n)
            Q = pi.values
        else:
            P = views['P']
            Q = views['Q']
        
        tau = self.confidence
        omega = np.diag(np.diag(P @ sigma @ P.T)) * (1 - self.confidence) / self.confidence
        bl_mean = pi + tau * sigma @ P.T @ np.linalg.inv(tau * P @ sigma @ P.T + omega) @ (Q - P @ pi)
        
        def objective(w):
            return -w @ bl_mean + 0.5 * w @ sigma @ w
        
        bounds = [(0, 1) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=returns_df.columns)

log("Step 4: 完成")

log("Step 5: 开始回测")
models = ['BL1']
model_names = ['BL模型1']

for model, name in zip(models, model_names):
    log(f"  回测 {name}...")
    
    lookback_window = 60
    
    def weight_generator(date):
        start_idx = max(0, asset_prices.index.get_loc(date) - lookback_window)
        lookback_data = asset_prices.iloc[start_idx:asset_prices.index.get_loc(date)]
        returns_df = lookback_data.pct_change().dropna()
        
        if len(returns_df) < 30:
            n = len(asset_prices.columns)
            return pd.Series(np.ones(n)/n, index=asset_prices.columns)
        
        m = BlackLittermanModel(confidence=0.5)
        return m.get_weights(returns_df)
    
    log("    初始化回测...")
    dates = asset_prices.index
    portfolio_value = pd.Series(index=dates)
    current_weights = None
    prev_weights = None
    
    portfolio_value.iloc[0] = 1.0
    log("    开始逐日计算...")
    
    for i, date in enumerate(dates):
        if i == 0:
            continue
        
        if date in rebalance_dates:
            current_weights = weight_generator(date)
            if prev_weights is not None:
                turnover = np.sum(np.abs(current_weights - prev_weights))
                cost = turnover * BACKTEST_CONFIG['transaction_cost']
                portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 - cost)
            prev_weights = current_weights.copy()
        
        if current_weights is None:
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1]
        else:
            returns = asset_prices.loc[date] / asset_prices.loc[dates[i-1]] - 1
            portfolio_return = np.sum(current_weights * returns)
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)
    
    log("    计算指标...")
    daily_returns = portfolio_value.pct_change().dropna()
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    log(f"    年化收益: {annualized_return:.2%}")
    log(f"    夏普比率: {sharpe_ratio:.2f}")
    
log("Step 5: 完成")
log("\n程序执行完成！")