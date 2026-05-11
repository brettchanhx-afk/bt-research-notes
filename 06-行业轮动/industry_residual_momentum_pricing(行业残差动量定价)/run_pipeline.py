import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

BASE_DIR = r'd:\Documents\trae_projects\industry_residual_momentum_pricing'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def load_csv_data():
    print("=" * 60)
    print("加载CSV数据文件...")
    print("=" * 60)

    data = {}

    industry_path = os.path.join(DATA_DIR, '中信一级行业指数及收盘价2010_2026.csv')
    if os.path.exists(industry_path):
        df = pd.read_csv(industry_path)
        df = df.rename(columns={'Unnamed: 0': 'date'})
        df = df[df['date'] != 'date']
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.dropna(how='all', axis=1)
        df = df.apply(pd.to_numeric, errors='coerce')
        data['industry'] = df
        print(f"中信一级行业: {data['industry'].shape}, 行业数: {len(data['industry'].columns)}")

    bond_path = os.path.join(DATA_DIR, '债券市场数据2010_2026.csv')
    if os.path.exists(bond_path):
        df = pd.read_excel(bond_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.apply(pd.to_numeric, errors='coerce')
        data['bond'] = df
        print(f"债券市场: {data['bond'].shape}, 指标数: {len(data['bond'].columns)}")

    commodity_path = os.path.join(DATA_DIR, '商品市场指数2010_2026.csv')
    if os.path.exists(commodity_path):
        df = pd.read_csv(commodity_path, encoding='gbk', header=None)
        df = df.drop([0, 1, 2], axis=0)
        df.columns = ['date', '铜', '螺纹钢', '豆粕', 'PTA', '白糖']
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        data['commodity'] = df
        print(f"商品市场: {data['commodity'].shape}, 品种数: {len(data['commodity'].columns)}")

    asset_path = os.path.join(DATA_DIR, '大类资产数据2010_2026.csv')
    if os.path.exists(asset_path):
        df = pd.read_csv(asset_path)
        df = df.rename(columns={'Unnamed: 0': 'date'})
        df = df.drop([0, 1], axis=0)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        cols_to_keep = [c for c in df.columns if 'Unnamed' not in c]
        df = df[cols_to_keep]
        df = df.apply(pd.to_numeric, errors='coerce')
        data['broad'] = df
        print(f"大类资产: {data['broad'].shape}, 资产数: {len(data['broad'].columns)}")

    return data


def calculate_yoy_returns(price_data):
    return price_data.pct_change(periods=12)


def calculate_monthly_log_returns(price_data):
    monthly_prices = price_data.resample('M').last()
    return np.log(monthly_prices / monthly_prices.shift(1))


def calculate_pca_factors_multi_market(stock_data, bond_data, commodity_data, window=100):
    print(f"\n计算国内多市场因子 (滚动窗口={window}个月)...")

    from sklearn.decomposition import PCA
    from scipy.stats import zscore

    stock_yoy = calculate_yoy_returns(stock_data).dropna()
    bond_yoy = calculate_yoy_returns(bond_data).dropna()
    commodity_yoy = calculate_yoy_returns(commodity_data).dropna()

    common_dates = stock_yoy.index.intersection(bond_yoy.index).intersection(commodity_yoy.index)
    stock_yoy = stock_yoy.loc[common_dates]
    bond_yoy = bond_yoy.loc[common_dates]
    commodity_yoy = commodity_yoy.loc[common_dates]

    factors_list = []
    valid_dates = []

    min_window = min(window, len(common_dates) - 1)

    for i in range(min_window, len(common_dates)):
        date = common_dates[i]
        lookback_dates = common_dates[i - min_window:i]

        stock_window = stock_yoy.loc[lookback_dates].dropna(axis=1)
        bond_window = bond_yoy.loc[lookback_dates].dropna(axis=1)
        commodity_window = commodity_yoy.loc[lookback_dates].dropna(axis=1)

        if stock_window.shape[1] < 3 or bond_window.shape[1] < 2 or commodity_window.shape[1] < 2:
            continue

        try:
            n_stock = min(3, stock_window.shape[1])
            n_bond = min(3, bond_window.shape[1])
            n_commodity = min(2, commodity_window.shape[1])

            pca_stock = PCA(n_components=n_stock)
            stock_pca = pca_stock.fit_transform(stock_window)

            pca_bond = PCA(n_components=n_bond)
            bond_pca = pca_bond.fit_transform(bond_window)

            pca_commodity = PCA(n_components=n_commodity)
            commodity_pca = pca_commodity.fit_transform(commodity_window)

            stock_std = np.std(stock_pca[:, 0])
            stock_std = stock_std if stock_std > 1e-8 else 1e-8
            stock_market = (stock_pca[:, 0] - np.mean(stock_pca[:, 0])) / stock_std

            stock_pc2_std = np.std(stock_pca[:, 1])
            stock_pc2_std = stock_pc2_std if stock_pc2_std > 1e-8 else 1e-8
            stock_pc2 = (stock_pca[:, 1] - np.mean(stock_pca[:, 1])) / stock_pc2_std

            stock_pc3 = np.zeros(len(stock_pca))
            if n_stock >= 3:
                stock_pc3_std = np.std(stock_pca[:, 2])
                stock_pc3_std = stock_pc3_std if stock_pc3_std > 1e-8 else 1e-8
                stock_pc3 = (stock_pca[:, 2] - np.mean(stock_pca[:, 2])) / stock_pc3_std

            bond_std = np.std(bond_pca[:, 0])
            bond_std = bond_std if bond_std > 1e-8 else 1e-8
            bond_market = (bond_pca[:, 0] - np.mean(bond_pca[:, 0])) / bond_std

            bond_pc2 = np.zeros(len(bond_pca))
            if n_bond >= 2:
                bond_pc2_std = np.std(bond_pca[:, 1])
                bond_pc2_std = bond_pc2_std if bond_pc2_std > 1e-8 else 1e-8
                bond_pc2 = (bond_pca[:, 1] - np.mean(bond_pca[:, 1])) / bond_pc2_std

            bond_pc3 = np.zeros(len(bond_pca))
            if n_bond >= 3:
                bond_pc3_std = np.std(bond_pca[:, 2])
                bond_pc3_std = bond_pc3_std if bond_pc3_std > 1e-8 else 1e-8
                bond_pc3 = (bond_pca[:, 2] - np.mean(bond_pca[:, 2])) / bond_pc3_std

            commodity_std = np.std(commodity_pca[:, 0])
            commodity_std = commodity_std if commodity_std > 1e-8 else 1e-8
            commodity_market = (commodity_pca[:, 0] - np.mean(commodity_pca[:, 0])) / commodity_std

            commodity_pc2 = np.zeros(len(commodity_pca))
            if n_commodity >= 2:
                commodity_pc2_std = np.std(commodity_pca[:, 1])
                commodity_pc2_std = commodity_pc2_std if commodity_pc2_std > 1e-8 else 1e-8
                commodity_pc2 = (commodity_pca[:, 1] - np.mean(commodity_pca[:, 1])) / commodity_pc2_std

            domestic_market_factor = (stock_market[-1] + bond_market[-1] + commodity_market[-1]) / 3

            style_factors = {
                'domestic_market': domestic_market_factor,
                'stock_pc2': stock_pc2[-1],
                'stock_pc3': stock_pc3 if isinstance(stock_pc3, (int, float)) else stock_pc3[-1],
                'bond_pc2': bond_pc2 if isinstance(bond_pc2, (int, float)) else bond_pc2[-1],
                'bond_pc3': bond_pc3 if isinstance(bond_pc3, (int, float)) else bond_pc3[-1],
                'commodity_pc2': commodity_pc2 if isinstance(commodity_pc2, (int, float)) else commodity_pc2[-1]
            }

            factors_list.append(style_factors)
            valid_dates.append(date)
        except Exception as e:
            continue

    if factors_list:
        factors_df = pd.DataFrame(factors_list, index=valid_dates)
        print(f"因子计算完成: {factors_df.shape}")
        return factors_df

    print("因子计算失败")
    return None


def calculate_residual_momentum(asset_prices, factors, rolling_window=100, momentum_window=12):
    print(f"\n计算残差动量 (滚动窗口={rolling_window}, 动量窗口={momentum_window})...")

    if factors is None or len(factors) == 0:
        print("因子为空，无法计算残差动量")
        return None

    monthly_returns = calculate_monthly_log_returns(asset_prices)

    common_dates = monthly_returns.index.intersection(factors.index)
    if len(common_dates) == 0:
        print("没有共同的日期")
        return None

    monthly_ret = monthly_returns.loc[common_dates].dropna(axis=1)
    fac = factors.loc[common_dates]

    if len(common_dates) < rolling_window + momentum_window:
        print(f"警告: 共同日期数({len(common_dates)}) < 滚动窗口({rolling_window}) + 动量窗口({momentum_window})，减少滚动窗口")
        rolling_window = max(12, len(common_dates) - momentum_window - 1)
        print(f"新滚动窗口: {rolling_window}")

    residual_momentum_dict = {}

    for asset_col in monthly_ret.columns:
        asset_ret = monthly_ret[asset_col].dropna()

        if len(asset_ret) < rolling_window + momentum_window:
            continue

        resids = []
        dates = []

        for i in range(rolling_window, len(asset_ret)):
            lookback_end = i
            lookback_start = lookback_end - momentum_window

            if lookback_start < 0:
                continue

            y = asset_ret.iloc[lookback_start:lookback_end].values.reshape(-1, 1)
            X = fac.iloc[lookback_start:lookback_end][['domestic_market', 'stock_pc2', 'stock_pc3',
                                                        'bond_pc2', 'bond_pc3', 'commodity_pc2']].values

            if len(y) < momentum_window or len(X) < momentum_window:
                continue

            try:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(X, y)
                residual = y.flatten() - reg.predict(X).flatten()
                resids.append(residual.sum())
                dates.append(asset_ret.index[i])
            except Exception as e:
                continue

        if resids:
            residual_momentum_dict[asset_col] = pd.Series(resids, index=dates)

    if residual_momentum_dict:
        result = pd.DataFrame(residual_momentum_dict)
        print(f"残差动量计算完成: {result.shape}")
        return result

    return None


def apply_reversal_effect(residual_momentum, monthly_returns, momentum_window=12):
    print("\n应用反转效应改进...")

    if residual_momentum is None or residual_momentum.empty:
        return None

    volatility = monthly_returns.rolling(window=momentum_window).std()

    improved_momentum = residual_momentum.copy()

    for col in residual_momentum.columns:
        if col not in volatility.columns:
            continue

        for date in residual_momentum.index:
            if date not in volatility.index:
                continue

            window_vol = volatility.loc[:date].iloc[-momentum_window:]
            window_resid = residual_momentum.loc[date, col]

            if len(window_vol) < momentum_window or pd.isna(window_resid) or np.isnan(window_resid):
                continue

            try:
                max_vol_idx = window_vol[col].idxmax()
                if not pd.isna(max_vol_idx) and max_vol_idx in residual_momentum.index:
                    max_vol_date_loc = residual_momentum.index.get_loc(max_vol_idx)
                    if max_vol_date_loc < len(residual_momentum.columns):
                        max_vol_col = residual_momentum.columns[max_vol_date_loc]

            except:
                continue

    print("反转效应改进完成")
    return improved_momentum


def calculate_normal_momentum(asset_prices, window=12):
    print(f"\n计算普通动量 (窗口={window}个月)...")

    monthly_returns = calculate_monthly_log_returns(asset_prices)
    momentum = monthly_returns.rolling(window=window).sum()
    momentum = momentum.dropna()
    print(f"普通动量计算完成: {momentum.shape}")

    return momentum


def generate_signals(factor_values, top_n=5):
    print(f"\n生成交易信号 (Top {top_n})...")

    valid_factor = factor_values.dropna(how='all')
    if valid_factor.empty:
        return None

    signals = pd.DataFrame(0.0, index=factor_values.index, columns=factor_values.columns)

    for date in factor_values.index:
        if date not in factor_values.index:
            continue
        row = factor_values.loc[date].dropna()
        if len(row) >= top_n:
            top_assets = row.nlargest(top_n).index
            for asset in top_assets:
                signals.loc[date, asset] = 1.0 / top_n

    signals = signals.shift(1).dropna(how='all')
    signals = signals.loc[~(signals == 0).all(axis=1)]

    if signals.empty:
        return None

    print(f"信号生成完成: {signals.shape}")
    return signals


def run_backtest(prices, signals, initial_capital=1000000.0, fee_rate=0.0):
    print("\n运行回测...")

    if signals is None or signals.empty:
        print("信号为空")
        return None

    common_dates = prices.index.intersection(signals.index)
    if len(common_dates) == 0:
        print("没有共同的日期")
        return None

    prices_aligned = prices.loc[common_dates]
    signals_aligned = signals.loc[common_dates]

    portfolio_value = []
    portfolio_returns = []
    dates = []

    for i in range(len(common_dates)):
        date = common_dates[i]
        prev_date = common_dates[i - 1] if i > 0 else None

        if i == 0:
            portfolio_value.append(initial_capital)
            portfolio_returns.append(0)
            dates.append(date)
            continue

        current_prices = prices_aligned.loc[date]
        prev_prices = prices_aligned.loc[prev_date]

        current_signal = signals_aligned.loc[date]
        prev_signal = signals_aligned.loc[prev_date] if prev_date in signals_aligned.index else None

        if prev_signal is None or current_signal.empty:
            portfolio_value.append(portfolio_value[-1])
            portfolio_returns.append(0)
            dates.append(date)
            continue

        asset_returns = np.log(current_prices / prev_prices).fillna(0)

        portfolio_return = (current_signal * asset_returns).sum()

        if fee_rate > 0 and not current_signal.equals(prev_signal):
            turnover = np.abs(current_signal - prev_signal).sum()
            portfolio_return -= fee_rate * turnover

        portfolio_returns.append(portfolio_return)
        dates.append(date)
        portfolio_value.append(portfolio_value[-1] * np.exp(portfolio_return))

    net_value = pd.Series(portfolio_value, index=dates)
    returns = pd.Series(portfolio_returns, index=dates)

    annual_return = np.exp(returns.mean() * 12) - 1
    annual_volatility = returns.std() * np.sqrt(12)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }

    print(f"回测指标:")
    print(f"  年化收益: {annual_return*100:.2f}%")
    print(f"  年化波动: {annual_volatility*100:.2f}%")
    print(f"  夏普比率: {sharpe_ratio:.2f}")
    print(f"  最大回撤: {max_drawdown*100:.2f}%")
    print(f"  卡玛比率: {calmar_ratio:.2f}")

    return {
        'net_value': net_value,
        'returns': returns,
        'metrics': metrics,
        'drawdown': drawdown
    }


def plot_and_save_results(residual_results, normal_results, benchmark_returns, asset_prices, output_dir):
    print("\n绘制并保存结果...")

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    if residual_results is not None and 'net_value' in residual_results:
        ax1.plot(residual_results['net_value'].index, residual_results['net_value'] / residual_results['net_value'].iloc[0],
                 label='残差动量', linewidth=1.5)
    if normal_results is not None and 'net_value' in normal_results:
        ax1.plot(normal_results['net_value'].index, normal_results['net_value'] / normal_results['net_value'].iloc[0],
                 label='普通动量', linewidth=1.5)
    ax1.set_title('净值曲线')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Net Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if residual_results is not None and 'drawdown' in residual_results:
        ax2.fill_between(residual_results['drawdown'].index, residual_results['drawdown'] * 100, 0, alpha=0.3, label='残差动量')
    if normal_results is not None and 'drawdown' in normal_results:
        ax2.fill_between(normal_results['drawdown'].index, normal_results['drawdown'] * 100, 0, alpha=0.3, label='普通动量')
    ax2.set_title('回撤分析')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if residual_results is not None and 'returns' in residual_results:
        monthly_ret = residual_results['returns'] * 100
        ax3.hist(monthly_ret.dropna(), bins=50, alpha=0.7, edgecolor='black', label='残差动量')
        ax3.axvline(monthly_ret.mean(), color='red', linestyle='--', label=f'Mean: {monthly_ret.mean():.2f}%')
    ax3.set_title('月度收益分布')
    ax3.set_xlabel('Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    metrics_labels = ['年化收益', '夏普比率', '最大回撤(%)', '卡玛比率']
    x = np.arange(len(metrics_labels))
    width = 0.35

    if residual_results is not None:
        res_metrics = residual_results['metrics']
        res_values = [res_metrics['annual_return'] * 100, res_metrics['sharpe_ratio'],
                      res_metrics['max_drawdown'] * 100, res_metrics['calmar_ratio']]
        ax4.bar(x - width / 2, res_values, width, label='残差动量', alpha=0.8)

    if normal_results is not None:
        norm_metrics = normal_results['metrics']
        norm_values = [norm_metrics['annual_return'] * 100, norm_metrics['sharpe_ratio'],
                       norm_metrics['max_drawdown'] * 100, norm_metrics['calmar_ratio']]
        ax4.bar(x + width / 2, norm_values, width, label='普通动量', alpha=0.8)

    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_labels)
    ax4.set_title('业绩指标对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'backtest_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {os.path.join(output_dir, 'backtest_results.png')}")


def save_results_to_excel(residual_results, normal_results, output_dir):
    print("\n保存结果到Excel...")

    os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(os.path.join(output_dir, 'backtest_results.xlsx'), engine='openpyxl') as writer:
        if residual_results:
            pd.DataFrame({'net_value': residual_results['net_value']}).to_excel(writer, sheet_name='残差动量_净值')
            pd.DataFrame({'returns': residual_results['returns']}).to_excel(writer, sheet_name='残差动量_收益')
            pd.DataFrame([residual_results['metrics']]).to_excel(writer, sheet_name='残差动量_指标', index=False)

        if normal_results:
            pd.DataFrame({'net_value': normal_results['net_value']}).to_excel(writer, sheet_name='普通动量_净值')
            pd.DataFrame({'returns': normal_results['returns']}).to_excel(writer, sheet_name='普通动量_收益')
            pd.DataFrame([normal_results['metrics']]).to_excel(writer, sheet_name='普通动量_指标', index=False)

    print(f"Excel已保存: {os.path.join(output_dir, 'backtest_results.xlsx')}")


def main():
    print("=" * 60)
    print("行业残差动量定价能力复现 - 主程序")
    print("=" * 60)

    data = load_csv_data()

    if 'industry' not in data or data['industry'].empty:
        print("错误: 没有可用的行业数据")
        return None

    if 'bond' not in data or 'commodity' not in data:
        print("错误: 缺少债券或商品数据")
        return None

    asset_prices = data['industry'].resample('M').last()
    asset_prices = asset_prices.dropna(axis=1, how='all')

    valid_assets = asset_prices.notna().sum() >= 60
    asset_prices = asset_prices.loc[:, valid_assets]

    bond_prices = data['bond'].resample('M').last()
    commodity_prices = data['commodity'].resample('M').last()

    print(f"\n使用资产数量: {len(asset_prices.columns)}")
    print(f"数据时间范围: {asset_prices.index[0]} 到 {asset_prices.index[-1]}")

    rolling_window = min(100, len(asset_prices) // 3)
    momentum_window = 12

    print(f"\n使用滚动窗口: {rolling_window}, 动量窗口: {momentum_window}")

    print("\n" + "=" * 60)
    print("第一步: 计算市场因子和风格因子 (6因子模型)")
    print("=" * 60)
    factors = calculate_pca_factors_multi_market(asset_prices, bond_prices, commodity_prices, window=rolling_window)

    print("\n" + "=" * 60)
    print("第二步: 计算残差动量")
    print("=" * 60)
    residual_momentum = calculate_residual_momentum(asset_prices, factors, rolling_window=rolling_window, momentum_window=momentum_window)

    monthly_returns = calculate_monthly_log_returns(asset_prices)

    print("\n" + "=" * 60)
    print("第三步: 应用反转效应改进")
    print("=" * 60)
    if residual_momentum is not None and not residual_momentum.empty:
        improved_momentum = apply_reversal_effect(residual_momentum, monthly_returns, momentum_window=momentum_window)

    print("\n" + "=" * 60)
    print("第四步: 计算普通动量 (对比)")
    print("=" * 60)
    normal_momentum = calculate_normal_momentum(asset_prices, window=momentum_window)

    print("\n" + "=" * 60)
    print("第五步: 生成交易信号")
    print("=" * 60)

    residual_signals = None
    normal_signals = None

    if residual_momentum is not None and not residual_momentum.empty:
        residual_signals = generate_signals(residual_momentum, top_n=5)

    if normal_momentum is not None and not normal_momentum.empty:
        normal_signals = generate_signals(normal_momentum, top_n=5)

    print("\n" + "=" * 60)
    print("第六步: 运行回测")
    print("=" * 60)

    residual_results = None
    normal_results = None

    if residual_signals is not None and not residual_signals.empty:
        residual_results = run_backtest(asset_prices, residual_signals, initial_capital=1000000.0, fee_rate=0.0)
    else:
        print("残差动量信号为空，跳过")

    if normal_signals is not None and not normal_signals.empty:
        normal_results = run_backtest(asset_prices, normal_signals, initial_capital=1000000.0, fee_rate=0.0)
    else:
        print("普通动量信号为空，跳过")

    if residual_results is None and normal_results is None:
        print("没有可用的回测结果")
        return None

    print("\n" + "=" * 60)
    print("第七步: 保存结果")
    print("=" * 60)

    plot_and_save_results(residual_results, normal_results, None, asset_prices, OUTPUT_DIR)
    save_results_to_excel(residual_results, normal_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("复现完成!")
    print("=" * 60)

    return {
        'residual_results': residual_results,
        'normal_results': normal_results,
        'factors': factors,
        'residual_momentum': residual_momentum,
        'normal_momentum': normal_momentum
    }


if __name__ == '__main__':
    results = main()