import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def calculate_returns(prices, method='simple'):
    if method == 'log':
        return np.log(prices / prices.shift(1))
    elif method == 'simple':
        return prices.pct_change()
    elif method == 'yoy':
        return prices.pct_change(periods=12)
    elif method == 'mom':
        return prices.pct_change(periods=1)
    return prices.pct_change()

def calculate_performance_metrics(returns, risk_free_rate=0.0):
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    else:
        returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {}

    annual_return = returns.mean() * 12
    annual_vol = returns.std() * np.sqrt(12)

    excess_returns = returns - risk_free_rate / 12
    sharpe_ratio = (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12)) if excess_returns.std() != 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    positive_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = positive_days / total_days if total_days > 0 else 0

    metrics = {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'total_return': cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0,
        'win_rate': win_rate,
        'mean_return': returns.mean(),
        'std_return': returns.std()
    }

    return metrics

def plot_net_value(net_values, labels=None, title='Net Value', save_path=None):
    if isinstance(net_values, pd.DataFrame):
        if labels is None:
            labels = net_values.columns.tolist()
        for col in net_values.columns:
            plt.plot(net_values.index, net_values[col], label=col)
    elif isinstance(net_values, pd.Series):
        plt.plot(net_values.index, net_values.values, label=labels if labels else 'Strategy')
    else:
        return

    plt.xlabel('Date')
    plt.ylabel('Net Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_cumulative_returns(cumulative_returns, labels=None, title='Cumulative Returns', save_path=None):
    if isinstance(cumulative_returns, pd.DataFrame):
        if labels is None:
            labels = cumulative_returns.columns.tolist()
        for col in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, (cumulative_returns[col] - 1) * 100, label=col)
    elif isinstance(cumulative_returns, pd.Series):
        plt.plot(cumulative_returns.index, (cumulative_returns - 1) * 100, label=labels if labels else 'Strategy')

    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_drawdown(drawdown, label='Drawdown', title='Drawdown', save_path=None):
    if isinstance(drawdown, pd.Series):
        plt.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, label=label)
        plt.plot(drawdown.index, drawdown * 100, label=label)
    else:
        return

    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_ic_analysis(ic_series, title='IC Analysis', save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ic_cumsum = ic_series.cumsum()

    axes[0].bar(ic_series.index, ic_series, alpha=0.5)
    axes[0].axhline(ic_series.mean(), color='red', linestyle='--', label=f'Mean IC: {ic_series.mean():.4f}')
    axes[0].set_ylabel('IC')
    axes[0].set_title(f'{title} - IC Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ic_cumsum.index, ic_cumsum)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative IC')
    axes[1].set_title(f'{title} - Cumulative IC')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_quantile_returns(quantile_returns, title='Quantile Returns', save_path=None):
    if isinstance(quantile_returns, pd.DataFrame):
        mean_returns = quantile_returns.mean() * 100
        std_returns = quantile_returns.std() * 100

        plt.bar(range(len(mean_returns)), mean_returns, yerr=std_returns, alpha=0.7, capsize=5)
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Quantile')
        plt.ylabel('Mean Return (%)')
        plt.title(title)
        plt.xticks(range(len(mean_returns)), [f'Q{i+1}' for i in range(len(mean_returns))])
        plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def resample_to_monthly(data, method='last'):
    if method == 'last':
        return data.resample('M').last()
    elif method == 'mean':
        return data.resample('M').mean()
    elif method == 'sum':
        return data.resample('M').sum()
    return data.resample('M').last()

def align_dataframes(*dfs, how='inner'):
    if len(dfs) == 0:
        return []

    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    aligned = [df.loc[common_idx] for df in dfs]
    return aligned

def format_performance_metrics(metrics):
    formatted = []
    formatted.append(f"年化收益: {metrics['annual_return']*100:.2f}%")
    formatted.append(f"年化波动: {metrics['annual_volatility']*100:.2f}%")
    formatted.append(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    formatted.append(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    formatted.append(f"卡玛比率: {metrics['calmar_ratio']:.2f}")
    formatted.append(f"总收益: {metrics['total_return']*100:.2f}%")
    return "\n".join(formatted)

def save_results_to_excel(results_dict, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for name, result in results_dict.items():
            if 'net_value' in result:
                result['net_value'].to_excel(writer, sheet_name=f'{name[:20]}_nav')
            if 'returns' in result:
                pd.DataFrame(result['returns']).to_excel(writer, sheet_name=f'{name[:20]}_returns')
            if 'metrics' in result:
                pd.DataFrame([result['metrics']]).to_excel(writer, sheet_name=f'{name[:20]}_metrics')

def create_summary_table(results_dict):
    summary_data = []
    for name, result in results_dict.items():
        if 'metrics' in result:
            metrics = result['metrics']
            summary_data.append({
                'Strategy': name,
                'Annual Return': f"{metrics['annual_return']*100:.2f}%",
                'Annual Vol': f"{metrics['annual_volatility']*100:.2f}%",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Max DD': f"{metrics['max_drawdown']*100:.2f}%",
                'Calmar': f"{metrics['calmar_ratio']:.2f}",
                'Total Return': f"{metrics['total_return']*100:.2f}%"
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df