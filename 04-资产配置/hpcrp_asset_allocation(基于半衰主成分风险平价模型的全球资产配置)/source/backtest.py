"""
回测引擎模块
实现资产配置策略的回测
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def run_backtest(returns, weights_func, model_name, rebalance_freq='quarterly', 
               window=240, start_date=None, end_date=None):
    """
    运行回测
    
    Args:
        returns: 收益率数据 (DataFrame)
        weights_func: 获取权重的函数 (model_name, returns_window) -> weights
        model_name: 模型名称
        rebalance_freq: 调仓频率 ('monthly', 'quarterly', 'yearly')
        window: 协方差估计窗口
        start_date: 回测开始日期
        end_date: 回测结束日期
    
    Returns:
        dict: 回测结果
    """
    # 筛选日期范围
    if start_date:
        returns = returns.loc[returns.index >= start_date]
    if end_date:
        returns = returns.loc[returns.index <= end_date]
    
    # 获取调仓日期
    if rebalance_freq == 'monthly':
        rebal_dates = returns.resample('M').last().index.tolist()
    elif rebalance_freq == 'quarterly':
        # 使用'Q'代替'QE'以兼容旧版pandas
        rebal_dates = returns.resample('Q').last().index.tolist()
    elif rebalance_freq == 'yearly':
        rebal_dates = returns.resample('Y').last().index.tolist()
    else:
        raise ValueError(f"不支持的频率: {rebalance_freq}")
    
    # 过滤调仓日期 (需要确保有足够的历史数据)
    valid_rebal_dates = [d for d in rebal_dates if d >= returns.index[window]]
    
    # 记录组合收益率和权重
    portfolio_returns = []
    weights_history = []
    dates_history = []
    
    current_weights = None
    
    for i, date in enumerate(returns.index):
        # 调仓日
        if date in valid_rebal_dates:
            # 获取历史窗口数据
            hist_data = returns.loc[returns.index < date].iloc[-window:]
            
            # 计算新权重
            current_weights = weights_func(model_name, hist_data)
            weights_history.append(current_weights)
            dates_history.append(date)
        
        # 计算组合收益
        if current_weights is not None:
            ret = (returns.loc[date].values * current_weights).sum()
            portfolio_returns.append(ret)
        else:
            portfolio_returns.append(0)
    
    # 构建结果DataFrame
    result = pd.DataFrame({
        ' returns': portfolio_returns
    }, index=returns.index)
    
    # 计算净值
    result['nav'] = (1 + result[' returns']).cumprod()
    
    return {
        'returns': result[' returns'],
        'nav': result['nav'],
        'weights': weights_history,
        'rebal_dates': dates_history,
        'model': model_name
    }


def calculate_metrics(returns, nav):
    """
    计算回测指标
    
    Args:
        returns: 组合收益率序列
        nav: 净值序列
    
    Returns:
        dict: 指标字典
    """
    # 年化收益率
    n_years = len(returns) / 252
    cumulative_return = nav.iloc[-1] - 1 if nav.iloc[-1] != 0 else nav.iloc[-1]
    annual_return = (1 + cumulative_return) ** (1 / n_years) - 1
    
    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252)
    
    # 夏普比率 (假设无风险利率为3%)
    risk_free = 0.03
    sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
    
    # 最大回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Calmar比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'cumulative_return': cumulative_return
    }


def run_multi_model_backtest(returns, models, rebalance_freq='quarterly', 
                        window=240, start_date=None, end_date=None):
    """
    运行多个模型的回测
    
    Args:
        returns: 收益率数据
        models: 模型名称列表
        rebalance_freq: 调仓频率
        window: 窗口大小
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        dict: 模型名称 -> 回测结果
    """
    from source.models import get_model_weights
    
    results = {}
    
    for model in models:
        print(f"运行 {model} 模型回测...")
        
        # 定义权重函数
        def weights_func(model_name, hist_data):
            if model_name == 'HPCRP':
                return get_model_weights(model_name, hist_data, half_life=120)
            else:
                return get_model_weights(model_name, hist_data)
        
        # 运行回测
        result = run_backtest(
            returns, weights_func, model,
            rebalance_freq=rebalance_freq,
            window=window,
            start_date=start_date,
            end_date=end_date
        )
        
        # 计算指标
        metrics = calculate_metrics(result['returns'], result['nav'])
        result['metrics'] = metrics
        
        results[model] = result
    
    return results


def compare_results(results):
    """
    比较回测结果
    
    Args:
        results: 多模型回测结果
    
    Returns:
        pd.DataFrame: 比较表格
    """
    rows = []
    
    for model, result in results.items():
        metrics = result['metrics']
        rows.append({
            'Model': model,
            'Annual Return': f"{metrics['annual_return']:.2%}",
            'Annual Vol': f"{metrics['annual_vol']:.2%}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
            'Calmar Ratio': f"{metrics['calmar_ratio']:.3f}"
        })
    
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    np.random.seed(42)
    
    test_returns = pd.DataFrame(
        np.random.randn(1000, 4) * 0.01,
        index=pd.date_range('2010-01-01', periods=1000, freq='D'),
        columns=['A', 'B', 'C', 'D']
    )
    
    from source.models import get_model_weights
    
    def wf(model, hist):
        if model == 'HPCRP':
            return get_model_weights(model, hist, half_life=120)
        return get_model_weights(model, hist)
    
    result = run_backtest(test_returns, wf, 'RP', rebalance_freq='quarterly', window=240)
    metrics = calculate_metrics(result['returns'], result['nav'])
    
    print("回测指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")