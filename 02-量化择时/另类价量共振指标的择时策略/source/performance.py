import numpy as np
import pandas as pd
import empyrical as ep

def _adjust_returns(returns, adjustment_factor):

    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns.copy()
    return returns - adjustment_factor


def information_ratio(returns, factor_returns):

    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error


# 风险指标
def strategy_performance(returns: pd.DataFrame,
                         mark_benchmark: str = 'benchmark',
                         periods: str = 'daily') -> pd.DataFrame:
 
    df: pd.DataFrame = pd.DataFrame()

    df['年化收益率'] = ep.annual_return(returns, period=periods)

    df['累计收益'] = returns.apply(lambda x: ep.cum_returns(x).iloc[-1])

    df['波动率'] = returns.apply(
        lambda x: ep.annual_volatility(x, period=periods))

    df['夏普'] = returns.apply(ep.sharpe_ratio, period=periods)

    df['最大回撤'] = returns.apply(lambda x: ep.max_drawdown(x))

    df['索提诺比率'] = returns.apply(lambda x: ep.sortino_ratio(x, period=periods))

    df['Calmar'] = returns.apply(lambda x: ep.calmar_ratio(x, period=periods))

    # 相对指标计算
    if mark_benchmark in returns.columns:

        select_col = [col for col in returns.columns if col != mark_benchmark]
        df['IR'] = returns[select_col].apply(
            lambda x: information_ratio(x, returns[mark_benchmark]))

        df['Alpha'] = returns[select_col].apply(
            lambda x: ep.alpha(x, returns[mark_benchmark], period=periods))

        df['Beta'] = returns[select_col].apply(
            lambda x: ep.beta(x, returns[mark_benchmark]))

        # 计算相对年化波动率
        df['超额收益率'] = df['年化收益率'] - \
            df.loc[mark_benchmark, '年化收益率']

    return df.T
