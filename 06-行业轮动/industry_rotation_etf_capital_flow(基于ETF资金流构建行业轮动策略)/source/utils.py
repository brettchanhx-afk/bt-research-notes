import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))


def ensure_dir(directory: Union[str, Path]) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_number(num: float, decimal_places: int = 2) -> str:
    return f"{num:.{decimal_places}f}"


def format_percent(num: float, decimal_places: int = 2) -> str:
    return f"{num:.{decimal_places}f}%"


def calculate_rolling_percentile(
    series: pd.Series,
    window: int,
    min_periods: int = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = window // 2

    rolling_data = series.rolling(window=window, min_periods=min_periods)

    percentile = rolling_data.apply(
        lambda x: (x < x[-1]).sum() / len(x) if len(x) > 0 else np.nan,
        raw=False,
    )

    return percentile


def get_trading_dates(
    start_date: str,
    end_date: str,
    freq: str = "D",
) -> List[pd.Timestamp]:
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    trading_dates = [
        d for d in dates if d.weekday() < 5
    ]

    return trading_dates


def get_week_boundaries(
    start_date: str,
    end_date: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    weeks = {}
    for date in dates:
        week_num = date.isocalendar()[1]
        year = date.year
        key = (year, week_num)

        if key not in weeks:
            weeks[key] = {"start": date, "end": date}
        else:
            weeks[key]["end"] = date

    week_list = [(v["start"], v["end"]) for v in weeks.values()]
    week_list.sort(key=lambda x: x[0])

    return week_list


def merge_on_date_range(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_date_col: str = "date",
    right_date_col: str = "date",
    left_industries: Optional[List[str]] = None,
    right_industries: Optional[List[str]] = None,
) -> pd.DataFrame:
    if left_industries is not None:
        left_df = left_df[left_df["industry"].isin(left_industries)]
    if right_industries is not None:
        right_df = right_df[right_df["industry"].isin(right_industries)]

    merged = pd.merge_asof(
        left_df.sort_values(left_date_col),
        right_df.sort_values(right_date_col),
        left_on=left_date_col,
        right_on=right_date_col,
        by="industry" if "industry" in left_df.columns and "industry" in right_df.columns else None,
        direction="backward",
    )

    return merged


def calculate_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    if len(returns) == 0:
        return {}

    returns = returns.dropna()

    total_return = (1 + returns / 100).prod() - 1
    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    annual_vol = returns.std() * np.sqrt(periods_per_year)

    cumulative = (1 + returns / 100).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    sharpe = (
        (annual_return - risk_free_rate) / annual_vol
        if annual_vol > 0
        else 0
    )

    calmar = (
        annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    )

    positive_count = (returns > 0).sum()
    total_count = len(returns)
    win_rate = positive_count / total_count if total_count > 0 else 0

    avg_positive = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_negative = returns[returns < 0].mean() if (returns < 0).any() else 0
    profit_loss_ratio = (
        abs(avg_positive / avg_negative) if avg_negative != 0 else np.inf
    )

    return {
        "total_return": total_return * 100,
        "annual_return": annual_return * 100,
        "annual_vol": annual_vol * 100,
        "max_drawdown": max_drawdown * 100,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate * 100,
        "profit_loss_ratio": profit_loss_ratio,
        "num_periods": n_periods,
    }


def resample_returns(
    returns: pd.Series,
    freq: str = "M",
) -> pd.Series:
    if len(returns) == 0:
        return pd.Series()

    df = pd.DataFrame({"return": returns})
    resampled = df.resample(freq).apply(
        lambda x: (1 + x / 100).prod() - 1
    )["return"] * 100

    return resampled


def create_tear_sheet(
    results: Dict[str, Any],
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "策略绩效报告",
) -> Dict[str, Any]:
    portfolio_returns = results.get("portfolio_returns", pd.Series())

    if isinstance(portfolio_returns, pd.DataFrame):
        if "return" in portfolio_returns.columns:
            portfolio_returns = portfolio_returns["return"]
        elif len(portfolio_returns) > 0:
            portfolio_returns = portfolio_returns.iloc[:, 0]

    metrics = calculate_performance_metrics(portfolio_returns)

    if benchmark_returns is not None:
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        info_ratio = (
            excess_returns.mean() * 252 / tracking_error
            if tracking_error > 0
            else 0
        )
        metrics["tracking_error"] = tracking_error
        metrics["info_ratio"] = info_ratio

    return {
        "metrics": metrics,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
    }


def save_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "strategy",
) -> Dict[str, str]:
    output_dir = ensure_dir(output_dir)

    saved_files = {}

    if "equity_curve" in results and len(results["equity_curve"]) > 0:
        equity_path = output_dir / f"{prefix}_equity_curve.csv"
        results["equity_curve"].to_csv(equity_path, index=False)
        saved_files["equity_curve"] = str(equity_path)

    if "portfolio_returns" in results and len(results["portfolio_returns"]) > 0:
        returns_path = output_dir / f"{prefix}_returns.csv"
        df = results["portfolio_returns"]
        if isinstance(df, pd.DataFrame):
            df.to_csv(returns_path, index=False)
        else:
            df.to_csv(returns_path)
        saved_files["portfolio_returns"] = str(returns_path)

    if "trade_history" in results and len(results["trade_history"]) > 0:
        trades_path = output_dir / f"{prefix}_trades.csv"
        results["trade_history"].to_csv(trades_path, index=False)
        saved_files["trade_history"] = str(trades_path)

    if "performance_metrics" in results:
        metrics_path = output_dir / f"{prefix}_metrics.csv"
        metrics_df = pd.DataFrame([results["performance_metrics"]])
        metrics_df.to_csv(metrics_path, index=False)
        saved_files["performance_metrics"] = str(metrics_path)

    return saved_files


def load_data_cache(
    data_dir: Union[str, Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)

    cache_files = {
        "index_returns": data_dir / "index_returns.csv",
        "etf_flow": data_dir / "etf_net_flow.pkl",
    }

    data = {}

    if cache_files["index_returns"].exists():
        df = pd.read_csv(cache_files["index_returns"])
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            if start_date:
                df = df[df["trade_date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["trade_date"] <= pd.to_datetime(end_date)]
        data["index_returns"] = df

    if cache_files["etf_flow"].exists():
        try:
            etf_flow = pd.read_pickle(cache_files["etf_flow"])
            data["etf_flow"] = etf_flow
        except Exception as e:
            print(f"加载ETF资金流缓存失败: {e}")

    return data


if __name__ == "__main__":
    print("工具模块测试...")

    print("\n1. 测试目录创建...")
    test_dir = Path(__file__).parent.parent / "test_output"
    created_dir = ensure_dir(test_dir)
    print(f"测试目录: {created_dir}")

    print("\n2. 测试百分比格式化...")
    print(f"格式化 0.1234: {format_percent(0.1234)}")

    print("\n3. 测试交易日生成...")
    dates = get_trading_dates("2024-01-01", "2024-01-31")
    print(f"2024年1月交易日数量: {len(dates)}")

    print("\n工具模块测试完成")
