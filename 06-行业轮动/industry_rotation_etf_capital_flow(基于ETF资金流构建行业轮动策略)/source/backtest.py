import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.0,
        slippage: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}

        self.trade_history = []
        self.equity_curve = []

    def reset(self):
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []

    def execute_trade(
        self,
        date: pd.Timestamp,
        symbol: str,
        signal: int,
        price: float,
        quantity: int = 0,
    ):
        if signal > 0:
            trade_type = "buy"
            cost = price * quantity * (1 + self.commission_rate + self.slippage)
            if cost <= self.cash:
                self.cash -= cost
                if symbol not in self.positions:
                    self.positions[symbol] = {"quantity": 0, "avg_price": 0}
                old_quantity = self.positions[symbol]["quantity"]
                old_value = old_quantity * self.positions[symbol]["avg_price"]
                new_value = old_value + price * quantity
                new_quantity = old_quantity + quantity
                self.positions[symbol]["avg_price"] = new_value / new_quantity if new_quantity > 0 else 0
                self.positions[symbol]["quantity"] = new_quantity

        elif signal < 0:
            trade_type = "sell"
            if symbol in self.positions and self.positions[symbol]["quantity"] >= quantity:
                proceeds = price * quantity * (1 - self.commission_rate - self.slippage)
                self.cash += proceeds
                self.positions[symbol]["quantity"] -= quantity
                if self.positions[symbol]["quantity"] == 0:
                    del self.positions[symbol]

        self.trade_history.append(
            {
                "date": date,
                "symbol": symbol,
                "type": trade_type if signal != 0 else "hold",
                "price": price,
                "quantity": quantity,
            }
        )

    def update_portfolio_value(self, date: pd.Timestamp, prices: Dict[str, float]):
        total_value = self.cash
        for symbol, pos in self.positions.items():
            if symbol in prices:
                total_value += pos["quantity"] * prices[symbol]
            else:
                total_value += pos["quantity"] * pos["avg_price"]

        self.portfolio_value = total_value
        self.equity_curve.append(
            {"date": date, "portfolio_value": total_value, "cash": self.cash}
        )

    def get_equity_curve(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)


class StrategyBacktester:
    def __init__(
        self,
        strategy,
        data_provider,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.0,
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
        )

    def run(
        self,
        start_date: str,
        end_date: str,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_prices: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        self.backtest_engine.reset()

        signals = signals.copy()
        signals["date"] = pd.to_datetime(signals["date"])

        prices = prices.copy()
        if "trade_date" in prices.columns:
            prices.rename(columns={"trade_date": "date"}, inplace=True)
        prices["date"] = pd.to_datetime(prices["date"])

        dates = sorted(signals["date"].unique())

        portfolio_returns = []

        for i, date in enumerate(dates):
            date_signals = signals[signals["date"] == date]

            date_prices = prices[prices["date"] == date]

            if len(date_prices) == 0:
                continue

            price_dict = dict(zip(date_prices["industry"], date_prices["close"]))

            for _, signal_row in date_signals.iterrows():
                industry = signal_row["industry"]
                signal_value = signal_row["signal"]

                if industry in price_dict:
                    current_price = price_dict[industry]

                    if signal_value > 0:
                        allocation = self.backtest_engine.portfolio_value * 0.1
                        quantity = int(allocation / current_price)
                        if quantity > 0:
                            self.backtest_engine.execute_trade(
                                date, industry, signal_value, current_price, quantity
                            )

                    elif signal_value < 0:
                        if industry in self.backtest_engine.positions:
                            quantity = self.backtest_engine.positions[industry]["quantity"]
                            self.backtest_engine.execute_trade(
                                date, industry, signal_value, current_price, quantity
                            )

            self.backtest_engine.update_portfolio_value(date, price_dict)

            if i > 0:
                prev_date = dates[i - 1]
                prev_prices = prices[prices["date"] == prev_date]
                curr_prices = prices[prices["date"] == date]

                if len(prev_prices) > 0 and len(curr_prices) > 0:
                    prev_price_dict = dict(zip(prev_prices["industry"], prev_prices["close"]))
                    curr_price_dict = dict(zip(curr_prices["industry"], curr_prices["close"]))

                    held_industries = list(self.backtest_engine.positions.keys())
                    if held_industries:
                        returns = []
                        for ind in held_industries:
                            if ind in prev_price_dict and ind in curr_price_dict:
                                ret = (curr_price_dict[ind] - prev_price_dict[ind]) / prev_price_dict[ind]
                                returns.append(ret)

                        if returns:
                            avg_return = np.mean(returns) * 100
                        else:
                            avg_return = 0
                    else:
                        avg_return = 0

                    portfolio_returns.append(
                        {
                            "date": date,
                            "return": avg_return,
                            "num_positions": len(self.backtest_engine.positions),
                        }
                    )

        equity_curve = self.backtest_engine.get_equity_curve()
        portfolio_returns_df = pd.DataFrame(portfolio_returns)

        results = {
            "equity_curve": equity_curve,
            "portfolio_returns": portfolio_returns_df,
            "trade_history": pd.DataFrame(self.backtest_engine.trade_history),
            "final_value": self.backtest_engine.portfolio_value,
            "total_return": (
                self.backtest_engine.portfolio_value / self.initial_capital - 1
            )
            * 100,
        }

        if benchmark_prices is not None and len(portfolio_returns_df) > 0:
            benchmark_returns = self._calculate_benchmark_returns(
                benchmark_prices, portfolio_returns_df
            )
            results["benchmark_returns"] = benchmark_returns
            results["excess_returns"] = (
                results["portfolio_returns"]["return"] - benchmark_returns
            )

        return results

    def _calculate_benchmark_returns(
        self, benchmark_prices: pd.DataFrame, portfolio_returns: pd.DataFrame
    ) -> pd.Series:
        if "trade_date" in benchmark_prices.columns:
            benchmark_prices = benchmark_prices.copy()
            benchmark_prices.rename(columns={"trade_date": "date"}, inplace=True)

        benchmark_prices["date"] = pd.to_datetime(benchmark_prices["date"])
        benchmark_prices = benchmark_prices.sort_values("date")

        benchmark_prices["return"] = benchmark_prices["close"].pct_change() * 100

        merged = portfolio_returns[["date"]].merge(
            benchmark_prices[["date", "return"]], on="date", how="left"
        )

        return merged["return"].fillna(0)

    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        equity_curve = results.get("equity_curve", pd.DataFrame())
        portfolio_returns = results.get("portfolio_returns", pd.DataFrame())

        if len(equity_curve) == 0 or len(portfolio_returns) == 0:
            return {}

        equity_curve = equity_curve.copy()
        equity_curve["date"] = pd.to_datetime(equity_curve["date"])
        portfolio_returns = portfolio_returns.copy()
        portfolio_returns["date"] = pd.to_datetime(portfolio_returns["date"])

        equity_curve = equity_curve.sort_values("date")
        portfolio_returns = portfolio_returns.sort_values("date")

        cumulative = equity_curve["portfolio_value"] / self.initial_capital

        total_return = cumulative.iloc[-1] - 1
        n_days = len(cumulative)
        n_years = n_days / 252

        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        daily_returns = equity_curve["portfolio_value"].pct_change().dropna()
        annual_vol = daily_returns.std() * np.sqrt(252)

        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        sharpe = (
            annual_return / annual_vol if annual_vol > 0 else 0
        )

        calmar = (
            annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        )

        returns_series = portfolio_returns["return"].dropna() / 100
        positive_count = (returns_series > 0).sum()
        total_count = len(returns_series)
        win_rate = positive_count / total_count if total_count > 0 else 0

        avg_win = returns_series[returns_series > 0].mean() if (returns_series > 0).any() else 0
        avg_loss = returns_series[returns_series < 0].mean() if (returns_series < 0).any() else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        metrics = {
            "total_return": total_return * 100,
            "annual_return": annual_return * 100,
            "annual_vol": annual_vol * 100,
            "max_drawdown": max_drawdown * 100,
            "sharpe": sharpe,
            "calmar": calmar,
            "win_rate": win_rate * 100,
            "profit_loss_ratio": profit_loss_ratio,
            "num_trades": len(results.get("trade_history", [])),
        }

        return metrics


def plot_equity_curve(
    results: Dict[str, Any],
    benchmark_results: Optional[Dict[str, Any]] = None,
    title: str = "策略净值曲线",
    save_path: Optional[str] = None,
):
    equity_curve = results.get("equity_curve", pd.DataFrame())

    if len(equity_curve) == 0:
        print("无净值数据可绘图")
        return

    equity_curve = equity_curve.copy()
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    equity_curve = equity_curve.sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        equity_curve["date"],
        equity_curve["portfolio_value"] / equity_curve["portfolio_value"].iloc[0],
        label="策略",
        linewidth=2,
    )

    if benchmark_results is not None:
        benchmark_curve = benchmark_results.get("equity_curve", pd.DataFrame())
        if len(benchmark_curve) > 0:
            benchmark_curve = benchmark_curve.copy()
            benchmark_curve["date"] = pd.to_datetime(benchmark_curve["date"])
            benchmark_curve = benchmark_curve.sort_values("date")
            ax.plot(
                benchmark_curve["date"],
                benchmark_curve["portfolio_value"]
                / benchmark_curve["portfolio_value"].iloc[0],
                label="基准",
                linewidth=1.5,
                alpha=0.7,
            )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("净值", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存至 {save_path}")

    plt.show()


def plot_drawdown(
    results: Dict[str, Any],
    title: str = "回撤曲线",
    save_path: Optional[str] = None,
):
    equity_curve = results.get("equity_curve", pd.DataFrame())

    if len(equity_curve) == 0:
        print("无净值数据可绘图")
        return

    equity_curve = equity_curve.copy()
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    equity_curve = equity_curve.sort_values("date")

    cumulative = equity_curve["portfolio_value"] / equity_curve["portfolio_value"].iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
    ax.plot(drawdown.index, drawdown, color="red", linewidth=1)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("回撤 (%)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存至 {save_path}")

    plt.show()


def generate_backtest_report(
    results: Dict[str, Any],
    metrics: Dict[str, float],
    strategy_name: str = "ETF行业轮动策略",
) -> str:
    report = f"""
{'='*60}
{strategy_name} - 回测报告
{'='*60}

一、收益概览
-----------------------------------
总收益率:     {metrics.get('total_return', 0):.2f}%
年化收益率:   {metrics.get('annual_return', 0):.2f}%
年化波动率:   {metrics.get('annual_vol', 0):.2f}%
最大回撤:     {metrics.get('max_drawdown', 0):.2f}%

二、风险收益指标
-----------------------------------
夏普比率:     {metrics.get('sharpe', 0):.2f}
卡尔玛比率:   {metrics.get('calmar', 0):.2f}
盈亏比:       {metrics.get('profit_loss_ratio', 0):.2f}

三、交易统计
-----------------------------------
胜率:         {metrics.get('win_rate', 0):.2f}%
总交易次数:   {metrics.get('num_trades', 0)}

四、期末状态
-----------------------------------
初始资金:     {results.get('equity_curve', pd.DataFrame()).iloc[0]['portfolio_value'] if len(results.get('equity_curve', pd.DataFrame())) > 0 else 0:.2f}
期末资金:     {results.get('final_value', 0):.2f}

{'='*60}
"""
    return report


if __name__ == "__main__":
    print("回测引擎模块测试...")

    engine = BacktestEngine(initial_capital=1000000)

    print(f"初始资金: {engine.initial_capital}")
    print(f"当前资金: {engine.cash}")

    print("\n回测引擎模块测试完成")
