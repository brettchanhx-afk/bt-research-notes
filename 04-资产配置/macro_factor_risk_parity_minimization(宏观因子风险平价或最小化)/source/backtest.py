"""
Backtest engine for macro risk parity and risk minimization strategies.
Uses provided CSV data files.
"""
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

from source.config import (
    BACKTEST_START,
    BACKTEST_END,
    FACTOR_NAMES,
    OUTPUT_DIR,
)
from source.data_loader import (
    load_asset_prices,
    load_high_freq_macro_factors,
    calculate_returns,
    resample_to_monthly,
)
from source.macro_factors import (
    build_mimicking_factors,
    compute_factor_exposures,
    compute_factor_covariance_rolling,
)
from source.optimization import (
    asset_risk_parity,
    macro_risk_parity,
    macro_risk_minimization,
)
from source.risk_attribution import compute_portfolio_factor_risk_contribution
from source.performance import (
    compute_performance_metrics,
    compute_cumulative_returns,
    compute_turnover,
    print_performance_summary,
    compare_strategies,
)


class MacroRiskBacktester:
    def __init__(self, start=BACKTEST_START, end=BACKTEST_END):
        self.start = start
        self.end = end
        self.asset_prices_daily = None
        self.asset_returns_daily = None
        self.asset_returns_monthly = None
        self.factor_values_daily = None
        self.factor_returns_daily = None
        self.factor_returns_monthly = None
        self.factor_cov_history = None
        self.residual_var_history = None
        self.betas_dict = None
        self.betas_array = None
        self.strategy_results = {}
        self.asset_names = None

    def load_data(self):
        print("\n" + "=" * 60)
        print("Step 1: Loading market data from CSV files...")
        print("=" * 60)

        self.asset_prices_daily = load_asset_prices()
        self.asset_returns_daily = self.asset_prices_daily.pct_change().dropna()
        self.asset_returns_daily = self.asset_returns_daily.replace([np.inf, -np.inf], np.nan)

        self.asset_returns_monthly = resample_to_monthly(self.asset_prices_daily)
        self.asset_returns_monthly = self.asset_returns_monthly.pct_change().dropna()
        self.asset_returns_monthly = self.asset_returns_monthly.replace([np.inf, -np.inf], np.nan)

        self.asset_names = self.asset_returns_monthly.columns.tolist()
        print(f"\nLoaded {len(self.asset_names)} assets: {self.asset_names}")
        print(f"Monthly returns shape: {self.asset_returns_monthly.shape}")
        print(f"Date range: {self.asset_returns_monthly.index[0]} to {self.asset_returns_monthly.index[-1]}")
        return self

    def build_factors(self):
        print("\n" + "=" * 60)
        print("Step 2: Building macro factors from high-frequency data...")
        print("=" * 60)

        hf_factors = load_high_freq_macro_factors()

        factor_values = build_mimicking_factors(hf_factors)
        self.factor_values_daily = factor_values

        self.factor_returns_daily = self.factor_values_daily.pct_change().dropna()
        self.factor_returns_daily = self.factor_returns_daily.replace([np.inf, -np.inf], np.nan)

        factor_values_monthly = resample_to_monthly(self.factor_values_daily)
        self.factor_returns_monthly = factor_values_monthly.pct_change().dropna()
        self.factor_returns_monthly = self.factor_returns_monthly.replace([np.inf, -np.inf], np.nan)

        start_dt = pd.to_datetime(self.start)
        end_dt = pd.to_datetime(self.end)
        self.factor_returns_monthly = self.factor_returns_monthly[
            (self.factor_returns_monthly.index >= start_dt) &
            (self.factor_returns_monthly.index <= end_dt)
        ]

        print(f"\nFactor returns (monthly) shape: {self.factor_returns_monthly.shape}")
        print(f"Factor returns date range: {self.factor_returns_monthly.index[0]} to {self.factor_returns_monthly.index[-1]}")
        return self

    def prepare_factor_data(self, lookback=36):
        print("\n" + "=" * 60)
        print("Step 3: Preparing factor covariance and betas...")
        print("=" * 60)

        common_dates = self.asset_returns_monthly.index.intersection(self.factor_returns_monthly.index)
        print(f"Common dates for backtest: {len(common_dates)} months")

        self.asset_returns = self.asset_returns_monthly.loc[common_dates].copy()
        self.factor_returns = self.factor_returns_monthly.loc[common_dates].copy()
        self.asset_names = self.asset_returns.columns.tolist()

        print(f"  Computing rolling factor covariance (window={lookback})...")
        self.factor_cov_history, factor_dates = compute_factor_covariance_rolling(
            self.factor_returns, window=lookback
        )
        print(f"  Factor covariance history: {self.factor_cov_history.shape}")

        print(f"  Computing rolling factor betas (window={lookback})...")
        self.betas_dict, self.betas_array = compute_factor_exposures(
            self.asset_returns, self.factor_returns, window=lookback
        )
        print(f"  Beta history shape: {self.betas_array.shape}")

        print(f"  Estimating residual variance...")
        self.residual_var_history = self._estimate_residual_var(lookback)
        print(f"  Residual variance history: {self.residual_var_history.shape}")
        return self

    def _estimate_residual_var(self, window=36):
        n = len(self.asset_returns)
        k = self.factor_returns.shape[1]
        resid_vars = []

        for i in range(n):
            lookback = min(window, i + 1)
            Y = self.asset_returns.iloc[i-lookback+1:i+1].values
            X = self.factor_returns.iloc[i-lookback+1:i+1].values
            X_with_const = np.column_stack([np.ones(len(X)), X])
            try:
                coef, _, _, _ = np.linalg.lstsq(X_with_const, Y, rcond=None)
                residuals = Y - X_with_const @ coef
                resid_var = np.var(residuals, axis=0) + 1e-8
            except Exception:
                resid_var = 0.01 * np.ones(self.asset_returns.shape[1])
            resid_vars.append(resid_var)

        return np.array(resid_vars)

    def _get_factor_cov_at_t(self, t):
        if t < len(self.factor_cov_history):
            return self.factor_cov_history[t]
        return self.factor_cov_history[-1]

    def _get_betas_at_t(self, t):
        n_assets = len(self.asset_names)
        n_factors = len(FACTOR_NAMES)
        B = np.zeros((n_assets, n_factors))
        for i, aname in enumerate(self.asset_names):
            for j, fname in enumerate(FACTOR_NAMES):
                key = (aname, fname)
                if key in self.betas_dict:
                    B[i, j] = self.betas_dict[key][t]
        return B

    def _get_idio_var_at_t(self, t):
        if t < len(self.residual_var_history):
            return self.residual_var_history[t]
        return 0.01 * np.ones(len(self.asset_names))

    def run_asset_risk_parity(self):
        print("\n" + "=" * 60)
        print("Running: Asset Risk Parity (Baseline)")
        print("=" * 60)
        return self._run_strategy(
            "Asset Risk Parity",
            self._arpt_weights,
        )

    def _arpt_weights(self, t):
        lookback = min(36, t + 1)
        rets = self.asset_returns.iloc[:t+1]
        cov = rets.cov().values
        try:
            w = asset_risk_parity(rets, cov)
        except Exception:
            w = np.ones(len(self.asset_names)) / len(self.asset_names)
        return w

    def run_macro_risk_parity(self):
        print("\n" + "=" * 60)
        print("Running: Macro Risk Parity")
        print("=" * 60)
        return self._run_strategy(
            "Macro Risk Parity",
            self._mrp_weights,
        )

    def _mrp_weights(self, t):
        lookback = min(36, t + 1)
        rets = self.asset_returns.iloc[:t+1]
        factor_rets = self.factor_returns.iloc[:t+1]
        B = self._get_betas_at_t(t)
        factor_cov = self._get_factor_cov_at_t(t)
        idio_var = self._get_idio_var_at_t(t)
        try:
            w = macro_risk_parity(rets, B, factor_cov, idio_var)
        except Exception:
            w = np.ones(len(self.asset_names)) / len(self.asset_names)
        return w

    def run_macro_risk_minimization(self):
        print("\n" + "=" * 60)
        print("Running: Macro Risk Minimization")
        print("=" * 60)
        return self._run_strategy(
            "Macro Risk Minimization",
            self._mrm_weights,
        )

    def _mrm_weights(self, t):
        lookback = min(36, t + 1)
        rets = self.asset_returns.iloc[:t+1]
        B = self._get_betas_at_t(t)
        factor_cov = self._get_factor_cov_at_t(t)
        idio_var = self._get_idio_var_at_t(t)
        try:
            w = macro_risk_minimization(rets, B, factor_cov, idio_var)
        except Exception:
            w = np.ones(len(self.asset_names)) / len(self.asset_names)
        return w

    def _run_strategy(self, name, weight_fn):
        dates = self.asset_returns.index.tolist()
        n_dates = len(dates)

        weights_history = []
        portfolio_returns = []
        frc_history = []

        for t, date in enumerate(dates):
            w = weight_fn(t)
            w = np.clip(w, 0, 1)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(len(self.asset_names)) / len(self.asset_names)

            weights_history.append(pd.Series(w, index=self.asset_names))

            if t < n_dates - 1:
                next_ret = self.asset_returns.iloc[t + 1].values
                port_ret = float(np.dot(w, next_ret))
                portfolio_returns.append(port_ret)

                B = self._get_betas_at_t(t + 1)
                factor_cov = self._get_factor_cov_at_t(t + 1)
                idio_var = self._get_idio_var_at_t(t + 1)
                frc, idio_frc = self._compute_frc(w, B, factor_cov, idio_var)
                frc_history.append(frc)

        ret_dates = dates[1:]
        weight_df = pd.DataFrame(weights_history, index=dates)
        ret_series = pd.Series(portfolio_returns, index=ret_dates)
        frc_df = pd.DataFrame(frc_history, index=ret_dates, columns=FACTOR_NAMES)

        metrics = compute_performance_metrics(ret_series, name)
        turnover = compute_turnover(weight_df)
        metrics["monthly_turnover"] = turnover * 100

        self.strategy_results[name] = {
            "weights": weight_df,
            "returns": ret_series,
            "metrics": metrics,
            "cumulative": compute_cumulative_returns(ret_series),
            "frc": frc_df,
        }

        print_performance_summary(metrics)
        print(f"  Monthly Turnover: {turnover*100:.1f}%")
        return self

    def _compute_frc(self, w, B, factor_cov, idio_var):
        if np.isscalar(idio_var):
            idio_arr = idio_var * np.ones(len(w))
        else:
            idio_arr = np.array(idio_var)
        frc, idio = compute_portfolio_factor_risk_contribution(w, B, factor_cov, idio_arr)
        return frc, idio

    def run_all_strategies(self):
        self.run_asset_risk_parity()
        self.run_macro_risk_parity()
        self.run_macro_risk_minimization()
        return self

    def compare_all_strategies(self):
        print("\n" + "=" * 80)
        print("Strategy Comparison")
        print("=" * 80)
        metrics_list = [v["metrics"] for v in self.strategy_results.values()]
        compare_strategies(metrics_list)
        return self

    def save_results(self, path=None):
        if path is None:
            path = os.path.join(OUTPUT_DIR, "backtest_results.pkl")
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "strategy_results": self.strategy_results,
                "asset_returns": self.asset_returns,
                "factor_returns": self.factor_returns,
            }, f)
        print(f"\nResults saved to: {path}")
        return self


def run_full_backtest():
    bt = MacroRiskBacktester()
    bt.load_data()
    bt.build_factors()
    bt.prepare_factor_data(lookback=36)
    bt.run_all_strategies()
    bt.compare_all_strategies()
    bt.save_results()
    return bt


if __name__ == "__main__":
    print("Starting full backtest with real data...")
    results = run_full_backtest()
    print("\nBacktest complete!")
