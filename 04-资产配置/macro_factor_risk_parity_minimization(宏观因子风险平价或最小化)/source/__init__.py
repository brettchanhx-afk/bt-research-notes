"""
Macro Factor Risk Parity & Risk Minimization Project.
Source package.
"""
from source.config import (
    FACTOR_NAMES,
    BACKTEST_START,
    BACKTEST_END,
    OUTPUT_DIR,
    DATA_DIR,
    ASSET_NAME_MAP,
    MACRO_FACTOR_COLS,
)
from source.data_loader import (
    load_asset_prices,
    load_high_freq_macro_factors,
    load_original_macro_factors,
    resample_to_monthly,
)
from source.macro_factors import (
    build_mimicking_factors,
    build_factor_returns_from_prices,
    compute_factor_exposures,
    compute_factor_covariance_rolling,
)
from source.risk_attribution import (
    compute_portfolio_factor_risk_contribution,
    compute_mrc_frc_series,
)
from source.optimization import (
    asset_risk_parity,
    macro_risk_parity,
    macro_risk_minimization,
)
from source.performance import (
    compute_performance_metrics,
    compute_cumulative_returns,
    compute_turnover,
    print_performance_summary,
    compare_strategies,
)
from source.backtest import MacroRiskBacktester, run_full_backtest

__all__ = [
    "FACTOR_NAMES",
    "BACKTEST_START", "BACKTEST_END",
    "OUTPUT_DIR", "DATA_DIR",
    "ASSET_NAME_MAP", "MACRO_FACTOR_COLS",
    "load_asset_prices",
    "load_high_freq_macro_factors",
    "load_original_macro_factors",
    "resample_to_monthly",
    "build_mimicking_factors",
    "build_factor_returns_from_prices",
    "compute_factor_exposures",
    "compute_factor_covariance_rolling",
    "compute_portfolio_factor_risk_contribution",
    "compute_mrc_frc_series",
    "asset_risk_parity",
    "macro_risk_parity",
    "macro_risk_minimization",
    "compute_performance_metrics",
    "compute_cumulative_returns",
    "compute_turnover",
    "print_performance_summary",
    "compare_strategies",
    "MacroRiskBacktester",
    "run_full_backtest",
]
