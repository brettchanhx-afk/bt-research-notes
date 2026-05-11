"""
桥水全天候策略与风险平价模型量化项目

项目结构：
- data_fetcher: 数据获取模块
- risk_parity: 风险平价模型核心模块
- risk_budget: 风险预算模型模块
- factor_risk_parity: 因子风险平价模型模块
- backtest: 回测引擎模块
- performance: 性能评估模块

使用方法：
    from source import fetch_all_assets, BacktestEngine, calculate_portfolio_metrics

    prices = fetch_all_assets('2008-01-01', '2023-04-30')
    engine = BacktestEngine(prices)
    results = engine.run_all_strategies()
"""

from .data_fetcher import (
    fetch_all_assets,
    calculate_returns,
    calculate_monthly_returns,
    save_data,
    load_data,
    ASSET_CONFIG
)

from .risk_parity import (
    calculate_covariance_matrix,
    calculate_marginal_risk_contribution,
    calculate_total_risk_contribution,
    calculate_portfolio_volatility,
    solve_risk_parity_weights,
    risk_parity_portfolio,
    equal_weight_portfolio,
    fixed_ratio_portfolio,
    risk_contribution_decomposition
)

from .risk_budget import (
    solve_risk_budget_weights,
    sharpe_squared_risk_budget_portfolio,
    leveraged_risk_parity_portfolio,
    custom_risk_budget_portfolio,
    calculate_sharpe_ratio_weights
)

from .factor_risk_parity import (
    extract_risk_factors,
    solve_factor_risk_parity_weights,
    principal_component_risk_parity_portfolio,
    factor_analysis_report
)

from .backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestEngine,
    run_comparative_backtest,
    calculate_portfolio_nav
)

from .performance import (
    calculate_annualized_return,
    calculate_max_drawdown,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    calculate_portfolio_metrics,
    calculate_rolling_metrics,
    calculate_drawdown_series,
    calculate_nav,
    calculate_yearly_returns,
    generate_performance_summary,
    generate_yearly_performance_table,
    print_performance_report
)

__version__ = '1.0.0'
__author__ = 'Risk Parity Research Team'