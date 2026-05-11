"""
大类资产配置策略模块
基于投资时钟的资产配置
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class AssetStrategy:
    def __init__(self, target_volatility=0.05, max_leverage=2.0, risk_free_rate=0.04):
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.weights_history = []
        self.returns_history = []

    def calculate_risk_budget_weights(self, returns_df, risk_budget=None):
        """
        计算风险预算权重
        """
        if returns_df.empty:
            return {}

        cov_matrix = returns_df.cov() * 12

        n_assets = len(returns_df.columns)

        if risk_budget is None:
            risk_budget = {col: 1.0 / n_assets for col in returns_df.columns}

        total_budget = sum(risk_budget.values())
        risk_budget = {k: v / total_budget for k, v in risk_budget.items()}

        def risk_contribution(w, cov):
            w = np.array(w)
            port_var = w @ cov @ w
            marginal_contrib = cov @ w
            risk_contrib = w * marginal_contrib / np.sqrt(port_var)
            return risk_contrib

        def objective(w, cov, target_budget):
            w = np.array(w)
            current_rc = risk_contribution(w, cov)
            total_rc = np.sum(current_rc)
            asset_names = list(target_budget.keys())
            target_rc = np.array([target_budget[asset_names[col]] * total_rc for col in range(len(w))])
            return np.sum((current_rc - target_rc) ** 2)

        n = len(returns_df.columns)
        w0 = np.ones(n) / n

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(
            objective,
            w0,
            args=(cov_matrix.values, risk_budget),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )

        if result.success:
            weights = dict(zip(returns_df.columns, result.x))
        else:
            weights = {col: 1.0 / n for col in returns_df.columns}

        return weights

    def adjust_weights_by_views(self, base_weights, factor_views, asset_mapping):
        """
        根据宏观观点调整权重
        """
        adjusted = base_weights.copy()

        for factor_name, view in factor_views.items():
            if factor_name not in asset_mapping:
                continue

            mapping = asset_mapping[factor_name]

            if view == 1:
                boost_factor = 2.0
                reduce_factor = 0.5
            elif view == -1:
                boost_factor = 0.5
                reduce_factor = 2.0
            else:
                continue

            best_asset = None
            worst_asset = None
            best_return = -np.inf
            worst_return = np.inf

            for asset_name in adjusted.keys():
                if asset_name in mapping:
                    up_return = mapping[asset_name].get('up', {}).get('mean', 0)
                    down_return = mapping[asset_name].get('down', {}).get('mean', 0)

                    if view == 1 and up_return > best_return:
                        best_return = up_return
                        best_asset = asset_name
                    elif view == -1 and down_return < worst_return:
                        worst_return = down_return
                        worst_asset = asset_name

            if best_asset and best_asset in adjusted:
                adjusted[best_asset] *= boost_factor
            if worst_asset and worst_asset in adjusted and worst_asset != best_asset:
                adjusted[worst_asset] *= reduce_factor

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def target_volatility_allocation(self, returns_df, base_weights, target_vol=None):
        """
        目标波动率调仓
        """
        if target_vol is None:
            target_vol = self.target_volatility

        if returns_df.empty or not base_weights:
            return base_weights

        weights = np.array([base_weights.get(col, 0) for col in returns_df.columns])
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        historical_vol = returns_df.std() * np.sqrt(12)

        total_vol = 0
        for i, col in enumerate(returns_df.columns):
            total_vol += weights[i] * historical_vol[col]

        if total_vol > 0:
            leverage = target_vol / total_vol
            leverage = min(max(leverage, 0), self.max_leverage)
        else:
            leverage = 1.0

        adjusted_weights = {col: leverage * base_weights.get(col, 0) for col in returns_df.columns}

        total = sum(adjusted_weights.values())
        if total > 1.0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def select_top_assets(self, returns_df, n_stock=1, n_commodity=2):
        """
        选择表现最好的资产
        """
        if returns_df.empty:
            return {}

        past_return = (1 + returns_df).prod() - 1

        stock_assets = [col for col in returns_df.columns if 'stock' in col.lower() or '000300' in col or '000905' in col or '399006' in col]
        commodity_assets = [col for col in returns_df.columns if 'commodity' in col.lower() or 'NH' in col or 'oil' in col.lower() or 'gold' in col.lower() or 'AU' in col]
        bond_assets = [col for col in returns_df.columns if 'bond' in col.lower() or 'CSI' in col or 'H11001' in col]

        selected = {}

        if stock_assets:
            stock_returns = past_return[stock_assets].sort_values(ascending=False)
            if len(stock_returns) >= n_stock:
                selected[stock_returns.index[0]] = 1.0
            else:
                for asset in stock_assets:
                    selected[asset] = 1.0 / len(stock_assets)

        if commodity_assets:
            commodity_returns = past_return[commodity_assets].sort_values(ascending=False)
            top_commodities = commodity_returns.head(n_commodity)
            for asset in top_commodities.index:
                selected[asset] = 1.0 / n_commodity

        if bond_assets:
            for asset in bond_assets:
                if asset not in selected:
                    selected[asset] = 1.0

        total = sum(selected.values())
        if total > 0:
            selected = {k: v / total for k, v in selected.items()}

        return selected

    def build_portfolio(self, returns_df, factor_views, asset_mapping,
                       risk_budget=None, selection_lookback=120):
        """
        构建投资组合
        """
        if returns_df.empty:
            return {}

        if len(returns_df) < 12:
            return {col: 1.0 / len(returns_df.columns) for col in returns_df.columns}

        lookback_returns = returns_df.tail(selection_lookback)

        base_weights = self.calculate_risk_budget_weights(lookback_returns, risk_budget)

        adjusted_weights = self.adjust_weights_by_views(base_weights, factor_views, asset_mapping)

        top_selected = self.select_top_assets(lookback_returns)

        final_weights = {}
        for asset in adjusted_weights:
            if asset in top_selected:
                final_weights[asset] = adjusted_weights[asset]

        if not final_weights:
            final_weights = top_selected

        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}

        final_weights = self.target_volatility_allocation(lookback_returns, final_weights)

        return final_weights

    def calculate_portfolio_return(self, weights, returns_series):
        """
        计算投资组合收益率
        """
        if not weights or returns_series.empty:
            return 0

        total_return = 0
        for asset, weight in weights.items():
            if asset in returns_series.index:
                total_return += weight * returns_series[asset]

        return total_return

    def rebalance_portfolio(self, current_weights, new_weights, threshold=0.1):
        """
        组合再平衡
        """
        changes = {}
        for asset in set(current_weights.keys()) | set(new_weights.keys()):
            old_w = current_weights.get(asset, 0)
            new_w = new_weights.get(asset, 0)
            changes[asset] = new_w - old_w

        need_rebalance = any(abs(c) > threshold for c in changes.values())

        if need_rebalance:
            return new_weights
        else:
            return current_weights

    def calculate_turnover(self, old_weights, new_weights):
        """
        计算换手率
        """
        total_turnover = 0
        for asset in set(old_weights.keys()) | set(new_weights.keys()):
            old_w = old_weights.get(asset, 0)
            new_w = new_weights.get(asset, 0)
            total_turnover += abs(new_w - old_w)

        return total_turnover / 2

    def get_strategy_summary(self, returns_series, weights_history):
        """
        获取策略摘要
        """
        if returns_series.empty:
            return {}

        cumulative_return = (1 + returns_series).prod() - 1
        annual_return = returns_series.mean() * 12
        annual_volatility = returns_series.std() * np.sqrt(12)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        running_max = (1 + returns_series).cumprod().cummax()
        drawdown = (1 + returns_series).cumprod() / running_max - 1
        max_drawdown = drawdown.min()

        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        winning_days = (returns_series > 0).sum()
        total_days = len(returns_series)
        win_rate = winning_days / total_days if total_days > 0 else 0

        avg_turnover = np.mean([self.calculate_turnover(w1, w2)
                               for w1, w2 in zip(weights_history[:-1], weights_history[1:])]) if len(weights_history) > 1 else 0

        return {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover
        }
