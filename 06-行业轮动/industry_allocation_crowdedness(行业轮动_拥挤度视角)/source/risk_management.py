import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    def __init__(self, max_drawdown_limit=0.2, max_position_loss=0.15, max_correlation=0.7):
        self.max_drawdown_limit = max_drawdown_limit
        self.max_position_loss = max_position_loss
        self.max_correlation = max_correlation
        self.current_drawdown = 0
        self.risk_events = []
        self.position_limits = {}

    def calculate_drawdown(self, equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0
        running_max = equity_curve.cummax()
        current_drawdown = (equity_curve.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        self.current_drawdown = current_drawdown
        return current_drawdown

    def check_drawdown_limit(self, equity_curve: pd.Series) -> Tuple[bool, str]:
        current_dd = self.calculate_drawdown(equity_curve)
        if current_dd < -self.max_drawdown_limit:
            return True, f"触及最大回撤限制 ({self.max_drawdown_limit:.1%})"
        return False, ""

    def calculate_portfolio_var(self, returns: pd.Series, confidence=0.95) -> float:
        if len(returns) < 30:
            return 0
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)

    def calculate_portfolio_cvar(self, returns: pd.Series, confidence=0.95) -> float:
        var = self.calculate_portfolio_var(returns, confidence)
        tail_returns = returns[returns <= -var]
        if len(tail_returns) > 0:
            return abs(tail_returns.mean())
        return var

    def calculate_position_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        if returns_df.shape[1] < 2:
            return pd.DataFrame()
        correlation_matrix = returns_df.corr()
        return correlation_matrix

    def check_correlation_limit(self, returns_df: pd.DataFrame) -> List[Tuple[str, str]]:
        violations = []
        if returns_df.shape[1] < 2:
            return violations
        corr_matrix = self.calculate_position_correlation(returns_df)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.max_correlation:
                    violations.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        return violations

    def calculate_position_risk(self, positions: Dict[str, float], prices: Dict[str, float],
                               volatility: Dict[str, float]) -> Dict[str, float]:
        position_values = {}
        total_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)
        for symbol, volume in positions.items():
            value = volume * prices.get(symbol, 0)
            weight = value / total_value if total_value > 0 else 0
            vol = volatility.get(symbol, 0.2)
            position_risk = weight * vol
            position_values[symbol] = {
                'value': value,
                'weight': weight,
                'volatility': vol,
                'risk_contribution': position_risk
            }
        return position_values

    def generate_risk_report(self, equity_curve: pd.Series, returns: pd.Series,
                            positions: Dict, prices: Dict) -> Dict:
        report = {
            'current_drawdown': self.calculate_drawdown(equity_curve),
            'var_95': self.calculate_portfolio_var(returns, 0.95),
            'cvar_95': self.calculate_portfolio_cvar(returns, 0.95),
            'max_drawdown_limit': self.max_drawdown_limit,
            'risk_events': self.risk_events,
            'position_count': len(positions),
            'total_exposure': sum(positions.values()) if positions else 0
        }
        return report

class CrowdMonitor:
    def __init__(self, crowdedness_signals: pd.DataFrame, thresholds: Dict[str, float]):
        self.signals = crowdedness_signals
        self.thresholds = thresholds

    def get_crowded_industries(self, date) -> List[str]:
        if date not in self.signals.index:
            return []
        return self.signals.loc[date][self.signals.loc[date] == True].index.tolist()

    def get_crowded_count(self, date) -> int:
        return len(self.get_crowded_industries(date))

    def is_market_crowded(self, date, threshold=10) -> bool:
        return self.get_crowded_count(date) > threshold

    def get_non_crowded_industries(self, date) -> List[str]:
        if date not in self.signals.index:
            return []
        return self.signals.loc[date][self.signals.loc[date] == False].index.tolist()

    def calculate_crowdedness_intensity(self, date) -> float:
        if date not in self.signals.index:
            return 0
        return self.signals.loc[date].mean()

class RiskAlerter:
    def __init__(self):
        self.alerts = []
        self.alert_thresholds = {
            'drawdown': -0.15,
            'volatility': 0.03,
            'crowded_industry_ratio': 0.5
        }

    def check_drawdown_alert(self, current_drawdown: float) -> Optional[Dict]:
        if current_drawdown < self.alert_thresholds['drawdown']:
            alert = {
                'type': 'drawdown',
                'level': 'warning',
                'message': f'组合回撤达到 {current_drawdown:.2%}',
                'timestamp': datetime.now()
            }
            self.alerts.append(alert)
            return alert
        return None

    def check_volatility_alert(self, current_volatility: float) -> Optional[Dict]:
        if current_volatility > self.alert_thresholds['volatility']:
            alert = {
                'type': 'volatility',
                'level': 'warning',
                'message': f'组合波动率上升至 {current_volatility:.2%}',
                'timestamp': datetime.now()
            }
            self.alerts.append(alert)
            return alert
        return None

    def check_crowdedness_alert(self, crowded_ratio: float) -> Optional[Dict]:
        if crowded_ratio > self.alert_thresholds['crowded_industry_ratio']:
            alert = {
                'type': 'crowdedness',
                'level': 'info',
                'message': f'市场拥挤度较高 ({crowded_ratio:.2%} 行业拥挤)',
                'timestamp': datetime.now()
            }
            self.alerts.append(alert)
            return alert
        return None

    def get_active_alerts(self, lookback_hours=24) -> List[Dict]:
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        return [a for a in self.alerts if a['timestamp'] > cutoff_time]

    def clear_old_alerts(self, days=7):
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]

def apply_risk_controls(positions: Dict, risk_manager: RiskManager,
                        crowd_monitor: CrowdMonitor, date,
                        current_prices: Dict, current_volatility: Dict) -> Tuple[Dict, List[Dict]]:
    adjusted_positions = positions.copy()
    actions_taken = []
    if crowd_monitor.is_market_crowded(date):
        crowded_industries = crowd_monitor.get_crowded_industries(date)
        for ind in crowded_industries:
            if ind in adjusted_positions:
                del adjusted_positions[ind]
                actions_taken.append({
                    'date': date,
                    'action': 'liquidate_crowded',
                    'industry': ind,
                    'reason': '行业拥挤度过高'
                })
    position_risks = risk_manager.calculate_position_risk(
        adjusted_positions, current_prices, current_volatility
    )
    for symbol, risk_info in position_risks.items():
        if risk_info['weight'] > 0.3:
            old_weight = adjusted_positions.get(symbol, 0)
            adjusted_positions[symbol] = old_weight * 0.5
            actions_taken.append({
                'date': date,
                'action': 'reduce_concentration',
                'industry': symbol,
                'reason': f'权重 {risk_info["weight"]:.2%} 超过30%限制'
            })
    return adjusted_positions, actions_taken

if __name__ == "__main__":
    print("风控模块测试...")
    risk_mgr = RiskManager(max_drawdown_limit=0.2)
    print(f"最大回撤限制: {risk_mgr.max_drawdown_limit:.1%}")
    print("风控模块测试完成")