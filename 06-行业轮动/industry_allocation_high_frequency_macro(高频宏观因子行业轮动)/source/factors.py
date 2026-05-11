import numpy as np
import pandas as pd
from scipy import stats
from config.settings import FACTOR_CONFIG, START_DATE, END_DATE

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

class FactorMimicking:
    def __init__(self, lookback_window=156, rolling_window=4):
        self.lookback_window = lookback_window
        self.rolling_window = rolling_window

    def calculate_weekly_returns(self, price_series):
        if len(price_series) < 2:
            return pd.Series(dtype=float)
        weekly_prices = price_series.resample('W-FRI').last()
        returns = weekly_prices.pct_change()
        return returns.dropna()

    def apply_rolling_ma(self, price_series):
        ma_prices = price_series.rolling(window=self.rolling_window).mean()
        return ma_prices.dropna()

    def calculate_inverse_volatility_weights(self, returns_series, min_periods=52):
        if len(returns_series) < min_periods:
            std = returns_series.std()
            if std == 0 or np.isnan(std):
                return np.array([1.0 / len(returns_series)] * len(returns_series))
            return np.array([1.0 / std for _ in returns_series])
        else:
            rolling_std = returns_series.tail(min_periods).std()
            if rolling_std == 0 or np.isnan(rolling_std):
                return np.array([1.0 / len(returns_series)] * len(returns_series))
            return np.array([1.0 / rolling_std for _ in returns_series])

    def normalize_weights(self, weights):
        weights = np.array(weights)
        if weights.sum() == 0:
            return np.ones(len(weights)) / len(weights)
        return weights / weights.sum()

    def build_factor_portfolio(self, asset_returns_dict, long_assets=None, short_assets=None):
        if not asset_returns_dict:
            return pd.Series(dtype=float)

        returns_df = pd.DataFrame(asset_returns_dict)
        if returns_df.empty:
            return pd.Series(dtype=float)

        weights = self.calculate_inverse_volatility_weights(returns_df.T.std())
        weights = self.normalize_weights(weights)

        long_weight = 1.0 if long_assets else 0.0
        short_weight = -1.0 if short_assets else 0.0

        portfolio_returns = pd.Series(dtype=float)
        for i, col in enumerate(returns_df.columns):
            asset_return = returns_df[col]
            asset_weight = weights[i]
            if long_assets and col in long_assets:
                asset_weight *= long_weight
            elif short_assets and col in short_assets:
                asset_weight *= short_weight
            portfolio_returns += asset_return * asset_weight

        return portfolio_returns

class HighFrequencyMacroFactors:
    def __init__(self, data_fetcher=None):
        self.factor_mimicking = FactorMimicking()
        self.data_fetcher = data_fetcher
        self.factors = {}
        self.factor_config = FACTOR_CONFIG
        self.use_akshare = AKSHARE_AVAILABLE

    def get_akshare_data(self, symbol, start_date=None, end_date=None):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df = df.set_index('date')
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                return df['close']
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return pd.Series(dtype=float)

    def get_akshare_bond_data(self, symbol, start_date=None, end_date=None):
        if not self.use_akshare:
            return pd.DataFrame()
        try:
            df = ak.bond_zh_daily(symbol=symbol)
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df = df.set_index('date')
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                return df['close']
        except Exception as e:
            print(f"Error fetching bond {symbol}: {e}")
        return pd.Series(dtype=float)

    def calculate_growth_factor(self, price_data):
        config = self.factor_config["growth"]
        long_assets = config.get("long_assets", [])
        short_assets = config.get("short_assets", [])

        long_returns = {}
        for asset in long_assets:
            if asset in price_data and len(price_data[asset]) > 0:
                long_returns[asset] = self.factor_mimicking.calculate_weekly_returns(price_data[asset])

        short_returns = {}
        for asset in short_assets:
            if asset in price_data and len(price_data[asset]) > 0:
                returns = self.factor_mimicking.calculate_weekly_returns(price_data[asset])
                short_returns[asset] = -returns

        all_returns = {**long_returns, **short_returns}
        if not all_returns:
            return pd.Series(dtype=float)

        returns_df = pd.DataFrame(all_returns)
        weights = self.factor_mimicking.calculate_inverse_volatility_weights(returns_df.T.std())
        weights = self.factor_mimicking.normalize_weights(weights)

        portfolio_returns = pd.Series(dtype=float)
        for i, col in enumerate(returns_df.columns):
            portfolio_returns += returns_df[col] * weights[i]

        return portfolio_returns

    def calculate_inflation_factor(self, price_data, factor_name):
        config = self.factor_config[factor_name]
        assets = config.get("assets", [])

        returns_dict = {}
        for asset in assets:
            if asset in price_data and len(price_data[asset]) > 0:
                returns = self.factor_mimicking.calculate_weekly_returns(price_data[asset])
                returns_dict[asset] = returns

        if not returns_dict:
            return pd.Series(dtype=float)

        returns_df = pd.DataFrame(returns_dict)
        weights = self.factor_mimicking.calculate_inverse_volatility_weights(returns_df.T.std())
        weights = self.factor_mimicking.normalize_weights(weights)

        portfolio_returns = pd.Series(dtype=float)
        for i, col in enumerate(returns_df.columns):
            portfolio_returns += returns_df[col] * weights[i]

        return portfolio_returns

    def calculate_rate_factor(self, price_data, factor_name):
        config = self.factor_config[factor_name]
        long_assets = config.get("long_assets", [])
        short_assets = config.get("short_assets", [])

        returns_dict = {}
        for asset in long_assets:
            if asset in price_data and len(price_data[asset]) > 0:
                returns_dict[asset] = self.factor_mimicking.calculate_weekly_returns(price_data[asset])

        for asset in short_assets:
            if asset in price_data and len(price_data[asset]) > 0:
                returns = self.factor_mimicking.calculate_weekly_returns(price_data[asset])
                returns_dict[asset] = -returns

        if not returns_dict:
            return pd.Series(dtype=float)

        returns_df = pd.DataFrame(returns_dict)
        weights = self.factor_mimicking.calculate_inverse_volatility_weights(returns_df.T.std())
        weights = self.factor_mimicking.normalize_weights(weights)

        portfolio_returns = pd.Series(dtype=float)
        for i, col in enumerate(returns_df.columns):
            portfolio_returns += returns_df[col] * weights[i]

        return portfolio_returns

    def calculate_all_factors(self, price_data_dict):
        self.factors = {}

        self.factors['growth'] = self.calculate_growth_factor(price_data_dict)
        self.factors['life_inflation'] = self.calculate_inflation_factor(price_data_dict, 'life_inflation')
        self.factors['production_inflation'] = self.calculate_inflation_factor(price_data_dict, 'production_inflation')
        self.factors['risk_free_rate'] = self.calculate_rate_factor(price_data_dict, 'risk_free_rate')
        self.factors['credit_spread'] = self.calculate_rate_factor(price_data_dict, 'credit_spread')
        self.factors['term_spread'] = self.calculate_rate_factor(price_data_dict, 'term_spread')
        self.factors['exchange_rate'] = self.calculate_rate_factor(price_data_dict, 'exchange_rate')

        return self.factors

    def get_factor_returns(self):
        return pd.DataFrame(self.factors)

    def calculate_factor_cumulative_returns(self, periods=52):
        if not self.factors:
            return pd.DataFrame()

        factor_returns = pd.DataFrame(self.factors)
        cumulative = (1 + factor_returns.tail(periods)).cumprod() - 1
        return cumulative

    def calculate_factor_yoy_returns(self):
        if not self.factors:
            return pd.DataFrame()

        factor_returns = pd.DataFrame(self.factors)
        yoy_returns = factor_returns.rolling(window=52).apply(lambda x: (1 + x).prod() - 1)
        return yoy_returns

    def calculate_factor_exposure(self, asset_returns):
        if not self.factors or asset_returns.empty:
            return pd.DataFrame()

        factor_returns = pd.DataFrame(self.factors)
        common_dates = factor_returns.index.intersection(asset_returns.index)

        if len(common_dates) < 52:
            return pd.DataFrame()

        factor_rets = factor_returns.loc[common_dates]
        asset_rets = asset_returns.loc[common_dates]

        exposures = {}
        for col in asset_rets.columns:
            valid_data = pd.concat([factor_rets, asset_rets[[col]]], axis=1).dropna()
            if len(valid_data) > 52:
                X = valid_data[factor_returns.columns].values
                y = valid_data[col].values
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    exposures[col] = beta
                except:
                    exposures[col] = np.zeros(len(factor_returns.columns))

        return pd.DataFrame(exposures, index=factor_returns.columns).T

def rolling_regression(y, X, window=52):
    results = []
    for i in range(window, len(y)):
        y_window = y.iloc[i-window:i]
        X_window = X.iloc[i-window:i]
        try:
            X_with_const = np.column_stack([np.ones(len(X_window)), X_window.values])
            beta = np.linalg.lstsq(X_with_const, y_window.values, rcond=None)[0]
            residuals = y_window.values - X_with_const @ beta
            r_squared = 1 - (residuals**2).sum() / ((y_window - y_window.mean())**2).sum()
            results.append({
                'beta': beta[1:],
                'r_squared': r_squared,
                'index': y.index[i]
            })
        except:
            results.append({
                'beta': np.zeros(X.shape[1]),
                'r_squared': 0,
                'index': y.index[i]
            })
    return results

def test_factor_stationarity(factor_returns):
    results = {}
    for col in factor_returns.columns:
        series = factor_returns[col].dropna()
        if len(series) > 0:
            try:
                adf_result = stats.normaltest(series)
                results[col] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis()
                }
            except:
                results[col] = {}
    return pd.DataFrame(results).T

DATA_GAPS = {
    'growth': {
        'missing_assets': ['HSI.HI', 'CRBRI.RB', 'NH0012.NHF', 'CBA00652.CS'],
        'alternative': 'Use HS300, CSI500, CSI1000 as growth proxies via tushare',
        'note': 'Original proxy assets (恒生指数, CRB工业现货, 南华沪铜, 国债净价指数) not available via tushare/akshare'
    },
    'life_inflation': {
        'missing_assets': ['NH0056.NHF'],
        'alternative': 'Use CPI data from akshare macro_cpi()',
        'note': 'Original proxy asset (南华生猪指数) not available'
    },
    'production_inflation': {
        'missing_assets': ['B00.IPE', 'NH0016.NHF', 'NH0030.NHF'],
        'alternative': 'Use PPI data from akshare ppi() or commodity ETFs',
        'note': 'Original proxy assets (布伦特原油, 南华螺纹钢, 南华动力煤) not available'
    },
    'risk_free_rate': {
        'missing_assets': ['CBA00621.CS'],
        'alternative': 'Use SHIBOR or bond ETFs from akshare',
        'note': 'Original proxy asset (国债财富指数) not available'
    },
    'credit_spread': {
        'missing_assets': ['CBA00621.CS', 'CBA02501.CS'],
        'alternative': 'Use corporate bond data or credit ETFs',
        'note': 'Original proxy assets not available via tushare/akshare'
    },
    'term_spread': {
        'missing_assets': ['CBA00651.CS', 'CBA00621.CS'],
        'alternative': 'Use bond yield curve data from akshare',
        'note': 'Original proxy assets not available'
    },
    'exchange_rate': {
        'missing_assets': ['AU9999.SGE'],
        'alternative': 'Use USD/CNY exchange rate from akshare',
        'note': 'Original proxy asset (伦敦金现) not available, can use gold ETF instead'
    }
}

def get_替代因子映射():
    return {
        'HSI.HI': '000300.SH',
        'CRBRI.RB': None,
        'NH0012.NHF': '159985.SZ',
        'CBA00652.CS': '511010.SH',
        'NH0056.NHF': None,
        'B00.IPE': 'USO',
        'NH0016.NHF': '螺纹钢ETF',
        'NH0030.NHF': None,
        'CBA00621.CS': '511010.SH',
        'CBA02501.CS': '511020.SH',
        'CBA00651.CS': '511030.SH',
        'AU9999.SGE': '518880.SH'
    }

if __name__ == "__main__":
    print("Factor Mimicking Module initialized successfully!")
    print("Available factors:", list(FACTOR_CONFIG.keys()))
    print("\n" + "="*60)
    print("Data Availability Notice:")
    print("="*60)
    for factor, info in DATA_GAPS.items():
        print(f"\n{factor}:")
        print(f"  Missing: {info['missing_assets']}")
        print(f"  Alternative: {info['alternative']}")
