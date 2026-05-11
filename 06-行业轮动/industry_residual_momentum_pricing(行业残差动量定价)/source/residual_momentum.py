import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class ResidualMomentumCalculator:
    def __init__(self, rolling_window=100, momentum_window=12):
        self.rolling_window = rolling_window
        self.momentum_window = momentum_window
        self.global_factors = None
        self.domestic_factors = None
        self.residuals_dict = {}

    def calculate_log_returns(self, price_data):
        return np.log(price_data / price_data.shift(1))

    def calculate_yoy_returns(self, price_data):
        return price_data.pct_change(periods=12)

    def calculate_mom_returns(self, price_data):
        return price_data.pct_change(periods=1)

    def calculate_monthly_log_returns(self, price_data):
        monthly_prices = price_data.resample('M').last()
        return np.log(monthly_prices / monthly_prices.shift(1))

    def rolling_pca_for_factors(self, data, n_components=3):
        factors_list = []
        dates = []

        for i in range(self.rolling_window, len(data)):
            window_data = data.iloc[i-self.rolling_window:i].dropna()
            if len(window_data) < self.rolling_window * 0.7:
                continue

            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(n_components, window_data.shape[1]))
                pca.fit(window_data)
                factors_list.append(pca.components_)
                dates.append(data.index[i])
            except Exception as e:
                continue

        return pd.DataFrame(factors_list, index=dates)

    def calculate_global_market_style_factors(self, stock_data, bond_data, commodity_data):
        from sklearn.decomposition import PCA

        stock_yoy = self.calculate_yoy_returns(stock_data).dropna()
        bond_yoy = self.calculate_yoy_returns(bond_data).dropna()
        commodity_yoy = self.calculate_yoy_returns(commodity_data).dropna()

        common_dates = stock_yoy.index.intersection(bond_yoy.index).intersection(commodity_yoy.index)
        stock_yoy = stock_yoy.loc[common_dates]
        bond_yoy = bond_yoy.loc[common_dates]
        commodity_yoy = commodity_yoy.loc[common_dates]

        factors_dict = {}

        for n_comp in range(1, 4):
            try:
                pca_stock = PCA(n_components=n_comp)
                stock_pca = pca_stock.fit_transform(stock_yoy)
                pca_bond = PCA(n_components=n_comp)
                bond_pca = pca_bond.fit_transform(bond_yoy)
                pca_commodity = PCA(n_components=n_comp)
                commodity_pca = pca_commodity.fit_transform(commodity_yoy)
            except:
                continue

        stock_market_factor = stock_pca[:, 0]
        stock_style_factor = stock_pca[:, 1] + stock_pca[:, 2] if stock_pca.shape[1] >= 3 else stock_pca[:, 1]

        bond_market_factor = bond_pca[:, 0] + bond_pca[:, 1] if bond_pca.shape[1] >= 2 else bond_pca[:, 0]
        bond_style_factor = bond_pca[:, 2] if bond_pca.shape[1] >= 3 else bond_pca[:, 1]

        commodity_market_factor = commodity_pca[:, 0]
        commodity_style_factor = commodity_pca[:, 1] if commodity_pca.shape[1] >= 2 else commodity_pca[:, 0]

        global_market = (zscore(stock_market_factor) + zscore(bond_market_factor) + zscore(commodity_market_factor)) / 3

        factors_dict['global_market'] = pd.Series(zscore(global_market), index=common_dates)
        factors_dict['stock_style'] = pd.Series(zscore(stock_style_factor), index=common_dates)
        factors_dict['bond_style'] = pd.Series(zscore(bond_style_factor), index=common_dates)
        factors_dict['commodity_style'] = pd.Series(zscore(commodity_style_factor), index=common_dates)

        self.global_factors = pd.DataFrame(factors_dict)
        return self.global_factors

    def calculate_domestic_market_style_factors(self, stock_data, bond_data, commodity_data):
        from sklearn.decomposition import PCA

        stock_yoy = self.calculate_yoy_returns(stock_data).dropna()
        bond_yoy = self.calculate_yoy_returns(bond_data).dropna()
        commodity_yoy = self.calculate_yoy_returns(commodity_data).dropna()

        common_dates = stock_yoy.index.intersection(bond_yoy.index).intersection(commodity_yoy.index)
        stock_yoy = stock_yoy.loc[common_dates]
        bond_yoy = bond_yoy.loc[common_dates]
        commodity_yoy = commodity_yoy.loc[common_dates]

        factors_dict = {}

        for n_comp in range(1, 7):
            try:
                pca_stock = PCA(n_components=n_comp)
                stock_pca = pca_stock.fit_transform(stock_yoy)
                pca_bond = PCA(n_components=n_comp)
                bond_pca = pca_bond.fit_transform(bond_yoy)
                pca_commodity = PCA(n_components=n_comp)
                commodity_pca = pca_commodity.fit_transform(commodity_yoy)
            except:
                continue

        stock_market_factor = stock_pca[:, 0]
        stock_pc2 = stock_pca[:, 1]
        stock_pc3 = stock_pca[:, 2] if stock_pca.shape[1] >= 3 else 0

        bond_market_factor = bond_pca[:, 0]
        bond_pc2 = bond_pca[:, 1] if bond_pca.shape[1] >= 2 else 0
        bond_pc3 = bond_pca[:, 2] if bond_pca.shape[1] >= 3 else 0

        commodity_market_factor = commodity_pca[:, 0]
        commodity_pc2 = commodity_pca[:, 1] if commodity_pca.shape[1] >= 2 else 0

        domestic_market = (zscore(stock_market_factor) + zscore(bond_market_factor) + zscore(commodity_market_factor)) / 3

        factors_dict['domestic_market'] = pd.Series(zscore(domestic_market), index=common_dates)
        factors_dict['stock_pc2'] = pd.Series(zscore(stock_pc2), index=common_dates)
        factors_dict['stock_pc3'] = pd.Series(zscore(stock_pc3), index=common_dates)
        factors_dict['bond_pc2'] = pd.Series(zscore(bond_pc2), index=common_dates)
        factors_dict['bond_pc3'] = pd.Series(zscore(bond_pc3), index=common_dates)
        factors_dict['commodity_pc2'] = pd.Series(zscore(commodity_pc2), index=common_dates)

        self.domestic_factors = pd.DataFrame(factors_dict)
        return self.domestic_factors

    def calculate_residuals(self, asset_returns, factors, is_domestic=False):
        common_dates = asset_returns.index.intersection(factors.index)
        asset_ret = asset_returns.loc[common_dates].dropna()
        fac = factors.loc[common_dates].dropna()

        common_dates_final = asset_ret.index.intersection(fac.index)
        asset_ret = asset_ret.loc[common_dates_final]
        fac = fac.loc[common_dates_final]

        residuals_list = []
        residual_dates = []

        for i in range(self.rolling_window, len(asset_ret)):
            y = asset_ret.iloc[i-self.momentum_window:i].values.reshape(-1, 1)
            X = fac.iloc[i-self.momentum_window:i].values

            if len(y) < self.momentum_window or X.shape[0] < self.momentum_window:
                continue

            try:
                reg = LinearRegression()
                reg.fit(X, y)
                residual = y.flatten() - reg.predict(X).flatten()
                residuals_list.append(residual)
                residual_dates.append(asset_ret.index[i])
            except Exception as e:
                continue

        if len(residuals_list) > 0:
            residuals_df = pd.DataFrame(residuals_list, index=residual_dates)
            return residuals_df
        return pd.DataFrame()

    def calculate_residual_momentum(self, residuals):
        if residuals.empty:
            return pd.Series(dtype=float)

        residual_momentum = residuals.sum(axis=1)
        return residual_momentum

    def apply_reversal_effect(self, asset_returns, factors):
        residuals = self.calculate_residuals(asset_returns, factors, is_domestic=True)

        if residuals.empty:
            return pd.Series(dtype=float)

        monthly_returns = self.calculate_mom_returns(asset_returns)
        volatility = monthly_returns.rolling(window=self.momentum_window).std()

        modified_residuals = residuals.copy()

        for date in residuals.index:
            if date in volatility.index:
                window_vol = volatility.loc[:date].iloc[-self.momentum_window:]
                window_resid = residuals.loc[date]

                if len(window_vol) == self.momentum_window and len(window_resid) == self.momentum_window:
                    max_vol_idx = window_vol.idxmax()
                    if max_vol_idx in residuals.columns:
                        max_vol_col_idx = list(residuals.columns).index(max_vol_idx)
                        modified_residuals.loc[date, residuals.columns[max_vol_col_idx]] *= -1

        residual_momentum = self.calculate_residual_momentum(modified_residuals)
        return residual_momentum

    def calculate_all_residual_momentum(self, asset_data, factors, use_reversal=True):
        results = {}

        for col in asset_data.columns:
            try:
                asset_ret = asset_data[col].dropna()
                if use_reversal:
                    residual_mom = self.apply_reversal_effect(asset_ret, factors)
                else:
                    residuals = self.calculate_residuals(asset_ret, factors)
                    residual_mom = self.calculate_residual_momentum(residuals)
                results[col] = residual_mom
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_factor_correlations(self, factors1, factors2):
        if factors1 is None or factors2 is None:
            return {}

        common_dates = factors1.index.intersection(factors2.index)
        correlations = {}

        for col1 in factors1.columns:
            for col2 in factors2.columns:
                corr = factors1.loc[common_dates, col1].corr(factors2.loc[common_dates, col2])
                correlations[f'{col1}_vs_{col2}'] = corr

        return correlations