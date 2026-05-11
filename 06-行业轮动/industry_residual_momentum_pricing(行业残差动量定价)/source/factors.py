import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore

class FactorCalculator:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.stock_weights = None
        self.bond_weights = None
        self.commodity_weights = None

    def perform_pca(self, data, n_components=None):
        if n_components is None:
            n_components = min(self.n_components, data.shape[1], data.shape[0])
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        return transformed, pca.components_, pca.explained_variance_ratio_

    def calculate_stock_factors(self, stock_data, is_domestic=False):
        valid_data = stock_data.dropna()
        if len(valid_data) < self.n_components:
            return None, None

        pca_result, components, variance_ratio = self.perform_pca(valid_data)

        market_factor = pca_result[:, 0]

        if is_domestic:
            style_factor = pca_result[:, 1] + pca_result[:, 2] if pca_result.shape[1] >= 3 else pca_result[:, 1]
        else:
            style_factor = pca_result[:, 1] + pca_result[:, 2] if pca_result.shape[1] >= 3 else pca_result[:, 1]

        self.stock_weights = components

        dates = valid_data.index
        factors = pd.DataFrame({
            'market_factor': market_factor,
            'style_factor': style_factor
        }, index=dates)

        return factors, variance_ratio

    def calculate_bond_factors(self, bond_data, is_domestic=False):
        valid_data = bond_data.dropna()
        if len(valid_data) < self.n_components:
            return None, None

        pca_result, components, variance_ratio = self.perform_pca(valid_data)

        if is_domestic:
            market_factor = pca_result[:, 0]
            style_factor = pca_result[:, 1] if pca_result.shape[1] >= 2 else pca_result[:, 0]
        else:
            market_factor = pca_result[:, 0] + pca_result[:, 1] if pca_result.shape[1] >= 2 else pca_result[:, 0]
            style_factor = pca_result[:, 2] if pca_result.shape[1] >= 3 else pca_result[:, 1]

        self.bond_weights = components

        dates = valid_data.index
        factors = pd.DataFrame({
            'market_factor': market_factor,
            'style_factor': style_factor
        }, index=dates)

        return factors, variance_ratio

    def calculate_commodity_factors(self, commodity_data):
        valid_data = commodity_data.dropna()
        if len(valid_data) < self.n_components:
            return None, None

        pca_result, components, variance_ratio = self.perform_pca(valid_data)

        market_factor = pca_result[:, 0]
        style_factor = pca_result[:, 1] if pca_result.shape[1] >= 2 else pca_result[:, 0]

        self.commodity_weights = components

        dates = valid_data.index
        factors = pd.DataFrame({
            'market_factor': market_factor,
            'style_factor': style_factor
        }, index=dates)

        return factors, variance_ratio

    def calculate_global_factors(self, stock_data, bond_data, commodity_data):
        stock_factors, stock_var = self.calculate_stock_factors(stock_data, is_domestic=False)
        bond_factors, bond_var = self.calculate_bond_factors(bond_data, is_domestic=False)
        commodity_factors, commodity_var = self.calculate_commodity_factors(commodity_data)

        combined_factors = pd.DataFrame(index=stock_factors.index)

        stock_market_norm = zscore(stock_factors['market_factor']) if stock_factors is not None else None
        bond_market_norm = zscore(bond_factors['market_factor']) if bond_factors is not None else None
        commodity_market_norm = zscore(commodity_factors['market_factor']) if commodity_factors is not None else None

        if stock_market_norm is not None and bond_market_norm is not None and commodity_market_norm is not None:
            combined_factors['global_market_factor'] = (stock_market_norm + bond_market_norm + commodity_market_norm) / 3

        if stock_factors is not None:
            combined_factors['stock_style_factor'] = zscore(stock_factors['style_factor'])
        if bond_factors is not None:
            combined_factors['bond_style_factor'] = zscore(bond_factors['style_factor'])
        if commodity_factors is not None:
            combined_factors['commodity_style_factor'] = zscore(commodity_factors['style_factor'])

        return combined_factors

    def calculate_domestic_factors(self, stock_data, bond_data, commodity_data):
        stock_factors, stock_var = self.calculate_stock_factors(stock_data, is_domestic=True)
        bond_factors, bond_var = self.calculate_bond_factors(bond_data, is_domestic=True)
        commodity_factors, commodity_var = self.calculate_commodity_factors(commodity_data)

        combined_factors = pd.DataFrame(index=stock_factors.index)

        stock_market_norm = zscore(stock_factors['market_factor']) if stock_factors is not None else None
        bond_market_norm = zscore(bond_factors['market_factor']) if bond_factors is not None else None
        commodity_market_norm = zscore(commodity_factors['market_factor']) if commodity_factors is not None else None

        if stock_market_norm is not None and bond_market_norm is not None and commodity_market_norm is not None:
            combined_factors['domestic_market_factor'] = (stock_market_norm + bond_market_norm + commodity_market_norm) / 3

        if stock_factors is not None and stock_factors.shape[1] >= 2:
            combined_factors['stock_pc2'] = zscore(stock_factors['style_factor'])
        if stock_factors is not None and stock_factors.shape[1] >= 3:
            combined_factors['stock_pc3'] = zscore(0)

        return combined_factors

    def rolling_pca_factors(self, data, window=100):
        factors_list = []
        dates = []

        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i].dropna()
            if len(window_data) >= window * 0.5:
                try:
                    pca_result, _, _ = self.perform_pca(window_data)
                    factors_list.append(pca_result[-1])
                    dates.append(data.index[i])
                except Exception as e:
                    continue

        if len(factors_list) > 0:
            result_df = pd.DataFrame(factors_list, index=dates)
            result_df.columns = [f'PC{i+1}' for i in range(result_df.shape[1])]
            return result_df
        return pd.DataFrame()

    def get_factor_weights(self, asset_type='stock'):
        if asset_type == 'stock':
            return self.stock_weights
        elif asset_type == 'bond':
            return self.bond_weights
        elif asset_type == 'commodity':
            return self.commodity_weights
        return None