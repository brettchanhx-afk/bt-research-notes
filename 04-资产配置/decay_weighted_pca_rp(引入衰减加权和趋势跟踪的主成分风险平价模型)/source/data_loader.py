"""
Data loader module for fetching market data.
Uses tushare as primary data source with fallback to other sources.
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union

from .config import (
    TUSHARE_TOKEN,
    TUSHARE_API_URL,
    DATA_DIR,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    ASSET_CLASSES
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for fetching and managing market data.
    Primary source: tushare
    """

    def __init__(self, token: str = None, api_url: str = None):
        """
        Initialize DataLoader with tushare credentials.

        Args:
            token: tushare API token
            api_url: tushare API URL
        """
        self.token = token or TUSHARE_TOKEN
        self.api_url = api_url or TUSHARE_API_URL

        ts.set_token(self.token)
        self.pro = ts.pro_api(self.token)
        self.pro._DataApi__token = self.token
        self.pro._DataApi__http_url = self.api_url

        logger.info("DataLoader initialized with tushare API")

    def get_index_daily(
        self,
        ts_code: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE
    ) -> pd.DataFrame:
        """
        Fetch daily index data from tushare.

        Args:
            ts_code: Index code (e.g., '000300.SH' for CSI 300)
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format

        Returns:
            DataFrame with index daily data
        """
        try:
            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)

                logger.info(f"Fetched {len(df)} records for {ts_code}")
                return df
            else:
                logger.warning(f"No data returned for {ts_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for {ts_code}: {str(e)}")
            return pd.DataFrame()

    def get_bond_index_daily(
        self,
        ts_code: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE
    ) -> pd.DataFrame:
        """
        Fetch bond index daily data from tushare.

        Args:
            ts_code: Bond index code
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format

        Returns:
            DataFrame with bond index daily data
        """
        try:
            df = self.pro.bond_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)

                logger.info(f"Fetched {len(df)} records for bond {ts_code}")
                return df
            else:
                logger.warning(f"No bond data returned for {ts_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching bond data for {ts_code}: {str(e)}")
            return pd.DataFrame()

    def get_stock_daily(
        self,
        ts_code: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE
    ) -> pd.DataFrame:
        """
        Fetch daily stock data from tushare.

        Args:
            ts_code: Stock code (e.g., '600000.SH')
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format

        Returns:
            DataFrame with stock daily data
        """
        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                df.set_index('trade_date', inplace=True)

                logger.info(f"Fetched {len(df)} records for stock {ts_code}")
                return df
            else:
                logger.warning(f"No data returned for stock {ts_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching stock data for {ts_code}: {str(e)}")
            return pd.DataFrame()

    def get_index_weights(
        self,
        index_code: str,
        trade_date: str
    ) -> pd.DataFrame:
        """
        Fetch index constituent weights from tushare.

        Args:
            index_code: Index code
            trade_date: Trade date in YYYYMMDD format

        Returns:
            DataFrame with constituent weights
        """
        try:
            df = self.pro.index_weight(
                index_code=index_code,
                trade_date=trade_date
            )

            if df is not None and not df.empty:
                logger.info(f"Fetched {len(df)} constituents for {index_code}")
                return df
            else:
                logger.warning(f"No weight data for {index_code} on {trade_date}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching weights: {str(e)}")
            return pd.DataFrame()

    def get_macro_data(
        self,
        indicator: str = "M2"
    ) -> pd.DataFrame:
        """
        Fetch macro economic data from tushare.

        Args:
            indicator: Macro indicator code

        Returns:
            DataFrame with macro data
        """
        try:
            df = self.pro.macro_data(indicator=indicator)

            if df is not None and not df.empty:
                logger.info(f"Fetched macro data for {indicator}")
                return df
            else:
                logger.warning(f"No macro data for {indicator}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching macro data: {str(e)}")
            return pd.DataFrame()

    def get_trade_calendar(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Fetch trade calendar from tushare.

        Args:
            start_date: Start date
            end_date: End date
            exchange: Exchange code (SSE, SZSE, etc.)

        Returns:
            DataFrame with trade calendar
        """
        try:
            df = self.pro.trade_cal(
                start_date=start_date,
                end_date=end_date,
                exchange=exchange
            )

            if df is not None and not df.empty:
                logger.info(f"Fetched trade calendar: {len(df)} records")
                return df
            else:
                logger.warning("No trade calendar data returned")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching trade calendar: {str(e)}")
            return pd.DataFrame()

    def fetch_asset_class_data(
        self,
        asset_config: Dict,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE
    ) -> pd.DataFrame:
        """
        Fetch data for an asset class based on configuration.

        Args:
            asset_config: Asset configuration dict
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data
        """
        asset_type = asset_config.get('type')
        code = asset_config.get('code')

        if asset_type == 'index':
            return self.get_index_daily(code, start_date, end_date)
        elif asset_type == 'bond':
            return self.get_bond_index_daily(code, start_date, end_date)
        elif asset_type == 'stock':
            return self.get_stock_daily(code, start_date, end_date)
        else:
            logger.warning(f"Unknown asset type: {asset_type}")
            return pd.DataFrame()

    def save_data(
        self,
        data: pd.DataFrame,
        filename: str,
        data_dir: Path = DATA_DIR
    ) -> Path:
        """
        Save data to CSV file.

        Args:
            data: DataFrame to save
            filename: Output filename
            data_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = data_dir / filename

        if not data.empty:
            data.to_csv(output_path)
            logger.info(f"Data saved to {output_path}")
        else:
            logger.warning("Cannot save empty DataFrame")

        return output_path

    def load_data(
        self,
        filename: str,
        data_dir: Path = DATA_DIR,
        index_col: str = None
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Input filename
            data_dir: Input directory
            index_col: Column to use as index

        Returns:
            DataFrame loaded from file
        """
        file_path = data_dir / filename

        if file_path.exists():
            df = pd.read_csv(file_path, index_col=index_col)
            logger.info(f"Data loaded from {file_path}")
            return df
        else:
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()


def convert_market_code(market_code: str) -> str:
    """
    Convert market code format for tushare API.

    Args:
        market_code: Market code (e.g., 'sh000300', '000300.SH')

    Returns:
        tushare format code (e.g., '000300.SH')
    """
    code = market_code.lower()

    if code.startswith('sh'):
        return code[2:].upper() + '.SH'
    elif code.startswith('sz'):
        return code[2:].upper() + '.SZ'
    elif '.' in code:
        return code.upper()
    else:
        if code.startswith('000') or code.startswith('399'):
            return code.upper() + '.SZ'
        else:
            return code.upper() + '.SH'


def create_portfolio_data_loader(
    assets: List[str],
    asset_types: Dict[str, str],
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE
) -> DataLoader:
    """
    Create a configured DataLoader for portfolio assets.

    Args:
        assets: List of asset codes
        asset_types: Dict mapping asset codes to types
        start_date: Start date
        end_date: End date

    Returns:
        Configured DataLoader instance
    """
    loader = DataLoader()

    asset_configs = {}
    for asset in assets:
        asset_configs[asset] = {
            'code': convert_market_code(asset),
            'type': asset_types.get(asset, 'index')
        }

    return loader, asset_configs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = DataLoader()

    test_index = '000300.SH'
    data = loader.get_index_daily(test_index, '20170101', '20171117')

    if not data.empty:
        print(f"Fetched {len(data)} records")
        print(data.head())
    else:
        print("Failed to fetch data")