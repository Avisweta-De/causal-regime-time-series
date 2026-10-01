"""
Data Processing & Management Module

Handles:
- Data loading (Yahoo Finance) and preprocessing
- Return calculation with data-quality checks
- Missing value handling and alignment
"""

import pandas as pd
import numpy as np
from datetime import datetime


def compute_returns(prices: pd.DataFrame, method: str = 'simple',
                    forward_fill: bool = True) -> pd.DataFrame:
    """
    Convert prices to daily returns and invalidate impossible values.

    Markets close on different holidays, so prices are forward-filled first
    (a closed market gets a 0% return that day). Any return computed from a
    non-positive price is set to NaN. This matters for WTI crude futures,
    which settled at -$37.63 on 2020-04-20: a naive pct_change gives -306%
    that day and -127% the next, which are not real returns for any holder
    and badly distort means, correlations and VAR coefficients.

    Parameters:
    -----------
    prices : pd.DataFrame
        Price levels, one column per asset
    method : str
        'simple' (pct change) or 'log'
    forward_fill : bool
        Forward-fill missing prices before computing returns

    Returns:
    --------
    pd.DataFrame of returns (first row dropped). Invalid days are NaN.
    """
    p = prices.ffill() if forward_fill else prices
    valid = (p > 0) & (p.shift(1) > 0)

    if method == 'log':
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.log(p / p.shift(1))
    elif method == 'simple':
        returns = p / p.shift(1) - 1
    else:
        raise ValueError(f"Unknown method: {method}")

    returns = returns.where(valid)
    return returns.iloc[1:]


def data_quality_report(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing prices, non-positive prices and invalidated returns per asset."""
    return pd.DataFrame({
        'missing_prices': prices.isna().sum(),
        'non_positive_prices': (prices <= 0).sum(),
        'invalid_returns_set_to_nan': returns.isna().sum(),
        'min_return': returns.min(),
        'max_return': returns.max(),
    })


class DataLoader:
    """Load and preprocess financial data from Yahoo Finance"""

    def __init__(self, tickers, start_date='2010-01-01', end_date=None):
        """
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date for data
        end_date : str, optional
            End date for data (default: today)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.prices = None
        self.returns = None

    def download_data(self):
        """
        Download daily closing prices from Yahoo Finance.

        Recent yfinance versions return split/dividend-adjusted prices in
        'Close' (auto_adjust=True) and no longer include 'Adj Close'.

        Returns:
        --------
        pd.DataFrame of prices, or None if the download fails
        """
        import yfinance as yf  # imported lazily so offline use works without it

        print(f'📥 Downloading data for {len(self.tickers)} assets...')
        print(f'   Period: {self.start_date} to {self.end_date}')

        try:
            data = yf.download(self.tickers, start=self.start_date,
                               end=self.end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']]
            prices.columns.name = None
            self.prices = prices
            print(f'✅ Downloaded {len(prices)} observations')
            return prices
        except Exception as e:
            print(f'❌ Error downloading data: {e}')
            return None

    def calculate_returns(self, prices=None, method='simple'):
        """
        Calculate returns from prices (see compute_returns for the cleaning rules).

        Parameters:
        -----------
        prices : pd.DataFrame, optional
        method : str
            'simple' (default) or 'log'
        """
        if prices is None:
            prices = self.prices
        self.returns = compute_returns(prices, method=method)
        return self.returns

    def calculate_volatility(self, returns=None, window=20):
        """Rolling standard deviation of returns."""
        if returns is None:
            returns = self.returns
        return returns.rolling(window).std()

    def prepare_var_data(self, prices=None):
        """Log returns with invalid days removed, ready for VAR modelling."""
        if prices is None:
            prices = self.prices
        return compute_returns(prices, method='log').dropna()


def align_multiasset_data(*dataframes):
    """
    Align multiple time series to their common dates.

    Returns:
    --------
    list of aligned dataframes (or a single dataframe if one was passed)
    """
    common = dataframes[0].index
    for df in dataframes[1:]:
        common = common.intersection(df.index)
    common = common.sort_values()
    aligned = [df.loc[common] for df in dataframes]
    return aligned if len(aligned) > 1 else aligned[0]


def handle_missing_values(data, method='forward_fill'):
    """
    Handle missing values in time series.

    Note: 'forward_fill' only fills forward. Backward filling would copy
    future values into the past, which is a form of look-ahead, so leading
    NaNs are left in place.

    method : 'forward_fill', 'interpolate', 'drop'
    """
    if method == 'forward_fill':
        return data.ffill()
    elif method == 'interpolate':
        return data.interpolate(method='linear', limit_direction='forward')
    elif method == 'drop':
        return data.dropna()
    else:
        raise ValueError(f'Unknown method: {method}')


def resample_data(data, freq='D'):
    """Resample to another frequency, keeping the last value of each period."""
    return data.resample(freq).last()
