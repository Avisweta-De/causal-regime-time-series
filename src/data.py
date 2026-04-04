"""
Data Processing & Management Module

Handles:
- Data loading and preprocessing
- Feature engineering
- Data validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class DataLoader:
    """Load and preprocess financial data from Yahoo Finance"""
    
    def __init__(self, tickers, start_date='2010-01-01', end_date=None):
        """
        Initialize data loader
        
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
        self.data = None
        self.prices = None
        self.returns = None
    
    def download_data(self):
        """
        Download price data from Yahoo Finance
        
        Returns:
        --------
        pd.DataFrame with Adj Close prices
        """
        print(f'📥 Downloading data for {len(self.tickers)} assets...')
        print(f'   Period: {self.start_date} to {self.end_date}')
        
        try:
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            # Extract Adj Close
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            else:
                prices = data[['Adj Close']]
            
            self.prices = prices
            print(f'✅ Downloaded {len(prices)} observations')
            
            return prices
        
        except Exception as e:
            print(f'❌ Error downloading data: {e}')
            return None
    
    def calculate_returns(self, prices=None, method='log'):
        """
        Calculate returns from prices
        
        Parameters:
        -----------
        prices : pd.DataFrame, optional
        method : str
            'log' for log returns, 'simple' for simple returns
            
        Returns:
        --------
        pd.DataFrame with returns
        """
        if prices is None:
            prices = self.prices
        
        if method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()
        
        self.returns = returns
        return returns
    
    def calculate_volatility(self, returns=None, window=20):
        """
        Calculate rolling volatility
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
        window : int
            Rolling window size
            
        Returns:
        --------
        pd.DataFrame with volatility
        """
        if returns is None:
            returns = self.returns
        
        volatility = returns.rolling(window).std()
        return volatility
    
    def prepare_var_data(self, prices=None, assets=None):
        """
        Prepare clean data for VAR modeling
        
        Returns:
        --------
        pd.DataFrame ready for VAR
        """
        if prices is None:
            prices = self.prices
        
        # Align dates
        prices_clean = prices.dropna()
        
        # Calculate returns
        returns = np.log(prices_clean / prices_clean.shift(1)).dropna()
        
        return returns


def align_multiasset_data(*dataframes):
    """
    Align multiple time series to common dates
    
    Parameters:
    -----------
    *dataframes : pd.DataFrame
        Multiple dataframes to align
        
    Returns:
    --------
    list of aligned dataframes
    """
    # Find common dates
    all_dates = set(dataframes[0].index)
    for df in dataframes[1:]:
        all_dates = all_dates.intersection(set(df.index))
    
    aligned = [df.loc[sorted(all_dates)] for df in dataframes]
    
    return aligned if len(aligned) > 1 else aligned[0]


def handle_missing_values(data, method='forward_fill'):
    """
    Handle missing values in time series
    
    Parameters:
    -----------
    data : pd.DataFrame
    method : str
        'forward_fill', 'backward_fill', 'interpolate', 'drop'
        
    Returns:
    --------
    pd.DataFrame with NaNs handled
    """
    if method == 'forward_fill':
        return data.fillna(method='ffill').fillna(method='bfill')
    elif method == 'backward_fill':
        return data.fillna(method='bfill').fillna(method='ffill')
    elif method == 'interpolate':
        return data.interpolate(method='linear')
    elif method == 'drop':
        return data.dropna()
    else:
        raise ValueError(f'Unknown method: {method}')


def resample_data(data, freq='D'):
    """
    Resample time series to different frequency
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    freq : str
        Frequency ('D', 'W', 'M', etc.)
        
    Returns:
    --------
    Resampled dataframe (using last value of period)
    """
    return data.resample(freq).last()
