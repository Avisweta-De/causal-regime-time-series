"""
Causal Inference Module

Provides functions for:
- Granger Causality Testing
- VAR (Vector AutoRegression) Modeling
- Impulse Response Functions
- Regime-Conditional Analysis
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')


class CausalityAnalyzer:
    """Comprehensive causal inference toolkit for multivariate time series"""
    
    def __init__(self, data, assets):
        """
        Initialize causal analyzer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with asset columns
        assets : list
            List of asset column names to analyze
        """
        self.data = data
        self.assets = assets
        self.gc_results = None
        self.var_model = None
        self.irf = None
        
    def test_stationarity(self, series, name='Series'):
        """
        Perform Augmented Dickey-Fuller test
        
        Returns:
        --------
        dict with test results
        """
        result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'observations': result[3],
            'is_stationary': result[1] <= 0.05,
            'critical_values': result[4]
        }
    
    def granger_causality_matrix(self, data=None, maxlag=5):
        """
        Compute Granger causality p-values for all asset pairs
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Data to use (default: self.data)
        maxlag : int
            Maximum lag for causality test
            
        Returns:
        --------
        pd.DataFrame
            Causality p-value matrix
        """
        if data is None:
            data = self.data
            
        gc_matrix = pd.DataFrame(
            np.zeros((len(self.assets), len(self.assets))),
            index=self.assets, columns=self.assets
        )
        
        for cause in self.assets:
            for effect in self.assets:
                if cause != effect:
                    try:
                        test_data = data[[effect, cause]].dropna()
                        result = grangercausalitytests(test_data, maxlag, verbose=False)
                        # Use lag-1 p-value
                        p_value = result[1][0][1]
                        gc_matrix.loc[cause, effect] = p_value
                    except:
                        gc_matrix.loc[cause, effect] = np.nan
        
        self.gc_results = gc_matrix
        return gc_matrix
    
    def get_significant_causality(self, threshold=0.05):
        """Extract significant causal relationships"""
        if self.gc_results is None:
            self.granger_causality_matrix()
        
        significant = []
        for cause in self.gc_results.index:
            for effect in self.gc_results.columns:
                p_val = self.gc_results.loc[cause, effect]
                if pd.notna(p_val) and p_val < threshold and cause != effect:
                    significant.append({
                        'cause': cause,
                        'effect': effect,
                        'p_value': p_val,
                        'significant': True
                    })
        
        return pd.DataFrame(significant).sort_values('p_value')
    
    def fit_var(self, data=None, maxlags=None, ic='AIC'):
        """
        Fit Vector AutoRegression model
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
        maxlags : int, optional
        ic : str
            Information criterion ('AIC' or 'BIC')
            
        Returns:
        --------
        Fitted VAR model results
        """
        if data is None:
            data = self.data[self.assets].dropna()
        
        model = VAR(data)
        
        if maxlags is None:
            # Auto-select lags
            lag_info = model.select_lags()
            optimal_lags = getattr(lag_info, ic.lower())
        else:
            optimal_lags = maxlags
        
        self.var_model = model.fit(optimal_lags)
        return self.var_model
    
    def get_impulse_response(self, periods=10):
        """
        Compute impulse response functions
        
        Parameters:
        -----------
        periods : int
            Number of periods to compute
            
        Returns:
        --------
        IRF object
        """
        if self.var_model is None:
            self.fit_var()
        
        self.irf = self.var_model.irf(periods)
        return self.irf
    
    def regime_conditional_causality(self, regime_column, regime_list=None):
        """
        Compute Granger causality separately for each regime
        
        Parameters:
        -----------
        regime_column : str
            Column name containing regime labels
        regime_list : list, optional
            Specific regimes to analyze
            
        Returns:
        --------
        dict
            Causality results by regime
        """
        results_by_regime = {}
        
        regimes = regime_list if regime_list else self.data[regime_column].unique()
        
        for regime in sorted(regimes):
            regime_data = self.data[self.data[regime_column] == regime][self.assets]
            
            if len(regime_data) > 10:
                gc_matrix = self.granger_causality_matrix(regime_data, maxlag=3)
                results_by_regime[regime] = gc_matrix
        
        return results_by_regime


def detect_shocks(returns_series, threshold=3.0, window=20):
    """
    Detect anomalies using rolling Z-score
    
    Parameters:
    -----------
    returns_series : pd.Series
        Time series of returns
    threshold : float
        Z-score threshold for anomaly
    window : int
        Rolling window size
        
    Returns:
    --------
    tuple : (z_scores, boolean mask of shocks)
    """
    rolling_mean = returns_series.rolling(window).mean()
    rolling_std = returns_series.rolling(window).std()
    
    z_score = np.abs((returns_series - rolling_mean) / rolling_std)
    shocks = z_score > threshold
    
    return z_score, shocks


def get_shock_events(returns_series, threshold=2.5, top_n=10):
    """
    Get top shock events sorted by magnitude
    
    Parameters:
    -----------
    returns_series : pd.Series
    threshold : float
    top_n : int
        Number of top shocks to return
        
    Returns:
    --------
    pd.DataFrame with shock events
    """
    z_scores, shocks = detect_shocks(returns_series, threshold=threshold)
    
    shock_indices = returns_series.index[shocks]
    shock_values = returns_series[shocks]
    
    shock_df = pd.DataFrame({
        'date': shock_indices,
        'return': shock_values.values,
        'z_score': z_scores[shocks].values,
        'event_type': ['Crash' if x < 0 else 'Rally' for x in shock_values.values]
    })
    
    return shock_df.nlargest(top_n, 'z_score')
