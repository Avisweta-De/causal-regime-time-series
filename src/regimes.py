"""
Regime Detection Module

Implements:
- Gaussian Mixture Model (GMM) for regime detection
- Hidden Markov Model (HMM) alternatives
- Regime labeling and characterization
- Regime transition analysis
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector:
    """Detect market regimes using Gaussian Mixture Models"""
    
    def __init__(self, returns_data, n_regimes=3, random_state=42):
        """
        Initialize regime detector
        
        Parameters:
        -----------
        returns_data : pd.DataFrame or pd.Series
            Daily returns data
        n_regimes : int
            Number of regimes to detect (default: 3 for Bull/Neutral/Crisis)
        random_state : int
            For reproducibility
        """
        self.returns = returns_data
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regimes = None
        self.regime_labels = None
        self.model_score = None
        
    def fit_gmm(self, features=None):
        """
        Fit Gaussian Mixture Model for regime detection
        
        Parameters:
        -----------
        features : pd.DataFrame, optional
            Features to use (default: returns only)
            Can include volatility, other indicators
            
        Returns:
        --------
        Fitted GMM model
        """
        if features is None:
            # Use returns as feature
            if isinstance(self.returns, pd.DataFrame):
                # Use first column if multiple
                X = self.returns.iloc[:, 0].values.reshape(-1, 1)
            else:
                X = self.returns.values.reshape(-1, 1)
        else:
            X = features.values
        
        # Fit GMM
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=self.random_state
        )
        
        self.model.fit(X)
        self.regimes = self.model.predict(X)
        self.model_score = self.model.score(X)
        
        print(f'✅ GMM fitted successfully')
        print(f'   BIC Score: {self.model.bic(X):.2f}')
        print(f'   AIC Score: {self.model.aic(X):.2f}')
        print(f'   Log-likelihood: {self.model_score:.2f}')
        
        return self.model
    
    def get_model_metrics(self, X=None):
        """
        Get model evaluation metrics
        
        Returns:
        --------
        dict with Silhouette score, Davies-Bouldin index, etc.
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        if X is None:
            if isinstance(self.returns, pd.DataFrame):
                X = self.returns.iloc[:, 0].values.reshape(-1, 1)
            else:
                X = self.returns.values.reshape(-1, 1)
        
        metrics = {
            'silhouette_score': silhouette_score(X, self.regimes),
            'davies_bouldin_index': davies_bouldin_score(X, self.regimes),
            'bic': self.model.bic(X),
            'aic': self.model.aic(X),
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_
        }
        
        return metrics
    
    def label_regimes(self, by='volatility'):
        """
        Label regimes by economic meaning
        
        Parameters:
        -----------
        by : str
            'volatility' - Low vol = Bull, Med = Neutral, High = Crisis
            'returns' - High ret = Bull, Low = Crisis, etc.
            
        Returns:
        --------
        pd.Series with regime labels
        """
        if self.regimes is None:
            raise ValueError('Fit model first using fit_gmm()')
        
        regime_stats = pd.DataFrame({
            'regime': self.regimes,
            'returns': self.returns.values if isinstance(self.returns, pd.Series) 
                      else self.returns.iloc[:, 0].values
        }).groupby('regime')['returns'].agg(['mean', 'std', 'count'])
        
        if by == 'volatility':
            # Sort by volatility (std dev)
            sorted_regimes = regime_stats['std'].sort_values().index
            labels = {
                sorted_regimes[0]: 'Bull',
                sorted_regimes[1]: 'Neutral',
                sorted_regimes[2]: 'Crisis'
            }
        elif by == 'returns':
            # Sort by average return
            sorted_regimes = regime_stats['mean'].sort_values(ascending=False).index
            labels = {
                sorted_regimes[0]: 'Bull',
                sorted_regimes[1]: 'Neutral',
                sorted_regimes[2]: 'Crisis'
            }
        else:
            raise ValueError(f'Unknown labeling method: {by}')
        
        self.regime_labels = pd.Series(
            [labels.get(r, f'Regime_{r}') for r in self.regimes],
            index=self.returns.index if hasattr(self.returns, 'index') else None
        )
        
        return self.regime_labels
    
    def get_regime_characteristics(self, data, regime_col='Regime_Label'):
        """
        Compute statistics for each regime
        
        Parameters:
        -----------
        data : pd.DataFrame
            Full dataset with regime labels
        regime_col : str
            Column name with regime labels
            
        Returns:
        --------
        pd.DataFrame with regime statistics
        """
        primary_asset = data.columns[0] if isinstance(self.returns, pd.DataFrame) else data.index.name
        
        if isinstance(self.returns, pd.DataFrame):
            primary_asset = self.returns.columns[0]
        else:
            primary_asset = '^GSPC'  # Default
        
        characteristics = data.groupby(regime_col)[primary_asset].agg([
            ('Mean Return', 'mean'),
            ('Volatility', 'std'),
            ('Sharpe Ratio', lambda x: (x.mean() / x.std()) * np.sqrt(252)),
            ('Skewness', 'skew'),
            ('Kurtosis', 'kurtosis'),
            ('Min Return', 'min'),
            ('Max Return', 'max'),
            ('Count', 'count'),
            ('% of Days', lambda x: (len(x) / len(data)) * 100)
        ])
        
        return characteristics.round(4)
    
    def analyze_regime_transitions(self, data):
        """
        Analyze regime switches and durations
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with regime labels
            
        Returns:
        --------
        dict with transition statistics
        """
        regime_col = 'Regime_Label' if 'Regime_Label' in data.columns else 'regime'
        
        # Count transitions
        transitions = (data[regime_col] != data[regime_col].shift()).sum() - 1
        
        # Calculate durations
        data['regime_change'] = (data[regime_col] != data[regime_col].shift()).astype(int)
        data['regime_duration'] = data.groupby(
            (data['regime_change']).cumsum()
        ).cumcount() + 1
        
        duration_stats = data.groupby(regime_col)['regime_duration'].agg([
            'mean', 'min', 'max', 'std'
        ])
        
        results = {
            'total_transitions': transitions,
            'avg_transitions_per_year': transitions / (len(data) / 252),
            'duration_statistics': duration_stats,
            'regime_frequencies': data[regime_col].value_counts(normalize=True)
        }
        
        return results


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection
    Note: Requires hmmlearn package
    """
    
    def __init__(self, returns_data, n_states=3):
        """
        Initialize HMM regime detector
        
        Parameters:
        -----------
        returns_data : pd.DataFrame or pd.Series
        n_states : int
            Number of hidden states
        """
        self.returns = returns_data
        self.n_states = n_states
        self.model = None
        self.states = None
        
        try:
            from hmmlearn.hmm import GaussianHMM
            self.GaussianHMM = GaussianHMM
            self.hmm_available = True
        except ImportError:
            print('⚠️  hmmlearn not installed. Install with: pip install hmmlearn')
            self.hmm_available = False
    
    def fit_hmm(self):
        """
        Fit Hidden Markov Model
        
        Returns:
        --------
        Fitted HMM model
        """
        if not self.hmm_available:
            raise ImportError('hmmlearn not installed')
        
        X = self.returns.values.reshape(-1, 1)
        
        self.model = self.GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=1000)
        self.model.fit(X)
        self.states = self.model.predict(X)
        
        print(f'✅ HMM fitted with {self.n_states} states')
        print(f'   Converged: {self.model.monitor_.converged}')
        
        return self.model
    
    def predict_next_state(self, lookback=5):
        """
        Predict next regime state
        
        Parameters:
        -----------
        lookback : int
            Number of recent periods to use
            
        Returns:
        --------
        Next predicted state
        """
        if self.model is None:
            raise ValueError('Fit model first')
        
        recent_data = self.returns.iloc[-lookback:].values.reshape(-1, 1)
        next_state = self.model.predict(recent_data)[-1]
        
        return next_state


def compare_regimes(data_dict):
    """
    Compare regimes across multiple assets
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of {asset_name: returns_series}
        
    Returns:
    --------
    dict with regime overlap statistics
    """
    detectors = {}
    regimes_all = {}
    
    for asset, returns_data in data_dict.items():
        detector = RegimeDetector(returns_data, n_regimes=3)
        detector.fit_gmm()
        detector.label_regimes()
        
        detectors[asset] = detector
        regimes_all[asset] = detector.regime_labels
    
    # Create comparison dataframe
    regime_comparison = pd.DataFrame(regimes_all)
    
    # Calculate regime agreement (% of time in same regime)
    agreement = {}
    for asset1 in data_dict.keys():
        for asset2 in data_dict.keys():
            if asset1 < asset2:
                agreement[f'{asset1} vs {asset2}'] = (
                    regime_comparison[asset1] == regime_comparison[asset2]
                ).sum() / len(regime_comparison)
    
    return {
        'detectors': detectors,
        'regime_comparison': regime_comparison,
        'regime_agreement': pd.Series(agreement)
    }
