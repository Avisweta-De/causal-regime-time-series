"""
Regime Forecasting & Transition Prediction

Predicts market regime changes for forward-looking trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class RegimeForecaster:
    """
    Predict regime changes 1-5 steps ahead.
    
    Methods:
    - Markov chains: Transition probability estimation
    - Machine learning: RF/GB classifiers for next regime
    - Technical indicators: RSI, Bollinger Bands, Momentum
    """
    
    def __init__(self, regimes: pd.Series, returns: pd.DataFrame):
        """
        Initialize regime forecaster
        
        Parameters:
        -----------
        regimes : pd.Series
            Historical regime labels
        returns : pd.DataFrame or pd.Series
            Daily returns (multi-asset or single)
        """
        self.regimes = regimes
        self.returns = returns
        self.transition_matrix = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.regime_names = {0: 'Bull', 1: 'Neutral', 2: 'Crisis'}
        
    def estimate_transition_matrix(self) -> np.ndarray:
        """
        Estimate Markov transition probabilities between regimes
        
        Returns:
        --------
        np.ndarray
            3x3 transition probability matrix
        """
        # Count transitions
        transition_counts = np.zeros((3, 3))
        
        for i in range(len(self.regimes) - 1):
            current = int(self.regimes.iloc[i])
            next_regime = int(self.regimes.iloc[i+1])
            transition_counts[current, next_regime] += 1
        
        # Convert to probabilities (add smoothing to avoid zero probabilities)
        transition_matrix = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-10)
        self.transition_matrix = transition_matrix
        
        return self.transition_matrix
    
    def forecast_next_regime_markov(self, current_regime: int, steps: int = 1) -> Dict[int, float]:
        """
        Forecast regime probabilities using Markov chains
        
        Parameters:
        -----------
        current_regime : int
            Current regime (0, 1, or 2)
        steps : int
            Number of steps ahead (default 1)
            
        Returns:
        --------
        dict
            Probability distribution over regimes
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        # Initialize probability vector
        prob_vector = np.zeros(3)
        prob_vector[current_regime] = 1.0
        
        # Apply transition matrix 'steps' times
        for _ in range(steps):
            prob_vector = prob_vector @ self.transition_matrix
        
        return {i: prob_vector[i] for i in range(3)}
    
    def compute_technical_features(self, returns: pd.Series, lookback: int = 20) -> pd.DataFrame:
        """
        Compute technical indicator features
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns (single asset)
        lookback : int
            Lookback period for indicators
            
        Returns:
        --------
        pd.DataFrame
            Features for ML model
        """
        # Price levels (from cumulative returns)
        prices = (1 + returns).cumprod()
        
        features = pd.DataFrame(index=returns.index)
        
        # Momentum
        features['momentum_20'] = returns.rolling(lookback).mean()
        features['momentum_60'] = returns.rolling(60).mean()
        
        # Volatility
        features['volatility_20'] = returns.rolling(lookback).std()
        features['volatility_ratio'] = returns.rolling(5).std() / returns.rolling(20).std()
        
        # Skewness (tail risk)
        features['skewness_20'] = returns.rolling(lookback).skew()
        
        # Rolling correlation (if multi-asset)
        if isinstance(self.returns, pd.DataFrame):
            assets = self.returns.columns
            if len(assets) > 1:
                features['correlation'] = self.returns[assets[0]].rolling(lookback).corr(
                    self.returns[assets[1]])
        
        # Drawdown indicator
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        features['drawdown'] = (cum_returns - running_max) / running_max
        
        # Price acceleration
        features['returns_lag1'] = returns.shift(1)
        features['returns_lag5'] = returns.shift(5)
        
        return features.dropna()
    
    def train_ml_forecaster(self, model_type: str = 'rf', steps_ahead: int = 1) -> Dict:
        """
        Train ML model to forecast regime
        
        Parameters:
        -----------
        model_type : str
            'rf' (Random Forest) or 'gb' (Gradient Boosting)
        steps_ahead : int
            Number of steps to predict ahead (1-5)
            
        Returns:
        --------
        dict
            Training metrics
        """
        # Get returns for a single asset
        if isinstance(self.returns, pd.DataFrame):
            asset_returns = self.returns.iloc[:, 0]
        else:
            asset_returns = self.returns
        
        # Compute features
        X = self.compute_technical_features(asset_returns)
        
        # Target: regime at t+steps_ahead
        y = pd.Series(index=X.index, dtype=int)
        valid_idx = X.index[:-steps_ahead]
        y.loc[valid_idx] = self.regimes.loc[X.index[steps_ahead:]].values[:len(valid_idx)]
        
        # Remove NaN targets
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if model_type == 'rf':
            self.ml_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                   random_state=42, n_jobs=-1)
        else:  # 'gb'
            self.ml_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                       learning_rate=0.1, random_state=42)
        
        self.ml_model.fit(X_scaled, y)
        
        # Compute in-sample accuracy
        # NOTE: This is training (in-sample) accuracy and will be optimistic.
        # Use cross-validation (e.g., TimeSeriesSplit) for unbiased estimates.
        accuracy = self.ml_model.score(X_scaled, y)
        
        # Feature importance
        feature_importance = pd.Series(
            self.ml_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        return {
            'accuracy': accuracy,
            'model_type': model_type,
            'steps_ahead': steps_ahead,
            'feature_importance': feature_importance
        }
    
    def predict_next_regime_ml(self, last_features: pd.Series) -> Dict[int, float]:
        """
        Predict next regime using trained ML model
        
        Parameters:
        -----------
        last_features : pd.Series
            Latest feature values
            
        Returns:
        --------
        dict
            Probability distribution over regimes
        """
        if self.ml_model is None:
            raise ValueError("Train ML model first using train_ml_forecaster()")
        
        # Scale features
        features_scaled = self.scaler.transform([last_features])
        
        # Get probabilities
        probs = self.ml_model.predict_proba(features_scaled)[0]
        
        return {i: probs[i] for i in range(3)}
    
    def get_regime_signals(self, 
                          current_regime: int, 
                          forecast_horizon: int = 5) -> Dict[str, Union[str, float]]:
        """
        Generate trading signals based on regime forecasts
        
        Parameters:
        -----------
        current_regime : int
            Current regime
        forecast_horizon : int
            Days ahead to forecast
            
        Returns:
        --------
        dict
            Trading signals and confidence
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        # Get Markov forecast
        forecast_probs = self.forecast_next_regime_markov(current_regime, forecast_horizon)
        
        # Most likely regime
        predicted_regime = max(forecast_probs, key=forecast_probs.get)
        confidence = forecast_probs[predicted_regime]
        
        # Generate signal
        signal = 'HOLD'
        if predicted_regime == 0 and current_regime != 0:  # Moving to Bull
            signal = 'BUY'
        elif predicted_regime == 2 and current_regime != 2:  # Moving to Crisis
            signal = 'SELL'
        
        return {
            'signal': signal,
            'current_regime': self.regime_names[current_regime],
            'predicted_regime': self.regime_names[predicted_regime],
            'confidence': confidence,
            'horizon_days': forecast_horizon,
            'regime_probabilities': {self.regime_names[k]: v for k, v in forecast_probs.items()}
        }
    
    def print_transition_matrix(self) -> None:
        """Pretty print transition matrix"""
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        regime_names_list = ['Bull', 'Neutral', 'Crisis']
        
        print("\n" + "="*70)
        print("MARKOV TRANSITION MATRIX (Regime Probabilities)")
        print("="*70)
        print("\nRows = From Regime | Columns = To Regime\n")
        
        df = pd.DataFrame(
            self.transition_matrix,
            index=regime_names_list,
            columns=regime_names_list
        )
        
        print(df.round(4))
        print("\n" + "="*70)
    
    def analyze_regime_persistence(self) -> Dict[str, float]:
        """
        Analyze how long regimes typically persist
        
        Returns:
        --------
        dict
            Average duration by regime (in days)
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        # Calculate persistence (stay probability)
        durations = {}
        for i, regime_name in self.regime_names.items():
            stay_prob = self.transition_matrix[i, i]
            
            # Expected duration = 1 / (1 - stay_prob)
            if stay_prob < 1:
                expected_duration = 1 / (1 - stay_prob)
            else:
                expected_duration = np.inf
            
            durations[regime_name] = expected_duration
        
        return durations
