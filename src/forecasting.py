"""
Regime Forecasting & Transition Analysis

- Markov transition matrix and multi-step regime probabilities
- Machine-learning regime forecasts evaluated OUT OF SAMPLE with
  time-ordered splits and compared with a naive "no change" baseline

Regime ids follow the package convention 0=Calm, 1=Elevated, 2=Crisis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .regimes import REGIME_NAMES


class RegimeForecaster:
    """Predict regimes 1-k steps ahead."""

    def __init__(self, regimes: pd.Series, returns: Union[pd.Series, pd.DataFrame],
                 n_regimes: int = 3):
        """
        Parameters:
        -----------
        regimes : pd.Series
            Ordinal regime per date
        returns : pd.Series or pd.DataFrame
            Daily returns (first column used as the primary asset)
        """
        self.regimes = regimes.dropna().astype(int)
        self.returns = returns
        self.n_regimes = n_regimes
        self.transition_matrix = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.regime_names = REGIME_NAMES

    def estimate_transition_matrix(self) -> np.ndarray:
        """
        Daily Markov transition probabilities (rows = from, columns = to).

        A regime that never appears gets a row that stays put (identity),
        so every row is a valid probability distribution.
        """
        k = self.n_regimes
        counts = np.zeros((k, k))
        values = self.regimes.values
        for a, b in zip(values[:-1], values[1:]):
            counts[a, b] += 1
        rows = counts.sum(axis=1, keepdims=True)
        tm = np.divide(counts, rows, out=np.eye(k), where=rows > 0)
        self.transition_matrix = tm
        return tm

    def forecast_next_regime_markov(self, current_regime: int, steps: int = 1) -> Dict[int, float]:
        """Probability of each regime `steps` days ahead given today's regime."""
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        p = np.zeros(self.n_regimes)
        p[current_regime] = 1.0
        p = p @ np.linalg.matrix_power(self.transition_matrix, steps)
        return {i: float(p[i]) for i in range(self.n_regimes)}

    def _primary_returns(self) -> pd.Series:
        if isinstance(self.returns, pd.DataFrame):
            return self.returns.iloc[:, 0]
        return self.returns

    def compute_technical_features(self, returns: pd.Series = None, lookback: int = 20) -> pd.DataFrame:
        """
        Features known at the close of each day (no future data).
        """
        r = (self._primary_returns() if returns is None else returns).dropna()
        f = pd.DataFrame(index=r.index)
        f['momentum_20'] = r.rolling(lookback).mean()
        f['momentum_60'] = r.rolling(60).mean()
        f['volatility_20'] = r.rolling(lookback).std()
        f['volatility_ratio'] = r.rolling(5).std() / r.rolling(20).std()
        f['skewness_20'] = r.rolling(lookback).skew()
        if isinstance(self.returns, pd.DataFrame) and self.returns.shape[1] > 1:
            a, b = self.returns.columns[:2]
            f['correlation'] = self.returns[a].rolling(lookback).corr(self.returns[b])
        wealth = (1 + r).cumprod()
        f['drawdown'] = wealth / wealth.cummax() - 1
        f['return_today'] = r
        f['return_lag5'] = r.shift(5)
        return f.dropna()

    def _dataset(self, steps_ahead: int):
        X = self.compute_technical_features()
        X = X.loc[X.index.intersection(self.regimes.index)]
        target = self.regimes.shift(-steps_ahead).reindex(X.index)
        current = self.regimes.reindex(X.index)
        keep = target.notna()
        return X[keep], target[keep].astype(int), current[keep].astype(int)

    def _new_model(self, model_type: str):
        if model_type == 'rf':
            return RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                          random_state=42, n_jobs=-1)
        return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.05, random_state=42)

    def train_ml_forecaster(self, model_type: str = 'rf', steps_ahead: int = 1,
                            n_splits: int = 5) -> Dict:
        """
        Train a classifier for the regime `steps_ahead` days ahead.

        Evaluation uses TimeSeriesSplit (train on the past, test on the next
        block). The reported numbers are out-of-sample and are shown next to
        a "no change" baseline that predicts today's regime, which is hard to
        beat because regimes are persistent.

        Caveat: if `regimes` came from a full-sample fit, the target labels
        themselves were defined using the whole history. Use walk-forward
        regimes for a fully out-of-sample exercise.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import balanced_accuracy_score

        X, y, current = self._dataset(steps_ahead)
        self.feature_columns = list(X.columns)

        model_acc, base_acc, model_bal, base_bal = [], [], [], []
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            scaler = StandardScaler().fit(X.iloc[train_idx])
            model = self._new_model(model_type).fit(scaler.transform(X.iloc[train_idx]), y.iloc[train_idx])
            pred = model.predict(scaler.transform(X.iloc[test_idx]))
            truth, naive = y.iloc[test_idx].values, current.iloc[test_idx].values
            model_acc.append((pred == truth).mean())
            base_acc.append((naive == truth).mean())
            model_bal.append(balanced_accuracy_score(truth, pred))
            base_bal.append(balanced_accuracy_score(truth, naive))

        # Final model on all data, for use on new observations
        self.scaler = StandardScaler().fit(X)
        self.ml_model = self._new_model(model_type).fit(self.scaler.transform(X), y)

        importance = pd.Series(self.ml_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        return {
            'model_type': model_type,
            'steps_ahead': steps_ahead,
            'oos_accuracy': float(np.mean(model_acc)),
            'baseline_accuracy': float(np.mean(base_acc)),
            'oos_balanced_accuracy': float(np.mean(model_bal)),
            'baseline_balanced_accuracy': float(np.mean(base_bal)),
            'feature_importance': importance,
        }

    def predict_next_regime_ml(self, last_features: pd.Series) -> Dict[int, float]:
        """Regime probabilities for one row of features."""
        if self.ml_model is None:
            raise ValueError("Train ML model first using train_ml_forecaster()")
        row = pd.DataFrame([last_features[self.feature_columns].values], columns=self.feature_columns)
        probs = self.ml_model.predict_proba(self.scaler.transform(row))[0]
        out = {i: 0.0 for i in range(self.n_regimes)}
        for cls, p in zip(self.ml_model.classes_, probs):
            out[int(cls)] = float(p)
        return out

    def get_regime_signals(self, current_regime: int, forecast_horizon: int = 5) -> Dict:
        """
        Allocation signal from the Markov forecast.

        Returns the most likely regime after `forecast_horizon` days and the
        probability of being in Crisis, which is usually the more useful
        risk number.
        """
        probs = self.forecast_next_regime_markov(current_regime, forecast_horizon)
        predicted = max(probs, key=probs.get)
        signal = 'HOLD'
        if predicted < current_regime:
            signal = 'INCREASE EXPOSURE'
        elif predicted > current_regime:
            signal = 'REDUCE EXPOSURE'
        return {
            'signal': signal,
            'current_regime': self.regime_names[current_regime],
            'predicted_regime': self.regime_names[predicted],
            'confidence': probs[predicted],
            'crisis_probability': probs.get(2, np.nan),
            'horizon_days': forecast_horizon,
            'regime_probabilities': {self.regime_names[k]: v for k, v in probs.items()},
        }

    def print_transition_matrix(self) -> None:
        """Pretty-print the transition matrix."""
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        names = [self.regime_names[i] for i in range(self.n_regimes)]
        print("Daily transition probabilities (rows = today, columns = tomorrow)")
        print(pd.DataFrame(self.transition_matrix, index=names, columns=names).round(3))

    def analyze_regime_persistence(self) -> Dict[str, float]:
        """Expected run length in days, 1 / (1 - P(stay)), for each regime."""
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        out = {}
        for i in range(self.n_regimes):
            stay = self.transition_matrix[i, i]
            out[self.regime_names[i]] = 1 / (1 - stay) if stay < 1 else np.inf
        return out
