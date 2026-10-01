"""
Regime Detection Module

Implements:
- Rolling regime features that only use information available at each date
- Gaussian Mixture Model (GMM) regime detection with volatility-ordered labels
- Hidden Markov Model (HMM) alternative (optional, needs hmmlearn)
- Regime characterisation and transition analysis

Label convention used everywhere in this package
------------------------------------------------
GMM component numbers are arbitrary and change between fits. After fitting,
components are re-numbered by their average volatility so that:

    0 = 'Calm'      (lowest volatility)
    1 = 'Elevated'  (middle volatility)
    2 = 'Crisis'    (highest volatility)

The labels describe volatility, not the direction of returns.
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

REGIME_NAMES = {0: 'Calm', 1: 'Elevated', 2: 'Crisis'}


def make_regime_features(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Rolling features for regime detection.

    The value on date t uses returns up to and including t, so it is known
    at the close of day t and can only be used to trade from day t+1.

    Returns:
    --------
    pd.DataFrame with columns 'mean', 'vol', 'skew' (leading NaN rows dropped)
    """
    returns = returns.dropna()
    return pd.DataFrame({
        'mean': returns.rolling(window).mean(),
        'vol': returns.rolling(window).std(),
        'skew': returns.rolling(window).skew(),
    }).dropna()


def order_components_by_volatility(model: GaussianMixture, vol_column: int) -> dict:
    """
    Map raw GMM component ids to ordinal regime ids (0 = lowest volatility).

    Parameters:
    -----------
    model : fitted GaussianMixture
    vol_column : int
        Index of the volatility feature in the fitted feature matrix.
        StandardScaler preserves order, so means in scaled units work.
    """
    order = np.argsort(model.means_[:, vol_column])
    return {int(raw): rank for rank, raw in enumerate(order)}


class RegimeDetector:
    """Detect market regimes using Gaussian Mixture Models"""

    def __init__(self, returns_data, n_regimes=3, random_state=42, window=20):
        """
        Parameters:
        -----------
        returns_data : pd.Series or pd.DataFrame
            Daily returns (first column is used if a DataFrame is passed)
        n_regimes : int
            Number of regimes (names are defined for 3)
        random_state : int
            For reproducibility
        window : int
            Rolling window for the default features
        """
        if isinstance(returns_data, pd.DataFrame):
            returns_data = returns_data.iloc[:, 0]
        self.returns = returns_data.dropna()
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.window = window
        self.model = None
        self.scaler = None
        self.features = None
        self.X = None
        self.raw_components = None
        self.regimes = None
        self.regime_labels = None
        self.model_score = None

    def fit_gmm(self, features=None, verbose=True):
        """
        Fit a GMM on standardised features and store ordinal regimes.

        By default the features are the rolling mean, volatility and skew of
        returns (see make_regime_features). Clustering single-day returns
        instead just sorts days by the size of that day's move, which gives
        "regimes" that last about one day.

        Note: this is an in-sample fit on the whole history, useful for
        description. For anything that trades or forecasts, use
        backtesting.walk_forward_regime_backtest, which refits on past data only.

        Parameters:
        -----------
        features : pd.DataFrame, optional
            Custom features. A 'vol' column is used for ordering if present,
            otherwise the first column.
        """
        self.features = (make_regime_features(self.returns, self.window)
                         if features is None else features.dropna())
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.features.values)

        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=self.random_state,
        )
        self.model.fit(self.X)
        self.raw_components = self.model.predict(self.X)

        cols = list(self.features.columns)
        vol_col = cols.index('vol') if 'vol' in cols else 0
        mapping = order_components_by_volatility(self.model, vol_col)
        self.regimes = pd.Series([mapping[c] for c in self.raw_components],
                                 index=self.features.index, name='Regime')
        self.model_score = self.model.score(self.X)

        if verbose:
            print('✅ GMM fitted successfully')
            print(f'   BIC: {self.model.bic(self.X):.2f}')
            print(f'   AIC: {self.model.aic(self.X):.2f}')
            print(f'   Avg log-likelihood per day: {self.model_score:.4f}')
        return self.model

    def get_model_metrics(self):
        """Silhouette, Davies-Bouldin, BIC/AIC and convergence of the fitted model."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        if self.model is None:
            raise ValueError('Fit model first using fit_gmm()')
        return {
            'silhouette_score': silhouette_score(self.X, self.regimes.values),
            'davies_bouldin_index': davies_bouldin_score(self.X, self.regimes.values),
            'bic': self.model.bic(self.X),
            'aic': self.model.aic(self.X),
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_,
        }

    def label_regimes(self):
        """
        Names for the ordinal regimes ('Calm', 'Elevated', 'Crisis').

        Returns:
        --------
        pd.Series of labels on the feature index
        """
        if self.regimes is None:
            raise ValueError('Fit model first using fit_gmm()')
        names = REGIME_NAMES if self.n_regimes == 3 else {i: f'Regime_{i}' for i in range(self.n_regimes)}
        self.regime_labels = self.regimes.map(names).rename('Regime_Label')
        return self.regime_labels

    def get_regime_characteristics(self, returns=None):
        """
        Statistics of daily returns in each regime.

        Columns marked '(next day)' describe the return on the day AFTER the
        regime was observed, which is what a decision-maker could act on.
        """
        if self.regimes is None:
            raise ValueError('Fit model first using fit_gmm()')
        base = self.returns if returns is None else returns
        same = base.reindex(self.regimes.index).groupby(self.regimes.map(REGIME_NAMES))
        ahead = base.shift(-1).reindex(self.regimes.index).groupby(self.regimes.map(REGIME_NAMES))
        out = pd.DataFrame({
            'Mean Return': same.mean(),
            'Volatility': same.std(),
            'Mean Return (next day)': ahead.mean(),
            'Volatility (next day)': ahead.std(),
            'Worst Day': same.min(),
            'Count': same.count(),
            '% of Days': same.count() / len(self.regimes) * 100,
        })
        return out.reindex([n for n in REGIME_NAMES.values() if n in out.index])

    def analyze_regime_transitions(self):
        """Number of switches, run lengths and the empirical daily transition matrix."""
        if self.regimes is None:
            raise ValueError('Fit model first using fit_gmm()')
        reg = self.regimes
        switches = int((reg != reg.shift()).sum() - 1)
        runs = reg.groupby((reg != reg.shift()).cumsum()).agg(['first', 'size'])
        runs['first'] = runs['first'].map(REGIME_NAMES)
        duration_stats = runs.groupby('first')['size'].agg(['mean', 'min', 'max', 'std'])
        if isinstance(reg.index, pd.DatetimeIndex):
            years = (reg.index[-1] - reg.index[0]).days / 365.25
        else:
            years = len(reg) / 252
        names = reg.map(REGIME_NAMES)
        transition = pd.crosstab(names.shift(), names, normalize='index')
        order = [n for n in REGIME_NAMES.values() if n in transition.index]
        return {
            'total_transitions': switches,
            'avg_transitions_per_year': switches / years,
            'duration_statistics': duration_stats.reindex(order),
            'transition_matrix': transition.reindex(index=order, columns=order),
            'regime_frequencies': names.value_counts(normalize=True).reindex(order),
        }


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection (requires the hmmlearn package).
    States are re-ordered by volatility like the GMM detector.
    """

    def __init__(self, returns_data, n_states=3, random_state=42):
        self.returns = returns_data.dropna()
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.states = None
        try:
            from hmmlearn.hmm import GaussianHMM
            self.GaussianHMM = GaussianHMM
            self.hmm_available = True
        except ImportError:
            self.hmm_available = False

    def fit_hmm(self):
        """
        Fit a Gaussian HMM on daily returns; states ordered by volatility.

        predict() uses the whole sample (Viterbi), so the resulting states are
        in-sample. Use filtered_state_probabilities() for a version that only
        uses data up to each date.
        """
        if not self.hmm_available:
            raise ImportError('hmmlearn not installed. Install with: pip install hmmlearn')
        X = self.returns.values.reshape(-1, 1)
        self.model = self.GaussianHMM(n_components=self.n_states, covariance_type='full',
                                      n_iter=1000, random_state=self.random_state)
        self.model.fit(X)
        raw = self.model.predict(X)
        order = np.argsort(self.model.covars_.reshape(self.n_states, -1)[:, 0])
        mapping = {int(r): i for i, r in enumerate(order)}
        self.states = pd.Series([mapping[s] for s in raw], index=self.returns.index)
        self._state_order = order
        return self.model

    def filtered_state_probabilities(self):
        """Probability of each (ordered) state at t using data up to t only."""
        if self.model is None:
            raise ValueError('Fit model first')
        X = self.returns.values.reshape(-1, 1)
        probs = np.vstack([self.model.predict_proba(X[: t + 1])[-1] for t in range(len(X))])
        return pd.DataFrame(probs[:, self._state_order], index=self.returns.index,
                            columns=[REGIME_NAMES.get(i, i) for i in range(self.n_states)])


def compare_regimes(data_dict):
    """
    Fit a regime detector per asset and measure how often their regimes agree.

    Parameters:
    -----------
    data_dict : dict of {asset_name: returns_series}
    """
    detectors, labels = {}, {}
    for asset, series in data_dict.items():
        det = RegimeDetector(series, n_regimes=3)
        det.fit_gmm(verbose=False)
        detectors[asset] = det
        labels[asset] = det.label_regimes()

    comparison = pd.DataFrame(labels).dropna()
    names = list(data_dict.keys())
    agreement = {f'{a} vs {b}': (comparison[a] == comparison[b]).mean()
                 for i, a in enumerate(names) for b in names[i + 1:]}

    return {
        'detectors': detectors,
        'regime_comparison': comparison,
        'regime_agreement': pd.Series(agreement),
    }
