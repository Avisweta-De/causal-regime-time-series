"""
Causal Inference Module

Provides:
- Granger causality F-tests (pure numpy/scipy, no statsmodels needed)
- Regime-conditional Granger tests that keep true calendar lags
- Multiple-testing adjustment (Bonferroni / Benjamini-Hochberg)
- Stationarity tests, VAR models and impulse responses (need statsmodels)
- Shock detection

Note on wording: Granger causality means "past values of X improve the
forecast of Y". It is predictive, not proof of economic causation. Failing
to find it means "no evidence of predictability at this lag", not "the
null hypothesis is confirmed".
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def granger_test(effect: pd.Series, cause: pd.Series, lag: int = 1,
                 mask: pd.Series = None) -> dict:
    """
    Granger causality F-test: do past values of `cause` improve a regression
    of `effect` on its own past values?

    Restricted model:   y_t = c + sum_i a_i y_{t-i}
    Unrestricted model: y_t = c + sum_i a_i y_{t-i} + sum_i b_i x_{t-i}
    F = ((SSR_r - SSR_u) / lag) / (SSR_u / (n - 2*lag - 1))

    This is the statistic statsmodels reports as `ssr_ftest`.

    Parameters:
    -----------
    effect, cause : pd.Series
        Series on a shared date index
    lag : int
        Number of lags of each series
    mask : pd.Series of bool, optional
        Keep only observations where mask is True (e.g. days in one regime).
        Lags still come from the full series, so "yesterday" always means
        the previous trading day, not the previous day in that regime.

    Returns:
    --------
    dict with 'f_stat', 'p_value', 'df_num', 'df_den', 'n_obs'
    """
    df = pd.DataFrame({'y': effect, 'x': cause})
    lagged = {}
    for i in range(1, lag + 1):
        lagged[f'y_l{i}'] = df['y'].shift(i)
        lagged[f'x_l{i}'] = df['x'].shift(i)
    design = pd.concat([df['y'], pd.DataFrame(lagged)], axis=1)
    if mask is not None:
        keep = mask.reindex(design.index).fillna(False).astype(bool)
        design = design[keep]
    design = design.dropna()

    n = len(design)
    k_u = 2 * lag + 1
    if n <= k_u + 1:
        return {'f_stat': np.nan, 'p_value': np.nan, 'df_num': lag, 'df_den': n - k_u, 'n_obs': n}

    y = design['y'].values
    ones = np.ones((n, 1))
    X_r = np.hstack([ones, design[[f'y_l{i}' for i in range(1, lag + 1)]].values])
    X_u = np.hstack([X_r, design[[f'x_l{i}' for i in range(1, lag + 1)]].values])
    ssr_r = np.sum((y - X_r @ np.linalg.lstsq(X_r, y, rcond=None)[0]) ** 2)
    ssr_u = np.sum((y - X_u @ np.linalg.lstsq(X_u, y, rcond=None)[0]) ** 2)

    df_den = n - k_u
    f_stat = ((ssr_r - ssr_u) / lag) / (ssr_u / df_den)
    return {'f_stat': f_stat, 'p_value': stats.f.sf(f_stat, lag, df_den),
            'df_num': lag, 'df_den': df_den, 'n_obs': n}


def adjust_pvalues(pvalues: pd.DataFrame, method: str = 'bonferroni') -> pd.DataFrame:
    """
    Adjust a matrix of p-values for multiple testing (diagonal/NaN ignored).

    method : 'bonferroni' or 'fdr_bh' (Benjamini-Hochberg)
    """
    flat = pvalues.stack().dropna()
    m = len(flat)
    if method == 'bonferroni':
        adj = (flat * m).clip(upper=1.0)
    elif method == 'fdr_bh':
        order = flat.sort_values()
        ranked = order * m / np.arange(1, m + 1)
        adj = ranked[::-1].cummin()[::-1].clip(upper=1.0).reindex(flat.index)
    else:
        raise ValueError(f'Unknown method: {method}')
    return adj.unstack().reindex(index=pvalues.index, columns=pvalues.columns)


class CausalityAnalyzer:
    """Causal inference toolkit for multivariate return series"""

    def __init__(self, data, assets):
        """
        Parameters:
        -----------
        data : pd.DataFrame
            Returns with one column per asset (extra columns are allowed)
        assets : list
            Columns to analyse
        """
        self.data = data
        self.assets = assets
        self.gc_results = None
        self.var_model = None
        self.irf = None

    def test_stationarity(self, series, name='Series'):
        """Augmented Dickey-Fuller test (requires statsmodels)."""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'name': name,
            'adf_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'observations': result[3],
            'is_stationary': result[1] <= 0.05,
            'critical_values': result[4],
        }

    def granger_causality_matrix(self, data=None, lag=1, mask=None, maxlag=None):
        """
        Granger p-values for every ordered pair (rows = cause, columns = effect).

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Defaults to self.data
        lag : int
            Number of lags in the test
        mask : pd.Series of bool, optional
            Restrict the tested observations (see granger_test)
        maxlag : int, optional
            Deprecated alias for `lag`

        Returns:
        --------
        pd.DataFrame of p-values with NaN on the diagonal
        """
        if maxlag is not None:
            lag = maxlag
        data = self.data if data is None else data
        gc = pd.DataFrame(np.nan, index=self.assets, columns=self.assets, dtype=float)
        for cause in self.assets:
            for effect in self.assets:
                if cause != effect:
                    gc.loc[cause, effect] = granger_test(data[effect], data[cause], lag=lag, mask=mask)['p_value']
        self.gc_results = gc
        return gc

    def get_significant_causality(self, threshold=0.05, adjust='bonferroni'):
        """
        Pairs that stay significant after adjusting for multiple tests.

        With 5 assets there are 20 ordered pairs, so about one pair would pass
        p < 0.05 by chance alone; adjustment guards against that.
        """
        if self.gc_results is None:
            self.granger_causality_matrix()
        adjusted = adjust_pvalues(self.gc_results, adjust) if adjust else self.gc_results
        rows = []
        for cause in self.gc_results.index:
            for effect in self.gc_results.columns:
                p_adj = adjusted.loc[cause, effect]
                if pd.notna(p_adj) and p_adj < threshold:
                    rows.append({'cause': cause, 'effect': effect,
                                 'p_value': self.gc_results.loc[cause, effect],
                                 'p_value_adjusted': p_adj})
        cols = ['cause', 'effect', 'p_value', 'p_value_adjusted']
        return pd.DataFrame(rows, columns=cols).sort_values('p_value_adjusted').reset_index(drop=True)

    def regime_conditional_causality(self, regimes: pd.Series, lag=1):
        """
        Granger p-values computed on the days of each regime.

        Lags are taken from the full series before filtering, so the test
        never treats two days that are weeks apart as consecutive.

        Parameters:
        -----------
        regimes : pd.Series
            Regime per date (any labels)

        Returns:
        --------
        dict {regime: (p-value matrix, number of days)}
        """
        out = {}
        for regime in sorted(regimes.dropna().unique()):
            mask = regimes == regime
            out[regime] = (self.granger_causality_matrix(lag=lag, mask=mask), int(mask.sum()))
        return out

    def fit_var(self, data=None, maxlags=None, ic='aic'):
        """
        Fit a Vector AutoRegression (requires statsmodels).

        Rows with any missing value are dropped first.
        """
        from statsmodels.tsa.api import VAR
        data = (self.data[self.assets] if data is None else data).dropna()
        model = VAR(data)
        if maxlags is None:
            lags = getattr(model.select_order(), 'selected_orders')[ic.lower()]
            lags = max(int(lags), 1)
        else:
            lags = maxlags
        self.var_model = model.fit(lags)
        return self.var_model

    def get_impulse_response(self, periods=10):
        """Impulse response functions from the fitted VAR (requires statsmodels)."""
        if self.var_model is None:
            self.fit_var()
        self.irf = self.var_model.irf(periods)
        return self.irf


def detect_shocks(returns_series, threshold=3.0, window=20):
    """
    Flag returns that are extreme relative to the PREVIOUS `window` days.

    The rolling mean and standard deviation are shifted by one day, so a
    shock is measured against what was known before it happened (otherwise
    the shock inflates its own yardstick).

    Returns:
    --------
    tuple : (absolute z-scores, boolean mask of shocks)
    """
    mean = returns_series.rolling(window).mean().shift(1)
    std = returns_series.rolling(window).std().shift(1)
    z = ((returns_series - mean) / std).abs()
    return z, z > threshold


def get_shock_events(returns_series, threshold=2.5, top_n=10, window=20):
    """Largest shocks by z-score, labelled as crash or rally."""
    z, shocks = detect_shocks(returns_series, threshold=threshold, window=window)
    events = pd.DataFrame({
        'date': returns_series.index[shocks],
        'return': returns_series[shocks].values,
        'z_score': z[shocks].values,
    })
    events['event_type'] = np.where(events['return'] < 0, 'Crash', 'Rally')
    return events.nlargest(top_n, 'z_score').reset_index(drop=True)
