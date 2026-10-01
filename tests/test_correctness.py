"""
Correctness tests: the properties the results depend on.

- No look-ahead: positions use only earlier signals, and changing future
  returns cannot change past walk-forward results
- Regime ids are ordered by volatility on every fit
- The Granger F-test detects a planted lead-lag and returns valid p-values
- Impossible returns from non-positive prices are removed
- Metrics match hand calculations
- LLM text is checked against computed facts
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def regime_switching_returns(n=1500, seed=0):
    """Returns with alternating calm and turbulent blocks of 60 days."""
    rng = np.random.default_rng(seed)
    vols = np.where((np.arange(n) // 60) % 3 == 2, 0.03, np.where((np.arange(n) // 60) % 3 == 1, 0.012, 0.005))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(0.0003, vols), index=idx)


# --------------------------------------------------------------------------
# Look-ahead
# --------------------------------------------------------------------------

class TestNoLookAhead:

    def test_position_uses_previous_day_signal(self):
        from src import RegimeStrategy
        idx = pd.bdate_range("2020-01-01", periods=6)
        regimes = pd.Series([0, 2, 1, 0, 2, 0], index=idx)
        returns = pd.Series([0.01, -0.05, 0.02, 0.01, -0.04, 0.03], index=idx)
        strat = RegimeStrategy(regimes)
        out = strat.compute_strategy_returns(returns)
        expected_positions = [1.0, 0.0, 0.5, 1.0, 0.0]   # regime of the day before
        assert list(strat.positions.dropna()) == expected_positions
        assert np.allclose(out.values, np.array(expected_positions) * returns.iloc[1:].values)

    def test_perfect_foresight_labels_do_not_leak(self):
        """Labels built from same-day returns must not let the strategy dodge those days."""
        from src import RegimeStrategy
        r = regime_switching_returns(n=800, seed=3)
        cheating_labels = pd.Series(np.where(r < 0, 2, 0), index=r.index)
        out = RegimeStrategy(cheating_labels).compute_strategy_returns(r)
        # With a same-day bug every held day would be a positive-return day
        assert (out < 0).sum() > 100

    def test_future_data_cannot_change_past_results(self):
        from src import walk_forward_regime_backtest
        r = regime_switching_returns()
        cut = r.index[1200]
        shocked = r.copy()
        shocked.loc[cut:] = shocked.loc[cut:] * -5 + 0.02   # rewrite the future completely

        a = walk_forward_regime_backtest(r, train_window=504, random_state=1)
        b = walk_forward_regime_backtest(shocked, train_window=504, random_state=1)

        before = a["strategy_returns"].index < cut
        assert before.sum() > 100
        pd.testing.assert_series_equal(a["strategy_returns"][before], b["strategy_returns"][before])
        months_before = a["regimes"].index.to_period("M") < cut.to_period("M")
        pd.testing.assert_series_equal(a["regimes"][months_before], b["regimes"][months_before])


# --------------------------------------------------------------------------
# Regime ordering
# --------------------------------------------------------------------------

class TestRegimeOrdering:

    def test_detector_ids_ordered_by_volatility(self):
        from src import RegimeDetector
        r = regime_switching_returns()
        for seed in range(5):
            det = RegimeDetector(r, random_state=seed)
            det.fit_gmm(verbose=False)
            vol_by_regime = det.features["vol"].groupby(det.regimes).mean()
            assert vol_by_regime.is_monotonic_increasing, seed

    def test_walk_forward_ids_ordered_by_volatility(self):
        from src import walk_forward_regime_backtest, make_regime_features
        r = regime_switching_returns()
        res = walk_forward_regime_backtest(r, train_window=504, random_state=7)
        vol = make_regime_features(r)["vol"].reindex(res["regimes"].index)
        assert vol.groupby(res["regimes"]).mean().is_monotonic_increasing

    def test_crisis_regime_gets_zero_allocation(self):
        from src import walk_forward_regime_backtest
        r = regime_switching_returns()
        res = walk_forward_regime_backtest(r, train_window=504, random_state=7)
        regimes_held = res["regimes"].shift(1).reindex(res["positions"].index)
        assert (res["positions"][regimes_held == 2] == 0.0).all()
        assert (res["positions"][regimes_held == 0] == 1.0).all()


# --------------------------------------------------------------------------
# Granger causality
# --------------------------------------------------------------------------

class TestGranger:

    def _pair(self, n=2000, beta=0.4, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = np.zeros(n)
        y[1:] = beta * x[:-1] + rng.normal(size=n - 1)
        idx = pd.bdate_range("2010-01-01", periods=n)
        return pd.Series(x, idx), pd.Series(y, idx)

    def test_detects_planted_lead_lag(self):
        from src import granger_test
        x, y = self._pair()
        assert granger_test(y, x, lag=1)["p_value"] < 1e-10
        assert granger_test(x, y, lag=1)["p_value"] > 0.001

    def test_p_values_uniform_under_null(self):
        from src import granger_test
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2010-01-01", periods=500)
        p = [granger_test(pd.Series(rng.normal(size=500), idx), pd.Series(rng.normal(size=500), idx))["p_value"]
             for _ in range(300)]
        rejection_rate = np.mean(np.array(p) < 0.05)
        assert 0.02 < rejection_rate < 0.09

    def test_matches_statsmodels(self):
        sm = pytest.importorskip("statsmodels.tsa.stattools")
        from src import granger_test
        x, y = self._pair(n=600, beta=0.1)
        for lag in (1, 3):
            ours = granger_test(y, x, lag=lag)
            theirs = sm.grangercausalitytests(pd.concat([y, x], axis=1), maxlag=lag, verbose=False)[lag][0]["ssr_ftest"]
            assert np.isclose(ours["f_stat"], theirs[0]) and np.isclose(ours["p_value"], theirs[1])

    def test_mask_keeps_calendar_lags(self):
        from src import granger_test
        x, y = self._pair(n=1000)
        mask = pd.Series(np.arange(1000) % 2 == 0, index=x.index)
        res = granger_test(y, x, lag=1, mask=mask)
        # Lags come from the full series, so the planted effect is still found
        assert res["p_value"] < 1e-5
        assert res["n_obs"] == 499

    def test_bonferroni(self):
        from src import adjust_pvalues
        p = pd.DataFrame([[np.nan, 0.01], [0.04, np.nan]], index=list("ab"), columns=list("ab"))
        adj = adjust_pvalues(p, "bonferroni")
        assert np.isclose(adj.loc["a", "b"], 0.02) and np.isclose(adj.loc["b", "a"], 0.08)
        assert np.isnan(adj.loc["a", "a"])


# --------------------------------------------------------------------------
# Data and metrics
# --------------------------------------------------------------------------

class TestDataAndMetrics:

    def test_negative_price_returns_removed(self):
        from src import compute_returns
        idx = pd.bdate_range("2020-04-15", periods=5)
        prices = pd.DataFrame({"OIL": [19.87, 18.27, -37.63, 10.01, 13.78],
                               "SPX": [100, 101, 99, 98, 100.0]}, index=idx)
        r = compute_returns(prices)
        assert r["OIL"].iloc[1:3].isna().all()            # the -306% and -127% days
        assert np.isclose(r["OIL"].iloc[3], 13.78 / 10.01 - 1)
        assert r["SPX"].notna().all()
        assert (r.dropna() > -1).all().all()

    def test_metrics_hand_calculation(self):
        from src import performance_metrics
        idx = pd.bdate_range("2020-01-01", periods=252)
        r = pd.Series(0.001, index=idx)
        r.iloc[100] = -0.10
        m = performance_metrics(r)
        total = 1.001 ** 251 * 0.9 - 1
        assert np.isclose(m["Total Return"], total)
        assert np.isclose(m["Annual Return"], total)        # exactly one year
        assert np.isclose(m["Max Drawdown"], -0.10)
        assert np.isclose(m["Sharpe Ratio"], r.mean() * 252 / (r.std() * np.sqrt(252)))

    def test_transition_matrix_with_missing_regime(self):
        from src import RegimeForecaster
        regimes = pd.Series([0, 0, 1, 0, 1, 1])
        tm = RegimeForecaster(regimes, pd.Series(np.zeros(6))).estimate_transition_matrix()
        assert np.allclose(tm.sum(axis=1), 1.0)
        assert tm[2, 2] == 1.0


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

class TestGroundedReports:

    def test_flags_invented_numbers(self):
        from src import ungrounded_numbers
        facts = {"cagr_pct": 8.2, "sharpe": 0.85}
        assert ungrounded_numbers("CAGR was 8.2% with a Sharpe of 0.85.", facts) == []
        assert 3616.0 in ungrounded_numbers("Total return was 3616%.", facts)
