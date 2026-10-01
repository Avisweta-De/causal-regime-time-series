"""
Smoke tests for causal-regime-time-series

Check that modules import and core objects run end to end on synthetic
data. They need no network access and no API keys. Correctness checks
(look-ahead, label ordering, test statistics) live in test_correctness.py.

Run with:
    python -m pytest tests/ -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_returns():
    """Synthetic daily returns for 5 assets (600 trading days)."""
    rng = np.random.default_rng(42)
    n = 600
    dates = pd.bdate_range(start="2022-01-03", periods=n)
    tickers = ["SPY", "QQQ", "GLD", "USO", "UUP"]
    return pd.DataFrame(rng.normal(0.0003, 0.01, size=(n, len(tickers))),
                        index=dates, columns=tickers)


@pytest.fixture
def single_returns(sample_returns):
    return sample_returns["SPY"]


@pytest.fixture
def fitted_detector(single_returns):
    from src import RegimeDetector
    det = RegimeDetector(single_returns, n_regimes=3, random_state=0)
    det.fit_gmm(verbose=False)
    return det


class TestImports:

    def test_public_api(self):
        import src
        for name in src.__all__:
            assert getattr(src, name) is not None, name

    def test_package_version(self):
        import src
        assert src.__version__ == "2.0.0"


class TestRegimeDetector:

    def test_fit_gmm_basic(self, fitted_detector, single_returns):
        # One row per day once the 20-day rolling window is full
        assert len(fitted_detector.regimes) == len(single_returns) - 19
        assert set(fitted_detector.regimes.unique()) <= {0, 1, 2}

    def test_label_names(self, fitted_detector):
        labels = fitted_detector.label_regimes()
        assert set(labels.unique()) <= {"Calm", "Elevated", "Crisis"}

    def test_get_model_metrics(self, fitted_detector):
        metrics = fitted_detector.get_model_metrics()
        for key in ["silhouette_score", "bic", "converged"]:
            assert key in metrics

    def test_characteristics_and_transitions(self, fitted_detector):
        chars = fitted_detector.get_regime_characteristics()
        assert abs(chars["% of Days"].sum() - 100) < 1e-6
        trans = fitted_detector.analyze_regime_transitions()
        assert np.allclose(trans["transition_matrix"].sum(axis=1), 1.0)


class TestCausalityAnalyzer:

    def test_stationarity_test(self, single_returns):
        pytest.importorskip("statsmodels")
        from src import CausalityAnalyzer
        analyzer = CausalityAnalyzer(pd.DataFrame({"SPY": single_returns}), assets=["SPY"])
        result = analyzer.test_stationarity(single_returns, name="SPY")
        assert result["is_stationary"]

    def test_granger_causality_matrix(self, sample_returns):
        from src import CausalityAnalyzer
        assets = ["SPY", "QQQ"]
        gc = CausalityAnalyzer(sample_returns[assets], assets=assets).granger_causality_matrix(lag=2)
        assert gc.shape == (2, 2)
        assert np.isnan(gc.loc["SPY", "SPY"])
        assert 0 <= gc.loc["SPY", "QQQ"] <= 1


class TestRegimeStrategy:

    def test_allocation_rules(self):
        from src import RegimeStrategy
        strategy = RegimeStrategy(pd.Series([0, 1, 2]))
        assert [strategy.get_allocation(i) for i in range(3)] == [1.0, 0.5, 0.0]

    def test_lag_zero_rejected(self):
        from src import RegimeStrategy
        with pytest.raises(ValueError):
            RegimeStrategy(pd.Series([0, 1, 2]), lag=0)

    def test_compute_strategy_returns(self, fitted_detector, single_returns):
        from src import RegimeStrategy
        strat = RegimeStrategy(fitted_detector.regimes)
        out = strat.compute_strategy_returns(single_returns)
        assert len(out) == len(fitted_detector.regimes) - 1
        assert (out.abs() <= single_returns.reindex(out.index).abs() + 1e-12).all()


class TestBacktestEngine:

    def test_run_backtest(self, single_returns):
        from src import BacktestEngine
        engine = BacktestEngine(single_returns * 0.8, single_returns)
        results = engine.run_backtest()
        assert {"Strategy", "Benchmark", "Outperformance"} <= set(results)
        assert results["Strategy"]["Max Drawdown"] <= 0
        assert "BACKTEST PERFORMANCE REPORT" in engine.get_summary_report()

    def test_monthly_metrics_compound(self, single_returns):
        from src import BacktestEngine
        monthly = BacktestEngine(single_returns, single_returns).compute_monthly_metrics()
        first_month = single_returns[single_returns.index.to_period("M") == single_returns.index[0].to_period("M")]
        assert np.isclose(monthly["Strategy"].iloc[0], (1 + first_month).prod() - 1)

    def test_walk_forward_runs(self, single_returns):
        from src import walk_forward_regime_backtest
        res = walk_forward_regime_backtest(single_returns, train_window=252)
        assert len(res["strategy_returns"]) > 0
        assert res["strategy_returns"].index.equals(res["benchmark_returns"].index)


class TestRegimeForecaster:

    def test_transition_matrix(self, fitted_detector, single_returns):
        from src import RegimeForecaster
        tm = RegimeForecaster(fitted_detector.regimes, single_returns).estimate_transition_matrix()
        assert tm.shape == (3, 3)
        assert np.allclose(tm.sum(axis=1), 1.0)

    def test_markov_forecast(self, fitted_detector, single_returns):
        from src import RegimeForecaster
        probs = RegimeForecaster(fitted_detector.regimes, single_returns).forecast_next_regime_markov(0, steps=5)
        assert set(probs) == {0, 1, 2}
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_ml_forecaster_reports_out_of_sample(self, fitted_detector, single_returns):
        from src import RegimeForecaster
        result = RegimeForecaster(fitted_detector.regimes, single_returns).train_ml_forecaster(n_splits=3)
        assert 0 <= result["oos_accuracy"] <= 1
        assert 0 <= result["baseline_accuracy"] <= 1


class TestUtilities:

    def test_metrics_calculator(self, single_returns):
        from src import MetricsCalculator
        assert MetricsCalculator.omega_ratio(single_returns) > 0
        assert abs(MetricsCalculator.calmar_ratio(None, 0.15, -0.20) - 0.75) < 1e-9

    def test_config_manager(self):
        from src import ConfigManager
        config = ConfigManager()
        assert config.get("trading_costs") == 0.0005
        assert config.get("signal_lag_days") == 1
        config.set("trading_costs", 0.001)
        assert config.get("trading_costs") == 0.001

    def test_data_validator_alignment(self, single_returns):
        from src import DataValidator
        assert DataValidator.check_data_alignment({"A": single_returns, "B": single_returns * 1.01})

    def test_experiment_tracker(self):
        from src import ExperimentTracker
        tracker = ExperimentTracker()
        tracker.log_experiment(name="exp", config={}, metrics={"Sharpe Ratio": 1.5})
        assert tracker.get_best_experiment()["name"] == "exp"


class TestHelperFunctions:

    def test_handle_missing_values(self, single_returns):
        from src import handle_missing_values
        s = single_returns.copy()
        s.iloc[5] = np.nan
        assert handle_missing_values(s, "forward_fill").isna().sum() == 0
        assert len(handle_missing_values(s, "drop")) == len(s) - 1

    def test_shocks(self, single_returns):
        from src import detect_shocks, get_shock_events
        z, shocks = detect_shocks(single_returns, threshold=3.0, window=20)
        assert len(z) == len(single_returns) and shocks.dtype == bool
        assert "z_score" in get_shock_events(single_returns, threshold=2.0, top_n=5).columns

    def test_align_multiasset_data(self, sample_returns):
        from src import align_multiasset_data
        a, b = align_multiasset_data(sample_returns[["SPY"]], sample_returns[["QQQ"]].iloc[10:])
        assert len(a) == len(b) == len(sample_returns) - 10
