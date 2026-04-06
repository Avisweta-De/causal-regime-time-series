"""
Smoke Tests for causal-regime-time-series

These tests validate that all modules can be imported cleanly and that
core classes can be instantiated without errors. They do NOT require
network access (no yfinance downloads) or an OpenAI API key.

Run with:
    python -m pytest tests/ -v
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns():
    """Synthetic daily log-returns for 5 assets (600 trading days)."""
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range(start="2022-01-01", periods=n)
    tickers = ["SPY", "QQQ", "GLD", "USO", "UUP"]
    data = pd.DataFrame(
        np.random.normal(0.0003, 0.01, size=(n, len(tickers))),
        index=dates,
        columns=tickers,
    )
    return data


@pytest.fixture
def single_returns(sample_returns):
    """Single-asset returns series."""
    return sample_returns["SPY"]


# ---------------------------------------------------------------------------
# 1. Import Tests
# ---------------------------------------------------------------------------

class TestImports:
    """All public imports from the `src` package must succeed."""

    def test_import_data_loader(self):
        from src import DataLoader
        assert DataLoader is not None

    def test_import_regime_detector(self):
        from src import RegimeDetector
        assert RegimeDetector is not None

    def test_import_hmm_regime_detector(self):
        from src import HMMRegimeDetector
        assert HMMRegimeDetector is not None

    def test_import_causality_analyzer(self):
        from src import CausalityAnalyzer
        assert CausalityAnalyzer is not None

    def test_import_regime_strategy(self):
        from src import RegimeStrategy
        assert RegimeStrategy is not None

    def test_import_backtest_engine(self):
        from src import BacktestEngine
        assert BacktestEngine is not None

    def test_import_regime_forecaster(self):
        from src import RegimeForecaster
        assert RegimeForecaster is not None

    def test_import_llm_insight_generator(self):
        """Must not crash even without OPENAI_API_KEY set."""
        from src import LLMInsightGenerator
        assert LLMInsightGenerator is not None

    def test_import_utilities(self):
        from src import MetricsCalculator, ConfigManager, VisualizationHelper
        from src import DataValidator, ExperimentTracker
        assert all(
            cls is not None
            for cls in [MetricsCalculator, ConfigManager, VisualizationHelper,
                        DataValidator, ExperimentTracker]
        )

    def test_package_version(self):
        import src
        assert hasattr(src, "__version__")
        assert src.__version__ == "1.0.0"


# ---------------------------------------------------------------------------
# 2. RegimeDetector Tests
# ---------------------------------------------------------------------------

class TestRegimeDetector:

    def test_fit_gmm_basic(self, single_returns):
        from src import RegimeDetector
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        model = detector.fit_gmm()
        assert model is not None
        assert detector.regimes is not None
        assert len(detector.regimes) == len(single_returns)

    def test_label_regimes_volatility(self, single_returns):
        from src import RegimeDetector
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        detector.fit_gmm()
        labels = detector.label_regimes(by="volatility")
        assert set(labels.unique()).issubset({"Bull", "Neutral", "Crisis"})

    def test_label_regimes_returns(self, single_returns):
        from src import RegimeDetector
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        detector.fit_gmm()
        labels = detector.label_regimes(by="returns")
        assert set(labels.unique()).issubset({"Bull", "Neutral", "Crisis"})

    def test_get_model_metrics(self, single_returns):
        from src import RegimeDetector
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        detector.fit_gmm()
        metrics = detector.get_model_metrics()
        assert "silhouette_score" in metrics
        assert "bic" in metrics
        assert "converged" in metrics


# ---------------------------------------------------------------------------
# 3. CausalityAnalyzer Tests
# ---------------------------------------------------------------------------

class TestCausalityAnalyzer:

    def test_stationarity_test(self, single_returns):
        from src import CausalityAnalyzer
        assets = ["SPY", "QQQ"]
        analyzer = CausalityAnalyzer(
            pd.DataFrame({"SPY": single_returns, "QQQ": single_returns * 1.02}),
            assets=assets,
        )
        result = analyzer.test_stationarity(single_returns, name="SPY")
        assert "p_value" in result
        assert "is_stationary" in result

    def test_granger_causality_matrix(self, sample_returns):
        from src import CausalityAnalyzer
        assets = ["SPY", "QQQ"]
        data = sample_returns[assets]
        analyzer = CausalityAnalyzer(data, assets=assets)
        gc_matrix = analyzer.granger_causality_matrix(maxlag=2)
        assert gc_matrix.shape == (2, 2)
        # Diagonal should be 0.0 (no self-causality)
        assert gc_matrix.loc["SPY", "SPY"] == 0.0


# ---------------------------------------------------------------------------
# 4. RegimeStrategy Tests
# ---------------------------------------------------------------------------

class TestRegimeStrategy:

    def test_allocation_rules(self):
        from src import RegimeStrategy
        labels = pd.Series([0, 0, 1, 2, 1, 0])
        strategy = RegimeStrategy(labels)
        assert strategy.get_allocation(0) == 1.0
        assert strategy.get_allocation(1) == 0.5
        assert strategy.get_allocation(2) == 0.0

    def test_compute_allocations(self):
        from src import RegimeStrategy
        labels = pd.Series([0, 1, 2, 0], dtype=int)
        strategy = RegimeStrategy(labels)
        allocs = strategy.compute_allocations()
        assert list(allocs) == [1.0, 0.5, 0.0, 1.0]

    def test_compute_strategy_returns(self, single_returns):
        from src import RegimeDetector, RegimeStrategy
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        detector.fit_gmm()
        numeric_labels = pd.Series(detector.regimes, index=single_returns.index)
        strategy = RegimeStrategy(numeric_labels)
        strat_returns = strategy.compute_strategy_returns(single_returns)
        assert len(strat_returns) == len(single_returns)
        # Strategy returns must be ≤ asset returns in absolute magnitude
        assert (strat_returns.abs() <= single_returns.abs() + 1e-10).all()


# ---------------------------------------------------------------------------
# 5. BacktestEngine Tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def test_compute_metrics(self, single_returns):
        from src import BacktestEngine
        engine = BacktestEngine(
            strategy_returns=single_returns * 0.7,
            benchmark_returns=single_returns,
        )
        metrics = engine.compute_metrics(single_returns, name="Test")
        assert "Sharpe Ratio" in metrics
        assert "Max Drawdown" in metrics
        assert metrics["Max Drawdown"] <= 0

    def test_run_backtest(self, single_returns):
        from src import BacktestEngine
        engine = BacktestEngine(
            strategy_returns=single_returns * 0.8,
            benchmark_returns=single_returns,
        )
        results = engine.run_backtest()
        assert "Strategy" in results
        assert "Benchmark" in results
        assert "Outperformance" in results

    def test_apply_transaction_costs(self, single_returns):
        from src import BacktestEngine, RegimeDetector, RegimeStrategy
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        detector.fit_gmm()
        numeric_labels = pd.Series(detector.regimes, index=single_returns.index)
        strategy = RegimeStrategy(numeric_labels)
        allocs = strategy.compute_allocations()
        strat_returns = strategy.compute_strategy_returns(single_returns)
        engine = BacktestEngine(
            strategy_returns=strat_returns,
            benchmark_returns=single_returns,
        )
        adjusted = engine.apply_transaction_costs(allocs, cost_per_trade=0.001)
        assert len(adjusted) == len(strat_returns)


# ---------------------------------------------------------------------------
# 6. RegimeForecaster Tests
# ---------------------------------------------------------------------------

class TestRegimeForecaster:

    def test_estimate_transition_matrix(self, single_returns):
        from src import RegimeDetector, RegimeForecaster
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        detector.fit_gmm()
        labels = pd.Series(detector.regimes, index=single_returns.index)
        forecaster = RegimeForecaster(labels, single_returns)
        tm = forecaster.estimate_transition_matrix()
        assert tm.shape == (3, 3)
        # Each row must sum to ~1
        assert np.allclose(tm.sum(axis=1), 1.0, atol=1e-6)

    def test_forecast_next_regime_markov(self, single_returns):
        from src import RegimeDetector, RegimeForecaster
        detector = RegimeDetector(single_returns, n_regimes=3, random_state=0)
        detector.fit_gmm()
        labels = pd.Series(detector.regimes, index=single_returns.index)
        forecaster = RegimeForecaster(labels, single_returns)
        probs = forecaster.forecast_next_regime_markov(current_regime=0, steps=1)
        assert set(probs.keys()) == {0, 1, 2}
        assert abs(sum(probs.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 7. Utilities Tests
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_metrics_calculator_omega_ratio(self, single_returns):
        from src import MetricsCalculator
        omega = MetricsCalculator.omega_ratio(single_returns)
        assert omega > 0

    def test_metrics_calculator_calmar_ratio(self):
        from src import MetricsCalculator
        calmar = MetricsCalculator.calmar_ratio(None, annual_return=0.15, max_drawdown=-0.20)
        assert abs(calmar - 0.75) < 1e-9

    def test_config_manager_defaults(self):
        from src import ConfigManager
        config = ConfigManager()
        assert config.get("trading_costs") == 0.001
        assert config.get("nonexistent_key", "default") == "default"

    def test_config_manager_set(self):
        from src import ConfigManager
        config = ConfigManager()
        config.set("trading_costs", 0.005)
        assert config.get("trading_costs") == 0.005

    def test_data_validator_alignment(self, single_returns):
        from src import DataValidator
        data_dict = {"SPY": single_returns, "QQQ": single_returns * 1.01}
        result = DataValidator.check_data_alignment(data_dict)
        assert result is True

    def test_experiment_tracker(self):
        from src import ExperimentTracker
        tracker = ExperimentTracker()
        tracker.log_experiment(
            name="test_exp",
            config={"param": 1},
            metrics={"Sharpe Ratio": 1.5, "Max Drawdown": -0.10},
        )
        best = tracker.get_best_experiment(metric="Sharpe Ratio")
        assert best["name"] == "test_exp"


# ---------------------------------------------------------------------------
# 8. Helper Functions Tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:

    def test_handle_missing_values_ffill(self, single_returns):
        from src.data import handle_missing_values
        data_with_nan = single_returns.copy()
        data_with_nan.iloc[5] = np.nan
        result = handle_missing_values(data_with_nan, method="forward_fill")
        assert result.isna().sum() == 0

    def test_handle_missing_values_drop(self, single_returns):
        from src.data import handle_missing_values
        data_with_nan = single_returns.copy()
        data_with_nan.iloc[5] = np.nan
        result = handle_missing_values(data_with_nan, method="drop")
        assert result.isna().sum() == 0
        assert len(result) == len(single_returns) - 1

    def test_detect_shocks(self, single_returns):
        from src import detect_shocks
        z_scores, shocks = detect_shocks(single_returns, threshold=3.0, window=20)
        assert len(z_scores) == len(single_returns)
        assert shocks.dtype == bool

    def test_get_shock_events(self, single_returns):
        from src import get_shock_events
        shock_df = get_shock_events(single_returns, threshold=2.0, top_n=5)
        assert isinstance(shock_df, pd.DataFrame)
        assert "z_score" in shock_df.columns

    def test_align_multiasset_data(self, sample_returns):
        from src import align_multiasset_data
        df1 = sample_returns[["SPY"]]
        df2 = sample_returns[["QQQ"]]
        result = align_multiasset_data(df1, df2)
        # When both have same index, alignment should preserve all rows
        assert len(result[0]) == len(df1)
