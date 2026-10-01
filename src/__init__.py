"""
Causal Regime Time-Series Analysis

Modules:
- data:         loading and cleaning returns (invalid prices handled)
- regimes:      volatility regimes from rolling features (0=Calm, 1=Elevated, 2=Crisis)
- causality:    Granger tests with multiple-testing control, VAR/IRF (statsmodels optional)
- strategy:     regime allocation with a one-day signal lag
- backtesting:  metrics, walk-forward backtest, robustness checks
- forecasting:  Markov transitions and out-of-sample ML regime forecasts
- llm_insights: reports built from computed facts, optional LLM narration

Optional dependencies are imported only when used: yfinance (downloads),
statsmodels (ADF, VAR, IRF), hmmlearn (HMM), openai (narration).
"""

from .data import (DataLoader, compute_returns, data_quality_report,
                   align_multiasset_data, handle_missing_values)
from .regimes import (RegimeDetector, HMMRegimeDetector, compare_regimes,
                      make_regime_features, REGIME_NAMES)
from .causality import (CausalityAnalyzer, granger_test, adjust_pvalues,
                        detect_shocks, get_shock_events)
from .strategy import RegimeStrategy
from .backtesting import (BacktestEngine, performance_metrics,
                          walk_forward_regime_backtest, robustness_check,
                          constant_mix_returns)
from .forecasting import RegimeForecaster
from .llm_insights import (LLMInsightGenerator, InsightGenerator, build_facts,
                           render_report, ungrounded_numbers)
from .utils import (MetricsCalculator, ConfigManager, VisualizationHelper,
                    DataValidator, ExperimentTracker)

__version__ = '2.0.0'

__all__ = [
    'DataLoader', 'compute_returns', 'data_quality_report',
    'align_multiasset_data', 'handle_missing_values',
    'RegimeDetector', 'HMMRegimeDetector', 'compare_regimes',
    'make_regime_features', 'REGIME_NAMES',
    'CausalityAnalyzer', 'granger_test', 'adjust_pvalues',
    'detect_shocks', 'get_shock_events',
    'RegimeStrategy',
    'BacktestEngine', 'performance_metrics', 'walk_forward_regime_backtest', 'robustness_check', 'constant_mix_returns',
    'RegimeForecaster',
    'LLMInsightGenerator', 'InsightGenerator', 'build_facts', 'render_report', 'ungrounded_numbers',
    'MetricsCalculator', 'ConfigManager', 'VisualizationHelper', 'DataValidator', 'ExperimentTracker',
]
