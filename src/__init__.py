"""
Causal Regime-Time Series Analysis

Production-ready modules for financial time series analysis:
- Data management and preprocessing
- Regime detection and forecasting
- Causal inference (Granger, VAR, IRF)
- Backtesting, strategy implementation, and optimization
- Utilities for metrics, configuration, and visualization
"""

# Core analysis modules
from .data import DataLoader, align_multiasset_data, handle_missing_values
from .regimes import RegimeDetector, HMMRegimeDetector, compare_regimes
from .causality import CausalityAnalyzer, detect_shocks, get_shock_events

# Trading modules
from .strategy import RegimeStrategy
from .backtesting import BacktestEngine
from .forecasting import RegimeForecaster

# Utilities
from .utils import (
    MetricsCalculator,
    ConfigManager,
    VisualizationHelper,
    DataValidator,
    ExperimentTracker
)

__version__ = '1.0.0'

__all__ = [
    # Data loading
    'DataLoader',
    'align_multiasset_data',
    'handle_missing_values',
    
    # Regime analysis
    'RegimeDetector',
    'HMMRegimeDetector',
    'compare_regimes',
    
    # Causal inference
    'CausalityAnalyzer',
    'detect_shocks',
    'get_shock_events',
    
    # Trading
    'RegimeStrategy',
    'BacktestEngine',
    'RegimeForecaster',
    
    # Utilities
    'MetricsCalculator',
    'ConfigManager',
    'VisualizationHelper',
    'DataValidator',
    'ExperimentTracker',
]
