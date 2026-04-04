"""
Causal Regime-Time Series Analysis

Production-ready modules for financial time series analysis:
- Data management and preprocessing
- Regime detection
- Causal inference (Granger, VAR)
- Backtesting and strategy implementation
"""

from .data import DataLoader, align_multiasset_data, handle_missing_values
from .regimes import RegimeDetector, HMMRegimeDetector, compare_regimes
from .causality import CausalityAnalyzer, detect_shocks, get_shock_events

__version__ = '1.0.0'
__all__ = [
    'DataLoader',
    'RegimeDetector',
    'HMMRegimeDetector',
    'compare_regimes',
    'CausalityAnalyzer',
    'detect_shocks',
    'get_shock_events',
    'align_multiasset_data',
    'handle_missing_values'
]
