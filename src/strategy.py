"""
Regime-Based Allocation Strategy

Tactical equity allocation driven by volatility regimes:
- Calm (0):     100% equities
- Elevated (1):  50% equities / 50% cash
- Crisis (2):     0% equities (100% cash)

Timing rule: a regime observed at the close of day t sets the allocation for
day t+1. Applying it to day t's own return would use information that was
not available when the position was taken (look-ahead bias).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from .regimes import REGIME_NAMES

DEFAULT_ALLOCATIONS = {0: 1.0, 1: 0.5, 2: 0.0}


class RegimeStrategy:
    """Tactical allocation strategy based on ordinal regimes (0=Calm, 1=Elevated, 2=Crisis)."""

    def __init__(self, regime_labels: pd.Series, allocations: Dict[int, float] = None,
                 lag: int = 1):
        """
        Parameters:
        -----------
        regime_labels : pd.Series
            Daily ordinal regimes, as produced by RegimeDetector.regimes
        allocations : dict, optional
            Equity weight per regime (default 1.0 / 0.5 / 0.0)
        lag : int
            Days between observing a regime and holding the position.
            Must be >= 1 for a tradeable backtest.
        """
        if lag < 1:
            raise ValueError('lag must be >= 1; lag=0 applies a signal to the return it was computed from')
        self.regime_labels = regime_labels
        self.allocation_map = dict(DEFAULT_ALLOCATIONS if allocations is None else allocations)
        self.lag = lag
        self.allocations = None
        self.positions = None
        self.strategy_returns = None

    def get_allocation(self, regime: int) -> float:
        """Equity weight (0-1) for an ordinal regime."""
        return self.allocation_map[int(regime)]

    def compute_allocations(self) -> pd.Series:
        """Target allocation decided at each date's close."""
        self.allocations = self.regime_labels.map(self.get_allocation)
        return self.allocations

    def compute_positions(self) -> pd.Series:
        """Allocation actually held on each date (the decision from `lag` days earlier)."""
        if self.allocations is None:
            self.compute_allocations()
        self.positions = self.allocations.shift(self.lag)
        return self.positions

    def compute_strategy_returns(self, asset_returns: pd.Series,
                                 cost_per_unit_turnover: float = 0.0) -> pd.Series:
        """
        Daily strategy returns: held position x asset return, minus trading costs.

        Days before the first available position are dropped, so the result
        starts `lag` days after the first regime label.

        Parameters:
        -----------
        asset_returns : pd.Series
            Daily returns of the traded asset
        cost_per_unit_turnover : float
            Cost as a fraction of the amount traded (0.0005 = 5 bps).
            Moving from 100% to 50% equities is a turnover of 0.5.
        """
        positions = self.compute_positions().dropna()
        rets = asset_returns.reindex(positions.index)
        valid = rets.notna()
        positions, rets = positions[valid], rets[valid]

        turnover = positions.diff().abs().fillna(0.0)
        self.strategy_returns = positions * rets - cost_per_unit_turnover * turnover
        return self.strategy_returns

    def get_regime_summary(self) -> Dict[str, dict]:
        """Days, share of time and allocation for each regime."""
        summary = {}
        for regime_id, name in REGIME_NAMES.items():
            count = int((self.regime_labels == regime_id).sum())
            summary[name] = {
                'days': count,
                'percentage': 100 * count / len(self.regime_labels),
                'allocation': self.allocation_map.get(regime_id),
            }
        return summary

    def get_regime_returns(self, asset_returns: pd.Series, next_day: bool = True) -> Dict[str, pd.Series]:
        """
        Asset returns grouped by regime.

        next_day=True (default) groups the return of day t+1 by the regime
        observed on day t, i.e. what an investor acting on the signal faced.
        """
        target = asset_returns.shift(-1) if next_day else asset_returns
        target = target.reindex(self.regime_labels.index)
        return {name: target[self.regime_labels == rid].dropna()
                for rid, name in REGIME_NAMES.items()}

    def plot_allocations(self, figsize: Tuple[int, int] = (14, 5)):
        """Plot the held equity allocation over time."""
        import matplotlib.pyplot as plt
        positions = self.compute_positions().dropna()
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(positions.index, positions * 100, alpha=0.6, step='post',
                        label='Equity allocation held')
        ax.set_ylabel('Equity allocation (%)')
        ax.set_xlabel('Date')
        ax.set_title('Regime-based equity allocation')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax
