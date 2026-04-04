"""
Regime-Based Trading Strategy Implementation

Provides tactical allocation rules based on market regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


class RegimeStrategy:
    """
    Tactical allocation strategy based on market regimes.
    
    Allocation rules:
    - Bull (0): 100% stocks (full exposure)
    - Neutral (1): 50% stocks / 50% cash (balanced)
    - Crisis (2): 0% stocks / 100% cash (preservation)
    """
    
    def __init__(self, regime_labels: pd.Series, asset_returns: pd.DataFrame = None):
        """
        Initialize strategy
        
        Parameters:
        -----------
        regime_labels : pd.Series
            Daily regime labels (0=Bull, 1=Neutral, 2=Crisis)
        asset_returns : pd.DataFrame, optional
            Returns for each asset (for multi-asset strategies)
        """
        self.regime_labels = regime_labels
        self.asset_returns = asset_returns
        self.allocations = None
        self.strategy_returns = None
        
    def get_allocation(self, regime: int) -> float:
        """
        Get stock allocation (0-1) based on regime
        
        Parameters:
        -----------
        regime : int
            0=Bull, 1=Neutral, 2=Crisis
            
        Returns:
        --------
        float
            Stock allocation percentage (0.0 to 1.0)
        """
        if regime == 0:      # Bull
            return 1.0       # 100% stocks
        elif regime == 1:    # Neutral
            return 0.5       # 50% stocks, 50% cash
        else:                # Crisis
            return 0.0       # 0% stocks (100% cash)
    
    def compute_allocations(self) -> pd.Series:
        """
        Compute daily allocations based on regimes
        
        Returns:
        --------
        pd.Series
            Daily stock allocation percentages
        """
        self.allocations = self.regime_labels.map(self.get_allocation)
        return self.allocations
    
    def compute_strategy_returns(self, asset_returns: pd.Series) -> pd.Series:
        """
        Apply allocation to asset returns
        
        Parameters:
        -----------
        asset_returns : pd.Series
            Daily returns of primary asset
            
        Returns:
        --------
        pd.Series
            Strategy returns with regimes applied
        """
        if self.allocations is None:
            self.compute_allocations()
        
        self.strategy_returns = asset_returns * self.allocations
        return self.strategy_returns
    
    def get_regime_summary(self) -> Dict[str, int]:
        """
        Get count and percentage of days in each regime
        
        Returns:
        --------
        dict
            Regime statistics
        """
        regime_names = {0: 'Bull', 1: 'Neutral', 2: 'Crisis'}
        summary = {}
        
        for regime_id, regime_name in regime_names.items():
            count = (self.regime_labels == regime_id).sum()
            pct = 100 * count / len(self.regime_labels)
            summary[regime_name] = {
                'days': count,
                'percentage': pct,
                'allocation': self.get_allocation(regime_id)
            }
        
        return summary
    
    def get_regime_returns(self, asset_returns: pd.Series) -> Dict[str, pd.Series]:
        """
        Get returns segmented by regime
        
        Parameters:
        -----------
        asset_returns : pd.Series
            Daily returns
            
        Returns:
        --------
        dict
            Returns by regime
        """
        regime_names = {0: 'Bull', 1: 'Neutral', 2: 'Crisis'}
        returns_by_regime = {}
        
        for regime_id, regime_name in regime_names.items():
            mask = self.regime_labels == regime_id
            returns_by_regime[regime_name] = asset_returns[mask]
        
        return returns_by_regime
    
    def optimize_allocation(self, target_volatility: float = 0.10) -> Dict[int, float]:
        """
        Optimize allocations to achieve target volatility
        
        Parameters:
        -----------
        target_volatility : float
            Target annualized volatility (10% default)
            
        Returns:
        --------
        dict
            Optimized allocations by regime {0: x, 1: y, 2: z}
        """
        # Simple approach: scale allocations to match target vol
        if self.strategy_returns is None:
            raise ValueError("Call compute_strategy_returns() first")
        
        current_vol = self.strategy_returns.std() * np.sqrt(252)
        scale_factor = target_volatility / current_vol if current_vol > 0 else 1.0
        
        optimized = {
            0: min(1.0 * scale_factor, 1.0),  # Bull - capped at 100%
            1: 0.5 * scale_factor,             # Neutral
            2: 0.0                            # Crisis - always 0%
        }
        
        return optimized
    
    def plot_allocations(self, figsize: Tuple[int, int] = (14, 5)):
        """
        Visualize allocation over time
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.allocations is None:
                self.compute_allocations()
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.fill_between(self.allocations.index, self.allocations, 
                           alpha=0.6, label='Stock Allocation')
            ax.set_ylabel('Allocation (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_title('Regime-Based Stock Allocation Over Time', 
                        fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig, ax
        except ImportError:
            print("Matplotlib not available for plotting")
            return None, None
