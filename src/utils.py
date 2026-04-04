"""
Utility Functions for Causal Regime Analysis

Common helpers for metrics, visualization, and configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Compute additional performance metrics"""
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, annual_return: float, max_drawdown: float) -> float:
        """
        Calmar ratio = Annual return / Max drawdown
        
        Higher is better (reward per unit of drawdown)
        """
        if max_drawdown < 0:
            return annual_return / abs(max_drawdown)
        return 0
    
    @staticmethod
    def return_drawdown_ratio(annual_return: float, max_drawdown: float) -> float:
        """Return to drawdown ratio for risk assessment"""
        if max_drawdown < 0:
            return annual_return / abs(max_drawdown)
        return np.inf
    
    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Omega ratio = Gains above threshold / Losses below threshold
        
        Higher is better (probability-weighted gain/loss ratio)
        """
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        
        return gains / losses if losses != 0 else np.inf
    
    @staticmethod
    def information_ratio(strategy_returns: pd.Series, 
                         benchmark_returns: pd.Series) -> float:
        """
        Information ratio = Excess return / Tracking error
        
        Measures skill in outperforming benchmark
        """
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        excess_annual = excess_returns.mean() * 252
        
        return excess_annual / tracking_error if tracking_error > 0 else 0
    
    @staticmethod
    def rolling_correlation(series1: pd.Series, series2: pd.Series, 
                           window: int = 20) -> pd.Series:
        """Compute rolling correlation"""
        return series1.rolling(window).corr(series2)
    
    @staticmethod
    def rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, 
                     window: int = 252) -> pd.Series:
        """
        Compute rolling beta relative to market
        
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        """
        # Align indices
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        rolling_cov = aligned['asset'].rolling(window).cov(aligned['market'])
        rolling_market_var = aligned['market'].rolling(window).var()
        
        return rolling_cov / rolling_market_var


class ConfigManager:
    """Manage strategy configuration"""
    
    DEFAULT_CONFIG = {
        'regime_labels': {0: 'Bull', 1: 'Neutral', 2: 'Crisis'},
        'allocations': {0: 1.0, 1: 0.5, 2: 0.0},
        'trading_costs': 0.001,  # 10 bps
        'rebalance_frequency': 'daily',  # 'daily', 'weekly', 'monthly'
        'lookback_period': 252,  # 1 year
        'forecast_horizon': 5,  # 1 week
    }
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize config manager"""
        self.config = self.DEFAULT_CONFIG.copy()
        if config_dict:
            self.config.update(config_dict)
    
    def get(self, key: str, default=None):
        """Get config value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set config value"""
        self.config[key] = value
    
    def print_config(self) -> None:
        """Print all config"""
        print("\n" + "="*60)
        print("STRATEGY CONFIGURATION")
        print("="*60)
        for key, value in self.config.items():
            print(f"  {key:20s}: {value}")
        print("="*60 + "\n")


class VisualizationHelper:
    """Helper functions for visualization"""
    
    @staticmethod
    def format_returns_plot(ax, title: str = "", xlabel: str = "Date", 
                           ylabel: str = "Cumulative Value") -> None:
        """Format returns/equity plot"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    
    @staticmethod
    def format_metric_table(metrics_dict: Dict) -> pd.DataFrame:
        """Format metrics dictionary as table"""
        df = pd.DataFrame(metrics_dict).T
        return df[['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']]
    
    @staticmethod
    def add_regime_shading(ax, regimes: pd.Series, regime_color_map: Dict = None) -> None:
        """
        Add shaded backgrounds for different regimes
        
        Parameters:
        -----------
        ax : matplotlib axis
            Plot axis
        regimes : pd.Series
            Regime label time series
        regime_color_map : dict
            Mapping of regime to color {0: 'green', 1: 'yellow', 2: 'red'}
        """
        if regime_color_map is None:
            regime_color_map = {0: 'green', 1: 'yellow', 2: 'red'}
        
        for i in range(len(regimes) - 1):
            regime = int(regimes.iloc[i])
            color = regime_color_map.get(regime, 'gray')
            ax.axvspan(regimes.index[i], regimes.index[i+1], 
                      alpha=0.1, color=color)


class DataValidator:
    """Validate input data quality"""
    
    @staticmethod
    def check_data_alignment(data_dict: Dict[str, pd.Series]) -> bool:
        """Check if all series have same length and index"""
        indices = [data.index for data in data_dict.values()]
        
        # All same length
        if len(set(len(idx) for idx in indices)) > 1:
            print("❌ Data series have different lengths")
            return False
        
        # All same index (at least first and last)
        first_indices = [idx[0] for idx in indices]
        last_indices = [idx[-1] for idx in indices]
        
        if len(set(first_indices)) > 1 or len(set(last_indices)) > 1:
            print("❌ Data series have different date ranges")
            return False
        
        print("✅ All data series properly aligned")
        return True
    
    @staticmethod
    def check_returns_distribution(returns: pd.Series) -> Dict[str, float]:
        """Check returns statistics"""
        stats = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max(),
            'pct_zero': (returns == 0).sum() / len(returns)
        }
        
        print("\n" + "="*60)
        print("RETURNS DISTRIBUTION ANALYSIS")
        print("="*60)
        for key, value in stats.items():
            print(f"  {key:15s}: {value:>12.4f}")
        print("="*60 + "\n")
        
        return stats
    
    @staticmethod
    def check_missing_values(data_dict: Dict[str, pd.Series]) -> None:
        """Check for missing values"""
        print("\n" + "="*60)
        print("MISSING VALUES CHECK")
        print("="*60)
        
        all_clean = True
        for name, data in data_dict.items():
            missing = data.isna().sum()
            if missing > 0:
                print(f"  ❌ {name:20s}: {missing:6d} missing ({100*missing/len(data):.2f}%)")
                all_clean = False
            else:
                print(f"  ✅ {name:20s}: Clean")
        
        print("="*60 + "\n")
        return all_clean


class ExperimentTracker:
    """Track backtest experiments and results"""
    
    def __init__(self):
        """Initialize experiment tracker"""
        self.experiments = []
    
    def log_experiment(self, 
                      name: str,
                      config: Dict,
                      metrics: Dict,
                      timestamp: Optional[str] = None) -> None:
        """
        Log experiment results
        
        Parameters:
        -----------
        name : str
            Experiment name
        config : dict
            Configuration used
        metrics : dict
            Performance metrics
        timestamp : str
            Timestamp (auto-filled if None)
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        experiment = {
            'name': name,
            'timestamp': timestamp,
            'config': config,
            'metrics': metrics
        }
        
        self.experiments.append(experiment)
    
    def get_best_experiment(self, metric: str = 'Sharpe Ratio') -> Dict:
        """Get best experiment by metric"""
        if not self.experiments:
            return None
        
        best = max(self.experiments, 
                  key=lambda x: x['metrics'].get(metric, 0))
        return best
    
    def compare_experiments(self) -> pd.DataFrame:
        """Compare all logged experiments"""
        results = []
        for exp in self.experiments:
            result = {'name': exp['name'], 'timestamp': exp['timestamp']}
            result.update(exp['metrics'])
            results.append(result)
        
        return pd.DataFrame(results)
    
    def print_summary(self) -> None:
        """Print experiment summary"""
        if not self.experiments:
            print("No experiments logged yet")
            return
        
        df = self.compare_experiments()
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(df.to_string())
        print("="*80 + "\n")
