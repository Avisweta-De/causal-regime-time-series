"""
Backtesting Engine for Strategy Evaluation

Computes performance metrics and conducts walk-forward analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from strategy import RegimeStrategy


class BacktestEngine:
    """
    Comprehensive backtesting engine for strategy evaluation.
    
    Features:
    - Daily/monthly performance metrics
    - Drawdown analysis
    - Risk-adjusted returns (Sharpe, Sortino)
    - Walk-forward testing
    - Transaction cost modeling
    """
    
    def __init__(self, 
                 strategy_returns: pd.Series, 
                 benchmark_returns: pd.Series,
                 risk_free_rate: float = 0.0):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            Daily strategy returns
        benchmark_returns : pd.Series
            Daily benchmark returns (e.g., buy-and-hold)
        risk_free_rate : float
            Annual risk-free rate (default 0%)
        """
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.metrics = {}
        
    def compute_metrics(self, returns: pd.Series, name: str = "Strategy") -> Dict[str, float]:
        """
        Compute comprehensive performance metrics
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns
        name : str
            Strategy name
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Basic returns
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized metrics (252 trading days)
        num_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / num_years) - 1
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio (penalizes downside volatility only)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum Drawdown
        cumulative_max = cumulative.cummax()
        drawdown = (cumulative - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_duration = self._calc_drawdown_duration(drawdown)
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Return statistics
        avg_daily_return = returns.mean()
        daily_vol = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Daily Volatility': daily_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Avg Drawdown Duration': drawdown_duration,
            'Win Rate': win_rate,
            'Avg Daily Return': avg_daily_return,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Cumulative': cumulative
        }
        
        self.metrics[name] = metrics
        return metrics
    
    def _calc_drawdown_duration(self, drawdown: pd.Series) -> float:
        """Calculate average drawdown duration in days"""
        in_drawdown = (drawdown < 0)
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_lengths = in_drawdown.groupby(groups).sum()
        return drawdown_lengths[drawdown_lengths > 0].mean() if len(drawdown_lengths[drawdown_lengths > 0]) > 0 else 0
    
    def run_backtest(self) -> Dict[str, Dict[str, float]]:
        """
        Run full backtest comparing strategy to benchmark
        
        Returns:
        --------
        dict
            Performance metrics for both strategy and benchmark
        """
        strat_metrics = self.compute_metrics(self.strategy_returns, "Strategy")
        bench_metrics = self.compute_metrics(self.benchmark_returns, "Benchmark")
        
        # Calculate outperformance
        outperformance = {
            'Return Alpha': strat_metrics['Total Return'] - bench_metrics['Total Return'],
            'Annual Alpha': strat_metrics['Annual Return'] - bench_metrics['Annual Return'],
            'Sharpe Outperformance': strat_metrics['Sharpe Ratio'] - bench_metrics['Sharpe Ratio'],
            'Drawdown Improvement': bench_metrics['Max Drawdown'] - strat_metrics['Max Drawdown']
        }
        
        self.metrics['Outperformance'] = outperformance
        
        return {
            'Strategy': strat_metrics,
            'Benchmark': bench_metrics,
            'Outperformance': outperformance
        }
    
    def compute_monthly_metrics(self) -> pd.DataFrame:
        """
        Compute monthly performance metrics
        
        Returns:
        --------
        pd.DataFrame
            Monthly returns and Sharpe ratios
        """
        strat_monthly = self.strategy_returns.resample('M').sum()
        bench_monthly = self.benchmark_returns.resample('M').sum()
        
        monthly_df = pd.DataFrame({
            'Strategy': strat_monthly,
            'Benchmark': bench_monthly,
            'Outperformance': strat_monthly - bench_monthly
        })
        
        return monthly_df
    
    def compute_rolling_metrics(self, window: int = 252) -> Dict[str, pd.Series]:
        """
        Compute rolling performance metrics
        
        Parameters:
        -----------
        window : int
            Rolling window in days (default 252 = 1 year)
            
        Returns:
        --------
        dict
            Rolling metrics
        """
        # Rolling Sharpe
        strat_rolling_sharpe = self.strategy_returns.rolling(window).mean() * 252 / \
                               (self.strategy_returns.rolling(window).std() * np.sqrt(252))
        bench_rolling_sharpe = self.benchmark_returns.rolling(window).mean() * 252 / \
                               (self.benchmark_returns.rolling(window).std() * np.sqrt(252))
        
        # Rolling returns
        strat_rolling_ret = self.strategy_returns.rolling(window).sum()
        bench_rolling_ret = self.benchmark_returns.rolling(window).sum()
        
        return {
            'Strategy Sharpe': strat_rolling_sharpe,
            'Benchmark Sharpe': bench_rolling_sharpe,
            'Strategy Return': strat_rolling_ret,
            'Benchmark Return': bench_rolling_ret
        }
    
    def apply_transaction_costs(self, 
                               allocations: pd.Series,
                               cost_per_trade: float = 0.001) -> pd.Series:
        """
        Adjust returns for transaction costs
        
        Parameters:
        -----------
        allocations : pd.Series
            Daily allocation changes
        cost_per_trade : float
            Cost per rebalance (0.1% default)
            
        Returns:
        --------
        pd.Series
            Adjusted strategy returns
        """
        # Calculate rebalancing days (when allocation changes)
        allocation_changes = allocations.diff().abs()
        rebalance_costs = allocation_changes * cost_per_trade
        
        adjusted_returns = self.strategy_returns - rebalance_costs
        return adjusted_returns
    
    def walk_forward_test(self, 
                         train_window: int = 252,
                         test_window: int = 63) -> pd.DataFrame:
        """
        Walk-forward analysis to test robustness
        
        Parameters:
        -----------
        train_window : int
            Training period in days (default 252 = 1 year)
        test_window : int
            Testing period in days (default 63 = 3 months)
            
        Returns:
        --------
        pd.DataFrame
            Out-of-sample performance metrics
        """
        results = []
        
        for i in range(train_window, len(self.strategy_returns) - test_window, test_window):
            test_period = self.strategy_returns.iloc[i:i+test_window]
            
            metrics = self.compute_metrics(test_period, f"Period_{i}")
            metrics['Start Date'] = self.strategy_returns.index[i]
            metrics['End Date'] = self.strategy_returns.index[i+test_window]
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def get_summary_report(self) -> str:
        """
        Generate text summary report
        
        Returns:
        --------
        str
            Formatted performance report
        """
        if 'Strategy' not in self.metrics:
            self.run_backtest()
        
        strat = self.metrics['Strategy']
        bench = self.metrics['Benchmark']
        outperf = self.metrics['Outperformance']
        
        report = f"""
{'='*70}
BACKTEST PERFORMANCE REPORT
{'='*70}

STRATEGY PERFORMANCE:
  Total Return:        {strat['Total Return']:>10.2%}
  Annual Return:       {strat['Annual Return']:>10.2%}
  Annual Volatility:   {strat['Annual Volatility']:>10.2%}
  Sharpe Ratio:        {strat['Sharpe Ratio']:>10.2f}
  Sortino Ratio:       {strat['Sortino Ratio']:>10.2f}
  Max Drawdown:        {strat['Max Drawdown']:>10.2%}
  Win Rate:            {strat['Win Rate']:>10.2%}

BENCHMARK (BUY & HOLD):
  Total Return:        {bench['Total Return']:>10.2%}
  Annual Return:       {bench['Annual Return']:>10.2%}
  Annual Volatility:   {bench['Annual Volatility']:>10.2%}
  Sharpe Ratio:        {bench['Sharpe Ratio']:>10.2f}
  Max Drawdown:        {bench['Max Drawdown']:>10.2%}
  Win Rate:            {bench['Win Rate']:>10.2%}

OUTPERFORMANCE:
  Return Alpha:        {outperf['Return Alpha']:>+10.2%}
  Annual Alpha:        {outperf['Annual Alpha']:>+10.2%}
  Sharpe Improvement:  {outperf['Sharpe Outperformance']:>+10.2f}
  Drawdown Reduction:  {outperf['Drawdown Improvement']:>+10.2%}

{'='*70}
"""
        return report
