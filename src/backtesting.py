"""
Backtesting Engine for Strategy Evaluation

- Performance metrics (CAGR, volatility, Sharpe, Sortino, drawdowns)
- A true walk-forward regime backtest: the model is refit each period on
  past data only, cluster labels are ordered by volatility on every refit,
  and signals are applied with a one-day lag
- Robustness checks across random seeds and training windows
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .strategy import RegimeStrategy
from .regimes import make_regime_features, order_components_by_volatility

TRADING_DAYS = 252


def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Standard performance metrics for a daily return series.

    Sharpe = annualised mean excess return / annualised volatility.
    Sortino uses downside deviation: sqrt(mean(min(excess, 0)^2)) * sqrt(252).

    Parameters:
    -----------
    returns : pd.Series
        Daily simple returns
    risk_free_rate : float
        Annual risk-free rate (default 0%)
    """
    returns = returns.dropna()
    excess = returns - risk_free_rate / TRADING_DAYS
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    years = len(returns) / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = excess.mean() * TRADING_DAYS / vol if vol > 0 else np.nan
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = excess.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()

    in_dd = drawdown < 0
    runs = in_dd.groupby((in_dd != in_dd.shift()).cumsum()).sum()
    runs = runs[runs > 0]

    return {
        'Total Return': total_return,
        'Annual Return': cagr,
        'Annual Volatility': vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': cagr / abs(max_dd) if max_dd < 0 else np.nan,
        'Avg Drawdown Duration': runs.mean() if len(runs) else 0.0,
        'Worst Day': returns.min(),
        'Win Rate': (returns > 0).mean(),
        'Avg Daily Return': returns.mean(),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Days': len(returns),
        'Start': returns.index[0],
        'End': returns.index[-1],
        'Cumulative': cumulative,
    }


def walk_forward_regime_backtest(returns: pd.Series,
                                 train_window: int = 504,
                                 feature_window: int = 20,
                                 n_regimes: int = 3,
                                 refit_freq: str = 'M',
                                 allocations: Optional[Dict[int, float]] = None,
                                 lag: int = 1,
                                 cost_per_unit_turnover: float = 0.0005,
                                 random_state: int = 42) -> Dict[str, pd.Series]:
    """
    Out-of-sample regime strategy backtest.

    For each period (month by default):
      1. Fit a scaler and a GMM on the `train_window` feature rows that end
         the day before the period starts (past data only).
      2. Order the fitted components by volatility (0 = Calm ... 2 = Crisis),
         because raw GMM component ids are arbitrary on every refit.
      3. Classify each day in the period from its rolling features, which use
         returns up to and including that day.
    The regime from day t then sets the position held on day t+lag.

    Parameters:
    -----------
    returns : pd.Series
        Daily simple returns of the traded asset
    train_window : int
        Number of past trading days used for each refit
    feature_window : int
        Rolling window of the regime features
    refit_freq : str
        Pandas period alias for refits ('M' monthly, 'Q' quarterly)
    allocations : dict, optional
        Equity weight per ordinal regime (default 1.0 / 0.5 / 0.0)
    lag : int
        Days between signal and position (>= 1)
    cost_per_unit_turnover : float
        Trading cost per unit of allocation changed (default 5 bps)
    random_state : int
        GMM seed

    Returns:
    --------
    dict with 'strategy_returns', 'benchmark_returns' (buy & hold over the
    same days), 'regimes' (out-of-sample) and 'positions'
    """
    returns = returns.dropna()
    features = make_regime_features(returns, feature_window)
    vol_col = list(features.columns).index('vol')
    periods = features.index.to_period(refit_freq)

    regimes = pd.Series(np.nan, index=features.index)
    for period in periods.unique():
        test_index = features.index[periods == period]
        start = features.index.get_loc(test_index[0])
        if start < train_window:
            continue
        train = features.iloc[start - train_window:start]  # strictly before the period

        scaler = StandardScaler().fit(train.values)
        gmm = GaussianMixture(n_components=n_regimes, covariance_type='full',
                              random_state=random_state).fit(scaler.transform(train.values))
        mapping = order_components_by_volatility(gmm, vol_col)
        raw = gmm.predict(scaler.transform(features.loc[test_index].values))
        regimes.loc[test_index] = [mapping[int(c)] for c in raw]

    regimes = regimes.dropna().astype(int)
    strategy = RegimeStrategy(regimes, allocations=allocations, lag=lag)
    strategy_returns = strategy.compute_strategy_returns(returns, cost_per_unit_turnover)
    positions = strategy.positions.reindex(strategy_returns.index)

    return {
        'strategy_returns': strategy_returns,
        'benchmark_returns': returns.reindex(strategy_returns.index),
        'regimes': regimes,
        'positions': positions,
    }


def constant_mix_returns(returns: pd.Series, equity_weight: float) -> pd.Series:
    """
    Daily-rebalanced fixed mix of the asset and cash (cash earns 0%).

    A regime strategy that holds 67% equities on average should be compared
    with a plain 67/33 mix as well as with 100% buy & hold; otherwise lower
    drawdowns could simply come from holding less equity.
    """
    return returns * equity_weight


def robustness_check(returns: pd.Series,
                     seeds: List[int] = tuple(range(10)),
                     train_windows: List[int] = (252, 504, 756),
                     **kwargs) -> pd.DataFrame:
    """
    Re-run the walk-forward backtest across seeds and training windows.

    Every run is compared over the same dates with buy & hold and with a
    constant mix holding the run's average equity weight. Use the spread of
    results, not a single run, to judge whether an edge is real.
    """
    rows = []
    for tw in train_windows:
        for seed in seeds:
            res = walk_forward_regime_backtest(returns, train_window=tw, random_state=seed, **kwargs)
            s = performance_metrics(res['strategy_returns'])
            b = performance_metrics(res['benchmark_returns'])
            w = res['positions'].mean()
            c = performance_metrics(constant_mix_returns(res['benchmark_returns'], w))
            rows.append({
                'train_window': tw, 'seed': seed,
                'start': s['Start'].date(),
                'strategy_cagr': s['Annual Return'], 'benchmark_cagr': b['Annual Return'],
                'strategy_vol': s['Annual Volatility'], 'benchmark_vol': b['Annual Volatility'],
                'strategy_sharpe': s['Sharpe Ratio'], 'benchmark_sharpe': b['Sharpe Ratio'],
                'strategy_max_dd': s['Max Drawdown'], 'benchmark_max_dd': b['Max Drawdown'],
                'avg_equity_allocation': w,
                'constant_mix_cagr': c['Annual Return'], 'constant_mix_vol': c['Annual Volatility'],
                'constant_mix_sharpe': c['Sharpe Ratio'], 'constant_mix_max_dd': c['Max Drawdown'],
            })
    return pd.DataFrame(rows)


class BacktestEngine:
    """
    Compare a strategy's daily returns with a benchmark's.

    The engine only evaluates return series you give it. It does not check
    how they were produced; build them with RegimeStrategy (lagged) or
    walk_forward_regime_backtest so the inputs are free of look-ahead.
    """

    def __init__(self, strategy_returns: pd.Series, benchmark_returns: pd.Series,
                 risk_free_rate: float = 0.0):
        """
        Parameters:
        -----------
        strategy_returns, benchmark_returns : pd.Series
            Daily returns; only their common dates are used
        risk_free_rate : float
            Annual risk-free rate (default 0%)
        """
        common = strategy_returns.dropna().index.intersection(benchmark_returns.dropna().index)
        self.strategy_returns = strategy_returns.loc[common]
        self.benchmark_returns = benchmark_returns.loc[common]
        self.risk_free_rate = risk_free_rate
        self.metrics = {}

    def compute_metrics(self, returns: pd.Series, name: str = "Strategy") -> Dict[str, float]:
        """Performance metrics for one return series (see performance_metrics)."""
        metrics = performance_metrics(returns, self.risk_free_rate)
        self.metrics[name] = metrics
        return metrics

    def run_backtest(self) -> Dict[str, Dict[str, float]]:
        """Metrics for strategy and benchmark plus their differences."""
        s = self.compute_metrics(self.strategy_returns, "Strategy")
        b = self.compute_metrics(self.benchmark_returns, "Benchmark")
        outperformance = {
            'Total Return Difference': s['Total Return'] - b['Total Return'],
            'CAGR Difference': s['Annual Return'] - b['Annual Return'],
            'Sharpe Difference': s['Sharpe Ratio'] - b['Sharpe Ratio'],
            'Max Drawdown Reduction': s['Max Drawdown'] - b['Max Drawdown'],
            'Volatility Reduction': b['Annual Volatility'] - s['Annual Volatility'],
        }
        self.metrics['Outperformance'] = outperformance
        return {'Strategy': s, 'Benchmark': b, 'Outperformance': outperformance}

    def compute_monthly_metrics(self) -> pd.DataFrame:
        """Monthly compounded returns of strategy and benchmark."""
        compound = lambda r: (1 + r).prod() - 1
        s = self.strategy_returns.resample('ME').apply(compound)
        b = self.benchmark_returns.resample('ME').apply(compound)
        return pd.DataFrame({'Strategy': s, 'Benchmark': b, 'Difference': s - b})

    def compute_rolling_metrics(self, window: int = TRADING_DAYS) -> Dict[str, pd.Series]:
        """Rolling Sharpe and rolling compounded return over `window` days."""
        def rolling_sharpe(r):
            return r.rolling(window).mean() * TRADING_DAYS / (r.rolling(window).std() * np.sqrt(TRADING_DAYS))

        def rolling_return(r):
            return (1 + r).rolling(window).apply(np.prod, raw=True) - 1

        return {
            'Strategy Sharpe': rolling_sharpe(self.strategy_returns),
            'Benchmark Sharpe': rolling_sharpe(self.benchmark_returns),
            'Strategy Return': rolling_return(self.strategy_returns),
            'Benchmark Return': rolling_return(self.benchmark_returns),
        }

    def apply_transaction_costs(self, positions: pd.Series,
                                cost_per_trade: float = 0.001) -> pd.Series:
        """
        Subtract costs from the strategy returns.

        Parameters:
        -----------
        positions : pd.Series
            Allocation HELD each day (e.g. RegimeStrategy.positions)
        cost_per_trade : float
            Cost per unit of allocation changed
        """
        turnover = positions.reindex(self.strategy_returns.index).diff().abs().fillna(0.0)
        return self.strategy_returns - turnover * cost_per_trade

    def period_metrics(self, period_days: int = 63) -> pd.DataFrame:
        """
        Metrics for consecutive non-overlapping blocks of the existing returns.

        This only slices returns that were already computed; it does not refit
        any model. For an out-of-sample test use walk_forward_regime_backtest.
        """
        rows = []
        r = self.strategy_returns
        for start in range(0, len(r) - period_days + 1, period_days):
            block = r.iloc[start:start + period_days]
            m = performance_metrics(block, self.risk_free_rate)
            rows.append({k: v for k, v in m.items() if k != 'Cumulative'})
        return pd.DataFrame(rows)

    def get_summary_report(self) -> str:
        """Plain-text comparison of strategy and benchmark."""
        if 'Strategy' not in self.metrics:
            self.run_backtest()
        s, b, o = self.metrics['Strategy'], self.metrics['Benchmark'], self.metrics['Outperformance']
        line = '=' * 70
        return f"""
{line}
BACKTEST PERFORMANCE REPORT  ({s['Start'].date()} to {s['End'].date()})
{line}
                      Strategy    Benchmark
  Total Return:     {s['Total Return']:>10.2%}  {b['Total Return']:>10.2%}
  CAGR:             {s['Annual Return']:>10.2%}  {b['Annual Return']:>10.2%}
  Volatility:       {s['Annual Volatility']:>10.2%}  {b['Annual Volatility']:>10.2%}
  Sharpe Ratio:     {s['Sharpe Ratio']:>10.2f}  {b['Sharpe Ratio']:>10.2f}
  Sortino Ratio:    {s['Sortino Ratio']:>10.2f}  {b['Sortino Ratio']:>10.2f}
  Max Drawdown:     {s['Max Drawdown']:>10.2%}  {b['Max Drawdown']:>10.2%}

  CAGR difference:         {o['CAGR Difference']:>+8.2%}
  Max drawdown reduction:  {o['Max Drawdown Reduction']:>+8.2%}
{line}
"""
