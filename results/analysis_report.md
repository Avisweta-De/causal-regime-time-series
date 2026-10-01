# Regime Risk Analysis: Summary Report

Data: S&P 500 and four cross-asset series, 2010-01-05 to 2024-12-30.

## Regimes (volatility states of the S&P 500)

| Regime | Share of days | Next-day daily volatility | Next-day mean return | P(still in regime tomorrow) |
|---|---|---|---|---|
| Calm | 63.5% | 0.71% | 0.042% | 97.3% |
| Elevated | 34.9% | 1.34% | 0.059% | 94.9% |
| Crisis | 1.6% | 3.81% | 0.238% | 96.7% |

Next-day volatility rises from regime to regime, so the regimes carry real information about near-term risk. Next-day mean returns are small and noisy by comparison, so the regimes say little about direction.

## Out-of-sample (walk-forward) backtest

Model refit on past data only (monthly, 504-day training window), signals applied after 1 day(s), 5 bps trading cost per unit of turnover. Period: 2012-02-02 to 2024-12-30.

| | Regime strategy | Fixed 66% equity mix | Buy & hold |
|---|---|---|---|
| CAGR | 8.9% | 8.4% | 12.3% |
| Annual volatility | 9.5% | 11.1% | 16.7% |
| Sharpe ratio | 0.94 | 0.78 | 0.78 |
| Max drawdown | -15.7% | -23.5% | -33.9% |

The fixed mix holds the strategy's average equity weight (66%) every day. It shows how much of the risk reduction comes from timing rather than simply from holding less equity.

## Robustness

Across 30 runs (different random seeds and training windows): the strategy had a smaller max drawdown in 30 runs and a higher Sharpe ratio in 10. CAGR minus buy-and-hold had a median of -5.0 points (range -6.3 to -3.1). Max drawdown was reduced by a median of 13.5 points (range 6.2 to 19.2).

Against the fixed mix with the same average equity weight, the strategy had a smaller max drawdown in 21 of 30 runs, a higher Sharpe ratio in 10 and a higher CAGR in 5.

## Lead-lag (Granger) tests

Pairs where yesterday's return of one asset helps predict today's return of another, after Bonferroni adjustment for 20 tests:

- ^GSPC → ^IXIC (adjusted p = 0.00013)
- ^GSPC → DX-Y.NYB (adjusted p = 0.0012)
- DX-Y.NYB → GC=F (adjusted p = 0.015)
- ^IXIC → DX-Y.NYB (adjusted p = 0.027)

Granger tests measure predictability, not economic causation. Assets that close at different times of day can show lead-lag effects mechanically, and this analysis did not test whether these effects are large enough to trade after costs.

## Correlations of daily returns

- S&P 500 vs NASDAQ: 0.95
- S&P 500 vs oil: 0.28
- S&P 500 vs gold: 0.05
- S&P 500 vs US dollar: -0.2
- Gold vs US dollar: -0.36

## Limitations

- One equity index and one historical period; results may not hold in other markets or periods.
- Cash is assumed to earn 0%, and taxes, slippage and market impact are not modelled.
- The regime model is a statistical description, not a forecast of crashes.
