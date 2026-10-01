# Key Findings

All numbers come from the executed notebooks and `results/summary_metrics.json`. Data: daily closes, January 2010 to December 2024.

> This file replaces an earlier version whose findings were based on regimes fitted to single-day returns, a backtest with look-ahead bias, Granger tests that never ran and corrupted oil data. Its recommendations (for example fixed gold and bond allocations) were not supported by any analysis in this repository and have been removed.

## 1. Regimes describe risk, not direction

The model groups days by rolling 20-day volatility, mean and skew of S&P 500 returns.

| Regime | Share of days | Next-day volatility (annualised) | Next-day mean return | Std. error of that mean |
|---|---|---|---|---|
| Calm | 63.5% | 11.3% | 0.042% | 0.015% |
| Elevated | 34.9% | 21.3% | 0.059% | 0.037% |
| Crisis | 1.6% | 60.4% | 0.238% | 0.491% |

- Volatility the next day is about twice as high in Elevated as in Calm, and about five times as high in Crisis. This is the useful signal.
- Next-day mean returns do not fall as risk rises, and the Crisis average is smaller than its own standard error. The regimes do not predict whether the market goes up or down.
- Regimes are persistent: on any day there is a 95–97% chance of staying in the same regime tomorrow, and runs last 20–37 days on average.
- Model fit: silhouette score 0.24, which indicates overlapping clusters. Volatility forms a continuum, and the regimes are a convenient way to cut it, not sharply separated states.
- The Crisis regime contains 60 days from two episodes (August–September 2011 and March–May 2020). Treat its statistics as illustrative.

## 2. Using regimes to control exposure

Walk-forward backtest, 2012–2024, monthly refits on past data only, one-day signal lag, 5 bps cost.

| | Regime strategy | Fixed 66% equity mix | Buy & hold |
|---|---|---|---|
| CAGR | 8.9% | 8.4% | 12.3% |
| Volatility | 9.5% | 11.1% | 16.7% |
| Sharpe | 0.94 | 0.78 | 0.78 |
| Max drawdown | −15.7% | −23.5% | −33.9% |

Across 30 runs (10 seeds × 3 training windows):

- Max drawdown was smaller than buy-and-hold in **30 of 30** runs (median −20.4% vs −33.9%).
- CAGR was lower than buy-and-hold in **30 of 30** runs (median gap −5.0 points).
- Against a fixed mix with the same average equity weight, max drawdown was smaller in 21 of 30 runs, Sharpe higher in 10 of 30 and CAGR higher in 5 of 30.
- Every run with a higher Sharpe ratio than either benchmark used the 2-year training window. The Sharpe improvement is therefore not robust.
- Cost sensitivity (main run): Sharpe 1.00 at 0 bps, 0.94 at 5 bps, 0.89 at 10 bps, 0.79 at 20 bps.

**Interpretation:** most of the risk reduction comes from holding less equity on average. Timing adds a smaller, setting-dependent improvement in drawdowns. This is consistent with volatility being persistent (so recent volatility predicts near-term volatility) while returns are not predictable.

## 3. Stress episodes (walk-forward main run)

| Window | Regime strategy | Buy & hold | Avg equity weight held |
|---|---|---|---|
| Aug 2015 (China devaluation) | −1.6% | −12.2% | 20% |
| Q4 2018 sell-off | −4.3% | −19.1% | 23% |
| Feb–Mar 2020 (COVID) | −2.2% | −33.6% | 8% |
| 2022 bear market | −10.6% | −24.9% | 45% |
| Rebound, 24 Mar – 8 Jun 2020 | +8.3% | +44.5% | 26% |

The signal reacts after volatility rises. In February 2020 the strategy was already at 50% equity because the preceding two years had been calm, took the first −3.4% day at half exposure and was in cash from the next day. The same caution cost most of the 2020 rebound.

## 4. Cross-asset relationships

Daily return correlations, 2010–2024: S&P 500 vs NASDAQ 0.95, vs oil 0.28, vs gold 0.05, vs US dollar −0.20; gold vs US dollar −0.36. (The earlier figure of −0.16 for stocks vs oil was caused by the corrupted April 2020 oil returns.)

Granger tests, significant after Bonferroni adjustment for 20 tests:

| Lag | Significant pairs |
|---|---|
| 1 day | S&P 500 → NASDAQ, S&P 500 → US dollar, US dollar → gold, NASDAQ → US dollar |
| 5 days | S&P 500 → oil, NASDAQ → oil, S&P 500 → NASDAQ, oil → S&P 500, US dollar → S&P 500 |

- These are predictive associations, not causal mechanisms.
- The series close at different times of day; nonsynchronous closes can create apparent next-day effects mechanically, which is a likely explanation for S&P 500 → NASDAQ.
- Within regimes, no pair is significant in the Calm or Elevated regime after adjustment. The Crisis regime has too few days for reliable tests.
- Whether any effect is large enough to trade after costs was not tested.

## 5. Forecasting regimes

Out-of-sample (5 time-ordered splits), random forest vs naive "no change" forecast:

| Horizon | Model accuracy | Naive accuracy |
|---|---|---|
| 1 day | 56.1% | 92.9% |
| 5 days | 56.9% | 80.0% |
| 20 days | 45.9% | 55.4% |

The model is worse than assuming today's regime continues. Persistence is the main source of predictability.

## 6. Relevance to risk management

- Volatility regimes are a simple, transparent input for **risk monitoring**: flagging when realised risk has moved into a higher state, sizing exposure limits or triggering reviews.
- They should not be presented as a return forecast or a crash predictor.
- Any de-risking rule should be judged against a fixed-exposure benchmark with the same average risk, not only against 100% buy-and-hold.
