# Architecture and Change Log

## Pipeline

```
01 Data           yfinance closes → compute_returns (forward-fill holidays, drop returns from prices ≤ 0)
       ↓
02 Regimes        rolling 20-day mean / volatility / skew → standardise → GMM (3 components)
                  → re-order components by volatility: 0 Calm, 1 Elevated, 2 Crisis
       ↓
03 Cross-asset    correlations · Granger F-tests + Bonferroni · regime-conditional tests · VAR/IRF · shocks
       ↓
04 Backtest       monthly walk-forward refits on past data · regime at close t → position at t+1
                  · 5 bps per unit turnover · fixed-mix and buy-and-hold benchmarks
                  · 30-run robustness grid · cost sensitivity · stress windows · OOS regime forecasts
       ↓
05 Report         computed facts → deterministic Markdown report → optional LLM narration with number check
```

## Modules

| Module | Main contents |
|---|---|
| `src/data.py` | `compute_returns`, `data_quality_report`, `DataLoader` (yfinance imported lazily) |
| `src/regimes.py` | `make_regime_features`, `RegimeDetector`, `order_components_by_volatility`, `HMMRegimeDetector` |
| `src/causality.py` | `granger_test`, `adjust_pvalues`, `CausalityAnalyzer`, `detect_shocks` |
| `src/strategy.py` | `RegimeStrategy` (lag ≥ 1 enforced, turnover costs) |
| `src/backtesting.py` | `performance_metrics`, `walk_forward_regime_backtest`, `robustness_check`, `constant_mix_returns`, `BacktestEngine` |
| `src/forecasting.py` | `RegimeForecaster` (Markov transitions, ML with TimeSeriesSplit vs naive baseline) |
| `src/llm_insights.py` | `build_facts`, `render_report`, `ungrounded_numbers`, `LLMInsightGenerator`, `InsightGenerator` |
| `src/utils.py` | `MetricsCalculator`, `ConfigManager`, `DataValidator`, `ExperimentTracker` |

Optional dependencies are imported only where used: `yfinance` (download), `statsmodels` (ADF, VAR, IRF), `hmmlearn` (HMM), `openai` (narration).

## Design rules

1. **No look-ahead.** Anything that trades or forecasts uses information available at the close of the previous day. The walk-forward backtest refits on data that ends before the period it classifies.
2. **Stable labels.** GMM component numbers are arbitrary, so every fit is re-ordered by volatility before labels or allocations are assigned.
3. **Fair benchmarks.** A de-risking rule is compared with a fixed mix at the same average equity weight, not only with 100% equities.
4. **Robustness over single runs.** Headline numbers are reported alongside a grid of seeds and training windows.
5. **Numbers come from code.** Reports are generated from computed results; LLM text is optional and checked against them.

## Tests

- `tests/test_smoke.py`: imports and end-to-end runs on synthetic data.
- `tests/test_correctness.py`: one-day lag, perfect-foresight labels cannot leak, changing future returns leaves past walk-forward results unchanged, volatility ordering of labels, Granger power and null behaviour, statsmodels agreement (when installed), invalid-price handling, metrics against hand calculations, and detection of invented numbers in LLM text.

## Change log

### v2.0 (October 2026): correctness fixes

| Problem in v1 | Effect | Fix |
|---|---|---|
| Regime of day *t* applied to day *t*'s own return | +3,616% headline; same-day "Crisis" days were the crash days | Positions use the previous day's regime; lag 0 rejected |
| Regimes fitted on single-day returns over the full sample | Neutral and Crisis "regimes" lasting about 1 day; in-sample labels used for evaluation | Rolling features; walk-forward refits for anything evaluated |
| Walk-forward mapped raw GMM ids to allocations | Allocations effectively arbitrary on each refit | Components ordered by volatility on every refit |
| Walk-forward still used same-day features and returns | Residual look-ahead | One-day lag between feature date and position |
| `BacktestEngine.walk_forward_test` never refit a model | Mislabelled as walk-forward | Replaced by `period_metrics`; real walk-forward in `walk_forward_regime_backtest` |
| README quoted ~8–12% CAGR and Sharpe 2.80 as realistic | Contradicted the notebook's own output (4.45% CAGR, Sharpe 0.39) | README rebuilt from computed results |
| Granger p-value read from the wrong place | Every test failed; reported as "no causality" | Correct F-test, Bonferroni adjustment |
| Regime-conditional Granger lagged after filtering | Non-consecutive days treated as consecutive | Lags taken before filtering |
| WTI −$37.63 price on 2020-04-20 | Returns of −306% and −127%; distorted correlations and VAR | Returns from non-positive prices set to missing |
| Notebook 05 printed hand-written "mock" LLM output | Report with figures that were never computed | Reports generated from computed facts; LLM optional and checked |
| `VAR.select_lags` (not a statsmodels method) | Automatic lag selection would fail | `select_order` |
| Monthly returns summed instead of compounded | Small errors in monthly figures | Compounded |
| Sharpe computed from CAGR | Non-standard ratio | Annualised mean excess return / annualised volatility |
| ML forecaster reported training accuracy | Optimistic | Time-ordered out-of-sample evaluation vs naive baseline |

### v1.0 (April 2026)

Initial version: data pipeline, GMM regimes, Granger/VAR analysis, regime allocation backtest, LLM report notebook.

## References

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*.
- Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*.
