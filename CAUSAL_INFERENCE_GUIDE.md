# Cross-Asset Analysis Guide (notebook 03)

How `notebooks/03_causal_inference.ipynb` and `src/causality.py` work, and how to read their output. Results are in the notebook itself and summarised in [INFERENCES.md](INFERENCES.md).

## Inputs

`data/processed/market_with_regimes.csv` (written by notebook 02):

| Column | Content |
|---|---|
| `^GSPC`, `^IXIC`, `GC=F`, `CL=F`, `DX-Y.NYB` | Daily simple returns. `CL=F` is missing on 2020-04-20 and 2020-04-21, when WTI prices were negative. |
| `Regime` | 0, 1, 2 from the full-sample GMM (blank for the first 19 days while the rolling window fills) |
| `Regime_Label` | Calm, Elevated, Crisis |

## Steps

1. **Stationarity (ADF).** Daily returns are tested for a unit root. Needs `statsmodels`.
2. **Correlations.** Pearson correlation of daily returns, shown as a heatmap.
3. **Granger causality.** For every ordered pair (cause X, effect Y) an F-test asks whether X's past returns improve a regression of Y on its own past returns. Run at 1 and 5 lags. `granger_test` implements the same statistic statsmodels reports as `ssr_ftest`, so it runs without statsmodels; when statsmodels is installed, a test checks the two agree.
4. **Multiple-testing adjustment.** Five assets give 20 ordered pairs, so about one would pass p < 0.05 by chance. `get_significant_causality` applies Bonferroni (or Benjamini-Hochberg with `adjust='fdr_bh'`).
5. **VAR and impulse responses.** A VAR(1) on all five series and orthogonalised impulse responses over 10 days. Needs `statsmodels`; rows with missing values are dropped first.
6. **Shocks.** Each day's return is compared with the mean and standard deviation of the previous 20 days (the window is shifted by one day so a shock does not inflate its own yardstick).
7. **Regime-conditional Granger tests.** The F-test is run on the days of each regime. Lags are taken from the full series before filtering, so "yesterday" is always the previous trading day, not the previous day that happened to be in the same regime.

## Reading the results

- A small p-value means **predictability**, not economic causation.
- With about 3,700 daily observations, very small effects become statistically significant. Significance says nothing about whether an effect is large enough to trade after costs.
- The series close at different times of day (US equity indices at 4 pm New York; futures and the dollar index on other schedules). Nonsynchronous closes can create apparent next-day lead-lag effects mechanically.
- The Crisis regime has only about 60 days, so tests within it have little power.

## What changed from the earlier version

- The earlier notebook read the p-value with `result[1][0][1]`. In statsmodels' output `result[lag][0]` is a dictionary of tests, so this raised `KeyError: 1` for every pair. The errors were caught and recorded as missing values, which were then reported as "no significant causality". The correct value is `result[lag][0]['ssr_ftest'][1]`.
- Regime-conditional tests previously filtered to a regime's days and then took lags, which treated days weeks apart as consecutive.
- The −306% and −127% oil "returns" from April 2020 are now removed; they had distorted correlations and the VAR.
- Illustrative outputs in the earlier guide (for example "Oil → S&P p=0.023" and impulse-response values) were not produced by the code and have been removed.

## Using the module directly

```python
import pandas as pd
from src import CausalityAnalyzer

data = pd.read_csv("data/processed/market_with_regimes.csv", index_col=0, parse_dates=True)
assets = ["^GSPC", "^IXIC", "GC=F", "CL=F", "DX-Y.NYB"]
analyzer = CausalityAnalyzer(data, assets)

pvalues = analyzer.granger_causality_matrix(lag=1)
significant = analyzer.get_significant_causality(threshold=0.05, adjust="bonferroni")
by_regime = analyzer.regime_conditional_causality(data["Regime_Label"], lag=1)
```
