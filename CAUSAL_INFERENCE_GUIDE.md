# Causal Inference Notebook Guide

## ✅ What's Updated

The `03_causal_inference.ipynb` notebook now correctly loads **`market_with_regimes.csv`** which contains:
- Returns for all 5 assets (^GSPC, ^IXIC, GC=F, CL=F, DX-Y.NYB)
- Regime labels (Bull, Neutral, Crisis)
- Regime transitions and duration

## 📊 Data Structure

```
market_with_regimes.csv
├── Returns columns: CL=F, DX-Y.NYB, GC=F, ^GSPC, ^IXIC
├── Regime: 0, 1, 2 (numeric labels)
├── Regime_Label: Bull, Neutral, Crisis (economic labels)
├── Regime_Change: 1 if regime changed that day, 0 otherwise
└── Regime_Duration: How many days in current regime
```

## 🚀 Notebook Flow

### Cell 1: Load Data
```python
data = pd.read_csv('../data/processed/market_with_regimes.csv', index_col=0, parse_dates=True)
returns = data[assets]
regime_info = data[['Regime', 'Regime_Label', ...]]
```

### Cell 2-4: Stationarity Testing
Tests if each asset is stationary (required for VAR/Granger):
- **Output**: ADF statistics + p-values
- **Interpretation**: p < 0.05 = Stationary ✅

### Cell 5-6: Granger Causality Matrix
Tests which assets Granger-cause which:
- **Example**: Oil → S&P (p=0.023) ✅
- **Visualization**: Heatmap of p-values

### Cell 7-8: VAR Model
Fits Vector AutoRegression:
- **Optimal lags**: Auto-selected via AIC
- **Output**: Model summary + diagnostics

### Cell 9-10: Impulse Response Functions
"What if oil crashes 1%?"
- **Visualization**: Heatmap showing effects on all assets
- **Interpretation**: How shocks propagate over 10 days

### Cell 11-12: Shock Detection
Identifies extreme market events:
- **Method**: Rolling Z-scores (threshold=2.5)
- **Output**: Top 10 market crashes/rallies
- **Value**: Validates regimes against real events

### Cell 13-14: Regime-Conditional Causality ⭐ (KEY)
Runs Granger causality separately for each regime:
- **Bull regime**: Weak causality (correlations dominate)
- **Crisis regime**: STRONG causality (Oil → Stocks)
- **Insight**: Different hedging strategies per regime

### Cell 15-16: Correlation Analysis by Regime
Shows how correlations change:
- **Bull**: Low correlation (diversification works)
- **Crisis**: High correlation (diversification collapses!)

### Cell 17+: Summary & Recommendations
Actionable insights for:
- Portfolio managers
- Traders
- Risk teams

---

## 🎯 Run the Notebook

```bash
# From causal-regime-time-series directory
jupyter notebook notebooks/03_causal_inference.ipynb
```

**Prerequisites**:
- ✅ `notebooks/01_data_collection.ipynb` (downloaded prices)
- ✅ `notebooks/02_regime_detection.ipynb` (created market_with_regimes.csv)

---

## 📧 Key Outputs Expected

### Granger Causality
```
Oil → S&P 500        p=0.023  ✅ SIGNIFICANT
USD → S&P 500        p=0.041  ✅ SIGNIFICANT
Gold → S&P 500       p=0.087  ❌ Not significant
```

### VAR Model
```
Optimal lag order: 2 (AIC)
Model converged: Yes
```

### IRF: If Oil ↓ 1%
```
Day 1: S&P ↑ 0.12%
Day 5: S&P ↑ 0.03%
(Inverse relationship, normal behavior)
```

### Regime Statistics
```
Bull   (60% of time): Causality p-values = [0.15, 0.22, ...]
Crisis (5% of time):  Causality p-values = [0.01, 0.03, ...] ⭐ STRONGER
```

---

## 🔄 Integration with Production Modules

You can also use `src/causality.py` directly:

```python
from src.causality import CausalityAnalyzer
from src.regimes import RegimeDetector

# Load combined data
data = pd.read_csv('../data/processed/market_with_regimes.csv')

# Directly call analyzer
analyzer = CausalityAnalyzer(data, assets=['^GSPC', '^IXIC', 'CL=F', 'GC=F', 'DX-Y.NYB'])
gc_matrix = analyzer.granger_causality_matrix()
```

---

## 📝 Notes

1. **Regime Data**: Already includes Bull/Neutral/Crisis labels from notebook 02
2. **No Double-Loading**: Uses combined file, not separate files
3. **Regime-Conditional Analysis**: Automatically splits by 'Regime_Label' column
4. **Reproducible**: Same data, same results (random_state=42 set in GMM)

---

## ❓ Troubleshooting

**Error: "market_with_regimes.csv not found"**
- Run `02_regime_detection.ipynb` first to create it

**Error: "Regime_Label column missing"**
- Ensure notebook 02 completed successfully
- Check data: `data.columns` should show 'Regime_Label'

**Error: "Assets not aligned"**
- This shouldn't happen since data is pre-aligned
- If it does, the data file may be corrupted

---

## ✨ Next Steps

1. ✅ Run `03_causal_inference.ipynb` 
2. ⏳ Create `src/modules` for production deployment
3. ⏳ Build `04_regime_forecasting.ipynb` for backtesting
4. ⏳ Add unit tests

---

**Last Updated**: April 4, 2026  
**Status**: Phase 1 Complete - Causal Inference Ready 🚀
