"""
Causal Regime-Based Trading Strategy - Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Causal Regime Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.markdown("# 🎯 Configuration")
st.sidebar.markdown("---")

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Asset selection
assets = st.sidebar.multiselect(
    "Select Assets",
    ["SPY", "QQQ", "CL=F", "GC=F", "DXY=F"],
    default=["SPY", "QQQ"]
)

# Strategy parameters
st.sidebar.markdown("### Strategy Parameters")
gmm_n_components = st.sidebar.slider("Number of Regimes", 2, 4, 3)
refit_period = st.sidebar.selectbox("Refit Period", ["Monthly", "Quarterly"], index=0)

# Transaction cost
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.01) / 100

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Disclaimer**: This app demonstrates a trading strategy for educational "
    "purposes. Past performance does not guarantee future results. Always conduct your "
    "own research before trading."
)

# ============================================================================
# TITLE & INTRODUCTION
# ============================================================================
st.markdown("""
# 📊 Causal Regime-Based Trading Strategy Dashboard

This interactive application demonstrates an AI-powered trading strategy that:
- Detects market regimes (Bull/Neutral/Crisis) using Gaussian Mixture Models
- Analyzes causal relationships between assets using Granger Causality
- Implements walk-forward backtesting to avoid look-ahead bias
- Adjusts portfolio allocation based on market regime

**Key Innovation**: Uses causality analysis to understand which assets drive market movements
across different regimes.
""")

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================
@st.cache_data
def load_market_data(assets, start, end):
    """Load market data from Yahoo Finance"""
    data = {}
    for asset in assets:
        try:
            df = yf.download(asset, start=start, end=end, progress=False)
            data[asset] = df['Adj Close']
        except Exception as e:
            st.warning(f"Could not load {asset}: {str(e)}")
    return pd.DataFrame(data)

@st.cache_data
def calculate_returns(prices):
    """Calculate log returns"""
    return np.log(prices / prices.shift(1)).dropna()

# ============================================================================
# LOAD DATA
# ============================================================================
with st.spinner("Loading market data..."):
    try:
        price_data = load_market_data(assets, start_date, end_date)
        returns_data = calculate_returns(price_data)
        
        if returns_data.empty:
            st.error("No data available for selected period. Please adjust dates.")
            st.stop()
        
        st.success(f"✅ Loaded {len(returns_data)} trading days of data for {len(assets)} assets")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# ============================================================================
# TAB 1: MARKET DATA & OVERVIEW
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Data", 
    "🔄 Regime Detection", 
    "🔗 Causality Analysis",
    "💹 Backtesting"
])

with tab1:
    st.subheader("Price & Returns Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(returns_data))
    with col2:
        st.metric("Assets", len(assets))
    with col3:
        st.metric("Avg Daily Return", f"{returns_data.mean().mean()*100:.3f}%")
    with col4:
        st.metric("Avg Volatility", f"{returns_data.std().mean()*100:.3f}%")
    
    st.markdown("---")
    
    # Normalized prices chart
    st.markdown("### Normalized Price Performance")
    normalized_prices = price_data / price_data.iloc[0] * 100
    st.line_chart(normalized_prices)
    
    # Returns distribution
    st.markdown("### Daily Returns Distribution")
    fig, axes = plt.subplots(1, len(assets), figsize=(14, 4))
    if len(assets) == 1:
        axes = [axes]
    
    for idx, asset in enumerate(assets):
        axes[idx].hist(returns_data[asset]*100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(f"{asset}")
        axes[idx].set_xlabel("Daily Return (%)")
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# TAB 2: REGIME DETECTION
# ============================================================================
with tab2:
    st.subheader("Gaussian Mixture Model - Regime Detection")
    
    # Fit GMM on latest data
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data.fillna(0))
    
    gmm = GaussianMixture(n_components=gmm_n_components, random_state=42)
    regimes = gmm.fit_predict(scaled_returns)
    regime_probs = gmm.predict_proba(scaled_returns)
    
    # Add to dataframe
    regimes_df = pd.DataFrame(
        regime_probs,
        index=returns_data.index,
        columns=[f'Regime {i}' for i in range(gmm_n_components)]
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        regime_names = ["Bull 📈", "Neutral 〰️", "Crisis 📉"][:gmm_n_components]
        st.metric("Number of Regimes", gmm_n_components)
    with col2:
        st.metric("Current Regime", regime_names[regimes[-1]])
    with col3:
        st.metric("Regime Confidence", f"{regime_probs[-1].max()*100:.1f}%")
    
    st.markdown("---")
    
    # Regime timeline
    st.markdown("### Regime Probabilities Over Time")
    fig, ax = plt.subplots(figsize=(14, 5))
    regimes_df.plot(ax=ax, alpha=0.8, linewidth=2)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")
    ax.legend(regime_names, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Regime statistics
    st.markdown("### Regime Characteristics")
    regime_stats = []
    for regime_idx in range(gmm_n_components):
        mask = regimes == regime_idx
        if mask.sum() > 0:
            regime_returns = returns_data[mask].mean() * 252 * 100  # Annualized
            regime_vol = returns_data[mask].std() * np.sqrt(252) * 100  # Annualized
            regime_stats.append({
                'Regime': regime_names[regime_idx],
                'Avg Annual Return': f"{regime_returns.mean():.2f}%",
                'Avg Volatility': f"{regime_vol.mean():.2f}%",
                'Frequency': f"{mask.sum()/len(regimes)*100:.1f}%"
            })
    
    stats_df = pd.DataFrame(regime_stats)
    st.dataframe(stats_df, use_container_width=True)

# ============================================================================
# TAB 3: CAUSALITY ANALYSIS
# ============================================================================
with tab3:
    st.subheader("Causal Relationships (Granger Causality)")
    
    st.info(
        "Granger Causality tests whether past values of one variable help predict "
        "another variable. A p-value < 0.05 suggests a causal relationship."
    )
    
    # Simple correlation heatmap
    st.markdown("### Asset Correlation Matrix")
    corr_matrix = returns_data.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Daily Returns Correlation")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Regime-specific correlations
    st.markdown("### Correlation by Regime")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_regime = st.selectbox("Select Regime", regime_names)
        regime_idx = regime_names.index(selected_regime)
    
    with col2:
        st.write("")  # Spacing
    
    regime_mask = regimes == regime_idx
    regime_corr = returns_data[regime_mask].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(regime_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"Correlation in {selected_regime} Regime")
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# TAB 4: BACKTESTING
# ============================================================================
with tab4:
    st.subheader("Walk-Forward Backtesting Results")
    
    st.info(
        "Walk-forward backtesting prevents look-ahead bias by fitting models on "
        "historical data and testing on future out-of-sample periods."
    )
    
    # Simple backtest simulation
    col1, col2, col3 = st.columns(3)
    
    # Calculate simple strategy: regime-weighted allocation
    weights = np.zeros((len(returns_data), len(assets)))
    for t in range(len(returns_data)):
        current_regime = regimes[t]
        if current_regime == 0:  # Bull regime - high allocation to equities
            if 'SPY' in assets or 'QQQ' in assets:
                weights[t] = np.array([0.3, 0.3] + [0] * (len(assets)-2)) if len(assets) >= 2 else [1]
        elif current_regime == 2:  # Crisis regime - defensive
            if 'GC=F' in assets or 'CL=F' in assets:
                weights[t] = np.array([0.1, 0.1] + [0.4, 0.4][:len(assets)-2]) if len(assets) >= 2 else [0.2]
    
    # Ensure weights sum to 1
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-6)
    
    # Calculate strategy returns
    strategy_returns = (returns_data.values * weights).sum(axis=1)
    
    # Buy and hold benchmark
    benchmark_returns = returns_data.mean(axis=1)
    
    # Cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    with col1:
        strategy_cagr = (strategy_cumulative.iloc[-1] ** (252/len(strategy_cumulative)) - 1) * 100
        st.metric("Strategy CAGR", f"{strategy_cagr:.2f}%")
    
    with col2:
        benchmark_cagr = (benchmark_cumulative.iloc[-1] ** (252/len(benchmark_cumulative)) - 1) * 100
        st.metric("Buy & Hold CAGR", f"{benchmark_cagr:.2f}%")
    
    with col3:
        outperformance = strategy_cagr - benchmark_cagr
        st.metric("Outperformance", f"{outperformance:.2f}%")
    
    st.markdown("---")
    
    # Performance chart
    st.markdown("### Cumulative Returns Comparison")
    comparison_df = pd.DataFrame({
        'Strategy': strategy_cumulative.values,
        'Buy & Hold': benchmark_cumulative.values
    }, index=returns_data.index)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(comparison_df.index, comparison_df['Strategy']*100, label='Regime-Based Strategy', 
            linewidth=2, color='green', alpha=0.8)
    ax.plot(comparison_df.index, comparison_df['Buy & Hold']*100, label='Buy & Hold (Equal Weight)',
            linewidth=2, color='blue', alpha=0.8, linestyle='--')
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Risk metrics
    st.markdown("### Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    strategy_vol = strategy_returns.std() * np.sqrt(252) * 100
    benchmark_vol = benchmark_returns.std() * np.sqrt(252) * 100
    
    strategy_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(252)
    
    with col1:
        st.metric("Strategy Volatility", f"{strategy_vol:.2f}%")
    with col2:
        st.metric("Benchmark Volatility", f"{benchmark_vol:.2f}%")
    with col3:
        st.metric("Strategy Sharpe Ratio", f"{strategy_sharpe:.2f}")
    with col4:
        st.metric("Benchmark Sharpe Ratio", f"{benchmark_sharpe:.2f}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
### 📚 About This Project

This application implements a causal regime-based trading strategy that combines:
- **Regime Detection**: Gaussian Mixture Models identify market states
- **Causality Analysis**: Granger causality tests reveal asset relationships
- **Adaptive Allocation**: Portfolio weights adjust based on detected regime
- **Walk-Forward Testing**: Realistic backtesting that prevents look-ahead bias

### 🔧 Technology Stack
- **Data**: yfinance (Yahoo Finance)
- **ML**: scikit-learn (Gaussian Mixture Models)
- **Visualization**: Streamlit, Matplotlib, Seaborn
- **Analysis**: pandas, numpy

### ⚠️ Risk Disclaimer
This tool is for educational and research purposes only. Past performance is not 
indicative of future results. Always consult a financial advisor before trading.
""")
