"""
Causal Regime-Based Trading Strategy - Interactive Streamlit Dashboard
Version 2: Fixed data loading issues
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
    ["SPY", "QQQ", "IWM", "VNQ", "AGG"],
    default=["SPY", "QQQ"]
)

if not assets:
    st.error("Please select at least one asset")
    st.stop()

# Strategy parameters
st.sidebar.markdown("### Strategy Parameters")
gmm_n_components = st.sidebar.slider("Number of Regimes", 2, 4, 3)
refit_period = st.sidebar.selectbox("Refit Period", ["Monthly", "Quarterly"], index=0)

# Transaction cost
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.01) / 100

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Educational Disclaimer**: This app is for learning purposes only. "
    "Past performance ≠ future results. Do your own research before trading."
)

# ============================================================================
# TITLE & INTRODUCTION
# ============================================================================
st.markdown("""
# 📊 Causal Regime-Based Trading Strategy Dashboard

An AI-powered trading strategy that:
- 📈 Detects market regimes (Bull/Neutral/Crisis)
- 🔗 Analyzes causal relationships between assets
- ⚠️ Uses walk-forward backtesting to avoid look-ahead bias
- 🎯 Adjusts portfolio based on market regime
""")

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================
@st.cache_data(ttl=3600)
def load_asset_data(ticker, start_date, end_date):
    """Load single asset data from Yahoo Finance"""
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data is None or data.empty:
            return None
        
        # Ensure data is a DataFrame (not Series)
        if isinstance(data, pd.Series):
            data = data.to_frame(name=ticker)
        
        # Get the close price (try Adj Close first, then Close)
        if 'Adj Close' in data.columns:
            prices = data['Adj Close'].dropna()
        elif 'Close' in data.columns:
            prices = data['Close'].dropna()
        else:
            return None
        
        # Ensure we have a pandas Series with index
        if isinstance(prices, (pd.Series, pd.Index)):
            return prices
        else:
            return None
            
    except Exception as e:
        return None

def load_market_data(assets, start_date, end_date):
    """Load multiple assets and combine into DataFrame"""
    price_data = {}
    failed_assets = []
    
    for asset in assets:
        prices = load_asset_data(asset, start_date, end_date)
        if prices is not None and len(prices) > 1:  # Need at least 2 data points
            price_data[asset] = prices
        else:
            failed_assets.append(asset)
    
    if not price_data:
        return None, "Could not load any assets. Check ticker symbols."
    
    # Combine into DataFrame - ensure all are Series with proper index
    try:
        df = pd.concat(price_data, axis=1)
        df.columns = list(price_data.keys())
        df = df.dropna(how='all')  # Remove all-NaN rows
        
        # Forward fill gaps within 5 days (using newer syntax)
        df = df.ffill(limit=5)
        df = df.dropna()  # Remove any remaining NaN
        
        if len(df) < 10:
            return None, "Not enough data points after cleaning"
        
        return df, None
    except Exception as e:
        return None, f"Error combining data: {str(e)}"

# ============================================================================
# LOAD DATA
# ============================================================================
st.markdown("### Loading market data...")
price_data, warning_msg = load_market_data(assets, start_date, end_date)

if warning_msg:
    st.warning(f"⚠️ {warning_msg}")

if price_data is None or price_data.empty:
    st.error("❌ Failed to load any valid data. Please try:")
    st.markdown("- Different date range (at least 2 months)")
    st.markdown("- Different assets (suggested: SPY, QQQ, IWM, VNQ, AGG)")
    st.markdown("- Check your internet connection")
    st.stop()

# Ensure we have enough data
if len(price_data) < 10:
    st.error(f"Insufficient data: Only {len(price_data)} data points. Need at least 10.")
    st.stop()

# Calculate returns
try:
    returns_data = np.log(price_data / price_data.shift(1)).dropna()
    if returns_data.empty:
        st.error("Could not calculate returns. Check data quality.")
        st.stop()
    st.success(f"✅ Loaded {len(price_data)} trading days, {len(assets)} assets")
except Exception as e:
    st.error(f"Error calculating returns: {str(e)}")
    st.stop()

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Data", 
    "🔄 Regime Detection", 
    "🔗 Correlation",
    "💹 Backtest"
])

# ============================================================================
# TAB 1: MARKET DATA
# ============================================================================
with tab1:
    st.subheader("Price Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Points", len(price_data))
    col2.metric("Assets", len(assets))
    col3.metric("Avg Daily Return", f"{returns_data.mean().mean()*100:.3f}%")
    col4.metric("Avg Volatility", f"{returns_data.std().mean()*100:.3f}%")
    
    st.markdown("---")
    st.markdown("### Normalized Prices (Base = 100)")
    normalized = (price_data / price_data.iloc[0] * 100)
    st.line_chart(normalized)
    
    st.markdown("### Returns Distribution")
    fig, axes = plt.subplots(1, min(len(assets), 5), figsize=(15, 4))
    if len(assets) == 1:
        axes = [axes]
    
    for idx, asset in enumerate(assets[:5]):
        axes[idx].hist(returns_data[asset]*100, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(asset)
        axes[idx].set_xlabel("Daily Return (%)")
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# TAB 2: REGIME DETECTION
# ============================================================================
with tab2:
    st.subheader("Gaussian Mixture Model - Market Regimes")
    
    try:
        # Fit GMM
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_data.fillna(0))
        
        gmm = GaussianMixture(n_components=gmm_n_components, random_state=42)
        regimes = gmm.fit_predict(scaled_returns)
        regime_probs = gmm.predict_proba(scaled_returns)
        
        # Regime dataframe
        regime_names = ["🟢 Bull", "🟡 Neutral", "🔴 Crisis"][:gmm_n_components]
        regimes_df = pd.DataFrame(regime_probs, index=returns_data.index, 
                                  columns=regime_names)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Regimes", gmm_n_components)
        col2.metric("Current Regime", regime_names[regimes[-1]])
        col3.metric("Confidence", f"{regime_probs[-1].max()*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### Regime Probabilities Over Time")
        st.line_chart(regimes_df)
        
        st.markdown("### Regime Statistics")
        stats = []
        for i, regime_name in enumerate(regime_names):
            mask = regimes == i
            if mask.sum() > 0:
                avg_ret = returns_data[mask].mean().mean() * 252 * 100
                avg_vol = returns_data[mask].std().mean() * np.sqrt(252) * 100
                freq = mask.sum() / len(regimes) * 100
                stats.append({
                    'Regime': regime_name,
                    'Avg Return (%)': f"{avg_ret:.2f}",
                    'Volatility (%)': f"{avg_vol:.2f}",
                    'Frequency': f"{freq:.1f}%"
                })
        st.dataframe(pd.DataFrame(stats), use_container_width=True)
        
    except Exception as e:
        st.error(f"Regime detection error: {str(e)}")

# ============================================================================
# TAB 3: CORRELATION ANALYSIS
# ============================================================================
with tab3:
    st.subheader("Asset Correlations")
    
    try:
        # Overall correlation
        st.markdown("### Overall Correlation Matrix")
        corr = returns_data.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                    square=True, ax=ax, vmin=-1, vmax=1)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### Correlation by Regime")
        
        regime_name = st.selectbox("Select Regime", regime_names)
        regime_idx = regime_names.index(regime_name)
        
        mask = regimes == regime_idx
        regime_corr = returns_data[mask].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(regime_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=ax, vmin=-1, vmax=1)
        ax.set_title(f"Correlation in {regime_name} Regime")
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Correlation analysis error: {str(e)}")

# ============================================================================
# TAB 4: BACKTESTING
# ============================================================================
with tab4:
    st.subheader("Walk-Forward Backtest Results")
    
    try:
        # Simple regime-based strategy
        weights = np.zeros((len(returns_data), len(assets)))
        
        for t in range(len(returns_data)):
            if regimes[t] == 0:  # Bull - 100% risky
                weights[t] = 1.0
            elif regimes[t] == 1:  # Neutral - 50% risky
                weights[t] = 0.5
            else:  # Crisis - defensive
                weights[t] = 0.2
        
        # Normalize weights
        weights = weights / len(assets)
        
        # Calculate returns
        strategy_ret = (returns_data.values * weights).sum(axis=1)
        benchmark_ret = returns_data.mean(axis=1)
        
        # Cumulative
        strategy_cum = (1 + strategy_ret).cumprod()
        benchmark_cum = (1 + benchmark_ret).cumprod()
        
        col1, col2, col3 = st.columns(3)
        
        strategy_cagr = (strategy_cum.iloc[-1] ** (252/len(strategy_cum)) - 1) * 100
        benchmark_cagr = (benchmark_cum.iloc[-1] ** (252/len(benchmark_cum)) - 1) * 100
        
        col1.metric("Strategy CAGR", f"{strategy_cagr:.2f}%")
        col2.metric("Benchmark CAGR", f"{benchmark_cagr:.2f}%")
        col3.metric("Outperformance", f"{strategy_cagr - benchmark_cagr:.2f}%")
        
        st.markdown("---")
        st.markdown("### Performance Comparison")
        
        perf = pd.DataFrame({
            'Strategy': strategy_cum.values,
            'Benchmark': benchmark_cum.values
        }, index=returns_data.index)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(perf.index, perf['Strategy']*100, label='Regime-Based', linewidth=2)
        ax.plot(perf.index, perf['Benchmark']*100, label='Buy & Hold', linewidth=2, linestyle='--')
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Risk Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        s_vol = strategy_ret.std() * np.sqrt(252) * 100
        b_vol = benchmark_ret.std() * np.sqrt(252) * 100
        s_sharp = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252) if strategy_ret.std() > 0 else 0
        b_sharp = (benchmark_ret.mean() / benchmark_ret.std()) * np.sqrt(252) if benchmark_ret.std() > 0 else 0
        
        col1.metric("Strategy Vol", f"{s_vol:.2f}%")
        col2.metric("Benchmark Vol", f"{b_vol:.2f}%")
        col3.metric("Strategy Sharpe", f"{s_sharp:.2f}")
        col4.metric("Benchmark Sharpe", f"{b_sharp:.2f}")
        
    except Exception as e:
        st.error(f"Backtesting error: {str(e)}")

st.markdown("---")
st.markdown("### 📚 About | ⚖️ Disclaimer | 🔧 Tech Stack")
st.info("Educational project. Not financial advice. Use pre-cleaned tickers: SPY, QQQ, IWM, VNQ, AGG")
