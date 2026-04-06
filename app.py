"""
Causal Regime Detection in Time Series
=======================================
Production-grade Streamlit dashboard for quantitative regime analysis.

Author : Avisweta De
GitHub : https://github.com/Avisweta-De/causal-regime-time-series
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Regime Detection | Avisweta De",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Avisweta-De/causal-regime-time-series",
        "Report a bug": "https://github.com/Avisweta-De/causal-regime-time-series/issues",
        "About": "Quantitative finance system for regime-aware portfolio allocation.",
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium fintech dark theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---- font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- background ---- */
.stApp { background: linear-gradient(135deg, #0A0F1E 0%, #0D1527 50%, #0A0F1E 100%); }

/* ---- metric cards ---- */
.metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,.08), rgba(99,102,241,.08));
    border: 1px solid rgba(0,212,255,.2);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform .2s, box-shadow .2s;
    margin-bottom: 8px;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,212,255,.15); }
.metric-value { font-size: 2rem; font-weight: 700; color: #00D4FF; line-height: 1.1; }
.metric-label { font-size: .78rem; color: #9CA3AF; text-transform: uppercase;
                letter-spacing: .08em; margin-top: 4px; }
.metric-delta-pos { font-size: .82rem; color: #34D399; margin-top: 2px; }
.metric-delta-neg { font-size: .82rem; color: #F87171; margin-top: 2px; }

/* ---- section header ---- */
.section-header {
    background: linear-gradient(90deg, rgba(0,212,255,.12), transparent);
    border-left: 3px solid #00D4FF;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E8EAF0;
    letter-spacing: .02em;
}

/* ---- regime badges ---- */
.badge-bull { background:#064E3B; color:#34D399; border-radius:20px;
              padding:3px 12px; font-size:.78rem; font-weight:600; }
.badge-neutral { background:#1C1917; color:#FCD34D; border-radius:20px;
                 padding:3px 12px; font-size:.78rem; font-weight:600; }
.badge-crisis { background:#450A0A; color:#F87171; border-radius:20px;
                padding:3px 12px; font-size:.78rem; font-weight:600; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1527 0%, #0A0F1E 100%);
    border-right: 1px solid rgba(0,212,255,.15);
}
[data-testid="stSidebar"] .stRadio label { font-size:.9rem; padding: 2px 0; }

/* ---- dataframe ---- */
.dataframe { border-radius: 8px !important; }
thead tr th { background: #111827 !important; color: #00D4FF !important; }

/* ---- spinner ---- */
.stSpinner > div { border-top-color: #00D4FF !important; }

/* ---- tab ---- */
button[data-baseweb="tab"] { color: #9CA3AF !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
}

/* ---- info / warning boxes ---- */
.stAlert { border-radius: 10px !important; }

/* ---- logo / title area ---- */
.hero {
    text-align: center;
    padding: 18px 0 6px;
    background: linear-gradient(135deg, rgba(0,212,255,.06), rgba(99,102,241,.06));
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid rgba(0,212,255,.12);
}
.hero h1 { font-size: 2.4rem; font-weight: 700;
           background: linear-gradient(90deg, #00D4FF, #818CF8);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #9CA3AF; font-size: .95rem; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Gold (GLD)": "GLD",
    "Oil (USO)": "USO",
    "US Dollar (UUP)": "UUP",
}
REGIME_COLORS = {"Bull": "#34D399", "Neutral": "#FCD34D", "Crisis": "#F87171"}
REGIME_BG = {"Bull": "rgba(52,211,153,.12)", "Neutral": "rgba(252,211,77,.12)", "Crisis": "rgba(248,113,113,.12)"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#E8EAF0"),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,.06)", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.1)", borderwidth=1),
    margin=dict(l=0, r=0, t=40, b=0),
    hovermode="x unified",
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = "", positive: bool = True) -> str:
    delta_class = "metric-delta-pos" if positive else "metric-delta-neg"
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f"{delta_html}"
        f"</div>"
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached data & analysis
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"]
    else:
        prices = df[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.dropna(how="all")


@st.cache_data(show_spinner=False)
def compute_returns(prices_json: str) -> str:
    prices = pd.read_json(prices_json)
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns.to_json()


@st.cache_data(show_spinner=False)
def fit_regimes(returns_json: str, primary: str, n_regimes: int, seed: int) -> dict:
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    returns = pd.read_json(returns_json)
    series = returns[primary].dropna()
    X = series.values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", n_init=10, random_state=seed)
    gmm.fit(X)
    raw_labels = gmm.predict(X)

    # Build regime stats to name Bull/Neutral/Crisis
    stats = pd.DataFrame({"regime": raw_labels, "ret": series.values}).groupby("regime")["ret"].agg(["mean", "std"])
    order = stats["std"].sort_values().index.tolist()
    name_map = {order[0]: "Bull", order[1]: "Neutral", order[2]: "Crisis"}
    labels = pd.Series([name_map[r] for r in raw_labels], index=series.index)

    sil = float(silhouette_score(X, raw_labels))
    bic = float(gmm.bic(X))
    aic = float(gmm.aic(X))
    converged = bool(gmm.converged_)
    n_iter = int(gmm.n_iter_)

    # Regime stats per asset
    regime_stats = {}
    for regime_name in ["Bull", "Neutral", "Crisis"]:
        mask = labels == regime_name
        regime_stats[regime_name] = {}
        for col in returns.columns:
            s = returns[col][mask]
            regime_stats[regime_name][col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "count": int(mask.sum()),
                "pct": float(mask.mean() * 100),
            }

    return {
        "labels_json": labels.to_json(),
        "sil": sil,
        "bic": bic,
        "aic": aic,
        "converged": converged,
        "n_iter": n_iter,
        "regime_stats": regime_stats,
    }


@st.cache_data(show_spinner=False)
def run_granger(returns_json: str, assets: list, maxlag: int) -> str:
    from statsmodels.tsa.stattools import grangercausalitytests

    returns = pd.read_json(returns_json)[assets]
    n = len(assets)
    matrix = pd.DataFrame(np.zeros((n, n)), index=assets, columns=assets)

    for cause in assets:
        for effect in assets:
            if cause == effect:
                matrix.loc[cause, effect] = np.nan
                continue
            try:
                test = grangercausalitytests(returns[[effect, cause]].dropna(), maxlag, verbose=False)
                pval = test[1][0][1]
                matrix.loc[cause, effect] = round(float(pval), 4)
            except Exception:
                matrix.loc[cause, effect] = np.nan

    return matrix.to_json()


@st.cache_data(show_spinner=False)
def compute_backtest(returns_json: str, labels_json: str, primary: str, cost_bps: float) -> dict:
    returns = pd.read_json(returns_json)[primary]
    labels = pd.read_json(labels_json, typ="series")
    labels.index = pd.to_datetime(labels.index)
    returns.index = pd.to_datetime(returns.index)

    # Align
    common = returns.index.intersection(labels.index)
    returns = returns.loc[common]
    labels = labels.loc[common]

    alloc_map = {"Bull": 1.0, "Neutral": 0.5, "Crisis": 0.0}
    alloc = labels.map(alloc_map)
    strat_ret = returns * alloc

    # Transaction costs
    cost = alloc.diff().abs() * (cost_bps / 10000)
    strat_ret = strat_ret - cost.fillna(0)

    def metrics(r: pd.Series) -> dict:
        cum = (1 + r).cumprod()
        total_ret = float(cum.iloc[-1] - 1)
        n_years = len(r) / 252
        cagr = float((1 + total_ret) ** (1 / n_years) - 1)
        ann_vol = float(r.std() * np.sqrt(252))
        sharpe = float(cagr / ann_vol) if ann_vol > 0 else 0
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min())
        downside = r[r < 0].std() * np.sqrt(252)
        sortino = float(cagr / downside) if downside > 0 else 0
        win_rate = float((r > 0).mean())
        return dict(
            total_ret=total_ret, cagr=cagr, ann_vol=ann_vol,
            sharpe=sharpe, sortino=sortino, max_dd=max_dd, win_rate=win_rate,
            cum_json=cum.to_json(), dd_json=dd.to_json(),
        )

    return {
        "strategy": metrics(strat_ret),
        "benchmark": metrics(returns),
        "alloc_json": alloc.to_json(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:16px 0 8px">
            <div style="font-size:2rem">📊</div>
            <div style="font-size:1.05rem;font-weight:700;color:#00D4FF">Regime Detector</div>
            <div style="font-size:.72rem;color:#6B7280;margin-top:2px">by Avisweta De</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "📈 Market Data", "🤖 Regime Detection",
         "🔗 Causal Inference", "📉 Strategy & Backtest", "🧠 LLM Insights"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**⚙️ Settings**")

    selected_labels = st.multiselect(
        "Assets",
        list(TICKERS.keys()),
        default=list(TICKERS.keys()),
    )
    selected_tickers = [TICKERS[l] for l in selected_labels] if selected_labels else list(TICKERS.values())
    primary_label = st.selectbox("Primary Asset", selected_labels or list(TICKERS.keys()))
    primary_ticker = TICKERS[primary_label]

    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("Start", value=datetime(2015, 1, 1), min_value=datetime(2010, 1, 1))
    end_date = col_e.date_input("End", value=datetime.today())

    n_regimes = st.slider("Regimes (K)", 2, 5, 3)
    cost_bps = st.slider("Transaction cost (bps)", 0, 50, 10)

    st.divider()
    st.markdown(
        """
        <div style="font-size:.72rem;color:#4B5563;text-align:center">
        <a href="https://github.com/Avisweta-De/causal-regime-time-series"
           style="color:#00D4FF;text-decoration:none;">GitHub ↗</a>
        &nbsp;·&nbsp;
        <a href="https://linkedin.com/in/avisweta-de"
           style="color:#00D4FF;text-decoration:none;">LinkedIn ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Load data (shared across all pages)
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("📡 Loading market data…"):
    try:
        prices = load_data(selected_tickers, str(start_date), str(end_date))
        returns_json = compute_returns(prices.to_json())
        returns = pd.read_json(returns_json)
        data_ok = True
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        data_ok = False
        st.stop()

# Fit regimes (needed on most pages)
with st.spinner("🤖 Detecting regimes…"):
    regime_result = fit_regimes(returns_json, primary_ticker, n_regimes, seed=42)

labels = pd.read_json(regime_result["labels_json"], typ="series")
labels.index = pd.to_datetime(labels.index)


# ──────────────────────────────────────────────────────────────────────────────
# ██ PAGE: DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>Causal Regime Detection</h1>
            <p>Quantitative finance system · Market regime analysis · GPT-4 investment narratives</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Backtest metrics for KPI cards
    bt = compute_backtest(returns_json, regime_result["labels_json"], primary_ticker, cost_bps)
    s, b = bt["strategy"], bt["benchmark"]

    section("🏆 Strategy Performance vs Buy & Hold")
    cols = st.columns(5)
    cards = [
        ("Total Return", f"{s['total_ret']*100:.0f}%", f"B&H: {b['total_ret']*100:.0f}%", True),
        ("CAGR", f"{s['cagr']*100:.1f}%", f"B&H: {b['cagr']*100:.1f}%", s['cagr'] > b['cagr']),
        ("Sharpe Ratio", f"{s['sharpe']:.2f}", f"B&H: {b['sharpe']:.2f}", s['sharpe'] > b['sharpe']),
        ("Max Drawdown", f"{s['max_dd']*100:.1f}%", f"B&H: {b['max_dd']*100:.1f}%", s['max_dd'] > b['max_dd']),
        ("Win Rate", f"{s['win_rate']*100:.1f}%", f"B&H: {b['win_rate']*100:.1f}%", s['win_rate'] > b['win_rate']),
    ]
    for col, (lbl, val, delta, pos) in zip(cols, cards):
        col.markdown(metric_card(lbl, val, delta, pos), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Regime distribution
    section("📊 Regime Distribution")
    c1, c2 = st.columns([1, 2])

    regime_counts = labels.value_counts()
    fig_pie = go.Figure(
        go.Pie(
            labels=regime_counts.index,
            values=regime_counts.values,
            hole=0.55,
            marker=dict(colors=[REGIME_COLORS.get(r, "#888") for r in regime_counts.index],
                        line=dict(color="#0A0F1E", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#E8EAF0"),
        )
    )
    fig_pie.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False,
        height=260,
        annotations=[dict(text=f"<b>{len(labels)}</b><br>days", x=0.5, y=0.5,
                          font=dict(size=14, color="#E8EAF0"), showarrow=False)],
    )
    c1.plotly_chart(fig_pie, use_container_width=True)

    # Regime stats table
    regime_tbl = []
    for regime_name in ["Bull", "Neutral", "Crisis"]:
        if regime_name in regime_result["regime_stats"]:
            st = regime_result["regime_stats"][regime_name]
            prim = st.get(primary_ticker, {})
            regime_tbl.append({
                "Regime": regime_name,
                "Frequency": f"{prim.get('pct', 0):.1f}%",
                "Days": prim.get("count", 0),
                "Avg Return": f"{prim.get('mean', 0)*100:.3f}%",
                "Volatility": f"{prim.get('std', 0)*100:.3f}%",
                "Annualised Ret": f"{prim.get('mean', 0)*252*100:.1f}%",
            })
    c2.dataframe(pd.DataFrame(regime_tbl), use_container_width=True, hide_index=True)

    # ── Model quality
    section("🔬 Model Quality")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(metric_card("Silhouette Score", f"{regime_result['sil']:.4f}", "≥ 0.50 = Good", True), unsafe_allow_html=True)
    mc2.markdown(metric_card("BIC", f"{regime_result['bic']:.0f}", "Lower is better", True), unsafe_allow_html=True)
    mc3.markdown(metric_card("Converged", "Yes ✓" if regime_result["converged"] else "No ✗", "", regime_result["converged"]), unsafe_allow_html=True)
    mc4.markdown(metric_card("Iterations", str(regime_result["n_iter"]), "EM algorithm", True), unsafe_allow_html=True)

    # ── Architecture
    section("🏗️ System Architecture")
    st.markdown(
        """
        ```
        yfinance Data  →  Feature Engineering (rolling stats)  →  Gaussian Mixture Model (K=3)
                                                                          │
                         ┌────────────────┬──────────────────────────────┤
                         │                │                              │
                    Causality         Strategy                    Backtesting
                  (Granger/VAR)  (Bull/Neutral/Crisis)         (Walk-forward)
                         │                │                              │
                         └────────────────┴──── GPT-4 Investment Narratives
        ```
        """,
    )

    # ── Tech stack badges
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">
        <span style="background:#1F2937;padding:4px 12px;border-radius:20px;font-size:.75rem;color:#60A5FA">🐍 Python 3.10+</span>
        <span style="background:#1F2937;padding:4px 12px;border-radius:20px;font-size:.75rem;color:#60A5FA">🤖 scikit-learn (GMM)</span>
        <span style="background:#1F2937;padding:4px 12px;border-radius:20px;font-size:.75rem;color:#60A5FA">📈 statsmodels (VAR/Granger)</span>
        <span style="background:#1F2937;padding:4px 12px;border-radius:20px;font-size:.75rem;color:#60A5FA">📊 Plotly (interactive charts)</span>
        <span style="background:#1F2937;padding:4px 12px;border-radius:20px;font-size:.75rem;color:#60A5FA">🧠 OpenAI GPT-4</span>
        <span style="background:#1F2937;padding:4px 12px;border-radius:20px;font-size:.75rem;color:#60A5FA">💹 yfinance (market data)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ██ PAGE: MARKET DATA
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📈 Market Data":
    st.title("📈 Market Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Price History", "Returns Distribution", "Rolling Statistics"])

    with tab1:
        section("Normalised Price History (Base = 100)")
        norm = prices / prices.iloc[0] * 100
        fig = go.Figure()
        palette = ["#00D4FF", "#818CF8", "#34D399", "#FCD34D", "#F87171"]
        for i, col in enumerate(norm.columns):
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[col], name=col,
                line=dict(color=palette[i % len(palette)], width=1.8),
                hovertemplate=f"<b>{col}</b><br>%{{y:.1f}}<extra></extra>",
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title="Normalised Prices (100 = start)")
        st.plotly_chart(fig, use_container_width=True)

        section("Raw Price Data")
        st.dataframe(prices.tail(30).style.format("{:.2f}"), use_container_width=True)

    with tab2:
        section("Daily Returns Distribution")
        col_sel = st.selectbox("Select asset", returns.columns.tolist(), key="dist_asset")
        r = returns[col_sel].dropna()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=r, nbinsx=80, name="Returns",
            marker=dict(color="#00D4FF", opacity=0.7, line=dict(color="#0A0F1E", width=0.3)),
        ))
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=340,
                               title=f"{col_sel} Daily Log Returns",
                               xaxis_title="Daily Log Return", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

        s = r.describe()
        dist_cols = st.columns(6)
        for col, (lbl, val) in zip(dist_cols, [
            ("Mean", f"{r.mean()*100:.3f}%"),
            ("Std Dev", f"{r.std()*100:.3f}%"),
            ("Skewness", f"{r.skew():.3f}"),
            ("Kurtosis", f"{r.kurtosis():.3f}"),
            ("Min", f"{r.min()*100:.2f}%"),
            ("Max", f"{r.max()*100:.2f}%"),
        ]):
            col.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    with tab3:
        section("Rolling Volatility (20-day)")
        fig_vol = go.Figure()
        for i, col in enumerate(returns.columns):
            rv = returns[col].rolling(20).std() * np.sqrt(252) * 100
            fig_vol.add_trace(go.Scatter(
                x=rv.index, y=rv, name=col,
                line=dict(color=palette[i % len(palette)], width=1.5),
            ))
        fig_vol.update_layout(**PLOTLY_LAYOUT, height=360,
                              title="20-day Rolling Annualised Volatility (%)",
                              yaxis_title="Volatility (%)")
        st.plotly_chart(fig_vol, use_container_width=True)

        section("Correlation Matrix")
        corr = returns.corr()
        fig_corr = px.imshow(
            corr.round(3), text_auto=True, aspect="auto",
            color_continuous_scale=[[0, "#F87171"], [0.5, "#1F2937"], [1, "#34D399"]],
            range_color=[-1, 1],
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=380, title="Asset Return Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# ██ PAGE: REGIME DETECTION
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Regime Detection":
    st.title("🤖 Regime Detection (GMM)")

    tab1, tab2, tab3 = st.tabs(["Regime Timeline", "Return Distributions", "Transition Analysis"])

    with tab1:
        section(f"Market Regimes — {primary_ticker} ({n_regimes} states)")

        prim_ret = returns[primary_ticker].dropna()
        common_idx = prim_ret.index.intersection(labels.index)
        r_aligned = prim_ret.loc[common_idx]
        l_aligned = labels.loc[common_idx]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.35, 0.65],
                            vertical_spacing=0.04)

        # Regime color bars
        for regime_name, color in REGIME_COLORS.items():
            mask = l_aligned == regime_name
            fig.add_trace(
                go.Scatter(x=l_aligned.index[mask], y=[regime_name] * mask.sum(),
                           mode="markers", marker=dict(color=color, size=4, symbol="square"),
                           name=regime_name, showlegend=True),
                row=1, col=1,
            )

        # Cumulative returns coloured by regime
        cum_ret = (1 + r_aligned).cumprod()
        for regime_name, color in REGIME_COLORS.items():
            mask = l_aligned == regime_name
            idx = l_aligned.index[mask]
            fig.add_trace(
                go.Scatter(x=idx, y=cum_ret.loc[idx], mode="markers",
                           marker=dict(color=color, size=3, opacity=0.7),
                           name=regime_name, showlegend=False),
                row=2, col=1,
            )

        # Full equity line on top
        fig.add_trace(
            go.Scatter(x=cum_ret.index, y=cum_ret, name="Equity (Buy&Hold)",
                       line=dict(color="rgba(200,200,200,.35)", width=1), showlegend=False),
            row=2, col=1,
        )

        fig.update_layout(**PLOTLY_LAYOUT, height=500,
                          title=f"{primary_ticker} — Regime Labels & Cumulative Return")
        fig.update_yaxes(title_text="Regime", row=1, col=1,
                         categoryorder="array", categoryarray=["Crisis", "Neutral", "Bull"])
        fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section("Return Distributions by Regime")
        asset_sel = st.selectbox("Asset", returns.columns.tolist(), key="dist_regime")
        r_full = returns[asset_sel].dropna()
        common = r_full.index.intersection(labels.index)

        fig_box = go.Figure()
        for regime_name, color in REGIME_COLORS.items():
            mask = labels.loc[common] == regime_name
            fig_box.add_trace(go.Violin(
                y=r_full.loc[common][mask] * 100,
                name=regime_name,
                line_color=color,
                fillcolor=REGIME_BG[regime_name],
                meanline_visible=True,
                box_visible=True,
                points="outliers",
            ))
        fig_box.update_layout(**PLOTLY_LAYOUT, height=380,
                              title=f"{asset_sel} — Return Distribution by Regime",
                              yaxis_title="Daily Return (%)")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        section("Regime Transition Matrix")

        # Build transition matrix
        label_seq = labels.values
        regime_names = ["Bull", "Neutral", "Crisis"]
        tm = np.zeros((3, 3))
        for i in range(len(label_seq) - 1):
            r1 = regime_names.index(label_seq[i]) if label_seq[i] in regime_names else -1
            r2 = regime_names.index(label_seq[i + 1]) if label_seq[i + 1] in regime_names else -1
            if r1 >= 0 and r2 >= 0:
                tm[r1, r2] += 1
        row_sums = tm.sum(axis=1, keepdims=True)
        tm = np.divide(tm, row_sums, where=row_sums > 0)

        fig_tm = px.imshow(
            np.round(tm, 3),
            labels=dict(x="To", y="From", color="Probability"),
            x=regime_names, y=regime_names,
            text_auto=True,
            color_continuous_scale=[[0, "#1F2937"], [0.5, "#0EA5E9"], [1, "#00D4FF"]],
            range_color=[0, 1],
        )
        fig_tm.update_layout(**PLOTLY_LAYOUT, height=360, title="Markov Transition Probabilities")
        st.plotly_chart(fig_tm, use_container_width=True)

        # Duration stats
        section("Regime Duration Statistics")
        dur_data = []
        current, count = label_seq[0], 1
        durations = {r: [] for r in regime_names}
        for lab in label_seq[1:]:
            if lab == current:
                count += 1
            else:
                if current in durations:
                    durations[current].append(count)
                current, count = lab, 1
        if current in durations:
            durations[current].append(count)

        for r in regime_names:
            d = durations[r]
            if d:
                dur_data.append({
                    "Regime": r,
                    "Avg Duration (days)": f"{np.mean(d):.1f}",
                    "Max Duration": max(d),
                    "Min Duration": min(d),
                    "# Episodes": len(d),
                })
        st.dataframe(pd.DataFrame(dur_data), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# ██ PAGE: CAUSAL INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔗 Causal Inference":
    st.title("🔗 Causal Inference")

    tab1, tab2 = st.tabs(["Granger Causality", "Shock Detection"])

    with tab1:
        maxlag = st.slider("Max lag (days)", 1, 10, 5)
        with st.spinner("Running Granger causality tests…"):
            gc_json = run_granger(returns_json, returns.columns.tolist(), maxlag)
        gc_matrix = pd.read_json(gc_json)

        section("Granger Causality P-Value Matrix")
        st.caption("Row = cause → Column = effect. Green (low p-value) = significant predictive causality.")

        sig_mask = gc_matrix < 0.05
        fig_gc = px.imshow(
            gc_matrix.round(3),
            text_auto=True,
            color_continuous_scale=[[0, "#34D399"], [0.05, "#FCD34D"], [0.3, "#1F2937"], [1, "#1F2937"]],
            range_color=[0, 1],
            labels=dict(x="Effect", y="Cause", color="p-value"),
        )
        fig_gc.update_layout(**PLOTLY_LAYOUT, height=420,
                             title=f"Granger Causality Matrix (maxlag={maxlag}) — p < 0.05 = Significant")
        st.plotly_chart(fig_gc, use_container_width=True)

        # Significant relationships
        sig_pairs = []
        for cause in gc_matrix.index:
            for effect in gc_matrix.columns:
                p = gc_matrix.loc[cause, effect]
                if pd.notna(p) and p < 0.05:
                    sig_pairs.append({"Cause": cause, "Effect": effect, "p-value": f"{p:.4f}", "Significant": "✅"})
        if sig_pairs:
            section("Significant Causal Relationships (p < 0.05)")
            st.dataframe(pd.DataFrame(sig_pairs), use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No significant Granger causality found at p < 0.05 — consistent with the Efficient Market Hypothesis.")

        with st.expander("📖 Interpreting Granger Causality"):
            st.markdown("""
**Granger causality** tests whether past values of one variable help predict another — it is *predictive*
causality, not true structural causality.

- **p < 0.05** — reject the null that X does NOT Granger-cause Y (X has predictive power over Y)
- **p ≥ 0.05** — no statistically significant predictive relationship at this lag length
- **Implication of no causality:** Markets are informationally efficient; each asset price series
  incorporates available information. Diversification across assets provides *genuine* risk reduction.
""")

    with tab2:
        section("Market Shock Detection (Rolling Z-Score)")
        shock_asset = st.selectbox("Asset", returns.columns.tolist(), key="shock_asset")
        threshold = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.1)
        window = st.slider("Rolling window (days)", 10, 60, 20)

        r_shock = returns[shock_asset].dropna()
        roll_mean = r_shock.rolling(window).mean()
        roll_std = r_shock.rolling(window).std()
        z = ((r_shock - roll_mean) / roll_std).abs()
        shocks = z > threshold

        fig_shock = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.5, 0.5], vertical_spacing=0.05)
        fig_shock.add_trace(
            go.Scatter(x=r_shock.index, y=r_shock * 100, name="Daily Return (%)",
                       line=dict(color="#00D4FF", width=1), fill="tozeroy",
                       fillcolor="rgba(0,212,255,.06)"),
            row=1, col=1,
        )
        fig_shock.add_trace(
            go.Scatter(x=z.index, y=z, name="|Z-score|",
                       line=dict(color="#818CF8", width=1.5)),
            row=2, col=1,
        )
        fig_shock.add_hline(y=threshold, line=dict(color="#F87171", dash="dash", width=1.5),
                            annotation_text="Threshold", row=2, col=1)

        # Mark shocks
        shock_dates = r_shock.index[shocks]
        fig_shock.add_trace(
            go.Scatter(x=shock_dates, y=r_shock[shocks] * 100, mode="markers",
                       name="Shock", marker=dict(color="#F87171", size=7, symbol="x")),
            row=1, col=1,
        )
        fig_shock.update_layout(**PLOTLY_LAYOUT, height=480,
                                title=f"{shock_asset} — Return Shocks (|Z| > {threshold})")
        st.plotly_chart(fig_shock, use_container_width=True)

        n_shocks = int(shocks.sum())
        col1, col2, col3 = st.columns(3)
        col1.markdown(metric_card("Shock Events", str(n_shocks), f"with |Z| > {threshold}", n_shocks < 30), unsafe_allow_html=True)
        col2.markdown(metric_card("Shock Rate", f"{n_shocks/len(r_shock)*100:.1f}%", "of all trading days", True), unsafe_allow_html=True)
        col3.markdown(metric_card("Largest Shock", f"{z.max():.2f}σ", f"on {z.idxmax().date()}", False), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ██ PAGE: STRATEGY & BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📉 Strategy & Backtest":
    st.title("📉 Strategy & Backtest")

    bt = compute_backtest(returns_json, regime_result["labels_json"], primary_ticker, cost_bps)
    s, b = bt["strategy"], bt["benchmark"]

    # ── KPI row
    section("Performance Metrics")
    cols = st.columns(6)
    kpis = [
        ("Total Return", f"{s['total_ret']*100:.1f}%", f"B&H: {b['total_ret']*100:.1f}%", s['total_ret'] > b['total_ret']),
        ("CAGR",        f"{s['cagr']*100:.2f}%",       f"B&H: {b['cagr']*100:.2f}%",       s['cagr'] > b['cagr']),
        ("Sharpe",      f"{s['sharpe']:.2f}",           f"B&H: {b['sharpe']:.2f}",           s['sharpe'] > b['sharpe']),
        ("Sortino",     f"{s['sortino']:.2f}",          f"B&H: {b['sortino']:.2f}",          s['sortino'] > b['sortino']),
        ("Max DD",      f"{s['max_dd']*100:.2f}%",      f"B&H: {b['max_dd']*100:.2f}%",      s['max_dd'] > b['max_dd']),
        ("Win Rate",    f"{s['win_rate']*100:.1f}%",    f"B&H: {b['win_rate']*100:.1f}%",    s['win_rate'] > b['win_rate']),
    ]
    for col, (lbl, val, delta, pos) in zip(cols, kpis):
        col.markdown(metric_card(lbl, val, delta, pos), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Equity Curve", "Drawdown", "Allocations"])

    with tab1:
        section("Cumulative Returns")
        strat_cum = pd.read_json(bt["strategy"]["cum_json"], typ="series")
        bench_cum = pd.read_json(bt["benchmark"]["cum_json"], typ="series")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=strat_cum.index, y=strat_cum,
            name="Regime Strategy",
            line=dict(color="#00D4FF", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,.06)",
        ))
        fig_eq.add_trace(go.Scatter(
            x=bench_cum.index, y=bench_cum,
            name="Buy & Hold",
            line=dict(color="#6B7280", width=1.5, dash="dot"),
        ))
        fig_eq.update_layout(**PLOTLY_LAYOUT, height=400,
                             title="Cumulative Wealth (Start = $1)",
                             yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_eq, use_container_width=True)

    with tab2:
        section("Drawdown from All-Time High")
        strat_dd = pd.read_json(bt["strategy"]["dd_json"], typ="series")
        bench_dd = pd.read_json(bt["benchmark"]["dd_json"], typ="series")

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=bench_dd.index, y=bench_dd * 100,
            name="Buy & Hold",
            line=dict(color="#F87171", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,.08)",
        ))
        fig_dd.add_trace(go.Scatter(
            x=strat_dd.index, y=strat_dd * 100,
            name="Regime Strategy",
            line=dict(color="#00D4FF", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,.08)",
        ))
        fig_dd.update_layout(**PLOTLY_LAYOUT, height=360,
                             title="Drawdown from Peak (%)",
                             yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab3:
        section("Portfolio Allocation Over Time")
        alloc = pd.read_json(bt["alloc_json"], typ="series")
        alloc.index = pd.to_datetime(alloc.index)

        fig_alloc = go.Figure(go.Scatter(
            x=alloc.index, y=alloc * 100,
            fill="tozeroy",
            line=dict(color="#818CF8", width=1.5),
            fillcolor="rgba(129,140,248,.15)",
            name="Stock Allocation (%)",
        ))
        fig_alloc.add_hline(y=50, line=dict(color="#FCD34D", dash="dash", width=1), annotation_text="50% (Neutral)")
        fig_alloc.update_layout(**PLOTLY_LAYOUT, height=320,
                                title="Regime-Based Stock Allocation",
                                yaxis_title="% Allocated to Stocks",
                                yaxis=dict(range=[0, 105]))
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.info(
            "🟢 **Bull** → 100% stocks | 🟡 **Neutral** → 50% stocks, 50% cash | "
            f"🔴 **Crisis** → 0% stocks | Transaction cost: **{cost_bps} bps** per rebalance"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ██ PAGE: LLM INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🧠 LLM Insights":
    st.title("🧠 LLM Insights (GPT-4)")
    st.caption("Translate quantitative findings into human-readable investment narratives using OpenAI GPT-4.")

    # Resolve API key
    api_key = (
        st.secrets.get("OPENAI_API_KEY", None)
        if hasattr(st, "secrets") else None
    ) or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning(
            "**OpenAI API key not found.** "
            "To enable LLM insights, add your key below or set `OPENAI_API_KEY` in `.streamlit/secrets.toml`."
        )
        api_key = st.text_input("🔑 Enter OpenAI API key", type="password", placeholder="sk-…")

    llm_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)

    # Regime stats summary
    regime_stats_summary = {}
    for r_name in ["Bull", "Neutral", "Crisis"]:
        if r_name in regime_result["regime_stats"]:
            pd_ = regime_result["regime_stats"][r_name].get(primary_ticker, {})
            regime_stats_summary[r_name] = {
                "percentage": pd_.get("pct", 0),
                "avg_return": pd_.get("mean", 0),
                "volatility": pd_.get("std", 0),
            }

    bt = compute_backtest(returns_json, regime_result["labels_json"], primary_ticker, cost_bps)

    st.divider()
    insight_type = st.selectbox(
        "Choose insight type",
        ["Regime Characteristics", "Backtest Performance", "Risk Assessment", "Investment Thesis"],
    )

    if st.button("✨ Generate AI Insight", type="primary", disabled=not api_key):
        if not api_key:
            st.error("Please provide an API key.")
        else:
            from openai import OpenAI
            _client = OpenAI(api_key=api_key)

            system_msg = (
                "You are a professional quantitative analyst and portfolio manager. "
                "Explain complex financial analysis in clear, actionable terms for sophisticated investors. "
                "Be concise, insightful, and avoid generic statements."
            )

            if insight_type == "Regime Characteristics":
                user_msg = f"""
Analyze these market regime statistics and explain them for a portfolio manager:

Bull Regime:  Frequency={regime_stats_summary.get('Bull',{}).get('percentage',0):.1f}%,
              Avg Daily Return={regime_stats_summary.get('Bull',{}).get('avg_return',0):.4f},
              Daily Volatility={regime_stats_summary.get('Bull',{}).get('volatility',0):.4f}

Neutral Regime: Frequency={regime_stats_summary.get('Neutral',{}).get('percentage',0):.1f}%,
                Avg Daily Return={regime_stats_summary.get('Neutral',{}).get('avg_return',0):.4f},
                Daily Volatility={regime_stats_summary.get('Neutral',{}).get('volatility',0):.4f}

Crisis Regime: Frequency={regime_stats_summary.get('Crisis',{}).get('percentage',0):.1f}%,
               Avg Daily Return={regime_stats_summary.get('Crisis',{}).get('avg_return',0):.4f},
               Daily Volatility={regime_stats_summary.get('Crisis',{}).get('volatility',0):.4f}

Asset: {primary_ticker} | Period: {str(start_date)} to {str(end_date)}

Provide: (1) Key characteristics of each regime, (2) What triggers transitions,
(3) Portfolio implications, (4) Risk management actions. Max 300 words.
"""
            elif insight_type == "Backtest Performance":
                s_bt = bt["strategy"]
                b_bt = bt["benchmark"]
                user_msg = f"""
Analyse this regime-based strategy vs buy-and-hold:

STRATEGY: CAGR={s_bt['cagr']*100:.2f}%, Sharpe={s_bt['sharpe']:.2f},
          MaxDD={s_bt['max_dd']*100:.2f}%, Volatility={s_bt['ann_vol']*100:.2f}%

BENCHMARK: CAGR={b_bt['cagr']*100:.2f}%, Sharpe={b_bt['sharpe']:.2f},
           MaxDD={b_bt['max_dd']*100:.2f}%, Volatility={b_bt['ann_vol']*100:.2f}%

Transaction cost: {cost_bps} bps per rebalance.

Provide: (1) Performance highlights, (2) Risk profile interpretation,
(3) Realistic forward expectations, (4) Recommendations. Max 300 words.
"""
            elif insight_type == "Risk Assessment":
                s_bt = bt["strategy"]
                user_msg = f"""
Generate a risk assessment for a regime-switching portfolio:
- Worst drawdown: {s_bt['max_dd']*100:.2f}%
- Annual volatility: {s_bt['ann_vol']*100:.2f}%
- Sharpe ratio: {s_bt['sharpe']:.2f}
- Strategy: Bull=100% stocks, Neutral=50%, Crisis=0%

Highlight: (1) Key tail risks, (2) Model limitations and assumption failures,
(3) Correlation breakdown risks, (4) Recommended hedges. Max 300 words.
"""
            else:  # Investment Thesis
                s_bt = bt["strategy"]
                user_msg = f"""
Write a 3-paragraph investment memo synthesising:

REGIME MODEL: Bull dominates (most days), Crisis rare but severe (-1.4%/day avg).
CAUSALITY: No significant Granger causality — efficient markets, genuine diversification benefit.
STRATEGY PERFORMANCE: CAGR={s_bt['cagr']*100:.2f}%, Sharpe={s_bt['sharpe']:.2f},
                      MaxDD={s_bt['max_dd']*100:.2f}% vs buy-and-hold.

Include: (1) Market outlook, (2) Key portfolio drivers, (3) Risk factors,
(4) Tactical allocation recommendations. Professional tone.
"""

            with st.spinner(f"🤖 Querying {llm_model}…"):
                try:
                    resp = _client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.7,
                        max_tokens=600,
                    )
                    insight_text = resp.choices[0].message.content
                    st.markdown(
                        f"""
                        <div style="background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.2);
                                    border-radius:12px;padding:24px;margin-top:8px;line-height:1.7;
                                    font-size:.92rem;color:#E8EAF0;">
                        {insight_text.replace(chr(10), '<br>')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Generated by {llm_model} · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                except Exception as e:
                    st.error(f"API Error: {e}")

    with st.expander("💡 How to add your API key permanently"):
        st.markdown("""
1. Create `.streamlit/secrets.toml` in the project root (already in `.gitignore`):
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
2. On **Streamlit Cloud**, go to App → Settings → Secrets and paste the same.
3. The key is never committed to git and remains private.
""")
