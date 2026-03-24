"""Macro Tactical Cockpit V1 -- Streamlit entry point."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ASSET_DISPLAY_NAMES, ASSET_UNIVERSE
from src.data import DataManager
from src.models import (
    fit_bsts,
    fit_hmm,
    forecast_bsts,
    map_regime_labels,
    predict_regimes,
    prepare_hmm_features,
)
from src.quant import ewma_correlation, lead_lag_rankings
from src.visuals import build_lead_lag_table, build_main_chart

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Macro Tactical Cockpit",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for polish ---
st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 2.5rem; }

    /* Style metric cards */
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    [data-testid="stMetric"] label { font-size: 0.78rem; opacity: 0.8; }

    /* Sidebar styling */
    [data-testid="stSidebar"] [data-testid="stDataFrame"] { font-size: 0.8rem; }

    /* Section headers */
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(160, 160, 160, 0.8);
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State: DataManager ---
@st.cache_resource
def get_data_manager() -> DataManager:
    dm = DataManager()
    dm.refresh()
    return dm


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Controls")

    if st.button("Refresh Data", use_container_width=True, type="primary"):
        st.cache_resource.clear()
        st.rerun()

    dm = get_data_manager()
    available = dm.available_assets

    if not available:
        st.error("No data loaded. Check your internet connection.")
        st.stop()

    display_options = [ASSET_DISPLAY_NAMES.get(a, a) for a in available]
    display_to_key = {ASSET_DISPLAY_NAMES.get(a, a): a for a in available}
    selected_display = st.selectbox(
        "Anchor Asset",
        options=display_options,
        index=0,
    )
    anchor = display_to_key[selected_display]

    st.divider()

    close_matrix = dm.get_close_matrix()

    # Lead-Lag Rankings
    st.markdown('<p class="section-header">Lead-Lag Rankings</p>', unsafe_allow_html=True)
    if len(close_matrix.columns) > 1:
        rankings = lead_lag_rankings(close_matrix, anchor=anchor)
        display_table = build_lead_lag_table(rankings)

        sidebar_table = display_table[["asset", "lag", "correlation", "direction"]].copy()
        sidebar_table["direction"] = sidebar_table["direction"].str.replace(
            "Contemporaneous", "Contemp."
        )
        st.dataframe(
            sidebar_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "asset": st.column_config.TextColumn("Asset", width="small"),
                "lag": st.column_config.NumberColumn("Lag", width="small"),
                "correlation": st.column_config.NumberColumn(
                    "Corr",
                    format="%.4f",
                    width="small",
                ),
                "direction": st.column_config.TextColumn("Dir.", width="small"),
            },
        )
    else:
        st.info("Need at least 2 assets for lead-lag analysis.")


# --- Main Area ---
ohlcv = dm.get_ohlcv(anchor)

# HMM Regime Detection
hmm_features, feat_means, feat_stds = prepare_hmm_features(ohlcv)
hmm_ok = False

try:
    hmm_model = fit_hmm(hmm_features)
    states, probs = predict_regimes(hmm_model, hmm_features)
    label_map = map_regime_labels(hmm_model, feat_means, feat_stds)

    feature_index = ohlcv.index[-len(states):]
    regime_series = None
    if len(feature_index) == len(states):
        regime_series = pd.Series(states, index=feature_index)

    hmm_ok = True
except Exception as e:
    st.warning(f"Regime model failed: {e}")
    label_map = {}
    regime_series = None

# --- Header row: title + price + regime badge (single HTML block for alignment) ---
latest_price = ohlcv["Close"].iloc[-1]
prev_price = ohlcv["Close"].iloc[-2] if len(ohlcv) > 1 else latest_price
price_change = latest_price - prev_price
delta_color = "#4ade80" if price_change >= 0 else "#f87171"
delta_arrow = "&#9650;" if price_change >= 0 else "&#9660;"

badge_html = ""
if hmm_ok:
    latest_probs = probs[-1]
    dominant_idx = int(np.argmax(latest_probs))
    dominant_label = label_map.get(dominant_idx, f"State {dominant_idx}")
    dominant_prob = latest_probs[dominant_idx]

    badge_colors = {
        "BULL": ("#065f46", "#d1fae5"),
        "BEAR": ("#7f1d1d", "#fecaca"),
        "HIGH_VOL": ("#78350f", "#fed7aa"),
        "NEUTRAL": ("#374151", "#e5e7eb"),
    }
    bg, fg = badge_colors.get(dominant_label, ("#374151", "#e5e7eb"))
    badge_html = (
        f'<span style="background:{fg}; color:{bg}; padding:5px 14px; '
        f'border-radius:20px; font-weight:700; font-size:0.85rem; '
        f'white-space:nowrap;">'
        f'{dominant_label} &middot; {dominant_prob:.0%}</span>'
    )

st.markdown(
    f'''<div style="display:flex; align-items:center; justify-content:space-between;
         flex-wrap:wrap; gap:0.5rem; margin-bottom:0.5rem;">
        <h2 style="margin:0; flex:1 1 auto;">Macro Tactical Cockpit — {anchor}</h2>
        <div style="display:flex; align-items:center; gap:1rem; flex-shrink:0;">
            <div style="text-align:right; line-height:1.3;">
                <span style="font-size:0.75rem; opacity:0.7;">Last Price</span><br/>
                <span style="font-size:1.4rem; font-weight:700;">{latest_price:,.2f}</span>
                <span style="font-size:0.8rem; color:{delta_color};"> {delta_arrow} {price_change:+,.2f}</span>
            </div>
            {badge_html}
        </div>
    </div>''',
    unsafe_allow_html=True,
)

# BSTS Forecast
close_prices = ohlcv["Close"]
try:
    bsts_result = fit_bsts(close_prices)
    fan_data = forecast_bsts(bsts_result)
except Exception:
    fan_data = None

# Main candlestick chart — full width
fig = build_main_chart(
    ohlcv,
    fan_data=fan_data,
    regime_states=regime_series,
    regime_label_map=label_map,
)
st.plotly_chart(fig, use_container_width=True)

# --- Bottom panels ---
st.divider()
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown('<p class="section-header">Regime Probabilities</p>', unsafe_allow_html=True)
    if hmm_ok:
        # Use 2x2 grid for regime probs to avoid truncation
        row1 = st.columns(2)
        row2 = st.columns(2)
        prob_cols = [*row1, *row2]
        for i, prob in enumerate(latest_probs):
            if i < len(prob_cols):
                label = label_map.get(i, f"State {i}")
                prob_cols[i].metric(label, f"{prob:.1%}")
    else:
        st.info("Regime model unavailable.")

with col2:
    st.markdown('<p class="section-header">EWMA Correlations</p>', unsafe_allow_html=True)
    if len(close_matrix.columns) > 1:
        try:
            ewma_corr = ewma_correlation(close_matrix, anchor=anchor)
            latest_corr = ewma_corr.dropna().iloc[-1].sort_values(ascending=False)

            # Build a horizontal Plotly bar chart for better control
            colors = ["#4ade80" if v > 0 else "#f87171" for v in latest_corr.values]
            fig_corr = go.Figure(go.Bar(
                x=latest_corr.values,
                y=latest_corr.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in latest_corr.values],
                textposition="auto",
                textfont=dict(size=12),
            ))
            fig_corr.update_layout(
                height=max(300, len(latest_corr) * 42),
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis=dict(range=[-1.15, 1.15], title="Correlation", zeroline=True,
                           zerolinecolor="rgba(128,128,128,0.3)", zerolinewidth=1),
                yaxis=dict(autorange="reversed"),
                template="plotly_dark",
                font=dict(size=13),
                bargap=0.3,
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning(f"EWMA correlation failed: {e}")
    else:
        st.info("Need at least 2 assets.")

# --- Footer: last update timestamp ---
last_ts = ohlcv.index[-1]
st.caption(f"Last data point: {last_ts.strftime('%Y-%m-%d %H:%M UTC')}")
