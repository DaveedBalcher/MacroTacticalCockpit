"""Macro Tactical Cockpit V1 -- Streamlit entry point."""

import logging

import numpy as np
import pandas as pd
import streamlit as st

from src.config import ASSET_UNIVERSE
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


# --- Session State: DataManager ---
@st.cache_resource
def get_data_manager() -> DataManager:
    dm = DataManager()
    dm.refresh()
    return dm


# --- Sidebar ---
with st.sidebar:
    st.header("Controls")

    if st.button("Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

    dm = get_data_manager()
    available = dm.available_assets

    if not available:
        st.error("No data loaded. Check your internet connection.")
        st.stop()

    anchor = st.selectbox(
        "Anchor Asset",
        options=available,
        index=0,
    )

    st.divider()
    st.subheader("Lead-Lag Rankings")

    close_matrix = dm.get_close_matrix()
    if len(close_matrix.columns) > 1:
        rankings = lead_lag_rankings(close_matrix, anchor=anchor)
        display_table = build_lead_lag_table(rankings)
        st.dataframe(display_table, hide_index=True, use_container_width=True)
    else:
        st.info("Need at least 2 assets for lead-lag analysis.")


# --- Main Area ---
st.title(f"Macro Tactical Cockpit — {anchor}")

ohlcv = dm.get_ohlcv(anchor)

# HMM Regime Detection
hmm_features = prepare_hmm_features(ohlcv)
hmm_ok = False

try:
    hmm_model = fit_hmm(hmm_features)
    states, probs = predict_regimes(hmm_model, hmm_features)
    label_map = map_regime_labels(hmm_model)

    # Align states to OHLCV index (features are shorter due to dropna)
    feature_index = ohlcv.index[-len(states):]
    regime_series = None
    if len(feature_index) == len(states):
        regime_series = pd.Series(states, index=feature_index)

    hmm_ok = True
except Exception as e:
    st.warning(f"Regime model failed: {e}")
    label_map = {}
    regime_series = None

# Regime summary — inline above the chart
if hmm_ok:
    latest_probs = probs[-1]
    dominant_idx = int(np.argmax(latest_probs))
    dominant_label = label_map.get(dominant_idx, f"State {dominant_idx}")
    dominant_prob = latest_probs[dominant_idx]

    regime_colors = {"BULL": "green", "BEAR": "red", "HIGH_VOL": "orange", "NEUTRAL": "gray"}
    color = regime_colors.get(dominant_label, "gray")
    st.markdown(
        f"**Regime:** :{color}[{dominant_label}] ({dominant_prob:.0%})",
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

# Bottom row: regime metrics + EWMA correlations side by side
col1, col2 = st.columns(2)

with col1:
    if hmm_ok:
        st.subheader("Regime Probabilities")
        cols = st.columns(len(latest_probs))
        for i, (col, prob) in enumerate(zip(cols, latest_probs)):
            label = label_map.get(i, f"State {i}")
            col.metric(label, f"{prob:.1%}")

with col2:
    if len(close_matrix.columns) > 1:
        st.subheader("EWMA Correlations")
        try:
            ewma_corr = ewma_correlation(close_matrix, anchor=anchor)
            latest_corr = ewma_corr.dropna().iloc[-1].sort_values(ascending=False)
            st.bar_chart(latest_corr)
        except Exception as e:
            st.warning(f"EWMA correlation failed: {e}")
