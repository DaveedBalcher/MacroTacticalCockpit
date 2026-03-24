"""Plotly figure builders for the Tactical Cockpit."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import REGIME_COLORS


def build_main_chart(
    ohlcv: pd.DataFrame,
    fan_data: pd.DataFrame | None = None,
    regime_states: pd.Series | None = None,
    regime_label_map: dict[int, str] | None = None,
) -> go.Figure:
    """Construct the main Plotly candlestick chart with BSTS overlay and regime background.

    Traces added:
      1. Candlestick (OHLCV)
      2. BSTS mean forecast line (if fan_data provided)
      3. Shaded 90% CI band (if fan_data provided)
      4. Shaded 50% CI band (if fan_data provided)

    Args:
        ohlcv: OHLCV DataFrame for the anchor asset.
        fan_data: Output of forecast_bsts() with CI columns.
        regime_states: Series of int states aligned to ohlcv index.
        regime_label_map: Dict from map_regime_labels().

    Returns:
        go.Figure ready for st.plotly_chart().
    """
    fig = go.Figure()

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["Open"],
        high=ohlcv["High"],
        low=ohlcv["Low"],
        close=ohlcv["Close"],
        name="Price",
    ))

    # 2-4. BSTS fan chart overlay
    if fan_data is not None:
        # 90% CI band (wider, added first so it's behind)
        if "ci_90_lower" in fan_data.columns:
            fig.add_trace(go.Scatter(
                x=fan_data.index,
                y=fan_data["ci_90_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=fan_data.index,
                y=fan_data["ci_90_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(100, 149, 237, 0.15)",
                name="90% CI",
            ))

        # 50% CI band (narrower, on top)
        if "ci_50_lower" in fan_data.columns:
            fig.add_trace(go.Scatter(
                x=fan_data.index,
                y=fan_data["ci_50_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=fan_data.index,
                y=fan_data["ci_50_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(100, 149, 237, 0.35)",
                name="50% CI",
            ))

        # Mean forecast line
        fig.add_trace(go.Scatter(
            x=fan_data.index,
            y=fan_data["mean"],
            mode="lines",
            line=dict(color="cornflowerblue", width=2),
            name="Forecast",
        ))

    # 5. Regime background colors via vrect shapes
    if regime_states is not None and regime_label_map is not None:
        _add_regime_backgrounds(fig, regime_states, regime_label_map)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis=dict(type="date"),
        template="plotly_dark",
        height=600,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig


def _add_regime_backgrounds(
    fig: go.Figure,
    regime_states: pd.Series,
    label_map: dict[int, str],
) -> None:
    """Add vertical rectangles for contiguous regime blocks."""
    if regime_states.empty:
        return

    # Group consecutive same-state bars into blocks
    states = regime_states.values
    timestamps = regime_states.index

    i = 0
    while i < len(states):
        state = states[i]
        j = i
        while j < len(states) and states[j] == state:
            j += 1

        label = label_map.get(int(state), "UNKNOWN")
        color = REGIME_COLORS.get(label, "rgba(128, 128, 128, 0.05)")

        fig.add_vrect(
            x0=timestamps[i],
            x1=timestamps[min(j, len(timestamps) - 1)],
            fillcolor=color,
            layer="below",
            line_width=0,
        )
        i = j


def build_regime_indicator(
    prob_vector: np.ndarray,
    label_map: dict[int, str],
) -> go.Figure:
    """Build a horizontal stacked bar showing current regime probability distribution.

    Args:
        prob_vector: 1D array of probabilities, one per state.
        label_map: Maps state index to label.

    Returns:
        go.Figure with stacked horizontal bar.
    """
    fig = go.Figure()

    state_colors = {
        "BULL": "rgba(0, 200, 0, 0.7)",
        "NEUTRAL": "rgba(128, 128, 128, 0.7)",
        "BEAR": "rgba(200, 0, 0, 0.7)",
        "HIGH_VOL": "rgba(255, 165, 0, 0.7)",
    }

    for state_idx in range(len(prob_vector)):
        label = label_map.get(state_idx, f"State {state_idx}")
        color = state_colors.get(label, "rgba(128, 128, 128, 0.7)")
        prob = prob_vector[state_idx]

        fig.add_trace(go.Bar(
            x=[prob],
            y=["Regime"],
            orientation="h",
            name=f"{label} ({prob:.0%})",
            marker_color=color,
            text=f"{label}" if prob > 0.1 else "",
            textposition="inside",
        ))

    fig.update_layout(
        barmode="stack",
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        xaxis=dict(range=[0, 1], showticklabels=False),
        yaxis=dict(showticklabels=False),
        template="plotly_dark",
    )

    return fig


def build_lead_lag_table(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Format lead-lag rankings for display in Streamlit sidebar.

    Adds directional description.

    Returns:
        Formatted DataFrame suitable for st.dataframe().
    """
    display = rankings.copy()

    def lag_description(lag: int) -> str:
        if lag > 0:
            return f"Leads by {lag} bars"
        elif lag < 0:
            return f"Lags by {abs(lag)} bars"
        return "Contemporaneous"

    display["direction"] = display["lag"].apply(lag_description)
    display["correlation"] = display["correlation"].round(4)

    return display[["asset", "lag", "correlation", "direction"]]
