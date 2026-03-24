"""Plotly figure builders for the Tactical Cockpit."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    FRESHNESS_STALE_MINUTES,
    PLOTLY_TEMPLATE,
    REGIME_BADGE_COLORS,
    REGIME_COLORS,
)


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
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.02,
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["Open"],
        high=ohlcv["High"],
        low=ohlcv["Low"],
        close=ohlcv["Close"],
        name="Price",
    ), row=1, col=1)

    # 2. Volume bar chart
    if "Volume" in ohlcv.columns:
        vol_colors = [
            "#4ade80" if c >= o else "#f87171"
            for c, o in zip(ohlcv["Close"], ohlcv["Open"])
        ]
        fig.add_trace(go.Bar(
            x=ohlcv.index,
            y=ohlcv["Volume"],
            marker_color=vol_colors,
            opacity=0.4,
            name="Volume",
            showlegend=False,
        ), row=2, col=1)

    # 3-5. BSTS fan chart overlay
    if fan_data is not None:
        # Connect forecast to last actual price for visual continuity
        last_close = ohlcv["Close"].iloc[-1]
        last_ts = ohlcv.index[-1]
        bridge_x = [last_ts] + list(fan_data.index)

        # 90% CI band (wider, added first so it's behind)
        if "ci_90_lower" in fan_data.columns:
            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=[last_close] + list(fan_data["ci_90_upper"]),
                mode="lines",
                line=dict(width=1, color="rgba(100, 149, 237, 0.4)"),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=[last_close] + list(fan_data["ci_90_lower"]),
                mode="lines",
                line=dict(width=1, color="rgba(100, 149, 237, 0.4)"),
                fill="tonexty",
                fillcolor="rgba(100, 149, 237, 0.2)",
                name="90% CI",
            ), row=1, col=1)

        # 50% CI band (narrower, on top)
        if "ci_50_lower" in fan_data.columns:
            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=[last_close] + list(fan_data["ci_50_upper"]),
                mode="lines",
                line=dict(width=1, color="rgba(100, 149, 237, 0.6)"),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bridge_x,
                y=[last_close] + list(fan_data["ci_50_lower"]),
                mode="lines",
                line=dict(width=1, color="rgba(100, 149, 237, 0.6)"),
                fill="tonexty",
                fillcolor="rgba(100, 149, 237, 0.4)",
                name="50% CI",
            ), row=1, col=1)

        # Mean forecast line
        fig.add_trace(go.Scatter(
            x=bridge_x,
            y=[last_close] + list(fan_data["mean"]),
            mode="lines",
            line=dict(color="cornflowerblue", width=2, dash="dot"),
            name="Forecast",
        ), row=1, col=1)

    # 5. Regime background colors via vrect shapes
    if regime_states is not None and regime_label_map is not None:
        _add_regime_backgrounds(fig, regime_states, regime_label_map)

    # Build rangebreaks from actual gaps in the OHLCV data.
    # We skip gaps that overlap with the forecast window so the
    # BSTS fan chart remains visible on the right edge.
    gap_breaks: list[dict] = []
    if len(ohlcv) > 1:
        diffs = ohlcv.index.to_series().diff()
        big_gaps = diffs[diffs > pd.Timedelta(minutes=5)]
        # Determine the forecast window to exclude from rangebreaks
        fan_start = fan_data.index[0] if fan_data is not None and len(fan_data) > 0 else None
        fan_end = fan_data.index[-1] if fan_data is not None and len(fan_data) > 0 else None

        for gap_end_ts, gap_dur in big_gaps.items():
            gap_start_ts = gap_end_ts - gap_dur
            # Skip this break if it overlaps with the forecast window
            if fan_start is not None and gap_start_ts < fan_end and gap_end_ts > fan_start:
                continue
            gap_breaks.append(dict(
                values=[str(gap_start_ts)],
                dvalue=int(gap_dur.total_seconds() * 1000),
            ))

    # Time-period buttons: anchor to last actual price, always include forecast
    x_right = fan_data.index[-1] if fan_data is not None and len(fan_data) > 0 else ohlcv.index[-1]
    last_actual = ohlcv.index[-1]
    period_buttons = []
    for label, minutes in [("1H", 60), ("4H", 240), ("1D", 1440), ("3D", 4320)]:
        x_left = last_actual - pd.Timedelta(minutes=minutes)
        period_buttons.append(dict(
            args=[{"xaxis.range": [str(x_left), str(x_right)]}],
            label=label,
            method="relayout",
        ))
    period_buttons.append(dict(
        args=[{"xaxis.autorange": True}],
        label="ALL",
        method="relayout",
    ))

    xaxis_cfg: dict = dict(
        type="date",
        rangebreaks=gap_breaks,
        autorange=True,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis=xaxis_cfg,
        xaxis2=dict(type="date", rangebreaks=gap_breaks, autorange=True),
        yaxis2=dict(showticklabels=False),
        template=PLOTLY_TEMPLATE,
        height=700,
        margin=dict(l=50, r=20, t=40, b=40),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            buttons=period_buttons,
            bgcolor="rgba(128,128,128,0.1)",
            font=dict(size=12),
        )],
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

    # Add legend entries for each regime label
    seen_labels: set[str] = set()
    for state_idx, label in label_map.items():
        if label in seen_labels:
            continue
        seen_labels.add(label)
        color = REGIME_COLORS.get(label, "rgba(128, 128, 128, 0.12)")
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=label,
            showlegend=True,
        ))


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
        template=PLOTLY_TEMPLATE,
    )

    return fig


def build_lead_lag_table(
    rankings: pd.DataFrame,
    compact: bool = False,
) -> pd.DataFrame:
    """Format lead-lag rankings for display in Streamlit sidebar.

    Args:
        rankings: DataFrame with asset, lag, correlation columns.
        compact: If True, use arrow notation for narrow sidebars.

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

    def lag_compact(lag: int) -> str:
        if lag > 0:
            return f"+{lag} \u2192"
        elif lag < 0:
            return f"{lag} \u2190"
        return "0 \u2194"

    display["direction"] = display["lag"].apply(lag_compact if compact else lag_description)
    display["correlation"] = display["correlation"].round(4)

    return display[["asset", "lag", "correlation", "direction"]]


def build_ewma_chart(latest_corr: pd.Series) -> go.Figure:
    """Build a horizontal bar chart of EWMA correlations.

    Args:
        latest_corr: Series of correlation values indexed by asset name.

    Returns:
        go.Figure with properly scaled height.
    """
    colors = ["#4ade80" if v > 0 else "#f87171" for v in latest_corr.values]
    fig = go.Figure(go.Bar(
        x=latest_corr.values,
        y=latest_corr.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in latest_corr.values],
        textposition="auto",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        height=max(250, len(latest_corr) * 50 + 60),
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(
            range=[-1.15, 1.15],
            title="Correlation",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.3)",
            zerolinewidth=1,
        ),
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        font=dict(size=13),
        bargap=0.3,
    )
    return fig


def build_regime_badge_html(
    label: str,
    probability: float,
    badge_colors: dict[str, tuple[str, str]] | None = None,
) -> str:
    """Build HTML for a regime badge with ambiguity cues.

    Args:
        label: Regime label (e.g. "BULL").
        probability: Dominant regime probability (0-1).
        badge_colors: Optional override; defaults to REGIME_BADGE_COLORS.

    Returns:
        HTML string for the badge.
    """
    if badge_colors is None:
        badge_colors = REGIME_BADGE_COLORS
    bg, fg = badge_colors.get(label, ("#374151", "#e5e7eb"))

    border_style = "none"
    warning = ""
    if probability < 0.6:
        border_style = "2px dashed rgba(128,128,128,0.5)"
        warning = "&#9888; "

    return (
        f'<span style="background:{fg}; color:{bg}; padding:5px 14px; '
        f'border-radius:20px; font-weight:700; font-size:0.85rem; '
        f'border:{border_style}; white-space:nowrap;">'
        f'{warning}{label} &middot; {probability:.0%}</span>'
    )


def build_freshness_banner(
    last_timestamp: pd.Timestamp,
    stale_threshold_minutes: int | None = None,
) -> str:
    """Build an HTML freshness indicator.

    Args:
        last_timestamp: Timestamp of last data point.
        stale_threshold_minutes: Minutes before data is considered stale.

    Returns:
        HTML string with colored pill.
    """
    if stale_threshold_minutes is None:
        stale_threshold_minutes = FRESHNESS_STALE_MINUTES

    now = pd.Timestamp.now(tz=last_timestamp.tzinfo)
    age = now - last_timestamp
    age_minutes = age.total_seconds() / 60

    ts_str = last_timestamp.strftime("%Y-%m-%d %H:%M %Z")

    if age_minutes < stale_threshold_minutes:
        color = "#4ade80"
        text_color = "#065f46"
        status = "LIVE"
    else:
        color = "#f87171"
        text_color = "#7f1d1d"
        status = "STALE"

    age_display = f"{int(age_minutes)}m ago" if age_minutes < 120 else f"{int(age_minutes / 60)}h ago"

    return (
        f'<div style="display:inline-flex; align-items:center; gap:0.5rem; '
        f'margin-bottom:0.5rem;">'
        f'<span style="background:{color}; color:{text_color}; padding:3px 10px; '
        f'border-radius:12px; font-weight:700; font-size:0.75rem;">{status}</span>'
        f'<span style="font-size:0.8rem; opacity:0.7;">{ts_str} ({age_display})</span>'
        f'</div>'
    )
