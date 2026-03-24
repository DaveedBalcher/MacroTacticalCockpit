"""Tests for Plotly figure builders."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visuals import (
    build_ewma_chart,
    build_freshness_banner,
    build_lead_lag_table,
    build_main_chart,
    build_regime_badge_html,
    build_regime_indicator,
)


@pytest.fixture
def fan_data():
    """Synthetic BSTS fan data."""
    idx = pd.date_range("2025-01-08", periods=60, freq="min", tz="UTC")
    return pd.DataFrame({
        "mean": np.linspace(5000, 5010, 60),
        "ci_50_lower": np.linspace(4995, 5005, 60),
        "ci_50_upper": np.linspace(5005, 5015, 60),
        "ci_90_lower": np.linspace(4990, 5000, 60),
        "ci_90_upper": np.linspace(5010, 5020, 60),
    }, index=idx)


@pytest.fixture
def regime_states(synthetic_ohlcv):
    """Synthetic regime states aligned to OHLCV index."""
    n = len(synthetic_ohlcv)
    # Alternate regimes in blocks of 2500
    states = np.zeros(n, dtype=int)
    states[2500:5000] = 1
    states[5000:7500] = 2
    states[7500:] = 3
    return pd.Series(states, index=synthetic_ohlcv.index)


@pytest.fixture
def regime_label_map():
    return {0: "BULL", 1: "NEUTRAL", 2: "BEAR", 3: "HIGH_VOL"}


# --- build_main_chart ---


def test_build_main_chart_candlestick_only(synthetic_ohlcv):
    """Figure with no fan/regime data has exactly 2 traces (candlestick + volume)."""
    fig = build_main_chart(synthetic_ohlcv)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Price"
    assert fig.data[1].name == "Volume"


def test_build_main_chart_with_fan(synthetic_ohlcv, fan_data):
    """Figure with fan_data has 7 traces: candles + volume + 2x90%CI + 2x50%CI + mean."""
    fig = build_main_chart(synthetic_ohlcv, fan_data=fan_data)
    # 1 candlestick + 1 volume + 2 (90% band) + 2 (50% band) + 1 (mean line) = 7
    assert len(fig.data) == 7


def test_build_main_chart_with_regime_shapes(synthetic_ohlcv, regime_states, regime_label_map):
    """Figure with regime_states has vrect shapes in layout."""
    fig = build_main_chart(
        synthetic_ohlcv,
        regime_states=regime_states,
        regime_label_map=regime_label_map,
    )
    # Should have at least 4 shapes (one per regime block)
    assert len(fig.layout.shapes) >= 4


def test_build_main_chart_has_time_period_buttons(synthetic_ohlcv):
    """Chart includes client-side updatemenus buttons for time zoom."""
    fig = build_main_chart(synthetic_ohlcv)
    assert len(fig.layout.updatemenus) == 1
    labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert labels == ["1H", "4H", "1D", "3D", "ALL"]


def test_build_main_chart_time_buttons_anchor_to_actual_data(synthetic_ohlcv, fan_data):
    """Period buttons use last actual price as right anchor, not forecast end."""
    fig = build_main_chart(synthetic_ohlcv, fan_data=fan_data)
    btn_1h = fig.layout.updatemenus[0].buttons[0]
    x_range = btn_1h.args[0]["xaxis.range"]
    x_left = pd.Timestamp(x_range[0])
    x_right = pd.Timestamp(x_range[1])
    last_actual = synthetic_ohlcv.index[-1]
    # Left edge should be ~60 min before last actual price
    assert abs((last_actual - x_left).total_seconds() - 3600) < 1
    # Right edge should extend to end of forecast, not last actual
    assert x_right > last_actual


# --- build_regime_indicator ---


def test_build_regime_indicator_trace_count(regime_label_map):
    """Regime indicator has n_states traces (one per state bar segment)."""
    prob_vector = np.array([0.5, 0.2, 0.2, 0.1])
    fig = build_regime_indicator(prob_vector, regime_label_map)
    assert len(fig.data) == 4


def test_build_regime_indicator_probabilities_displayed(regime_label_map):
    """Traces have correct probability values."""
    prob_vector = np.array([0.5, 0.2, 0.2, 0.1])
    fig = build_regime_indicator(prob_vector, regime_label_map)
    displayed_probs = [trace.x[0] for trace in fig.data]
    np.testing.assert_allclose(displayed_probs, prob_vector)


# --- build_lead_lag_table ---


def test_build_lead_lag_table_columns():
    """Formatted table has expected columns."""
    rankings = pd.DataFrame({
        "asset": ["NQ", "GC", "CL"],
        "lag": [5, -3, 0],
        "correlation": [0.85, -0.62, 0.45],
    })
    result = build_lead_lag_table(rankings)
    assert list(result.columns) == ["asset", "lag", "correlation", "direction"]


def test_build_lead_lag_table_direction_labels():
    """Direction labels are human-readable."""
    rankings = pd.DataFrame({
        "asset": ["A", "B", "C"],
        "lag": [5, -3, 0],
        "correlation": [0.8, -0.6, 0.4],
    })
    result = build_lead_lag_table(rankings)
    assert result.iloc[0]["direction"] == "Leads by 5 bars"
    assert result.iloc[1]["direction"] == "Lags by 3 bars"
    assert result.iloc[2]["direction"] == "Contemporaneous"


# --- Phase 1: Theme consistency ---


def test_build_main_chart_uses_light_template(synthetic_ohlcv):
    fig = build_main_chart(synthetic_ohlcv)
    assert "dark" not in str(fig.layout.template).lower() or "plotly_white" in str(fig.layout.template)


def test_build_regime_indicator_uses_light_template(regime_label_map):
    prob_vector = np.array([0.5, 0.2, 0.2, 0.1])
    fig = build_regime_indicator(prob_vector, regime_label_map)
    assert "dark" not in str(fig.layout.template).lower() or "plotly_white" in str(fig.layout.template)


# --- Phase 2: EWMA chart ---


def test_build_ewma_chart_returns_figure():
    corr_series = pd.Series({"NQ": 0.85, "GC": -0.4, "CL": 0.3})
    fig = build_ewma_chart(corr_series)
    assert isinstance(fig, go.Figure)


def test_build_ewma_chart_height_scales_with_assets():
    small = pd.Series({"NQ": 0.85, "GC": -0.4})
    large = pd.Series({f"A{i}": 0.1 * i for i in range(10)})
    fig_small = build_ewma_chart(small)
    fig_large = build_ewma_chart(large)
    assert fig_large.layout.height > fig_small.layout.height
    assert fig_small.layout.height >= 250


def test_build_ewma_chart_bar_count_matches_input():
    corr_series = pd.Series({"NQ": 0.85, "GC": -0.4, "CL": 0.3})
    fig = build_ewma_chart(corr_series)
    assert len(fig.data[0].y) == 3


def test_build_ewma_chart_uses_light_template():
    corr_series = pd.Series({"NQ": 0.85})
    fig = build_ewma_chart(corr_series)
    assert "dark" not in str(fig.layout.template).lower()


# --- Phase 3: Volume subplot ---


def test_build_main_chart_has_volume_trace(synthetic_ohlcv):
    fig = build_main_chart(synthetic_ohlcv)
    volume_traces = [t for t in fig.data if t.name == "Volume"]
    assert len(volume_traces) == 1


def test_build_main_chart_volume_is_bar(synthetic_ohlcv):
    fig = build_main_chart(synthetic_ohlcv)
    volume_traces = [t for t in fig.data if t.name == "Volume"]
    assert isinstance(volume_traces[0], go.Bar)


def test_build_main_chart_candlestick_still_first(synthetic_ohlcv):
    fig = build_main_chart(synthetic_ohlcv)
    assert fig.data[0].name == "Price"


# --- Phase 4: Regime legend ---


def test_regime_legend_entries_present(synthetic_ohlcv, regime_states, regime_label_map):
    fig = build_main_chart(
        synthetic_ohlcv,
        regime_states=regime_states,
        regime_label_map=regime_label_map,
    )
    legend_names = {t.name for t in fig.data if t.showlegend}
    for label in ["BULL", "NEUTRAL", "BEAR", "HIGH_VOL"]:
        assert label in legend_names, f"Missing legend entry for {label}"


# --- Phase 5: Regime badge ---


def test_build_regime_badge_high_confidence():
    html = build_regime_badge_html("BULL", 0.85)
    assert "BULL" in html
    assert "85%" in html
    assert "dashed" not in html


def test_build_regime_badge_ambiguous():
    html = build_regime_badge_html("BULL", 0.52)
    assert "BULL" in html
    assert "52%" in html
    assert "dashed" in html


def test_build_regime_badge_unknown_label():
    html = build_regime_badge_html("UNKNOWN", 0.70)
    assert "UNKNOWN" in html


# --- Phase 6: Compact lead-lag ---


def test_build_lead_lag_table_compact_mode():
    rankings = pd.DataFrame({
        "asset": ["A", "B", "C"],
        "lag": [5, -3, 0],
        "correlation": [0.8, -0.6, 0.4],
    })
    result = build_lead_lag_table(rankings, compact=True)
    assert "\u2192" in result.iloc[0]["direction"]
    assert "\u2190" in result.iloc[1]["direction"]
    assert "\u2194" in result.iloc[2]["direction"]


def test_build_lead_lag_table_default_is_verbose():
    rankings = pd.DataFrame({
        "asset": ["A"], "lag": [5], "correlation": [0.8],
    })
    result = build_lead_lag_table(rankings)
    assert result.iloc[0]["direction"] == "Leads by 5 bars"


# --- Phase 7: Freshness banner ---


def test_build_freshness_banner_recent():
    ts = pd.Timestamp.now(tz="US/Eastern") - pd.Timedelta(minutes=5)
    html = build_freshness_banner(ts)
    assert "#4ade80" in html or "green" in html.lower()


def test_build_freshness_banner_stale():
    ts = pd.Timestamp.now(tz="US/Eastern") - pd.Timedelta(hours=2)
    html = build_freshness_banner(ts)
    assert "#f87171" in html or "red" in html.lower()


def test_build_freshness_banner_contains_timestamp():
    ts = pd.Timestamp("2025-06-15 14:30:00", tz="US/Eastern")
    html = build_freshness_banner(ts)
    assert "2025-06-15" in html
    assert "14:30" in html
