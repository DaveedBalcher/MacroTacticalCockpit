"""Tests for Plotly figure builders."""

import numpy as np
import pandas as pd
import pytest

from src.visuals import build_lead_lag_table, build_main_chart, build_regime_indicator


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
    """Figure with no fan/regime data has exactly 1 trace (candlestick)."""
    fig = build_main_chart(synthetic_ohlcv)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], type(fig.data[0]))  # it's a trace
    assert fig.data[0].name == "Price"


def test_build_main_chart_with_fan(synthetic_ohlcv, fan_data):
    """Figure with fan_data has 6 traces: candles + 2x90%CI + 2x50%CI + mean."""
    fig = build_main_chart(synthetic_ohlcv, fan_data=fan_data)
    # 1 candlestick + 2 (90% band) + 2 (50% band) + 1 (mean line) = 6
    assert len(fig.data) == 6


def test_build_main_chart_with_regime_shapes(synthetic_ohlcv, regime_states, regime_label_map):
    """Figure with regime_states has vrect shapes in layout."""
    fig = build_main_chart(
        synthetic_ohlcv,
        regime_states=regime_states,
        regime_label_map=regime_label_map,
    )
    # Should have at least 4 shapes (one per regime block)
    assert len(fig.layout.shapes) >= 4


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
