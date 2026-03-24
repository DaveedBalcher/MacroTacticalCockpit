"""Tests for EWMA correlation and lead-lag engines."""

import numpy as np
import pandas as pd
import pytest

from src.quant import cross_correlation_lag, ewma_correlation, lead_lag_rankings


# --- EWMA Correlation ---


def test_ewma_correlation_shape(synthetic_close_matrix):
    """Output has correct shape: same rows (minus 1 for returns), N-1 columns."""
    result = ewma_correlation(synthetic_close_matrix, anchor="ANCHOR")
    # Should exclude anchor column
    assert "ANCHOR" not in result.columns
    assert len(result.columns) == len(synthetic_close_matrix.columns) - 1


def test_ewma_correlation_values_bounded(synthetic_close_matrix):
    """All correlation values in [-1, 1]."""
    result = ewma_correlation(synthetic_close_matrix, anchor="ANCHOR")
    # Drop initial NaN rows from EWMA warmup
    result = result.dropna()
    assert (result >= -1.0 - 1e-6).all().all()
    assert (result <= 1.0 + 1e-6).all().all()


def test_ewma_correlation_invalid_anchor(synthetic_close_matrix):
    """Invalid anchor raises ValueError."""
    with pytest.raises(ValueError, match="not in close_matrix"):
        ewma_correlation(synthetic_close_matrix, anchor="NONEXISTENT")


# --- Cross-Correlation Lag ---


def test_cross_correlation_identifies_known_lag(synthetic_sine_pair):
    """Given two sine waves with known offset, engine finds the correct lag."""
    series_a, series_b, known_lag = synthetic_sine_pair
    detected_lag, corr = cross_correlation_lag(series_a, series_b, max_lag=30)
    # series_b leads series_a by known_lag, so detected_lag should be positive
    assert abs(detected_lag) == known_lag
    assert abs(corr) > 0.8  # strong correlation at the optimal lag


def test_cross_correlation_zero_lag_for_identical():
    """Identical series should return lag=0, correlation~1.0."""
    np.random.seed(99)
    s = pd.Series(np.random.randn(500))
    lag, corr = cross_correlation_lag(s, s, max_lag=30)
    assert lag == 0
    assert corr > 0.95


def test_cross_correlation_respects_max_lag():
    """Result lag is within [-max_lag, +max_lag]."""
    np.random.seed(42)
    a = pd.Series(np.random.randn(500))
    b = pd.Series(np.random.randn(500))
    max_lag = 15
    lag, _ = cross_correlation_lag(a, b, max_lag=max_lag)
    assert -max_lag <= lag <= max_lag


# --- Lead-Lag Rankings ---


def test_lead_lag_rankings_sorted_by_abs_correlation(synthetic_close_matrix):
    """Rankings are sorted by absolute correlation descending."""
    rankings = lead_lag_rankings(synthetic_close_matrix, anchor="ANCHOR")
    abs_corrs = rankings["correlation"].abs().values
    assert all(abs_corrs[i] >= abs_corrs[i + 1] for i in range(len(abs_corrs) - 1))


def test_lead_lag_rankings_excludes_anchor(synthetic_close_matrix):
    """Anchor asset does not appear in its own rankings."""
    rankings = lead_lag_rankings(synthetic_close_matrix, anchor="ANCHOR")
    assert "ANCHOR" not in rankings["asset"].values


def test_lead_lag_rankings_has_expected_columns(synthetic_close_matrix):
    """Rankings DataFrame has the right columns."""
    rankings = lead_lag_rankings(synthetic_close_matrix, anchor="ANCHOR")
    assert list(rankings.columns) == ["asset", "lag", "correlation"]


def test_lead_lag_rankings_invalid_anchor(synthetic_close_matrix):
    """Invalid anchor raises ValueError."""
    with pytest.raises(ValueError, match="not in close_matrix"):
        lead_lag_rankings(synthetic_close_matrix, anchor="FAKE")
