"""Shared test fixtures for Macro Tactical Cockpit."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 10_000 rows of synthetic 1-min OHLCV data."""
    np.random.seed(42)
    n = 10_000
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")

    # Random walk for close prices starting at 5000
    returns = np.random.normal(0, 0.0002, n)
    close = 5000.0 * np.exp(np.cumsum(returns))

    # Derive OHLV from close
    noise = np.abs(np.random.normal(0, 0.5, n))
    high = close + noise
    low = close - noise
    open_ = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(100, 10_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def synthetic_close_matrix() -> pd.DataFrame:
    """Generate a close-price matrix for 5 synthetic assets with known correlations."""
    np.random.seed(42)
    n = 2000
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")

    # Base random walk
    base_returns = np.random.normal(0, 0.001, n)
    base = 100.0 * np.exp(np.cumsum(base_returns))

    assets = {}
    assets["ANCHOR"] = base

    # Correlated asset (rho ~ 0.8)
    corr_returns = 0.8 * base_returns + 0.2 * np.random.normal(0, 0.001, n)
    assets["CORR_HIGH"] = 100.0 * np.exp(np.cumsum(corr_returns))

    # Anti-correlated asset
    anti_returns = -0.6 * base_returns + 0.4 * np.random.normal(0, 0.001, n)
    assets["ANTI_CORR"] = 100.0 * np.exp(np.cumsum(anti_returns))

    # Uncorrelated asset
    uncorr_returns = np.random.normal(0, 0.001, n)
    assets["UNCORR"] = 100.0 * np.exp(np.cumsum(uncorr_returns))

    # Lagged asset (leads ANCHOR by 10 bars)
    lead_returns = np.zeros(n)
    lead_returns[:-10] = base_returns[10:]
    assets["LEADER"] = 100.0 * np.exp(np.cumsum(lead_returns))

    return pd.DataFrame(assets, index=idx)


@pytest.fixture
def synthetic_sine_pair() -> tuple[pd.Series, pd.Series, int]:
    """
    Two sine-wave series where series_b leads series_a by `known_lag` bars.
    Returns (series_a, series_b, known_lag).
    """
    n = 1000
    known_lag = 10
    t = np.arange(n, dtype=float)
    period = 100

    series_a = pd.Series(np.sin(2 * np.pi * t / period), name="anchor")
    # series_b is shifted ahead by known_lag (it leads)
    series_b = pd.Series(np.sin(2 * np.pi * (t + known_lag) / period), name="leader")

    return series_a, series_b, known_lag
