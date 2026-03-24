"""Tests for data ingestion and state management."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import ROLLING_WINDOW_BARS
from src.data import (
    DataManager,
    align_timestamps,
    fetch_ohlcv,
    rolling_trim,
)


# --- fetch_ohlcv ---


def test_fetch_ohlcv_invalid_ticker_raises():
    """Unknown alias raises ValueError."""
    with pytest.raises(ValueError, match="Unknown ticker alias"):
        fetch_ohlcv("INVALID_TICKER")


def test_fetch_ohlcv_returns_expected_columns(synthetic_ohlcv):
    """OHLCV DataFrame has Open, High, Low, Close, Volume columns."""
    # We mock yfinance to return our synthetic data
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = synthetic_ohlcv

    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        df = fetch_ohlcv("ES")

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.tz is not None  # timezone-aware
    assert len(df) == len(synthetic_ohlcv)


def test_fetch_ohlcv_empty_data_raises():
    """Empty yfinance response raises RuntimeError."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(RuntimeError, match="no data"):
            fetch_ohlcv("ES")


# --- rolling_trim ---


def test_rolling_trim_enforces_max_rows(synthetic_ohlcv):
    """DataFrame with 10_000 rows trimmed to specified max."""
    max_rows = 5000
    trimmed = rolling_trim(synthetic_ohlcv, max_rows=max_rows)
    assert len(trimmed) == max_rows


def test_rolling_trim_preserves_most_recent(synthetic_ohlcv):
    """After trim, last row matches original last row."""
    max_rows = 5000
    trimmed = rolling_trim(synthetic_ohlcv, max_rows=max_rows)
    pd.testing.assert_series_equal(trimmed.iloc[-1], synthetic_ohlcv.iloc[-1])


def test_rolling_trim_noop_when_under_limit(synthetic_ohlcv):
    """No trimming when DataFrame is already under the limit."""
    trimmed = rolling_trim(synthetic_ohlcv, max_rows=20_000)
    assert len(trimmed) == len(synthetic_ohlcv)


# --- align_timestamps ---


def test_align_timestamps_forward_fills():
    """Gaps <= MAX_FFILL_BARS are forward-filled."""
    idx = pd.date_range("2025-01-01", periods=20, freq="min", tz="UTC")

    df_a = pd.DataFrame({"Close": range(20)}, index=idx)

    # df_b has a 3-bar gap (rows 5,6,7 missing)
    idx_b = idx.delete([5, 6, 7])
    df_b = pd.DataFrame({"Close": range(17)}, index=idx_b)

    result = align_timestamps({"A": df_a, "B": df_b}, max_ffill=5)

    assert "A" in result
    assert "B" in result
    assert not result["B"].isna().any().any()
    assert len(result["A"]) == len(result["B"])


def test_align_timestamps_drops_asset_with_no_overlap():
    """Asset with almost no overlapping timestamps is excluded."""
    idx_a = pd.date_range("2025-01-01", periods=100, freq="min", tz="UTC")
    idx_b = pd.date_range("2025-01-01", periods=100, freq="min", tz="UTC")
    idx_c = pd.date_range("2025-02-01", periods=100, freq="min", tz="UTC")  # no overlap

    df_a = pd.DataFrame({"Close": range(100)}, index=idx_a)
    df_b = pd.DataFrame({"Close": range(100)}, index=idx_b)
    df_c = pd.DataFrame({"Close": range(100)}, index=idx_c)

    result = align_timestamps({"A": df_a, "B": df_b, "C": df_c}, max_ffill=5)

    assert "A" in result
    assert "B" in result
    assert "C" not in result  # no overlap with A and B


def test_align_timestamps_empty_input():
    """Empty dict returns empty dict."""
    assert align_timestamps({}) == {}


# --- DataManager ---


def test_data_manager_refresh_and_get(synthetic_ohlcv):
    """Mock yfinance.download, verify DataManager stores aligned frames."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = synthetic_ohlcv

    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        dm = DataManager(assets=["ES", "NQ"], max_rows=5000)
        dm.refresh()

    assert "ES" in dm.available_assets
    assert "NQ" in dm.available_assets

    ohlcv = dm.get_ohlcv("ES")
    assert len(ohlcv) <= 5000
    assert list(ohlcv.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_data_manager_get_close_matrix(synthetic_ohlcv):
    """Close matrix has correct shape and column names."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = synthetic_ohlcv

    with patch("src.data.yf.Ticker", return_value=mock_ticker):
        dm = DataManager(assets=["ES", "NQ"], max_rows=5000)
        dm.refresh()

    matrix = dm.get_close_matrix()
    assert "ES" in matrix.columns
    assert "NQ" in matrix.columns
    assert len(matrix) <= 5000


def test_data_manager_get_ohlcv_missing_asset():
    """Requesting unloaded asset raises KeyError."""
    dm = DataManager(assets=["ES"])
    with pytest.raises(KeyError):
        dm.get_ohlcv("NONEXISTENT")
