"""Data ingestion and state management for 1-minute OHLCV data."""

import logging
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    ASSET_UNIVERSE,
    FETCH_INTERVAL,
    FETCH_PERIOD,
    MAX_FFILL_BARS,
    ROLLING_WINDOW_BARS,
    TICKER_MAP,
)

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(
    ticker_alias: str,
    period: str = FETCH_PERIOD,
    interval: str = FETCH_INTERVAL,
) -> pd.DataFrame:
    """Fetch 1-minute OHLCV data for a single asset from yfinance.

    Args:
        ticker_alias: Human-readable name (e.g., "ES"). Mapped via TICKER_MAP.
        period: yfinance period string. Default "7d" (max for 1m).
        interval: yfinance interval string. Default "1m".

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and
        DatetimeIndex (UTC-normalized).

    Raises:
        ValueError: If ticker_alias not in TICKER_MAP.
        RuntimeError: If yfinance returns empty data.
    """
    if ticker_alias not in TICKER_MAP:
        raise ValueError(
            f"Unknown ticker alias '{ticker_alias}'. "
            f"Valid aliases: {list(TICKER_MAP.keys())}"
        )

    yf_ticker = TICKER_MAP[ticker_alias]
    ticker = yf.Ticker(yf_ticker)

    # Retry with backoff on rate-limit errors (common on Streamlit Cloud cold start)
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = ticker.history(period=period, interval=interval)
            break
        except yf.exceptions.YFRateLimitError:
            if attempt < 2:
                wait = 2 ** attempt  # 1s, 2s
                logger.warning("Rate-limited fetching %s, retrying in %ds...", ticker_alias, wait)
                time.sleep(wait)
            else:
                raise

    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for '{ticker_alias}' ({yf_ticker})"
        )

    # Keep only OHLCV columns (yfinance may include Dividends, Stock Splits)
    df = df[[c for c in EXPECTED_COLUMNS if c in df.columns]]

    # Normalize timezone to UTC
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC")

    df.index.name = "datetime"
    return df


def fetch_universe(
    assets: list[str] | None = None,
    period: str = FETCH_PERIOD,
    interval: str = FETCH_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all assets in the universe (or a subset).

    Returns:
        Dict mapping ticker_alias -> OHLCV DataFrame.
        Assets that fail to download are logged and omitted.
    """
    assets = assets or ASSET_UNIVERSE
    frames: dict[str, pd.DataFrame] = {}

    for alias in assets:
        try:
            frames[alias] = fetch_ohlcv(alias, period=period, interval=interval)
        except (ValueError, RuntimeError, yf.exceptions.YFRateLimitError) as e:
            logger.warning("Skipping %s: %s", alias, e)

    return frames


def align_timestamps(
    frames: dict[str, pd.DataFrame],
    max_ffill: int = MAX_FFILL_BARS,
) -> dict[str, pd.DataFrame]:
    """Align all DataFrames to a common DatetimeIndex.

    Strategy: Build a consensus index from timestamps where the majority of
    assets have data, then forward-fill small gaps per asset.

    Returns:
        Dict of aligned DataFrames (same index across all).
    """
    if not frames:
        return {}

    if len(frames) == 1:
        return dict(frames)

    # Build a consensus index: keep timestamps where at least half the assets
    # have data (handles BTC 24/7 vs futures ~23hr gracefully)
    all_timestamps = pd.Index([], dtype="datetime64[ns, UTC]")
    for df in frames.values():
        all_timestamps = all_timestamps.union(df.index)

    # Count how many assets have data at each timestamp
    counts = pd.Series(0, index=all_timestamps)
    for df in frames.values():
        counts.loc[counts.index.isin(df.index)] += 1

    min_coverage = max(len(frames) // 2, 2)
    consensus_idx = counts[counts >= min_coverage].index.sort_values()

    if consensus_idx.empty:
        logger.warning("No overlapping timestamps found across assets")
        return {}

    # Reindex each asset to the consensus index
    aligned: dict[str, pd.DataFrame] = {}

    for alias, df in frames.items():
        reindexed = df.reindex(consensus_idx)
        reindexed = reindexed.ffill(limit=max_ffill)

        nan_pct = reindexed.isna().any(axis=1).mean()
        if nan_pct > 0.5:
            logger.warning(
                "Dropping %s: %.0f%% missing after ffill (limit=%d)",
                alias, nan_pct * 100, max_ffill,
            )
            continue

        # Drop remaining NaN rows from the edges
        reindexed = reindexed.dropna()
        aligned[alias] = reindexed

    # Final pass: intersect indices so all assets share the same rows
    if len(aligned) > 1:
        common_idx = aligned[next(iter(aligned))].index
        for df in aligned.values():
            common_idx = common_idx.intersection(df.index)
        aligned = {alias: df.loc[common_idx] for alias, df in aligned.items()}

    return aligned


def rolling_trim(
    df: pd.DataFrame,
    max_rows: int = ROLLING_WINDOW_BARS,
) -> pd.DataFrame:
    """Trim DataFrame to keep only the last `max_rows` rows.

    Returns:
        Trimmed DataFrame (tail slice, copy).
    """
    if len(df) <= max_rows:
        return df.copy()
    return df.iloc[-max_rows:].copy()


_SYNTHETIC_INDEX: pd.DatetimeIndex | None = None


def _get_synthetic_index(n: int = 2000) -> pd.DatetimeIndex:
    """Return a shared time index for all synthetic assets."""
    global _SYNTHETIC_INDEX
    if _SYNTHETIC_INDEX is None or len(_SYNTHETIC_INDEX) != n:
        _SYNTHETIC_INDEX = pd.date_range(
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=n),
            periods=n,
            freq="min",
            tz="UTC",
        )
    return _SYNTHETIC_INDEX


def _generate_synthetic_ohlcv(alias: str, n: int = 2000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing/demo purposes."""
    np.random.seed(hash(alias) % 2**31)
    idx = _get_synthetic_index(n)
    returns = np.random.normal(0, 0.0003, n)
    close = 5000.0 * np.exp(np.cumsum(returns))
    noise = np.abs(np.random.normal(0, 0.5, n))
    return pd.DataFrame({
        "Open": close + np.random.normal(0, 0.3, n),
        "High": close + noise,
        "Low": close - noise,
        "Close": close,
        "Volume": np.random.randint(100, 10_000, n),
    }, index=idx)


class DataManager:
    """Stateful manager holding the current universe of aligned OHLCV DataFrames.

    Supports incremental refresh and rolling trim.
    Set MOCK_DATA=1 environment variable to use synthetic data (for testing).
    """

    def __init__(
        self,
        assets: list[str] | None = None,
        max_rows: int = ROLLING_WINDOW_BARS,
    ):
        self.assets = assets or list(TICKER_MAP.keys())
        self.max_rows = max_rows
        self._frames: dict[str, pd.DataFrame] = {}

    def refresh(self) -> None:
        """Fetch latest data, align, trim, and store."""
        if os.environ.get("MOCK_DATA") == "1":
            self._frames = {
                alias: rolling_trim(_generate_synthetic_ohlcv(alias), self.max_rows)
                for alias in self.assets
            }
            return

        raw = fetch_universe(self.assets)
        aligned = align_timestamps(raw)
        self._frames = {
            alias: rolling_trim(df, self.max_rows) for alias, df in aligned.items()
        }

    def get_close_matrix(self) -> pd.DataFrame:
        """Return a single DataFrame of Close prices with assets as columns."""
        if not self._frames:
            return pd.DataFrame()
        return pd.DataFrame(
            {alias: df["Close"] for alias, df in self._frames.items()}
        )

    def get_ohlcv(self, asset: str) -> pd.DataFrame:
        """Return OHLCV DataFrame for a single asset."""
        if asset not in self._frames:
            raise KeyError(f"Asset '{asset}' not loaded. Available: {self.available_assets}")
        return self._frames[asset]

    @property
    def available_assets(self) -> list[str]:
        """Assets that successfully loaded data."""
        return list(self._frames.keys())
