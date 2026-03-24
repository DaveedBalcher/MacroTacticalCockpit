"""EWMA correlation and lead-lag cross-correlation engines."""

import numpy as np
import pandas as pd
from scipy import signal

from src.config import EWMA_SPAN, MAX_LAG_BARS


def ewma_correlation(
    close_matrix: pd.DataFrame,
    anchor: str,
    span: int = EWMA_SPAN,
) -> pd.DataFrame:
    """Compute rolling EWMA correlation of all assets against the anchor asset.

    Args:
        close_matrix: DataFrame with assets as columns, DatetimeIndex as index.
                      Values are Close prices.
        anchor: Column name of the anchor asset.
        span: EWMA half-life in bars.

    Returns:
        DataFrame with same index, columns = non-anchor assets,
        values = rolling EWMA correlation with anchor.
    """
    if anchor not in close_matrix.columns:
        raise ValueError(f"Anchor '{anchor}' not in close_matrix columns")

    # Compute log returns
    log_returns = np.log(close_matrix / close_matrix.shift(1)).dropna()

    anchor_ret = log_returns[anchor]
    other_cols = [c for c in log_returns.columns if c != anchor]

    correlations = {}
    for col in other_cols:
        aux_ret = log_returns[col]

        # EWMA covariance and variances
        ewm = pd.DataFrame({"anchor": anchor_ret, "aux": aux_ret}).ewm(span=span)
        cov_matrix = ewm.cov()

        # Extract pairwise covariance and variances
        # cov() returns a multi-level index: (time, variable) x variable
        cov_vals = cov_matrix.xs("anchor", level=1)["aux"]
        var_anchor = cov_matrix.xs("anchor", level=1)["anchor"]
        var_aux = cov_matrix.xs("aux", level=1)["aux"]

        corr = cov_vals / np.sqrt(var_anchor * var_aux)
        correlations[col] = corr

    return pd.DataFrame(correlations)


def cross_correlation_lag(
    anchor_returns: pd.Series,
    aux_returns: pd.Series,
    max_lag: int = MAX_LAG_BARS,
) -> tuple[int, float]:
    """Find the lag at which aux_returns maximally correlates with anchor_returns.

    Uses normalized cross-correlation over range [-max_lag, +max_lag].

    Args:
        anchor_returns: Log returns of the anchor asset.
        aux_returns: Log returns of the auxiliary asset.
        max_lag: Maximum lag to search (both positive and negative).

    Returns:
        (best_lag, correlation_value)
        - best_lag > 0 means aux LEADS anchor by that many bars.
        - best_lag < 0 means aux LAGS anchor.
        - best_lag == 0 means contemporaneous.
    """
    a = anchor_returns.values.astype(float)
    b = aux_returns.values.astype(float)

    # Normalize to zero mean, unit variance
    a = (a - np.mean(a)) / (np.std(a) + 1e-10)
    b = (b - np.mean(b)) / (np.std(b) + 1e-10)

    # Full cross-correlation
    full_corr = signal.correlate(a, b, mode="full")
    full_corr = full_corr / len(a)  # normalize

    # The center of the full correlation is at index len(a) - 1
    center = len(a) - 1
    lag_range = np.arange(-center, center + 1)

    # Restrict to [-max_lag, +max_lag]
    mask = (lag_range >= -max_lag) & (lag_range <= max_lag)
    restricted_corr = full_corr[mask]
    restricted_lags = lag_range[mask]

    best_idx = np.argmax(np.abs(restricted_corr))
    best_lag = int(restricted_lags[best_idx])
    best_corr = float(restricted_corr[best_idx])

    return best_lag, best_corr


def lead_lag_rankings(
    close_matrix: pd.DataFrame,
    anchor: str,
    max_lag: int = MAX_LAG_BARS,
) -> pd.DataFrame:
    """For each non-anchor asset, compute lead-lag vs anchor.

    Returns:
        DataFrame with columns [asset, lag, correlation], sorted by
        abs(correlation) descending. Positive lag = asset leads anchor.
    """
    if anchor not in close_matrix.columns:
        raise ValueError(f"Anchor '{anchor}' not in close_matrix columns")

    log_returns = np.log(close_matrix / close_matrix.shift(1)).dropna()
    anchor_ret = log_returns[anchor]

    results = []
    for col in log_returns.columns:
        if col == anchor:
            continue
        lag, corr = cross_correlation_lag(anchor_ret, log_returns[col], max_lag)
        results.append({"asset": col, "lag": lag, "correlation": corr})

    rankings = pd.DataFrame(results)
    rankings = rankings.sort_values("correlation", key=abs, ascending=False)
    rankings = rankings.reset_index(drop=True)
    return rankings
