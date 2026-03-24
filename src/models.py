"""HMM regime detection and BSTS probabilistic forecasting."""

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.statespace.structural import UnobservedComponents

from src.config import (
    BSTS_CI_LEVELS,
    BSTS_FORECAST_HORIZON,
    HMM_COVARIANCE_TYPE,
    HMM_N_ITER,
    HMM_N_STATES,
    REGIME_LABELS,
)

logger = logging.getLogger(__name__)


# ─── HMM REGIME DETECTION ────────────────────────────────────────────


def prepare_hmm_features(
    ohlcv: pd.DataFrame,
    vol_window: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare feature matrix for HMM: [log_returns, realized_volatility].

    Args:
        ohlcv: OHLCV DataFrame for the anchor asset.
        vol_window: Rolling window for realized volatility.

    Returns:
        (standardized_features, means, stds) — the raw statistics are
        needed to un-standardize HMM state means for regime labeling.
    """
    close = ohlcv["Close"]
    log_returns = np.log(close / close.shift(1))
    realized_vol = log_returns.rolling(window=vol_window).std()

    features = pd.DataFrame({
        "log_return": log_returns,
        "realized_vol": realized_vol,
    }).dropna()

    # Standardize features for numerical stability
    values = features.values
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds < 1e-10] = 1.0  # prevent division by zero
    standardized = (values - means) / stds

    return standardized, means, stds


def fit_hmm(
    features: np.ndarray,
    n_states: int = HMM_N_STATES,
    covariance_type: str = HMM_COVARIANCE_TYPE,
    n_iter: int = HMM_N_ITER,
    random_state: int = 42,
) -> GaussianHMM:
    """Fit a Gaussian HMM to the feature matrix.

    Returns:
        Fitted GaussianHMM model.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=0.01,
    )
    model.fit(features)

    if not model.monitor_.converged:
        logger.warning("HMM did not converge within %d iterations", n_iter)

    return model


def predict_regimes(
    model: GaussianHMM,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict regime states and posterior probabilities.

    Returns:
        (states, probabilities)
        - states: 1D array of int, shape (T,), values in {0, ..., n_states-1}
        - probabilities: 2D array, shape (T, n_states), rows sum to 1.0
    """
    states = model.predict(features)
    probabilities = model.predict_proba(features)
    return states, probabilities


def map_regime_labels(
    model: GaussianHMM,
    feature_means: np.ndarray | None = None,
    feature_stds: np.ndarray | None = None,
) -> dict[int, str]:
    """Dynamically assign human-readable labels to raw HMM states.

    Uses un-standardized state means so that labeling reflects actual
    return/volatility magnitudes rather than relative rankings of noise.

    Strategy:
      1. Convert standardized HMM means back to raw scale.
      2. States whose raw mean return exceeds +1 std of the return
         distribution -> BULL; below -1 std -> BEAR.
      3. Among unlabeled states, highest raw volatility (above median) -> HIGH_VOL.
      4. Everything else -> NEUTRAL.

    This avoids labeling the dominant quiet-market cluster as BEAR/BULL
    just because it has a slightly positive/negative standardized mean.

    Returns:
        Dict mapping raw state int -> label string.
    """
    n_states = model.n_components
    means = model.means_  # shape (n_states, n_features), standardized scale

    # Un-standardize if stats are available
    if feature_means is not None and feature_stds is not None:
        raw_means = means * feature_stds + feature_means
    else:
        raw_means = means

    raw_returns = raw_means[:, 0]   # mean log return per state
    raw_vols = raw_means[:, 1]      # mean realized vol per state

    # Threshold: 0.5 std of the cross-state return spread
    ret_spread = np.std(raw_returns)
    ret_center = np.mean(raw_returns)
    bull_thresh = ret_center + 0.5 * ret_spread
    bear_thresh = ret_center - 0.5 * ret_spread

    vol_median = np.median(raw_vols)

    label_map: dict[int, str] = {}
    unlabeled = list(range(n_states))

    # Assign BULL / BEAR only to states clearly above/below threshold
    bull_candidates = [s for s in unlabeled if raw_returns[s] > bull_thresh]
    bear_candidates = [s for s in unlabeled if raw_returns[s] < bear_thresh]

    if bull_candidates:
        best_bull = max(bull_candidates, key=lambda s: raw_returns[s])
        label_map[best_bull] = "BULL"
        unlabeled.remove(best_bull)

    if bear_candidates:
        best_bear = min(bear_candidates, key=lambda s: raw_returns[s])
        label_map[best_bear] = "BEAR"
        unlabeled.remove(best_bear)

    # Among remaining, highest volatility -> HIGH_VOL (if above median)
    if unlabeled:
        high_vol_candidate = max(unlabeled, key=lambda s: raw_vols[s])
        if raw_vols[high_vol_candidate] > vol_median:
            label_map[high_vol_candidate] = "HIGH_VOL"
            unlabeled.remove(high_vol_candidate)

    # Everything else -> NEUTRAL
    for s in unlabeled:
        label_map[s] = "NEUTRAL"

    return label_map


# ─── BSTS FORECASTING ────────────────────────────────────────────────


def fit_bsts(
    close_prices: pd.Series,
) -> object:
    """Fit a BSTS model using statsmodels UnobservedComponents.

    Args:
        close_prices: Series of close prices with DatetimeIndex.

    Returns:
        Fitted UnobservedComponents result object.
    """
    model = UnobservedComponents(
        close_prices,
        level="local linear trend",
    )
    result = model.fit(disp=False)
    return result


def forecast_bsts(
    fitted_model,
    horizon: int = BSTS_FORECAST_HORIZON,
    ci_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Generate probabilistic forecast with confidence intervals.

    Args:
        fitted_model: Result from fit_bsts().
        horizon: Number of steps to forecast.
        ci_levels: List of CI levels (e.g., [0.50, 0.90]).

    Returns:
        DataFrame with columns:
            [mean, ci_50_lower, ci_50_upper, ci_90_lower, ci_90_upper]
        and a DatetimeIndex extending beyond the training data.
    """
    if ci_levels is None:
        ci_levels = BSTS_CI_LEVELS

    forecast = fitted_model.get_forecast(steps=horizon)

    result = pd.DataFrame(index=forecast.predicted_mean.index)
    result["mean"] = forecast.predicted_mean

    for level in sorted(ci_levels):
        alpha = 1.0 - level
        ci = forecast.conf_int(alpha=alpha)
        pct = int(level * 100)
        result[f"ci_{pct}_lower"] = ci.iloc[:, 0]
        result[f"ci_{pct}_upper"] = ci.iloc[:, 1]

    # Ensure forecast has a proper DatetimeIndex (statsmodels may return
    # a numeric RangeIndex when it can't infer frequency from 1-min data).
    if not isinstance(result.index, pd.DatetimeIndex):
        training_index = fitted_model.model.data.dates
        if training_index is not None and len(training_index) > 1:
            last_ts = training_index[-1]
            freq = training_index[-1] - training_index[-2]
            result.index = pd.date_range(
                start=last_ts + freq,
                periods=horizon,
                freq=freq,
            )
        else:
            # Fallback: 1-minute intervals from now
            result.index = pd.date_range(
                start=pd.Timestamp.now(tz="UTC"),
                periods=horizon,
                freq="min",
            )

    return result
