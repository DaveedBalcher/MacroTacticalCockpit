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
) -> np.ndarray:
    """Prepare feature matrix for HMM: [log_returns, realized_volatility].

    Args:
        ohlcv: OHLCV DataFrame for the anchor asset.
        vol_window: Rolling window for realized volatility.

    Returns:
        2D numpy array of shape (T, 2), NaN-free.
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
    values = (values - means) / stds

    return values


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
    ewma_corr_matrix: pd.DataFrame | None = None,
) -> dict[int, str]:
    """Dynamically assign human-readable labels to raw HMM states.

    Strategy: Sort states by their learned mean log-return (first feature):
      - Highest mean return -> BULL
      - Lowest mean return -> BEAR
      - Among remaining, highest mean volatility -> HIGH_VOL
      - Remaining -> NEUTRAL

    Returns:
        Dict mapping raw state int -> label string.
    """
    n_states = model.n_components
    means = model.means_  # shape (n_states, n_features)

    # Mean log return is first feature
    mean_returns = means[:, 0]
    # Mean realized vol is second feature
    mean_vols = means[:, 1]

    # Sort states by mean return
    sorted_by_return = np.argsort(mean_returns)

    label_map: dict[int, str] = {}
    label_map[int(sorted_by_return[-1])] = "BULL"  # highest return
    label_map[int(sorted_by_return[0])] = "BEAR"   # lowest return

    # Among the remaining states, assign HIGH_VOL to highest volatility
    remaining = [int(s) for s in sorted_by_return[1:-1] if int(s) not in label_map]

    if remaining:
        vol_order = sorted(remaining, key=lambda s: mean_vols[s], reverse=True)
        label_map[vol_order[0]] = "HIGH_VOL"
        for s in vol_order[1:]:
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
