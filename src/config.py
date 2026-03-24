"""Central configuration for Macro Tactical Cockpit."""

# === TICKER MAPPING ===
# PRD name -> yfinance ticker
# BTC1! is TradingView notation; yfinance uses BTC-USD for spot
# DXY is DX-Y.NYB on Yahoo Finance
# MOVE is ^MOVE (1m data may be unavailable)
# VIX is ^VIX (cash index, volume will be 0)
TICKER_MAP: dict[str, str] = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "ZB": "ZB=F",
    "CL": "CL=F",
    "GC": "GC=F",
    "BTC": "BTC-USD",
    "DXY": "DX-Y.NYB",
    "MOVE": "^MOVE",
    "VIX": "^VIX",
}

ASSET_UNIVERSE: list[str] = list(TICKER_MAP.keys())

# Human-readable display names: "Full Name (SHORT)"
ASSET_DISPLAY_NAMES: dict[str, str] = {
    "ES": "S&P 500 Futures (ES)",
    "NQ": "Nasdaq 100 Futures (NQ)",
    "ZB": "30-Year Treasury Futures (ZB)",
    "CL": "Crude Oil Futures (CL)",
    "GC": "Gold Futures (GC)",
    "BTC": "Bitcoin (BTC)",
    "DXY": "US Dollar Index (DXY)",
    "MOVE": "Bond Volatility Index (MOVE)",
    "VIX": "Equity Volatility Index (VIX)",
}

# === DATA PARAMETERS ===
ROLLING_WINDOW_BARS: int = 7 * 23 * 60  # ~7 days of 1-min bars (~9660)
MAX_FFILL_BARS: int = 5  # forward-fill up to 5 missing bars
FETCH_PERIOD: str = "7d"  # yfinance max for 1m data
FETCH_INTERVAL: str = "1m"

# === QUANT PARAMETERS ===
EWMA_SPAN: int = 60  # 60-bar EWMA half-life
MAX_LAG_BARS: int = 30  # cross-correlation search up to +/- 30 bars

# === MODEL PARAMETERS ===
HMM_N_STATES: int = 4
HMM_COVARIANCE_TYPE: str = "full"
HMM_N_ITER: int = 100
BSTS_FORECAST_HORIZON: int = 60  # 60 bars forward (~1 hour)
BSTS_CI_LEVELS: list[float] = [0.50, 0.90]

# === REGIME LABELS (canonical reference, dynamically mapped) ===
REGIME_LABELS: list[str] = ["BULL", "NEUTRAL", "BEAR", "HIGH_VOL"]
REGIME_COLORS: dict[str, str] = {
    "BULL": "rgba(0, 200, 0, 0.12)",
    "NEUTRAL": "rgba(128, 128, 128, 0.12)",
    "BEAR": "rgba(200, 0, 0, 0.12)",
    "HIGH_VOL": "rgba(255, 165, 0, 0.12)",
}

# === VISUAL PARAMETERS ===
PLOTLY_TEMPLATE: str = "plotly_white"

REGIME_BADGE_COLORS: dict[str, tuple[str, str]] = {
    "BULL": ("#065f46", "#d1fae5"),
    "BEAR": ("#7f1d1d", "#fecaca"),
    "HIGH_VOL": ("#78350f", "#fed7aa"),
    "NEUTRAL": ("#374151", "#e5e7eb"),
}

FRESHNESS_STALE_MINUTES: int = 30
