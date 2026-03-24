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
    "BULL": "rgba(0, 200, 0, 0.08)",
    "NEUTRAL": "rgba(128, 128, 128, 0.08)",
    "BEAR": "rgba(200, 0, 0, 0.08)",
    "HIGH_VOL": "rgba(255, 165, 0, 0.08)",
}
