# **Product Requirements Document (PRD): Macro Tactical Cockpit V1 (MVP)**

## **1\. Executive Summary**

The Macro Tactical Cockpit V1 is a local-first, Python-based Streamlit application engineered to digitize discretionary macro-trading methodologies. By adhering strictly to "Informed Simplicity," this MVP focuses entirely on building a highly performant, in-memory mathematical kernel. It provides a unified, quantitative view of a single "Anchor Asset," contextualized by cross-asset lead-lag relationships, Gaussian Hidden Markov Model (HMM) regime probabilities, and Bayesian probabilistic forecasting. Complex UI overlays, alerting systems, and narrative intelligence features are deferred to ensure the core data and quantitative pipelines are flawless.

## **2\. Scope Definition**

### **In-Scope for V1 MVP**

* Local execution environment (monolithic Python process).  
* Ingestion and memory management of 1-minute OHLCV data for a defined asset universe of continuous-session instruments (e.g., ES, NQ, ZB, CL, GC, BTC1!, DXY, MOVE, VIX). Natural Gas (NG) and Silver (SI) are excluded from V1. Bank equities (GS, JPM) are deferred to V2 pending resolution of the trading hours mismatch.
* Mathematical engines operating strictly in-memory using vectorized operations:  
  * Gaussian HMM (Regime Classification).  
  * EWMA (Dynamic Correlation).  
  * Variable-lag Cross-Correlation (Lead-Lag).  
  * BSTS (Probabilistic Forecasting).  
* "Tactical Cockpit" UI utilizing Streamlit and Plotly for core visual outputs.  
* Comprehensive test-driven development (TDD) workflow for autonomous coding execution.

### **Out-of-Scope for V1 MVP**

* Large Language Model (LLM) API integrations, sentiment Z-scoring, and "News Failure" algorithms.  
* Intermarket Network Graphs, 5:1 Reward-to-Risk UI overlays, and Correlation Velocity Alerts.  
* Cloud databases, Kubernetes clusters, or microservices.  
* Millisecond tick-level data processing and full order-book depth.
* Bank equities (GS, JPM) — deferred to V2 pending session-aware alignment logic.

## ---

**3\. Core Functional Requirements**

### **3.1 Data Ingestion & State Management**

* **Requirement:** The system must fetch, buffer, and align 1-minute OHLCV data for the 15-asset universe.
* **Execution:** Data must be stored in memory using vectorized Pandas DataFrames.
* **Constraint:** Implement a strict rolling window mechanism (e.g., keeping only the last $N$ periods) to prevent memory leaks during continuous local operation. Handle missing data via forward-filling up to a maximum threshold before dropping the asset from the current calculation window.
* **Deferred — Trading Hours Mismatch:** Bank equities (GS, JPM) were considered but removed from V1. Futures and crypto trade ~23 hours/day; equities only trade 9:30 AM–4:00 PM EST. Mixing the two session types introduces stale-data artifacts in cross-asset correlation calculations. V1 avoids this problem entirely by restricting the universe to continuous-session instruments. GS and JPM will be reintroduced in V2 with explicit session-aware alignment logic.

### **3.2 The Quantitative Engine**

* **EWMA Dynamic Correlation:** Calculate rolling Exponentially Weighted Moving Average (EWMA) correlations across the asset universe against the selected Anchor Asset.  
* **Lead-Lag Engine:** Implement a cross-correlation function with variable lags to mathematically determine if an auxiliary asset (the "tail") is leading the Anchor Asset (the "dog").  
* **HMM Regime Detection:** Implement a 4-state **unsupervised** Gaussian Hidden Markov Model trained on the Anchor Asset's log returns and realized volatility. The model outputs raw states (0, 1, 2, 3) and a probability vector — it does not hardcode qualitative labels. A separate **dynamic mapping function** inspects how the Anchor Asset's EWMA correlations are behaving within each discovered state (e.g., equity/bond decoupling, commodity convergence) and assigns human-readable labels at runtime. The canonical reference labels are **BULL**, **NEUTRAL**, **BEAR**, and **HIGH VOL**, but the mapping is data-driven and reconfigurable without retraining the kernel. The UI must surface the full probability vector, not a single hard classification, to reflect ambiguity and regime transitions.

### **3.3 The Tactical Cockpit UI**

* **BSTS Probabilistic Fan Charts:** Utilize Bayesian Structural Time Series to generate forward-looking price projections. The core visualization must be a Plotly Candlestick chart overlaid with these projections, rendering shaded 50% and 90% confidence interval "cones."  
* **Regime Visualization:** The Plotly chart background or a prominent indicator must dynamically change color to reflect the active mathematical state outputted by the HMM.  
* **Analytical Sidebar:** A streamlined panel to handle Anchor Asset selection and display a ranked list of the top lead-lag correlation drivers.

## ---

**4\. Technical Architecture**

* **Project Setup:** uv + pyproject.toml (Python 3.12+)
* **Framework:** Streamlit (Frontend/Routing)
* **Data Source:** yfinance (1-minute OHLCV; 7-day rolling window max)
* **Data Manipulation:** pandas, numpy
* **Quantitative Models:** scipy (Signal processing/Cross-correlation), hmmlearn (Regime detection), statsmodels UnobservedComponents (BSTS implementation)
* **Visualization:** plotly (Interactive charting)
* **Testing:** pytest, pytest-playwright (E2E)

*Note on Extensibility:* The data pipelines and state management must be constructed with modular decoupling to allow the seamless reintroduction of the Narrative Intelligence Pipeline (LLM API calls) and websocket streaming in V2 without rewriting the core mathematical kernel.

### **Ticker Mapping (yfinance)**

| PRD Name | yfinance Ticker | Notes |
|----------|----------------|-------|
| ES | ES=F | E-mini S&P 500 continuous front-month |
| NQ | NQ=F | E-mini Nasdaq 100 |
| ZB | ZB=F | 30-Year Treasury Bond |
| CL | CL=F | Crude Oil WTI |
| GC | GC=F | Gold |
| BTC | BTC-USD | Bitcoin spot proxy (BTC1! is TradingView notation) |
| DXY | DX-Y.NYB | US Dollar Index |
| MOVE | ^MOVE | ICE BofA MOVE Index (1m data may be unavailable) |
| VIX | ^VIX | CBOE Volatility Index |

## ---

**5\. Implementation & Validation Plan**

This plan is structured for execution by an autonomous coding agent (Claude Code) utilizing a strict test-driven workflow. Every component must be validated before moving to the next.

### **Step 1: Data Pipeline & Memory Management**

* **Target Files:** src/data.py, tests/test\_data.py  
* **Task:** Build the 1m OHLCV fetcher and rolling-window DataFrame manager.  
* **Validation Gate:** Write unit tests to assert that the DataFrame maintains its maximum row limit after sequential updates and that missing data is correctly forward-filled.  
* **Pass Command:** pytest tests/test\_data.py \-v

### **Step 2: Correlation & Lead-Lag Engines**

* **Target Files:** src/quant.py, tests/test\_quant.py  
* **Task:** Implement the EWMA correlation matrix and the variable-lag cross-correlation logic.  
* **Validation Gate:** Feed deterministic, synthetic sine-wave data with a known lag offset into the engine. Assert that the engine correctly identifies the lagging asset and the exact period offset.  
* **Pass Command:** pytest tests/test\_quant.py \-k "test\_lead\_lag" \-v

### **Step 3: HMM & BSTS Modeling**

* **Target Files:** src/models.py, tests/test\_models.py  
* **Task:** Implement the 4-state unsupervised Gaussian HMM, the dynamic label-mapping function, and the BSTS forecasting model.
* **Validation Gate:** Assert the HMM outputs exactly 4 raw states (0–3) and that the probability vector sums to 1.0. Assert the dynamic mapping function returns a string label for each state given a mock EWMA correlation matrix. Assert the BSTS outputs upper and lower bounds for the 50% and 90% intervals.  
* **Pass Command:** pytest tests/test\_models.py \-v

### **Step 4: UI Construction (Plotly Integration)**

* **Target Files:** app.py, src/visuals.py, tests/test\_visuals.py  
* **Task:** Assemble the Streamlit layout. Create build\_main\_chart(ohlcv, fan\_data, regime) to construct the Plotly go.Figure.  
* **Validation Gate:** Write unit tests to assert the generated go.Figure object contains the correct number of traces (1 for candles, multiple for the BSTS confidence interval bands).  
* **Pass Command:** pytest tests/test\_visuals.py \-v

### **Step 5: End-to-End Validation via Playwright**

* **Target Files:** tests/test\_e2e\_ui.py  
* **Dependencies:** pytest-playwright  
* **Task:** Write an asynchronous script to spawn the Streamlit process, navigate to localhost, interact with the sidebar to change the Anchor Asset, and verify the Plotly canvas renders in the DOM without JavaScript console errors.  
* **Validation Gate:** The Playwright browser successfully automates the UI, confirms the data load, and tears down the Streamlit server cleanly.  
* **Pass Command:** pytest tests/test\_e2e\_ui.py \-v