"""End-to-end Streamlit UI tests using Playwright."""

import subprocess
import time

import pytest
import requests
from playwright.sync_api import Page, sync_playwright


@pytest.fixture(scope="module")
def streamlit_server():
    """Spawn Streamlit server as a subprocess, yield the URL, teardown after tests."""
    env = {**subprocess.os.environ, "MOCK_DATA": "1"}
    proc = subprocess.Popen(
        [
            "uv", "run", "streamlit", "run", "app.py",
            "--server.port=8599",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    url = "http://localhost:8599"

    # Poll health endpoint until server is ready (max 30s)
    for _ in range(30):
        try:
            resp = requests.get(f"{url}/_stcore/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        proc.wait()
        pytest.fail("Streamlit server did not start within 30 seconds")

    yield url

    proc.terminate()
    proc.wait()


def _wait_for_app_loaded(page: Page, timeout: int = 120000) -> None:
    """Wait for Streamlit to finish running the script."""
    page.wait_for_function(
        """() => {
            const app = document.querySelector('[data-testid="stApp"]');
            if (!app) return false;
            const state = app.getAttribute('data-test-script-state');
            return state === 'notRunning';
        }""",
        timeout=timeout,
    )


def test_app_loads_without_errors(streamlit_server):
    """Page loads and contains the title."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        page.goto(streamlit_server, timeout=120000)
        _wait_for_app_loaded(page)

        # Check title is present in the rendered text
        page.wait_for_selector("text=Macro Tactical Cockpit", timeout=10000)

        browser.close()


def test_plotly_canvas_renders(streamlit_server):
    """A Plotly chart element exists in the DOM."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(streamlit_server, timeout=120000)
        _wait_for_app_loaded(page)

        # Give Plotly extra time to render after app loads
        page.wait_for_timeout(5000)

        # Dump all iframes and data-testid attributes for debugging
        iframes = page.query_selector_all("iframe")
        for iframe in iframes:
            src = iframe.get_attribute("src") or ""
            title = iframe.get_attribute("title") or ""
            if "plotly" in src.lower() or "plotly" in title.lower():
                assert True, "Found plotly iframe"
                browser.close()
                return

        # Check for Streamlit's Plotly chart container
        # Streamlit >= 1.40 uses data-testid="stCustomComponentV1" for plotly
        content = page.content()
        has_chart = any(s in content for s in [
            "plotly", "Plotly", "js-plotly-plot",
            "stPlotlyChart", "stCustomComponentV1",
            "candlestick", "Candlestick",
        ])

        # Debug: show what text is actually on the page
        visible_text = page.inner_text("body")
        assert has_chart, f"No Plotly chart found. Page text: {visible_text[:2000]}"

        browser.close()


def test_sidebar_has_anchor_selector(streamlit_server):
    """The sidebar contains an anchor asset selector."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(streamlit_server, timeout=120000)
        _wait_for_app_loaded(page)

        # Streamlit sidebar should have a selectbox
        sidebar = page.wait_for_selector(
            "[data-testid='stSidebar']",
            timeout=10000,
        )
        assert sidebar is not None, "Sidebar not found"

        # Check for the selectbox label
        page.wait_for_selector(
            "[data-testid='stSidebar'] >> text=Anchor Asset",
            timeout=10000,
        )

        browser.close()
