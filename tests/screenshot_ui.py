"""Playwright script to screenshot the Streamlit app for UI/UX review."""

import subprocess
import sys
import time

import requests
from playwright.sync_api import sync_playwright


def main():
    env = {**subprocess.os.environ, "MOCK_DATA": "1"}
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8598",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    url = "http://localhost:8598"
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
        print("ERROR: Streamlit server did not start")
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.goto(url, timeout=120000)

            # Wait for app to finish loading
            page.wait_for_function(
                """() => {
                    const app = document.querySelector('[data-testid="stApp"]');
                    if (!app) return false;
                    return app.getAttribute('data-test-script-state') === 'notRunning';
                }""",
                timeout=120000,
            )
            # Extra wait for charts to render
            page.wait_for_timeout(3000)

            # Full page screenshot
            page.screenshot(path="tests/screenshot_full.png", full_page=True)
            # Viewport screenshot
            page.screenshot(path="tests/screenshot_viewport.png")

            print("Screenshots saved to tests/screenshot_full.png and tests/screenshot_viewport.png")
            browser.close()
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
