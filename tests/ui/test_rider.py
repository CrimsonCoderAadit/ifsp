"""Browser smoke checks on real Commons inputs; set BUS_FINDER_URL to run."""
import os
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
URL = os.getenv("BUS_FINDER_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Set BUS_FINDER_URL to a running local server")


@pytest.mark.parametrize("width", [375, 1280])
@pytest.mark.parametrize("filename,route", [
    ("own/bus_102a.png", "102A"),
    ("commons/MTC_Bus_21G.JPG", "21G"),
    ("commons/Front_profile_chennai_bus.JPG", "D70"),
    ("commons/Metropolitan_Transport_Corporation_Chennai_bus_1.JPG", "22"),
    ("commons/Chennai_bus.jpg", None),
    ("commons/MTC-ticket.jpg", None),
])
def test_real_upload_result(width, filename, route):
    path = ROOT / "data/real" / filename
    if not path.exists():
        pytest.skip(f"Real photo missing: {filename}")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 900})
        page.goto(URL, wait_until="networkidle")
        page.locator("#upload").set_input_files(str(path))
        page.wait_for_function("document.querySelector('#detected').textContent !== '…'", timeout=45000)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        if route:
            assert "✓" in page.locator("#detected").inner_text()
            assert route in page.locator("#route").inner_text()
            assert page.locator("#ocr").inner_text() != "…"
            assert page.locator("#stops").is_visible()
            directions = page.locator("#directions > .direction")
            assert directions.count() == 2
            for index in range(2):
                direction = directions.nth(index)
                assert f"Direction {index + 1}" in direction.locator("summary").text_content()
                assert direction.locator("li").count() > 5
        else:
            assert page.locator("#route").inner_text() == "✗"
            assert not page.locator("#stops").is_visible()
        out = ROOT / "docs" / f"rider-{path.stem}-{width}.png"
        page.screenshot(path=str(out), full_page=True)
        browser.close()
