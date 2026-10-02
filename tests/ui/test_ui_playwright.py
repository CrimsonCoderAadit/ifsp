"""Browser tests for the web UI against the real server and local models.

Run: python -m pytest tests/ui -q   (needs: pip install playwright; python -m playwright install chromium)
Set UI_RECORD_DEMO=1 to also record logs/ui/demo.webm.
"""
import json
import os
import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / "logs/ui"
AXE = Path(__file__).with_name("vendor") / "axe.min.js"
SIZES = [(1440, 900), (1024, 768), (375, 812)]
THEMES = ["dark", "light", "hc"]
MODES = ["rider", "engineer"]

pytestmark = pytest.mark.ui


def no_horizontal_scroll(page):
    return page.evaluate("() => document.documentElement.scrollWidth <= document.documentElement.clientWidth")


@pytest.mark.parametrize("width,height", SIZES)
def test_layouts_have_no_horizontal_scroll_in_every_mode_and_theme(session, width, height):
    SHOTS.mkdir(parents=True, exist_ok=True)
    ui = session(width=width, height=height)
    expected_default = "rider" if width < 768 else "engineer"
    expect(ui.page.locator("body")).to_have_attribute("data-mode", expected_default)
    assert no_horizontal_scroll(ui.page), f"start overlay scrolls sideways at {width}px"
    ui.start()
    ui.command("find bus 21G")
    ui.sample(0)
    ui.wait_confirmed()
    for drawer in ("#ocrDrawer", "#logDrawer", ".system-card", ".type-instead"):
        ui.page.locator(drawer).evaluate("element => element.open = true")
    for mode in MODES:
        ui.mode(mode)
        for theme in THEMES:
            ui.theme(theme)
            ui.page.evaluate("() => window.scrollTo(0, 0)")
            ui.page.wait_for_timeout(250)
            sizes = ui.page.evaluate("() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]")
            ui.page.screenshot(path=str(SHOTS / f"{width}x{height}-{mode}-{theme}.png"), full_page=True)
            assert sizes[0] <= sizes[1], f"{width}px {mode} {theme}: scrollWidth {sizes[0]} > clientWidth {sizes[1]}"
    assert not ui.errors, ui.errors


def test_typed_command_and_board_confirm_bus_through_every_stage(session):
    ui = session()
    ui.start()
    ui.command("find bus 21G")
    expect(ui.page.locator("#ledRoute")).to_have_text("21G")
    expect(ui.page.locator("#ledMarquee .marquee-copy").first).to_contain_text("→")
    ui.sample(0)
    expect(ui.page.locator("#boardMode")).to_be_checked()
    ui.wait_confirmed()
    expect(ui.page.locator("[data-node=tts] .node-value")).to_contain_text("Bus 21G detected", timeout=15_000)
    nodes = ui.page.locator(".node").evaluate_all(
        "nodes => nodes.map(n => ({name: n.dataset.node, state: n.dataset.state, value: n.querySelector('.node-value').textContent}))")
    assert [node["name"] for node in nodes] == ["mic", "stt", "intent", "detect", "ocr", "normalise", "gtfs", "scheduler", "tts"]
    for node in nodes:
        assert node["state"] in {"lit", "done"}, node
        assert node["value"] not in {"", "—"}, node
    by_name = {node["name"]: node["value"] for node in nodes}
    assert by_name["intent"] == "FIND_BUS (21G)"
    assert by_name["ocr"] == "“21 G”" and by_name["normalise"] == "21G"
    assert by_name["gtfs"].startswith("21G")
    first_attempt = ui.page.locator("#ocrDrawer")
    first_attempt.evaluate("element => element.open = true")
    expect(ui.page.locator("#ocrRows tr").first).to_contain_text("full")
    expect(ui.page.locator("#ocrRows tr.winner")).to_contain_text("21G")
    assert not ui.errors, ui.errors


def test_high_risk_preempts_bus_and_bus_replays(session):
    ui = session()
    ui.start()
    ui.command("find bus 21G")
    expect(ui.page.locator("#lastSpoken")).to_have_text("Looking for bus 21G.", timeout=10_000)
    ui.page.wait_for_timeout(2500)
    ui.sample(0)
    expect(ui.page.locator("#lastSpoken")).to_have_attribute("data-priority", "bus", timeout=15_000)
    ui.page.locator("#riskFire").click()
    banner = ui.page.locator("#riskBanner")
    expect(banner).to_be_visible(timeout=3000)
    expect(banner).to_have_attribute("data-direction", "right")
    expect(ui.page.locator("#riskText")).to_have_text("Vehicle approaching right.")
    expect(ui.page.locator("#riskArrowLabel")).to_have_text("Right")
    rotation = ui.page.locator(".risk-arrow .icon").evaluate("icon => getComputedStyle(icon).transform")
    assert rotation == "matrix(0, 1, -1, 0, 0, 0)", f"arrow should point right (rotate 90deg), got {rotation}"
    expect(ui.page.locator("#riskLive")).to_have_text("HIGH warning: Vehicle approaching right.")
    expect(banner).to_be_hidden(timeout=8000)
    expect(ui.page.locator("#lastSpoken")).to_have_text("Bus 21G detected ahead.", timeout=8000)
    ui.mode("engineer")
    expect(ui.page.locator("#timeline")).to_have_attribute("data-preempted", "1", timeout=5000)
    expect(ui.page.locator("#timeline")).to_have_attribute("data-requeued", "1", timeout=5000)
    expect(ui.page.locator("#timelineSummary")).to_contain_text("1 pre-empted, 1 requeued")
    assert not ui.errors, ui.errors


def test_spam_is_suppressed_by_cooldown(session):
    ui = session()
    ui.start()
    ui.page.locator("#riskSpam").click()
    expect(ui.page.locator("#riskSpam")).to_be_disabled()
    expect(ui.page.locator("#riskSpam")).to_be_enabled(timeout=10_000)
    ui.page.wait_for_timeout(1500)
    data = ui.timeline()
    badges = json.loads(data["badges"])
    high = [badge for badge in badges if badge["lane"] == "HIGH"]
    assert high and high[0]["count"] > 50, badges
    assert high[0]["text"] == f"×{high[0]['count']} suppressed"
    assert int(data["spoken"]) <= 2, data
    assert not ui.errors, ui.errors


FOCUS_CHECK = """element => {
  const proxy = element.matches('input') && getComputedStyle(element).opacity === '0' ? element.nextElementSibling : element;
  const style = getComputedStyle(proxy);
  const box = proxy.getBoundingClientRect();
  const label = element.id || element.getAttribute('aria-label') || element.textContent.trim().slice(0, 30) || element.tagName;
  return {label: `${element.tagName.toLowerCase()} ${label}`,
          visible: box.width > 0 && box.height > 0 && style.visibility !== 'hidden',
          ring: style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) >= 2,
          inViewport: box.bottom > 0 && box.top < innerHeight};
}"""


def tab_cycle(page, limit=250):
    page.evaluate("() => { document.activeElement?.blur(); window.scrollTo(0, 0); }")
    seen, problems = [], []
    for _ in range(limit):
        page.keyboard.press("Tab")
        handle = page.evaluate_handle("() => document.activeElement")
        if page.evaluate("element => element === document.body", handle):
            continue
        info = page.evaluate(FOCUS_CHECK, handle)
        if seen and info["label"] == seen[0]["label"]:
            break
        seen.append(info)
        if page.evaluate("element => element.tagName === 'SUMMARY' && !element.parentElement.open", handle):
            page.keyboard.press("Enter")
        if not (info["visible"] and info["ring"] and info["inViewport"]):
            problems.append(info)
    return seen, problems


@pytest.mark.parametrize("mode", MODES)
def test_keyboard_tab_order_shows_focus_on_every_control(session, mode):
    ui = session()
    expect(ui.page.locator("#startButton")).to_be_focused()
    ui.page.keyboard.press("Tab")
    expect(ui.page.locator("#startButton")).to_be_focused()
    ui.page.keyboard.press("Enter")
    expect(ui.page.locator("#connection")).to_have_attribute("data-state", "connected", timeout=10_000)
    ui.page.evaluate("() => document.activeElement.blur()")
    if ui.page.locator("body").get_attribute("data-mode") != mode:
        ui.page.keyboard.press("m")
    expect(ui.page.locator("body")).to_have_attribute("data-mode", mode)
    tab_cycle(ui.page)  # first lap opens each collapsed drawer with Enter
    seen, problems = tab_cycle(ui.page)
    assert len(seen) >= (30 if mode == "engineer" else 12), [item["label"] for item in seen]
    assert not problems, problems
    labels = " | ".join(item["label"] for item in seen)
    print(f"\n{mode}: {len(seen)} tab stops: {labels}")
    for required in ("ptt", "theme", "helpButton", "Repeat") + (("riskFire", "riskSpam", "boardMode") if mode == "engineer" else ()):
        assert required in labels, f"{required} not reachable by Tab: {labels}"


def test_every_shortcut_works_from_the_keyboard(session):
    ui = session()
    page = ui.page
    page.keyboard.press("Enter")
    expect(page.locator("#connection")).to_have_attribute("data-state", "connected", timeout=10_000)
    page.evaluate("() => document.activeElement.blur()")
    keyboard = page.keyboard

    keyboard.down(" ")
    expect(page.locator("#ptt")).to_have_attribute("aria-pressed", "true", timeout=5000)
    page.wait_for_timeout(3000)
    keyboard.up(" ")
    expect(page.locator("#transcript")).to_contain_text("find bus twenty one", timeout=10_000)
    expect(page.locator("#intentLine")).to_contain_text("FIND_BUS 21G")
    expect(page.locator("#ledRoute")).to_have_text("21G")
    expect(page.locator("[data-metric=stt] .big-value")).not_to_have_text("—")

    for key, expected in (("a", "WHATS_AHEAD"), ("r", "REPEAT"), ("s", "STOP"), ("c", "CANCEL_BUS")):
        keyboard.press(key)
        expect(page.locator("#intentLine")).to_contain_text(expected, timeout=5000)
    expect(page.locator("#ledRoute")).to_have_text("- - -")

    start_mode = page.locator("body").get_attribute("data-mode")
    keyboard.press("m")
    expect(page.locator("body")).not_to_have_attribute("data-mode", start_mode)
    keyboard.press("m")
    expect(page.locator("body")).to_have_attribute("data-mode", start_mode)

    keyboard.press("h")
    expect(page.locator("html")).to_have_attribute("data-theme", "hc")
    keyboard.press("h")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")

    keyboard.press("p")
    expect(page.locator("#pauseFrames")).to_have_attribute("aria-pressed", "true")
    keyboard.press("p")
    expect(page.locator("#pauseFrames")).to_have_attribute("aria-pressed", "false")

    keyboard.press("1")
    expect(page.locator("#lastSpoken")).to_have_attribute("data-priority", "info", timeout=8000)
    keyboard.press("2")
    expect(page.locator("#riskLevelLabel")).to_have_text("HIGH", timeout=5000)
    expect(page.locator("#riskBanner")).to_be_hidden(timeout=8000)
    keyboard.press("3")
    expect(page.locator("#riskLevelLabel")).to_have_text("CRITICAL", timeout=5000)

    keyboard.press("?")
    expect(page.locator("#helpDialog")).to_be_visible()
    expect(page.locator("#shortcutTable tr")).to_have_count(12)
    keyboard.press("Escape")
    expect(page.locator("#helpDialog")).to_be_hidden()
    assert not ui.errors, ui.errors


def axe_violations(page):
    page.add_script_tag(path=str(AXE))
    return page.evaluate("""async () => {
      const result = await axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']}});
      return result.violations.map(v => ({id: v.id, impact: v.impact, help: v.help,
        nodes: v.nodes.slice(0, 4).map(n => n.target.join(' ') + ' :: ' + n.failureSummary)}));
    }""")


@pytest.mark.skipif(not AXE.exists(), reason="Run scripts/fetch_axe.py once to vendor axe-core")
@pytest.mark.parametrize("theme", THEMES)
def test_axe_has_no_serious_violations(session, theme):
    ui = session()
    ui.page.evaluate("theme => localStorage.setItem('theme', theme)", theme)
    ui.page.reload()
    expect(ui.page.locator("html")).to_have_attribute("data-theme", theme)
    results = {"start overlay": axe_violations(ui.page)}
    ui.start()
    ui.command("find bus 21G")
    ui.sample(0)
    ui.wait_confirmed()
    for drawer in ("#ocrDrawer", "#logDrawer", ".system-card"):
        ui.page.locator(drawer).evaluate("element => element.open = true")
    ui.page.wait_for_timeout(600)
    for mode in MODES:
        ui.mode(mode)
        results[mode] = axe_violations(ui.page)
    serious = {where: [v for v in found if v["impact"] in {"serious", "critical"}] for where, found in results.items()}
    report = SHOTS / f"axe-{theme}.json"
    SHOTS.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    assert not any(serious.values()), json.dumps(serious, indent=2, ensure_ascii=False)


CAPTION = """text => {
  let box = document.getElementById('demoCaption');
  if (!box) {
    box = document.createElement('div');
    box.id = 'demoCaption';
    box.setAttribute('aria-hidden', 'true');
    box.style.cssText = 'position:fixed;left:16px;bottom:16px;z-index:90;max-width:640px;padding:12px 18px;' +
      'background:#000;color:#fff;border:3px solid #ffb000;border-radius:12px;font:700 22px/1.3 system-ui';
    document.body.append(box);
  }
  box.textContent = text;
}"""


def scroll_to(page, selector, block="start"):
    page.locator(selector).evaluate(
        "(element, block) => element.scrollIntoView({behavior: 'smooth', block})", block)
    page.wait_for_timeout(900)


@pytest.mark.skipif(os.environ.get("UI_RECORD_DEMO") != "1", reason="Set UI_RECORD_DEMO=1 to record logs/ui/demo.webm")
def test_record_demo_video(session, tmp_path):
    ui = session(record_video_dir=str(tmp_path / "video"), record_video_size={"width": 1440, "height": 900})
    page = ui.page
    caption = lambda text: page.evaluate(CAPTION, text)
    started = time.monotonic()
    page.wait_for_timeout(1200)
    ui.start()
    ui.mode("engineer")
    caption("1 · Voice command: hold Space and say “find bus twenty one G”")
    page.wait_for_timeout(1200)
    page.evaluate("() => document.activeElement.blur()")
    page.keyboard.down(" ")
    page.wait_for_timeout(3000)
    page.keyboard.up(" ")
    expect(page.locator("#intentLine")).to_contain_text("FIND_BUS 21G", timeout=10_000)
    expect(page.locator("#lastSpoken")).to_have_text("Looking for bus 21G.", timeout=10_000)
    page.wait_for_timeout(2200)

    caption("2 · Camera sees a 21G board → OCR → GTFS match → confirmed")
    scroll_to(page, ".workbench")
    page.locator(".sample").nth(0).evaluate("button => button.click()")
    ui.wait_confirmed()
    expect(page.locator("#lastSpoken")).to_have_attribute("data-priority", "bus", timeout=15_000)
    page.wait_for_timeout(250)

    caption("3 · HIGH risk (car, right) interrupts the bus message")
    page.keyboard.press("2")
    expect(page.locator("#riskBanner")).to_be_visible(timeout=3000)
    page.wait_for_timeout(1500)

    caption("4 · Bus information resumes after the warning")
    scroll_to(page, ".spoken-card", "center")
    expect(page.locator("#lastSpoken")).to_have_text("Bus 21G detected ahead.", timeout=10_000)
    page.wait_for_timeout(2500)
    scroll_to(page, "#timeline", "center")
    caption("Timeline: bus bar cut ✕ by the warning, dashed line to its replay")
    expect(page.locator("#timeline")).to_have_attribute("data-preempted", "1", timeout=3000)
    expect(page.locator("#timeline")).to_have_attribute("data-requeued", "1", timeout=3000)
    page.wait_for_timeout(3500)

    caption("5 · 150 HIGH warnings in 5 s: cooldown speaks only a few")
    page.locator("#riskSpam").evaluate("button => button.click()")
    expect(page.locator("#riskSpam")).to_be_enabled(timeout=10_000)
    caption("5 · Spam done: suppressed by cooldown, see the ×N badge")
    remaining = 45 - (time.monotonic() - started)
    page.wait_for_timeout(max(1000, remaining * 1000))
    timeline = ui.timeline()
    assert any(badge["count"] > 50 for badge in json.loads(timeline["badges"])), timeline
    video = page.video
    ui.close()
    SHOTS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(video.path(), SHOTS / "demo.webm")
    assert (SHOTS / "demo.webm").stat().st_size > 100_000
