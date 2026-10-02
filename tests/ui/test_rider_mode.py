"""Rider-mode regressions: destinations, found state, status strip, risk banner timing, "Ready.".

Some tests proxy the WebSocket to strip additive fields, reproducing an older server.
"""
import json

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.ui

# Records speak/stop_audio received, controls sent, and banner visibility, with page timestamps.
INSTRUMENT = """
window.__events = [];
const NativeSocket = window.WebSocket;
window.WebSocket = class extends NativeSocket {
  constructor(...args) {
    super(...args);
    this.addEventListener('message', event => {
      if (typeof event.data !== 'string') return;
      const message = JSON.parse(event.data);
      if (message.type === 'speak' || message.type === 'stop_audio')
        window.__events.push({kind: 'in', type: message.type, id: message.id, priority: message.priority, text: message.text, t: performance.now()});
    });
  }
  send(data) {
    if (typeof data === 'string') {
      const message = JSON.parse(data);
      window.__events.push({kind: 'out', type: message.type, id: message.id, t: performance.now()});
    }
    return super.send(data);
  }
};
document.addEventListener('DOMContentLoaded', () => {
  const banner = document.getElementById('riskBanner');
  new MutationObserver(() => window.__events.push({kind: 'banner', hidden: banner.hidden, text: document.getElementById('riskText').textContent, t: performance.now()}))
    .observe(banner, {attributes: true, attributeFilter: ['hidden']});
});
"""


def rider(ui):
    ui.page.evaluate("() => document.activeElement.blur()")
    if ui.page.locator("body").get_attribute("data-mode") != "rider":
        ui.page.keyboard.press("m")
    expect(ui.page.locator("body")).to_have_attribute("data-mode", "rider")


def load_sample_then_ride(ui, index=0):
    ui.sample(index)
    rider(ui)


def proxy_websocket(page, rewrite):
    def handler(route):
        server = route.connect_to_server()
        route.on_message(lambda message: server.send(message))
        def downstream(message):
            if isinstance(message, str):
                payload = json.loads(message)
                rewrite(payload)
                message = json.dumps(payload)
            route.send(message)
        server.on_message(downstream)
    page.route_web_socket("**/ws", handler)


def strip_long_names(payload):
    for holder in (payload, payload.get("state") or {}):
        if isinstance(holder.get("target_bus"), dict):
            holder["target_bus"].pop("long_names", None)


def collect_state_messages(page, store):
    def on_socket(socket):
        socket.on("framereceived", lambda frame: store.append(json.loads(frame)) if isinstance(frame, str)
                  and frame.startswith('{"type":"state"') else None)
    page.on("websocket", on_socket)


# 1 — destinations and the endpoints behind them
def test_destination_comes_from_gtfs_and_state_carries_long_names(session):
    ui = session()
    states = []
    collect_state_messages(ui.page, states)
    ui.page.reload()
    ui.start()
    rider(ui)
    for path in ("/api/route/21G", "/api/health"):
        response = ui.page.request.get(ui.page.url.rstrip("/") + path)
        assert response.status == 200, path
    route = ui.page.request.get(ui.page.url.rstrip("/") + "/api/route/21G").json()
    assert route["route"] == "21G" and any("Kilambakkam" in name for name in route["long_names"])
    ui.command("find bus 21G")
    expect(ui.page.locator("#ledRoute")).to_have_text("21G")
    destination = ui.page.locator("#ledMarquee .marquee-copy").first
    expect(destination).to_contain_text("→")
    expect(destination).to_contain_text("Kilambakkam")
    expect(destination).not_to_contain_text("GTFS")
    ui.page.wait_for_timeout(800)
    targets = [message["target_bus"] for message in states if message.get("target_bus")]
    assert targets and targets[-1]["long_names"] == route["long_names"]


@pytest.mark.parametrize("api,expected", [
    ({"status": 200, "body": {"route": "21G", "variants": ["21G"], "long_names": ["Tambaram West TO Besant Nagar"]}},
     "Tambaram West → Besant Nagar"),
    ({"status": 500, "body": {"detail": "boom"}}, "Destination unavailable"),
    ({"status": 404, "body": {"detail": "Not Found"}}, "Destination unavailable"),
    ({"status": 404, "body": {"detail": "Unknown route 21G"}}, "Not in GTFS feed"),
    ({"status": 200, "body": {"route": "21G", "variants": ["21G"], "long_names": []}}, None),
])
def test_destination_fallbacks_when_state_lacks_long_names(session, api, expected):
    ui = session()
    proxy_websocket(ui.page, strip_long_names)
    ui.page.route("**/api/route/**", lambda route: route.fulfill(status=api["status"], json=api["body"]))
    ui.page.reload()
    ui.start()
    rider(ui)
    ui.command("find bus 21G")
    expect(ui.page.locator("#ledRoute")).to_have_text("21G")
    line = ui.page.locator(".led-dest")
    if expected is None:
        expect(line).to_be_hidden()
    else:
        expect(line).to_be_visible()
        expect(ui.page.locator("#ledMarquee .marquee-copy").first).to_have_text(expected)


# 2 — found state on the LED board
def test_found_state_announces_and_reverts_after_track_loss(session):
    ui = session()
    ui.start()
    ui.command("find bus 21G")
    status = ui.page.locator("#ledStatus")
    expect(status).to_have_text("Looking for bus 21G…")
    load_sample_then_ride(ui)
    expect(status).to_have_text("21G is here, ahead", timeout=15_000)
    expect(ui.page.locator("#ledBoard")).to_have_attribute("data-found", "true")
    expect(ui.page.locator("#ledBoard .found-badge")).to_be_visible()
    expect(ui.page.locator("#busInfo")).to_have_text("Bus 21G is here, ahead.")
    assert ui.page.locator("#busInfo").get_attribute("aria-live") == "polite"
    assert "found-pulse" in ui.page.locator("#ledBoard").get_attribute("class")
    ui.page.keyboard.press("p")
    ui.page.wait_for_timeout(3500)
    expect(status).to_have_text("21G is here, ahead")
    expect(status).to_have_text("Looking for bus 21G…", timeout=4000)
    expect(ui.page.locator("#ledBoard")).to_have_attribute("data-found", "false")
    expect(ui.page.locator("#busInfo")).to_have_text("Lost sight of bus 21G. Looking again.")


# 3 — camera status strip
def strip_status(payload):
    strip_long_names(payload)
    if payload.get("type") == "overlay":
        payload.pop("status", None)
        payload.pop("status_text", None)


@pytest.mark.parametrize("old_server", [False, True])
def test_status_strip_follows_frames_with_and_without_overlay_status(session, old_server):
    ui = session()
    if old_server:
        proxy_websocket(ui.page, strip_status)
        ui.page.reload()
    ui.start()
    ui.command("find bus 21G")
    strip = ui.page.locator("#statusStrip")
    expect(strip).to_contain_text("No frames yet")
    load_sample_then_ride(ui)
    expect(strip).to_contain_text("21G confirmed — your bus", timeout=15_000)
    expect(strip).to_have_attribute("data-kind", "ok")
    for _ in range(6):
        ui.page.wait_for_timeout(250)
        assert "No frames yet" not in strip.inner_text()


# 4 — risk banner appears on send, slides in ≤150 ms, stays until playback_done
def time_banner_after_key(page, key):
    return page.evaluate("""async key => {
      const banner = document.getElementById('riskBanner');
      const started = performance.now();
      document.dispatchEvent(new KeyboardEvent('keydown', {key, code: 'Digit' + key, bubbles: true}));
      let shown = null;
      while (performance.now() - started < 2000) {
        await new Promise(resolve => requestAnimationFrame(resolve));
        const top = banner.getBoundingClientRect().top;
        if (!banner.hidden && shown === null) shown = performance.now() - started;
        if (!banner.hidden && top >= -0.5) return {shown, inPlace: performance.now() - started,
          animation: getComputedStyle(banner).animationName, duration: getComputedStyle(banner).animationDuration, started};
      }
      return {shown, inPlace: null, started};
    }""", key)


def test_risk_banner_is_immediate_and_held_until_playback_done(session):
    ui = session()
    ui.page.add_init_script(INSTRUMENT)
    ui.page.reload()
    ui.start()
    rider(ui)
    timing = time_banner_after_key(ui.page, "2")
    assert timing["shown"] is not None and timing["shown"] < 50, timing
    assert timing["inPlace"] is not None and timing["inPlace"] <= 200, timing
    assert float(timing["duration"].rstrip("s")) <= 0.15, timing
    expect(ui.page.locator("#riskText")).to_have_text("Vehicle approaching right.")
    expect(ui.page.locator("#riskBanner")).to_be_hidden(timeout=10_000)
    events = ui.page.evaluate("() => window.__events")
    speak = next(e for e in events if e["kind"] == "in" and e["type"] == "speak" and e["priority"] == 1)
    done = next(e for e in events if e["kind"] == "out" and e["type"] == "playback_done" and e["id"] == speak["id"])
    shown = next(e for e in events if e["kind"] == "banner" and not e["hidden"] and e["t"] >= timing["started"])
    hidden = next(e for e in events if e["kind"] == "banner" and e["hidden"] and e["t"] > shown["t"])
    assert shown["t"] < speak["t"], "banner must appear before the TTS speak message arrives"
    assert hidden["t"] >= done["t"], f"banner hid {done['t'] - hidden['t']:.0f} ms before playback_done"
    assert hidden["t"] - done["t"] < 250, "banner should clear promptly after playback_done"


def test_risk_banner_is_instant_with_reduced_motion(session):
    ui = session(reduced_motion="reduce")
    ui.start()
    rider(ui)
    timing = time_banner_after_key(ui.page, "3")
    assert timing["animation"] == "none", timing
    assert timing["inPlace"] is not None and timing["inPlace"] <= 50, timing
    expect(ui.page.locator("#riskText")).to_have_text("Stop. Car right.")


def test_suppressed_repeat_does_not_hold_banner(session):
    ui = session()
    ui.start()
    rider(ui)
    ui.page.keyboard.press("2")
    expect(ui.page.locator("#riskBanner")).to_be_visible()
    expect(ui.page.locator("#riskBanner")).to_be_hidden(timeout=10_000)
    ui.page.keyboard.press("2")
    ui.page.wait_for_timeout(300)
    expect(ui.page.locator("#riskBanner")).to_be_hidden()


# 5 — "Ready." under Last spoken
def test_start_says_ready_under_last_spoken(session):
    ui = session()
    expect(ui.page.locator("#lastSpoken")).to_have_text("—")
    ui.start()
    rider(ui)
    expect(ui.page.locator("#lastSpoken")).to_have_text("Ready.")
    expect(ui.page.locator("#lastSpokenPriority")).to_have_text("SYSTEM")
    assert not ui.errors, ui.errors
