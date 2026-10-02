"""Fixtures for the browser tests: real server, fake microphone, Chromium, sessions."""
import socket
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    port = free_port()
    log = tmp_path_factory.mktemp("server") / "server.log"
    with log.open("w") as stream:
        process = subprocess.Popen([sys.executable, "-m", "app.server", "--port", str(port)],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"Server exited early:\n{log.read_text()[-3000:]}")
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.3)
    else:
        process.terminate()
        pytest.fail("Server did not become healthy within 90 s")
    yield url
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="session")
def fake_mic(tmp_path_factory):
    """The Piper command clip padded with silence, used as Chromium's fake microphone."""
    with wave.open(str(ROOT / "data/test_audio/find_bus_twenty_one_g.wav")) as source:
        rate = source.getframerate()
        speech = np.frombuffer(source.readframes(source.getnframes()), "<i2")
    padded = np.concatenate([np.zeros(int(rate * 0.3), np.int16), speech, np.zeros(int(rate * 3), np.int16)])
    path = tmp_path_factory.mktemp("audio") / "mic.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(padded.tobytes())
    return path


@pytest.fixture(scope="session")
def browser(fake_mic):
    with sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch(args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={fake_mic}", "--autoplay-policy=user-gesture-required"])
        except Exception as error:
            pytest.skip(f"Chromium unavailable: {error}")
        yield instance
        instance.close()


class Session:
    def __init__(self, browser, server, width=1440, height=900, **context_options):
        self.context = browser.new_context(viewport={"width": width, "height": height}, **context_options)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("console", lambda message: message.type == "error" and self.errors.append(message.text))
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(server + "/")

    def start(self):
        self.page.get_by_role("button", name="Start assistant").click()
        expect(self.page.locator("#connection")).to_have_attribute("data-state", "connected", timeout=10_000)
        expect(self.page.locator("#startOverlay")).to_be_hidden()

    def mode(self, mode):
        self.page.locator(f"[data-mode-button={mode}]").click()
        expect(self.page.locator("body")).to_have_attribute("data-mode", mode)

    def theme(self, theme):
        self.page.locator("#theme").select_option(theme)
        expect(self.page.locator("html")).to_have_attribute("data-theme", theme)

    def command(self, text):
        details = self.page.locator(".type-instead")
        if not details.evaluate("element => element.open"):
            details.locator("summary").click()
        self.page.locator("#commandInput").fill(text)
        self.page.locator("#commandInput").press("Enter")

    def sample(self, index=0):
        self.mode("engineer")
        self.page.locator(".sample").nth(index).click()

    def wait_confirmed(self):
        expect(self.page.locator("#statusStrip")).to_have_attribute("data-kind", "ok", timeout=15_000)
        expect(self.page.locator("#statusStrip")).to_contain_text("21G confirmed")

    def timeline(self):
        return self.page.locator("#timeline").evaluate("canvas => ({...canvas.dataset})")

    def close(self):
        self.context.close()


@pytest.fixture
def session(browser, server):
    sessions = []
    def open_session(**options):
        sessions.append(Session(browser, server, **options))
        return sessions[-1]
    yield open_session
    for item in sessions:
        item.close()
