import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from core.events import BusEvent, RiskEvent, RiskLevel, SystemEvent, from_dict, risk_to_phrase
from speech.scheduler import AudioItem, AudioScheduler, NullSink, Priority, SchedulerConfig


class FakeTTS:
    def synthesize(self, text):
        return np.ones(100, dtype=np.int16), 1000


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def wait_for(scheduler, decision, item_id=None, timeout=1):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        found = [row for row in list(scheduler.decisions) if row["decision"] == decision
                 and (item_id is None or row["item_id"] == item_id)]
        if found:
            return found[-1]
        assert not any(row["decision"] == "scheduler_error" for row in list(scheduler.decisions))
        time.sleep(.001)
    raise AssertionError(f"Missing {decision} for {item_id}: {list(scheduler.decisions)}")


@pytest.fixture
def scheduler():
    sink = NullSink(duration_s=.15)
    with AudioScheduler(FakeTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001)) as instance:
        yield instance, sink


def test_high_preempts_bus_within_50_ms(scheduler):
    audio, sink = scheduler
    bus = audio.submit(BusEvent("21G", .99, None, time.time()))
    wait_for(audio, "played", bus)
    before = time.monotonic()
    risk = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    stopped = wait_for(audio, "pre-empted", bus)
    wait_for(audio, "played", risk)
    assert sink.stop_history[0]["item_id"] == bus
    assert stopped["monotonic"]-before < .05
    assert audio.wait_idle()
    assert len([row for row in audio.decisions if row["decision"] == "played" and row["item_id"] == bus]) == 2


def test_critical_beats_high(scheduler):
    audio, sink = scheduler
    high = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    wait_for(audio, "played", high)
    critical = audio.submit(RiskEvent("motorcycle", "left", "CRITICAL", time.time()))
    wait_for(audio, "pre-empted", high)
    wait_for(audio, "played", critical)
    assert sink.stop_history[-1]["item_id"] == high


def test_risk_every_frame_at_30fps_for_five_seconds():
    clock = Clock()
    sink = NullSink(duration_s=.01, clock=clock)
    with AudioScheduler(FakeTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001),
                        clock=clock, event_clock=clock) as audio:
        for frame in range(150):
            item_id = audio.submit(RiskEvent("motorcycle", "right", "HIGH", clock()))
            if frame in {0, 121}:
                wait_for(audio, "played", item_id)
            else:
                wait_for(audio, "suppressed_cooldown", item_id)
            clock.advance(1/30)
        played = [row for row in audio.decisions if row["decision"] == "played"]
        assert len(played) == 2
        assert played[1]["monotonic"]-played[0]["monotonic"] >= 4


def test_escalation_medium_to_high_bypasses_cooldown(scheduler):
    audio, _ = scheduler
    medium = audio.submit(RiskEvent("car", "right", "MEDIUM", time.time()))
    wait_for(audio, "played", medium)
    high = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    wait_for(audio, "played", high)
    wait_for(audio, "pre-empted", medium)


def test_stale_bus_is_dropped(scheduler):
    audio, sink = scheduler
    item_id = audio.submit(BusEvent("21G", 1, None, time.time()-3.1))
    wait_for(audio, "dropped_ttl", item_id)
    assert audio.wait_idle() and not sink.play_history


def test_repeat_and_stop_flush(scheduler):
    audio, sink = scheduler
    first = audio.submit("hello", identity="hello")
    wait_for(audio, "played", first)
    assert audio.wait_idle()
    audio.repeat()
    deadline = time.monotonic()+1
    while time.monotonic() < deadline:
        repeated = wait_for(audio, "played")
        if repeated["item_id"] != first:
            break
        time.sleep(.001)
    assert repeated["text"] == "hello" and repeated["item_id"] != first
    bus = audio.submit(BusEvent("21G", 1, None, time.time()))
    info = audio.submit("info", identity="info")
    chat = audio.submit("chat", priority=Priority.CHAT)
    for item_id in (bus, info, chat):
        wait_for(audio, "queued", item_id)
    audio.stop()
    wait_for(audio, "stop")
    for item_id in (bus, info, chat):
        wait_for(audio, "flushed", item_id)
    assert audio.wait_idle() and not sink.is_playing


def test_repeat_while_playing_is_queued(scheduler):
    audio, _ = scheduler
    original = audio.submit("hello")
    wait_for(audio, "played", original)
    audio.repeat()
    assert audio.wait_idle()
    assert len([row for row in audio.decisions if row["decision"] == "played" and row["text"] == "hello"]) == 2


@pytest.mark.parametrize("priority", [Priority.INFO, Priority.CHAT])
def test_lower_priority_interruption_does_not_requeue(scheduler, priority):
    audio, _ = scheduler
    item = audio.submit("lower", priority=priority)
    wait_for(audio, "played", item)
    risk = audio.submit(RiskEvent("car", "left", "HIGH", time.time()))
    wait_for(audio, "played", risk)
    assert audio.wait_idle()
    assert not any(row["decision"] == "requeued" for row in audio.decisions)


def test_bus_requeued_only_once():
    sink = NullSink(duration_s=.3)
    with AudioScheduler(FakeTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        bus = audio.submit(BusEvent("21G", 1, None, time.time()))
        wait_for(audio, "played", bus)
        high = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
        wait_for(audio, "played", high)
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            if len([row for row in audio.decisions if row["decision"] == "played" and row["item_id"] == bus]) == 2:
                break
            time.sleep(.001)
        else:
            pytest.fail("Bus never resumed")
        critical = audio.submit(RiskEvent("motorcycle", "left", "CRITICAL", time.time()))
        wait_for(audio, "played", critical)
        assert audio.wait_idle()
        assert len([row for row in audio.decisions if row["decision"] == "requeued"]) == 1


def test_ttl_expiring_while_queued(scheduler):
    audio, _ = scheduler
    risk = audio.submit(RiskEvent("car", "left", "HIGH", time.time()))
    wait_for(audio, "played", risk)
    bus = audio.submit(BusEvent("21G", 1, None, time.time()), ttl=.03)
    wait_for(audio, "dropped_ttl", bus)


def test_interrupted_bus_loses_relevance(scheduler):
    audio, _ = scheduler
    bus = audio.submit(BusEvent("21G", 1, None, time.time()))
    wait_for(audio, "played", bus)
    audio.is_relevant = lambda item: item.priority != Priority.BUS
    risk = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    wait_for(audio, "played", risk)
    wait_for(audio, "dropped_irrelevant", bus)
    assert not any(row["decision"] == "requeued" for row in audio.decisions)


def test_identical_queued_text_deduped(scheduler):
    audio, _ = scheduler
    high = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    wait_for(audio, "played", high)
    first = audio.submit("same", identity="first")
    duplicate = audio.submit("same", identity="second")
    wait_for(audio, "queued", first)
    wait_for(audio, "deduped", duplicate)


def test_identical_text_does_not_suppress_higher_priority(scheduler):
    audio, _ = scheduler
    info = audio.submit("Vehicle approaching right.", priority=Priority.INFO)
    wait_for(audio, "played", info)
    high = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    wait_for(audio, "pre-empted", info)
    wait_for(audio, "played", high)


def test_submit_does_not_wait_for_slow_synthesis():
    entered, release = threading.Event(), threading.Event()
    class SlowTTS(FakeTTS):
        def synthesize(self, text):
            if text == "slow":
                entered.set()
                release.wait(2)
            return super().synthesize(text)
    with AudioScheduler(SlowTTS(), NullSink(), SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        try:
            audio.submit("slow", priority=Priority.BUS)
            assert entered.wait(1)
            before = time.monotonic()
            risk = audio.submit(RiskEvent("car", "right", "CRITICAL", time.time()))
            assert time.monotonic()-before < .01
            wait_for(audio, "played", risk)
        finally:
            release.set()


def test_preempts_even_while_another_synthesis_is_slow(scheduler):
    audio, _ = scheduler
    bus = audio.submit(BusEvent("21G", 1, None, time.time()))
    wait_for(audio, "played", bus)
    # Simulate a warning synth that is slow; playback abort is independent.
    release = threading.Event()
    original = audio.synthesizer.synthesize
    audio.synthesizer.synthesize = lambda text: (release.wait(1), original(text))[1]
    try:
        before = time.monotonic()
        audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
        stopped = wait_for(audio, "pre-empted", bus)
        assert stopped["monotonic"]-before < .05
    finally:
        release.set()


def test_stop_keeps_queued_safety_alerts(scheduler):
    audio, _ = scheduler
    critical = audio.submit(RiskEvent("car", "left", "CRITICAL", time.time()))
    wait_for(audio, "played", critical)
    high = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    wait_for(audio, "queued", high)
    audio.stop()
    wait_for(audio, "stopped", critical)
    wait_for(audio, "played", high)


def test_concurrent_submitters(scheduler):
    audio, _ = scheduler
    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(lambda _: audio.submit("same concurrent text"), range(100)))
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        decided = {row["item_id"] for row in list(audio.decisions)}
        if set(ids) <= decided:
            break
        time.sleep(.001)
    assert set(ids) <= decided
    assert audio.wait_idle()
    assert len([row for row in audio.decisions if row["decision"] == "played"]) == 1


def test_csv_decision_log(tmp_path):
    import csv
    path = tmp_path / "latency.csv"
    with AudioScheduler(FakeTTS(), NullSink(.01), SchedulerConfig(log_path=path)) as audio:
        item = audio.submit("hello")
        assert audio.wait_idle()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert {"queued", "played", "completed"} <= {row["decision"] for row in rows}
    assert all(row["timestamp"] and row["monotonic"] for row in rows)


@pytest.mark.parametrize("risk", list(RiskLevel))
def test_contract_and_critical_phrases(risk):
    event = from_dict({"type": "risk", "object": "car", "direction": "right",
                       "risk": risk.name, "timestamp": 123.4, "unexpected": "ignored"})
    assert event.risk == risk and event.timestamp == 123.4
    assert isinstance(event, RiskEvent)
    if risk == RiskLevel.CRITICAL:
        assert len(risk_to_phrase(event).split()) <= 4
    if risk == RiskLevel.HIGH:
        assert risk_to_phrase(event) == "Vehicle approaching right."


def test_other_contract_events():
    assert from_dict({"type": "bus", "route": "21G", "confidence": .9, "bbox": [1,2,3,4], "timestamp": 123, "extra": True}).bbox == (1,2,3,4)
    assert isinstance(from_dict({"type": "system", "text": "help", "timestamp": 123}), SystemEvent)
    assert risk_to_phrase(RiskEvent("motorcycle", "right", "CRITICAL", 0)) == "Motorcycle right!"
    with pytest.raises(ValueError):
        from_dict({"type": "unknown"})


def test_closed_scheduler_rejects_requests():
    audio = AudioScheduler(FakeTTS(), NullSink(), SchedulerConfig(log_path=None))
    audio.close()
    for operation in (lambda: audio.submit("hello"), audio.stop, audio.repeat):
        with pytest.raises(RuntimeError):
            operation()


def test_bus_cooldown_during_tracking():
    clock = Clock()
    sink = NullSink(.01, clock=clock)
    with AudioScheduler(FakeTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001),
                        clock=clock, event_clock=clock) as audio:
        first = audio.submit(BusEvent("21G", 1, None, clock()))
        wait_for(audio, "played", first)
        for _ in range(5):
            clock.advance(1)
            item = audio.submit(BusEvent("21G", 1, None, clock()))
            wait_for(audio, "suppressed_cooldown", item)
        clock.advance(1)
        second = audio.submit(BusEvent("21G", 1, None, clock()))
        wait_for(audio, "played", second)
        assert len(sink.play_history) == 2


def test_equal_priority_does_not_interrupt(scheduler):
    audio, sink = scheduler
    first = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
    wait_for(audio, "played", first)
    second = audio.submit(RiskEvent("motorcycle", "left", "HIGH", time.time()))
    wait_for(audio, "queued", second)
    assert not sink.stop_history
    wait_for(audio, "played", second)


def test_priority_fifo_and_capacity(scheduler):
    audio, _ = scheduler
    audio.config = replace(audio.config, max_pending=2)
    critical = audio.submit(RiskEvent("car", "right", "CRITICAL", time.time()))
    wait_for(audio, "played", critical)
    info = audio.submit("info")
    chat = audio.submit("chat", priority=Priority.CHAT)
    for item in (info, chat):
        wait_for(audio, "queued", item)
    bus = audio.submit(BusEvent("21G", 1, None, time.time()))
    wait_for(audio, "dropped_capacity", chat)
    wait_for(audio, "queued", bus)
    assert audio.wait_idle()
    played = [row["item_id"] for row in audio.decisions if row["decision"] == "played"]
    assert played == [critical, bus, info]


def test_queued_escalation_supersedes_lower_warning(scheduler):
    audio, _ = scheduler
    critical = audio.submit(RiskEvent("car", "left", "CRITICAL", time.time()))
    wait_for(audio, "played", critical)
    medium = audio.submit(RiskEvent("motorcycle", "right", "MEDIUM", time.time()))
    wait_for(audio, "queued", medium)
    high = audio.submit(RiskEvent("motorcycle", "right", "HIGH", time.time()))
    wait_for(audio, "superseded", medium)
    wait_for(audio, "played", high)
    assert not any(row["decision"] == "played" and row["item_id"] == medium for row in audio.decisions)


def test_stop_during_synthesis_discards_late_audio():
    entered, release = threading.Event(), threading.Event()
    class SlowTTS(FakeTTS):
        def synthesize(self, text):
            entered.set()
            release.wait(2)
            return super().synthesize(text)
    sink = NullSink()
    with AudioScheduler(SlowTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        try:
            item = audio.submit("slow info")
            assert entered.wait(1)
            audio.stop()
            wait_for(audio, "cancelled_synthesis", item)
            release.set()
            assert audio.wait_idle()
            time.sleep(.02)
            assert not sink.play_history
        finally:
            release.set()


def test_ttl_expires_during_synthesis():
    class SlowTTS(FakeTTS):
        def synthesize(self, text):
            time.sleep(.08)
            return super().synthesize(text)
    sink = NullSink()
    with AudioScheduler(SlowTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        item = audio.submit("bus stale during inference", priority=Priority.BUS, ttl=.02)
        wait_for(audio, "dropped_ttl", item)
        assert audio.wait_idle()
        assert not sink.play_history


def test_resumed_bus_expires_before_risk_finishes():
    sink = NullSink(.15)
    with AudioScheduler(FakeTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        bus = audio.submit(BusEvent("21G", 1, None, time.time()), ttl=.05)
        wait_for(audio, "played", bus)
        high = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
        wait_for(audio, "played", high)
        wait_for(audio, "dropped_ttl", bus)
        assert audio.wait_idle()
        assert len([row for row in audio.decisions if row["decision"] == "played" and row["item_id"] == bus]) == 1


def test_repeat_waits_for_current_safety_warning(scheduler):
    audio, sink = scheduler
    critical = audio.submit(RiskEvent("car", "left", "CRITICAL", time.time()))
    wait_for(audio, "played", critical)
    audio.repeat()
    wait_for(audio, "queued")
    assert not sink.stop_history
    assert audio.wait_idle()
    played = [row for row in audio.decisions if row["decision"] == "played"]
    assert [row["priority"] for row in played] == [0, 4]


def test_synthesis_failure_is_logged_and_worker_recovers(scheduler):
    audio, _ = scheduler
    original = audio.synthesizer.synthesize
    def fail(text):
        if text == "bad":
            raise RuntimeError("fake failure")
        return original(text)
    audio.synthesizer.synthesize = fail
    failed = audio.submit("bad")
    wait_for(audio, "synthesis_error", failed)
    good = audio.submit("good")
    wait_for(audio, "played", good)


def test_logging_failure_keeps_audio_running(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    with AudioScheduler(FakeTTS(), NullSink(.01), SchedulerConfig(log_path=blocker/"latency.csv")) as audio:
        item = audio.submit("still audible")
        wait_for(audio, "logging_error")
        wait_for(audio, "played", item)
        assert audio.log_error and audio.wait_idle()


def test_preemption_stress_20_cycles():
    sink = NullSink(.025)
    with AudioScheduler(FakeTTS(), sink, SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        delays = []
        for index in range(20):
            bus = audio.submit(f"bus {index}", priority=Priority.BUS, identity=f"bus:{index}")
            wait_for(audio, "played", bus)
            before = time.monotonic()
            risk = audio.submit(f"warning {index}", priority=Priority.HIGH, identity=f"risk:{index}")
            stopped = wait_for(audio, "pre-empted", bus)
            delays.append((stopped["monotonic"]-before)*1000)
            wait_for(audio, "played", risk)
            assert audio.wait_idle()
        print(f"\n20-cycle preemption: max={max(delays):.3f}ms mean={np.mean(delays):.3f}ms")
        assert max(delays) < 50


def test_custom_monotonic_event_clock():
    clock = Clock()
    with AudioScheduler(FakeTTS(), NullSink(.01, clock=clock), SchedulerConfig(log_path=None),
                        clock=clock, event_clock=clock) as audio:
        expired = audio.submit({"type": "bus", "route": "21G", "timestamp": clock()-4})
        wait_for(audio, "dropped_ttl", expired)
        fresh = audio.submit({"type": "risk", "object": "car", "direction": "right", "risk": "HIGH", "timestamp": clock()})
        wait_for(audio, "played", fresh)


@pytest.mark.parametrize("config", [
    SchedulerConfig(poll_seconds=0), SchedulerConfig(max_pending=0),
    SchedulerConfig(cooldowns={}), SchedulerConfig(ttls={}),
])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        AudioScheduler(FakeTTS(), NullSink(), config)


def test_local_sink_reuses_stream_and_stops_next_block(monkeypatch):
    import sys
    from types import SimpleNamespace
    from speech.scheduler import LocalSink
    streams = []
    class Stream:
        def __init__(self, **options):
            self.callback = options["callback"]
            self.latency = .005
            self.abort_calls = self.close_calls = 0
            streams.append(self)
        def start(self):
            pass
        def abort(self):
            self.abort_calls += 1
        def close(self):
            self.close_calls += 1
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=Stream))
    sink = LocalSink(sample_rate=1000)
    sink.prepare(1000)
    assert len(streams) == 1 and not sink.is_playing
    sink.play(np.ones(100, dtype=np.int16), 1000, "bus")
    output = np.empty((10, 1), dtype=np.int16)
    streams[0].callback(output, 10, None, False)
    assert np.all(output == 1)
    sink.stop()
    streams[0].callback(output, 10, None, False)
    assert np.all(output == 0) and not sink.is_playing
    assert streams[0].abort_calls == streams[0].close_calls == 0
    sink.play(np.full(5, 9, dtype=np.int16), 1000, "risk")
    assert len(streams) == 1
    streams[0].callback(output, 10, None, False)
    assert np.all(output[:5] == 9) and np.all(output[5:] == 0)
    streams[0].callback(output, 10, None, False)
    assert not sink.is_playing
    sink.close()
    assert streams[0].abort_calls == streams[0].close_calls == 1


@pytest.mark.parametrize("first,second", [("MEDIUM", "HIGH"), ("HIGH", "CRITICAL")])
def test_same_identity_escalation(scheduler, first, second):
    audio, _ = scheduler
    original = audio.submit(RiskEvent("car", "right", first, time.time()))
    wait_for(audio, "played", original)
    escalated = audio.submit(RiskEvent("car", "right", second, time.time()))
    wait_for(audio, "pre-empted", original)
    wait_for(audio, "played", escalated)


def test_submit_does_not_block_on_sink_stop():
    entered, release = threading.Event(), threading.Event()
    class SlowSink(NullSink):
        def stop(self):
            entered.set()
            release.wait(1)
            super().stop()
    sink = SlowSink(duration_s=1)
    with AudioScheduler(FakeTTS(), sink, SchedulerConfig(log_path=None)) as audio:
        try:
            bus = audio.submit(BusEvent("21G", 1, None, time.time()))
            wait_for(audio, "played", bus)
            audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
            assert entered.wait(1)
            before = time.monotonic()
            audio.submit(RiskEvent("motorcycle", "left", "CRITICAL", time.time()))
            assert time.monotonic()-before < .01
        finally:
            release.set()


def test_clear_cooldowns_allows_same_bus_again_but_keeps_risk_cooldown():
    with AudioScheduler(FakeTTS(), NullSink(duration_s=.01), SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        for event in (BusEvent("21G", .99, None, time.time()), RiskEvent("car", "right", "HIGH", time.time())):
            audio.submit(event)
            assert audio.wait_idle()
        audio.clear_cooldowns("bus:")
        bus = audio.submit(BusEvent("21G", .99, None, time.time()))
        risk = audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
        assert audio.wait_idle()
        wait_for(audio, "played", bus)
        wait_for(audio, "suppressed_cooldown", risk)


def test_snapshot_log_collapses_consecutive_identical_decisions():
    with AudioScheduler(FakeTTS(), NullSink(duration_s=.01), SchedulerConfig(log_path=None, poll_seconds=.001)) as audio:
        for _ in range(30):
            audio.submit(RiskEvent("car", "right", "HIGH", time.time()))
            assert audio.wait_idle()
        time.sleep(.01)
        log = audio.snapshot()["log"]
        assert len(audio.decisions) >= 32
    suppressed = [row for row in log if row["decision"] == "suppressed_cooldown"]
    assert len(suppressed) == 1 and suppressed[0]["count"] == 29
    assert suppressed[0]["first_timestamp"] <= suppressed[0]["last_timestamp"] == suppressed[0]["timestamp"]
    assert all(row["count"] == 1 for row in log if row["decision"] != "suppressed_cooldown")
    assert not any((a["decision"], a["text"]) == (b["decision"], b["text"]) for a, b in zip(log, log[1:]))
