"""Threaded priority playback. submit() never synthesizes, writes CSV, or plays.

Clocks: queue deadlines use monotonic time. Event timestamps are wall-clock by
default; inject event_clock=time.monotonic for monotonic CV producers. For video
relative timestamps use event_clock=lambda: video_position_seconds. Never mix
clock domains: stale detections must carry their original capture timestamp.
"""
from __future__ import annotations

import csv
import heapq
import math
import queue
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from core.config import ROOT
from core.events import BusEvent, RiskEvent, RiskLevel, SystemEvent, from_dict, risk_to_phrase


class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    BUS = 2
    INFO = 3
    CHAT = 4


@dataclass(frozen=True)
class AudioItem:
    text: str
    priority: Priority
    identity: str
    created_at: float = field(default_factory=time.monotonic)
    ttl: float = 3.0
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    risk_level: RiskLevel | None = None
    source_timestamp: float | None = None
    requeues: int = 0
    bypass_cooldown: bool = False


class AudioSink(Protocol):
    def play(self, pcm: np.ndarray, sr: int, item_id: str) -> None: ...
    def stop(self) -> None: ...
    @property
    def is_playing(self) -> bool: ...


class NullSink:
    """No audio output; fake durations and clock enable deterministic tests."""
    def __init__(self, duration_s: float | None = .1, *, clock=time.monotonic):
        self.duration_s = duration_s
        self.clock = clock
        self.play_history = []
        self.stop_history = []
        self._until = 0.0
        self._item_id = None
        self._lock = threading.Lock()

    def play(self, pcm, sr, item_id):
        with self._lock:
            duration = len(pcm)/sr if self.duration_s is None else self.duration_s
            self._until = self.clock()+duration
            self._item_id = item_id
            self.play_history.append({"item_id": item_id, "at": self.clock(), "duration_s": duration})

    def stop(self):
        with self._lock:
            if self._item_id is not None and self.clock() < self._until:
                self.stop_history.append({"item_id": self._item_id, "at": self.clock()})
            self._until = self.clock()
            self._item_id = None

    @property
    def is_playing(self):
        with self._lock:
            return self._item_id is not None and self.clock() < self._until


class LocalSink:
    """Persistent sounddevice stream; stop swaps to silence without driver I/O.

    Physical silence occurs within a callback block plus the device's output
    latency. Opening/closing Core Audio can exceed 100ms, so neither happens
    during pre-emption. close() releases the device at scheduler shutdown.
    """
    def __init__(self, device=None, *, sample_rate=None, block_seconds=.005):
        import sounddevice as sd
        self.sd = sd
        self.device = device
        self._target_sr = int(sample_rate or sd.query_devices(device, "output")["default_samplerate"])
        if self._target_sr <= 0 or not math.isfinite(block_seconds) or block_seconds <= 0:
            raise ValueError("LocalSink needs a positive sample rate and block duration")
        self.block_duration_ms = block_seconds*1000
        self._block_seconds = block_seconds
        self._resampled = OrderedDict()
        self._stream = None
        self._packet = None
        self._sr = None
        self._resampled.clear()
        self.output_latency_ms = 0.0
        self.underflows = 0

    def play(self, pcm, sr, item_id):
        if sr != self._target_sr:
            from speech.stt import resample_pcm
            key = (id(pcm), sr)
            if key not in self._resampled:
                self._resampled[key] = (pcm, resample_pcm(pcm, sr, self._target_sr))
                if len(self._resampled) > 16:
                    self._resampled.popitem(last=False)
            self._resampled.move_to_end(key)
            pcm = self._resampled[key][1]
            sr = self._target_sr
        self._packet = {"pcm": pcm, "offset": 0, "item_id": item_id}
        if self._stream is not None:
            return
        self._sr = sr

        def callback(outdata, frames, timing, status):
            outdata.fill(0)
            if status and status.output_underflow:
                self.underflows += 1
            packet = self._packet
            if packet is None:
                return
            offset = packet["offset"]
            count = min(frames, len(packet["pcm"])-offset)
            if count == 0:
                if self._packet is packet:
                    self._packet = None
                return
            outdata[:count, 0] = packet["pcm"][offset:offset+count]
            packet["offset"] += count

        try:
            self._stream = self.sd.OutputStream(samplerate=sr, channels=1, dtype="int16",
                                                blocksize=max(1, int(sr*self._block_seconds)), latency="low",
                                                device=self.device, callback=callback)
            self._stream.start()
            self.output_latency_ms = float(self._stream.latency)*1000
        except Exception:
            self.close()
            raise

    def stop(self):
        self._packet = None

    def prepare(self, sample_rate=22050):
        """Open/resample a silent sample at startup, before event TTLs begin."""
        self.play(np.zeros(1,dtype=np.int16),sample_rate,"startup-silence")
        self.stop()

    def close(self):
        self.stop()
        stream, self._stream = self._stream, None
        self._sr = None
        if stream is not None:
            try:
                stream.abort()
            finally:
                stream.close()

    @property
    def is_playing(self):
        return self._packet is not None


@dataclass(frozen=True)
class SchedulerConfig:
    cooldowns: dict = field(default_factory=lambda: {
        Priority.CRITICAL: 2.0, Priority.HIGH: 4.0, Priority.BUS: 6.0,
        Priority.INFO: 6.0, Priority.CHAT: 8.0})
    ttls: dict = field(default_factory=lambda: {
        Priority.CRITICAL: 1.5, Priority.HIGH: 2.0, Priority.BUS: 3.0,
        Priority.INFO: 5.0, Priority.CHAT: 10.0})
    poll_seconds: float = .005
    max_pending: int = 256
    log_path: Path | None = ROOT / "logs/scheduler_decisions.csv"


class AudioScheduler:
    def __init__(self, synthesizer, sink: AudioSink | None = None,
                 config: SchedulerConfig | None = None, *, clock=time.monotonic,
                 event_clock=time.time, is_relevant: Callable[[AudioItem], bool] | None = None):
        self.synthesizer = synthesizer
        self.sink = sink if sink is not None else LocalSink()
        self.config = config or SchedulerConfig()
        self.config = replace(self.config, cooldowns=dict(self.config.cooldowns), ttls=dict(self.config.ttls))
        if not math.isfinite(self.config.poll_seconds) or self.config.poll_seconds <= 0 or self.config.max_pending <= 0:
            raise ValueError("poll_seconds and max_pending must be positive")
        for priority in Priority:
            cooldown = self.config.cooldowns.get(priority, -1)
            ttl = self.config.ttls.get(priority, 0)
            if not math.isfinite(cooldown) or cooldown < 0 or not math.isfinite(ttl) or ttl <= 0:
                raise ValueError("Each priority needs a finite nonnegative cooldown and positive TTL")
        started = time.perf_counter()
        prepare = getattr(self.sink,"prepare",None)
        if prepare:
            prepare(getattr(getattr(synthesizer,"backend",None),"sample_rate",22050))
        self.sink_prepare_time_ms = (time.perf_counter()-started)*1000
        self.clock, self.event_clock = clock, event_clock
        self.is_relevant = is_relevant or (lambda item: True)
        self.last_spoken: AudioItem | None = None
        self._status = {"current": None, "queue": [], "last_spoken": None, "log": []}
        self.decisions = deque(maxlen=10000)
        self._collapsed = deque(maxlen=40)
        self._collapsed_lock = threading.Lock()
        self._inbox = queue.SimpleQueue()
        self._submission_lock = threading.Lock()
        self._logs = queue.SimpleQueue()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._pending = {}
        self._heap = []
        self._seq = 0
        self._current = None
        self._preparing = None
        self._audio = {}
        self._last_identity = {}
        self.last_error = None
        self.log_error = None
        # A bus synthesis cannot occupy the urgent task queue. Slow backend
        # locks may still delay NEW warning audio, but never playback abort.
        self._urgent = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-risk")
        self._normal = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-normal")
        self._logger = threading.Thread(target=self._write_logs, name="audio-log", daemon=True)
        self._worker = threading.Thread(target=self._run, name="audio-scheduler", daemon=True)
        self._logger.start()
        self._worker.start()

    def _log(self, decision, item=None, detail=""):
        now = self.clock()
        row = {"timestamp": time.time(), "monotonic": now, "decision": decision,
               "item_id": item.item_id if item else "", "identity": item.identity if item else "",
               "priority": int(item.priority) if item else "", "text": item.text if item else "",
               "age_ms": max(0, (now-item.created_at)*1000) if item else "",
               "source_timestamp": item.source_timestamp if item else "", "detail": detail}
        self.decisions.append(row)
        self._logs.put(row)
        with self._collapsed_lock:
            last = self._collapsed[-1] if self._collapsed else None
            if last and (last["decision"], last["text"]) == (decision, row["text"]):
                last.update(row, count=last["count"]+1, first_timestamp=last["first_timestamp"],
                            last_timestamp=row["timestamp"])
            else:
                self._collapsed.append({**row, "count": 1, "first_timestamp": row["timestamp"],
                                        "last_timestamp": row["timestamp"]})

    def _write_logs(self):
        stream = None
        writer = None
        try:
            if self.config.log_path is not None:
                path = Path(self.config.log_path)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = path.open("a", newline="", buffering=1)
                except Exception as error:
                    self.log_error = str(error)
                    self._log("logging_error", detail=str(error))
            while True:
                row = self._logs.get()
                if row is None:
                    break
                if stream:
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if stream.tell() == 0:
                            writer.writeheader()
                    try:
                        writer.writerow(row)
                    except Exception as error:
                        self.log_error = str(error)
                        self._log("logging_error", detail=str(error))
                        stream.close()
                        stream = None
        finally:
            if stream:
                stream.close()

    def submit(self, event, *, priority=Priority.INFO, identity=None, ttl=None, text_override=None) -> str:
        """Enqueue quickly; decisions happen asynchronously on the worker.

        Returns item_id, including for subsequently suppressed/dropped items.
        AudioItem.created_at is monotonic; event.timestamp uses event_clock.
        """
        if self._closed.is_set():
            raise RuntimeError("Scheduler is closed")
        now = self.clock()
        if isinstance(event, dict):
            event = from_dict(event)
        if isinstance(event, AudioItem):
            item = event
        else:
            source = getattr(event, "timestamp", None)
            created = now-max(0, self.event_clock()-source) if source is not None else now
            level = None
            if isinstance(event, RiskEvent):
                level = event.risk
                priority = {RiskLevel.CRITICAL: Priority.CRITICAL, RiskLevel.HIGH: Priority.HIGH,
                            RiskLevel.MEDIUM: Priority.INFO, RiskLevel.LOW: Priority.CHAT}[level]
                text = risk_to_phrase(event)
                identity = identity or f"risk:{event.object.lower()}:{event.direction.lower()}"
            elif isinstance(event, BusEvent):
                priority = Priority.BUS
                text = text_override or f"Bus {event.route} detected."
                identity = identity or f"bus:{event.route}"
            elif isinstance(event, SystemEvent):
                priority = Priority.CHAT if event.kind == "chat" else Priority.INFO
                text = event.message
            elif isinstance(event, str):
                text = event
            else:
                raise TypeError("Expected AudioItem, event dataclass/dict, or text")
            priority = Priority(priority)
            item = AudioItem(text.strip(), priority, identity or f"text:{text.strip()}", created,
                             self.config.ttls[priority] if ttl is None else ttl,
                             risk_level=level, source_timestamp=source)
        if not item.text or not math.isfinite(item.ttl) or item.ttl <= 0 or not math.isfinite(item.created_at):
            raise ValueError("Text and a positive finite TTL/deadline are required")
        self._publish("submit", item)
        return item.item_id

    def _publish(self, kind, value=None):
        # Tiny metadata-only lock prevents wait_idle()/submit()/close() races.
        # Synthesis, sink operations and CSV writes never hold this lock.
        with self._submission_lock:
            if self._closed.is_set():
                raise RuntimeError("Scheduler is closed")
            self._idle.clear()
            self._inbox.put((kind, value))
            self._wake.set()

    def repeat(self):
        self._publish("repeat")

    def stop(self):
        self._publish("stop")

    def clear_cooldowns(self, prefix):
        """Forget cooldowns whose identity starts with prefix, e.g. "bus:"."""
        self._publish("clear_cooldowns", prefix)

    def _valid(self, item):
        if self.clock()-item.created_at >= item.ttl:
            self._log("dropped_ttl", item)
            return False
        if not self.is_relevant(item):
            self._log("dropped_irrelevant", item)
            return False
        return True

    def _queue(self, item, decision="queued"):
        self._seq += 1
        self._pending[item.item_id] = item
        heapq.heappush(self._heap, (int(item.priority), self._seq, item.item_id))
        self._log(decision, item)

    def _interrupt(self, incoming=None, *, stopping=False):
        item = self._current
        if item is not None:
            try:
                self.sink.stop()
            except Exception as error:
                self._log("sink_error", item, str(error))
            self._current = None
            self._log("stopped" if stopping else "pre-empted", item,
                      f"by:{incoming.item_id}" if incoming else "STOP")
            if not stopping and item.priority == Priority.BUS and item.requeues == 0 and self._valid(item):
                self._queue(replace(item, requeues=1), "requeued")
            else:
                self._audio.pop(item.item_id, None)
        if self._preparing is not None:
            item, future = self._preparing
            future.cancel()
            self._preparing = None
            self._log("cancelled_synthesis", item)
            # Not yet spoken: preserve a pending bus without spending its retry.
            if not stopping and item.priority == Priority.BUS and self._valid(item):
                self._queue(item, "requeued_pending")

    def _accept(self, item):
        if not self._valid(item):
            return
        previous = self._last_identity.get(item.identity)
        escalates = bool(previous and item.risk_level is not None and previous[1] is not None
                         and item.risk_level > previous[1])
        if not item.bypass_cooldown and not escalates and previous:
            if self.clock()-previous[0] < self.config.cooldowns[item.priority]:
                self._log("suppressed_cooldown", item)
                return
        # Escalation replaces a queued lower-level warning before text dedupe.
        for queued in list(self._pending.values()):
            if ((queued.identity == item.identity and queued.risk_level is not None
                 and item.risk_level is not None and item.risk_level > queued.risk_level)
                    or (queued.text == item.text and item.priority < queued.priority)):
                self._pending.pop(queued.item_id)
                self._audio.pop(queued.item_id, None)
                self._log("superseded", queued)
        active = [entry for entry in (self._current, self._preparing[0] if self._preparing else None) if entry]
        if item.bypass_cooldown:
            active = []
        for queued in list(self._pending.values())+active:
            if queued.text == item.text or queued.identity == item.identity:
                if item.priority < queued.priority:
                    continue
                if item.risk_level is not None and queued.risk_level is not None and item.risk_level > queued.risk_level:
                    continue
                self._log("deduped", item)
                return
        if len(self._pending) >= self.config.max_pending:
            worst = max(self._pending.values(), key=lambda entry: (entry.priority, -entry.created_at))
            if item.priority < worst.priority:
                self._pending.pop(worst.item_id)
                self._audio.pop(worst.item_id, None)
                self._log("dropped_capacity", worst)
            else:
                self._log("dropped_capacity", item)
                return
        target = self._current or (self._preparing[0] if self._preparing else None)
        if item.priority <= Priority.HIGH and target and item.priority < target.priority:
            self._interrupt(item)
        self._queue(item)

    def _control(self, kind, value=None):
        if kind == "clear_cooldowns":
            for identity in [key for key in self._last_identity if key.startswith(value)]:
                del self._last_identity[identity]
            self._log("cleared_cooldowns", detail=value)
        elif kind == "stop":
            self._interrupt(stopping=True)
            for item in list(self._pending.values()):
                if item.priority >= Priority.BUS:
                    self._pending.pop(item.item_id)
                    self._audio.pop(item.item_id, None)
                    self._log("flushed", item, "STOP")
            self._log("stop")
        elif kind == "repeat" and self.last_spoken:
            # REPEAT is user-requested CHAT, so it cannot interrupt a warning.
            item = replace(self.last_spoken, priority=Priority.CHAT, created_at=self.clock(),
                           ttl=self.config.ttls[Priority.CHAT], item_id=uuid.uuid4().hex[:12],
                           identity=f"repeat:{self.last_spoken.identity}", requeues=0,
                           bypass_cooldown=True, risk_level=None)
            self._accept(item)

    def _start_playback(self, item, audio):
        if not self._valid(item):
            self._audio.pop(item.item_id, None)
            return
        try:
            prepare_item = getattr(self.sink, "prepare_item", None)
            if prepare_item:
                prepare_item(item)
            self.sink.play(*audio, item.item_id)
        except Exception as error:
            self._log("sink_error", item, str(error))
            try:
                self.sink.stop()
            except Exception as stop_error:
                self._log("sink_error", item, str(stop_error))
            self._audio.pop(item.item_id, None)
            return
        self._current = item
        self._audio[item.item_id] = audio
        self.last_spoken = item
        self._last_identity[item.identity] = (self.clock(), item.risk_level)
        self._log("played", item)

    def _tick(self):
        if len(self._heap) > self.config.max_pending*4:
            self._heap = [entry for entry in self._heap if entry[2] in self._pending]
            heapq.heapify(self._heap)
        for item in list(self._pending.values()):
            if not self._valid(item):
                self._pending.pop(item.item_id, None)
                self._audio.pop(item.item_id, None)
        if self._current is not None:
            if self.sink.is_playing:
                return
            self._log("completed", self._current)
            self._audio.pop(self._current.item_id, None)
            self._current = None
        if self._preparing is not None:
            item, future = self._preparing
            if not future.done():
                if not self._valid(item):
                    future.cancel()
                    self._preparing = None
                return
            self._preparing = None
            try:
                audio = future.result()
                self._start_playback(item, audio)
            except Exception as error:
                self._log("synthesis_error", item, str(error))
            return
        while self._heap:
            _, _, item_id = heapq.heappop(self._heap)
            item = self._pending.pop(item_id, None)
            if item is None:
                continue
            if not self._valid(item):
                continue
            if item_id in self._audio:
                self._start_playback(item, self._audio[item_id])
            else:
                executor = self._urgent if item.priority <= Priority.HIGH else self._normal
                future = executor.submit(self.synthesizer.synthesize, item.text)
                future.add_done_callback(lambda _: self._wake.set())
                self._preparing = item, future
            break

    def _run(self):
        try:
            while not self._closed.is_set():
                self._wake.clear()
                for _ in range(256):
                    try:
                        kind, value = self._inbox.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "submit":
                        self._accept(value)
                    else:
                        self._control(kind, value)
                self._tick()
                describe = lambda item: {"id": item.item_id, "text": item.text, "priority": int(item.priority)} if item else None
                self._status = {"current": describe(self._current),
                                "queue": [describe(item) for item in sorted(self._pending.values(), key=lambda item: (item.priority,item.created_at))],
                                "last_spoken": describe(self.last_spoken), "log": self._collapsed_log()}
                with self._submission_lock:
                    if not self._pending and self._current is None and self._preparing is None and self._inbox.empty():
                        self._idle.set()
                self._wake.wait(self.config.poll_seconds)
        except Exception as error:
            self.last_error = str(error)
            self._log("scheduler_error", detail=str(error))
            self._closed.set()
        finally:
            self._interrupt(stopping=True)
            for item in self._pending.values():
                self._log("flushed", item, "shutdown")
            self._pending.clear()
            while not self._inbox.empty():
                kind, item = self._inbox.get_nowait()
                if kind == "submit":
                    self._log("flushed", item, "shutdown")
            self._idle.set()

    def _collapsed_log(self):
        with self._collapsed_lock:
            return [dict(row) for row in self._collapsed]

    def wait_idle(self, timeout=5.0):
        idle = self._idle.wait(timeout)
        if self.last_error:
            raise RuntimeError(f"Scheduler failed: {self.last_error}")
        return idle

    def snapshot(self):
        """Worker-published debug state; avoids reading mutable queue internals."""
        import copy
        return copy.deepcopy(self._status)

    def close(self):
        with self._submission_lock:
            self._closed.set()
        self._wake.set()
        self._worker.join(timeout=2)
        if self._worker.is_alive():
            raise RuntimeError("Audio worker did not stop; sink/backend may be blocked")
        closer = getattr(self.sink, "close", None)
        try:
            if closer:
                closer()
        finally:
            self._urgent.shutdown(wait=False, cancel_futures=True)
            self._normal.shutdown(wait=False, cancel_futures=True)
            self._logs.put(None)
            self._logger.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
