"""One orchestrator for CLI/web; latest-frame mailbox keeps capture responsive."""
from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from dataclasses import asdict

from core.config import PipelineConfig
from core.events import BusEvent, RiskEvent
from core.state import AssistantState, TargetBus
from speech.intents import IntentType, parse_intent
from speech.scheduler import AudioScheduler, Priority
from speech.tts import SpeechSynthesizer
from transit.bus_detector import BusDetector
from transit.gtfs_loader import load_route_metadata, load_routes
from transit.ocr import BoardReader, create_ocr_backend

LOGGER = logging.getLogger("assistant")


def bbox_direction(bbox, frame_width):
    center = (bbox[0]+bbox[2])/2/frame_width
    return "on your left" if center < 1/3 else "on your right" if center > 2/3 else "ahead"


def is_approaching(areas, min_growth=1.12):
    """Require sustained growth; a single jittery box is not an approach cue."""
    return (len(areas) >= 3 and areas[-1] >= areas[0] * min_growth
            and all(later > earlier for earlier, later in zip(areas, list(areas)[1:])))


def detection_status(boxes, reads):
    """Summarise why nothing is announced: no bus, still reading, unreadable or unknown route."""
    if not boxes:
        return "no_bus_in_view", None
    if any(box["confirmed_route"] for box in boxes):
        return "confirmed", None
    attempts = [attempt for box in boxes if reads.get(box["track_id"])
                for attempt in reads[box["track_id"]].attempts]
    if not attempts or any(attempt["route"] for attempt in attempts):
        return "reading_board", None
    candidates = [candidate for attempt in attempts for candidate in attempt.get("candidates", ())]
    if candidates:
        return "route_not_in_gtfs", candidates[0]
    texts = [(conf, text) for attempt in attempts
             for text, conf in zip(attempt["text"], attempt.get("conf", [0]*len(attempt["text"]))) if text.strip()]
    return "route_unreadable", max(texts)[1] if texts else None


class Assistant:
    def __init__(self, detector=None, ocr=None, scheduler=None, routes=None, config=None,
                 *, clock=time.monotonic, wall_clock=time.time, reader=None, route_info=None):
        self.config = config or PipelineConfig()
        if (self.config.max_fps <= 0 or self.config.ocr_every_n < 1
                or not 1 <= self.config.vote_k <= self.config.vote_n
                or self.config.approach_min_growth <= 1 or self.config.approach_window < 3):
            raise ValueError("Invalid FPS/OCR cadence/vote configuration")
        self.clock, self.wall_clock = clock, wall_clock
        self.routes = set(load_routes() if routes is None else routes)
        self.route_info = route_info if route_info is not None else load_route_metadata() if routes is None else {}
        self.detector = detector if detector is not None else BusDetector()
        self.reader = reader if reader is not None else BoardReader(ocr or create_ocr_backend(), self.routes)
        self.scheduler = scheduler if scheduler is not None else AudioScheduler(SpeechSynthesizer())
        self._owns_scheduler = scheduler is None
        self.state = AssistantState()
        self._state_lock = threading.RLock()
        self._condition = threading.Condition()
        self._pending = None
        self._processing = False
        self._closed = False
        self._last_submit = float("-inf")
        self._submitted = self._processed = self._dropped = self._fps_skipped = 0
        self._cache = {}
        self._seen = {}
        self._areas = {}
        self._approaching = {}
        self._read_attempts = 0
        self._overlay = {"frame_id": 0, "boxes": [], "ocr_text": [], "matched_route": None,
                         "captured_at": None, "processed_at": None, "error": None,
                         "status": "no_bus_in_view", "status_text": None}
        previous = getattr(self.scheduler, "is_relevant", lambda item: True)
        self._previous_relevance = previous
        def relevance(item):
            if not previous(item):
                return False
            if item.priority != Priority.BUS:
                return True
            with self._state_lock:
                target = self.state.target_bus
                return bool(target and item.identity == f"bus:{target.route}" and any(
                    value["route"] == target.route and value.get("visible", False)
                    and (not self.config.approaching_only or self._approaching.get(track_id, False))
                    and self.wall_clock()-value.get("last_seen", value["timestamp"]) < self.config.recent_seconds
                    for track_id, value in self.state.confirmed.items()))
        self._relevance = relevance
        self.scheduler.is_relevant = relevance
        self._worker = threading.Thread(target=self._run, name="bus-pipeline", daemon=True)
        self._worker.start()

    def _say(self, text, *, priority=Priority.INFO, identity=None):
        item_id = self.scheduler.submit(text, priority=priority, identity=identity)
        self.state.last_announcement = text
        return item_id

    def handle_transcript(self, text):
        intent = parse_intent(text, self.routes)
        acknowledgement = None
        with self._state_lock:
            if intent.type == IntentType.CANCEL_BUS or (intent.type == IntentType.FIND_BUS and intent.target):
                self._reset_bus_search()
            if intent.type == IntentType.FIND_BUS:
                if intent.target is None:
                    acknowledgement = intent.prompt
                else:
                    long_names = tuple(self.route_info.get(intent.target, {}).get("long_names", ()))
                    self.state.target_bus = TargetBus(intent.target, self.wall_clock(), long_names)
                    self.state.mode = "find_bus"
                    acknowledgement = f"Looking for bus {intent.target}."
            elif intent.type == IntentType.CANCEL_BUS:
                self.state.target_bus = None
                self.state.mode = "idle"
                acknowledgement = "Bus search cancelled."
            elif intent.type == IntentType.WHATS_AHEAD:
                recent = self.state.recent_routes(self.wall_clock(), self.config.recent_seconds)
                acknowledgement = (f"Bus {recent[0]} ahead." if len(recent) == 1
                                   else "Buses "+", ".join(recent[:3])+" ahead." if recent
                                   else "No confirmed buses yet.")
            elif intent.type == IntentType.REPEAT:
                self.scheduler.repeat()
            elif intent.type == IntentType.STOP:
                self.scheduler.stop()
            elif intent.type == IntentType.HELP:
                acknowledgement = "Say find bus, what is ahead, repeat, stop, or cancel bus."
            else:
                acknowledgement = "Command not understood. Say help."
            if acknowledgement:
                self._say(acknowledgement, identity=f"intent:{intent.type.value}:{intent.target}")
            state = self.state.payload()
        LOGGER.info("TRANSCRIPT %r -> %s(%s); target_bus=%s; ACK %r", intent.raw,
                    intent.type.value, intent.target, state["target_bus"], acknowledgement)
        return {"transcript": intent.raw, "intent": intent.type.value, "target": intent.target,
                "acknowledgement": acknowledgement, "state": state}

    def _reset_bus_search(self):
        """A new or cancelled search must not inherit the old target's cooldown or votes."""
        if self.state.target_bus:
            self.state.forget_route(self.state.target_bus.route)
        clear = getattr(self.scheduler, "clear_cooldowns", None)
        if clear:
            clear("bus:")

    def handle_risk(self, event):
        risk = RiskEvent.from_dict(event) if isinstance(event, dict) else event
        if not isinstance(risk, RiskEvent):
            raise TypeError("Expected risk event")
        item_id = self.scheduler.submit(risk)
        LOGGER.info("RISK %s %s %s -> scheduler %s", risk.risk.name, risk.object, risk.direction, item_id)
        return item_id

    def handle_frame(self, frame, *, timestamp=None):
        """Never detect/OCR on the caller. Return the latest completed overlay.

        Caller retains its frame; the accepted frame is copied once. At most
        one processing frame and one replaceable pending frame are retained.
        timestamp is wall-clock capture time, not video-position seconds.
        """
        if frame is None or not frame.size:
            raise ValueError("Empty frame")
        now = self.clock()
        with self._condition:
            if self._closed:
                raise RuntimeError("Assistant is closed")
            submitted_id = None
            if now-self._last_submit < 1/self.config.max_fps:
                self._fps_skipped += 1
                status = "skipped_fps"
            else:
                self._last_submit = now
                self._submitted += 1
                submitted_id = self._submitted
                status = "busy_replaced" if self._pending is not None else "queued"
                if self._pending is not None:
                    self._dropped += 1
                self._pending = (frame.copy(), submitted_id, self.wall_clock() if timestamp is None else timestamp)
                self._condition.notify_all()
            result = copy.deepcopy(self._overlay)
            result["overlay_status"] = result.get("status")
            result.update({"status": status, "submitted_frame_id": submitted_id,
                           "processing": self._processing, "metrics": self.metrics()})
        with self._state_lock:
            result["state"] = self.state.payload()
        return result

    def metrics(self):
        return {"submitted": self._submitted, "processed": self._processed,
                "dropped_busy": self._dropped, "skipped_fps": self._fps_skipped,
                "ocr_reads": self._read_attempts}

    def overlay(self):
        with self._condition:
            result = copy.deepcopy(self._overlay)
            result["metrics"] = self.metrics()
            result["processing"] = self._processing
        with self._state_lock:
            result["state"] = self.state.payload()
        return result

    def wait_for_frame(self, frame_id=None, timeout=5):
        with self._condition:
            target = self._submitted if frame_id is None else frame_id
            return self._condition.wait_for(lambda: self._overlay["frame_id"] >= target or self._closed, timeout)

    def _process(self, frame, frame_id, captured_at):
        began = time.perf_counter()
        if self.wall_clock()-captured_at >= self.config.max_frame_age_seconds:
            return {"frame_id": frame_id, "boxes": [], "ocr_text": [], "matched_route": None,
                    "captured_at": captured_at, "processed_at": self.wall_clock(), "error": "dropped_stale_frame",
                    "status": self._overlay.get("status"), "status_text": self._overlay.get("status_text")}
        detections = self.detector.detect(frame)
        active = {detection.track_id for detection in detections}
        # Preserve briefly occluded tracks that the detector's tracker retains.
        tracker = getattr(self.detector, "tracker", None)
        retained = set(tracker.tracks) if tracker is not None else active
        with self._state_lock:
            self.state.prune(retained)
            for track_id,value in self.state.confirmed.items():
                value["visible"] = track_id in active
                if value["visible"]:
                    value["last_seen"] = captured_at
        for mapping in (self._cache, self._seen, self._areas, self._approaching):
            for track_id in list(mapping):
                if track_id not in retained:
                    del mapping[track_id]
        boxes, all_text = [], []
        reads = 0
        for detection in sorted(detections, key=lambda item: -item.conf):
            track_id = detection.track_id
            x1, y1, x2, y2 = detection.bbox
            area = max(0.0, x2-x1) * max(0.0, y2-y1) / (frame.shape[0]*frame.shape[1])
            history = self._areas.setdefault(track_id, deque(maxlen=self.config.approach_window))
            history.append(area)
            approaching = is_approaching(history, self.config.approach_min_growth)
            self._approaching[track_id] = approaching
            seen = self._seen.get(track_id, 0)
            self._seen[track_id] = seen+1
            due = seen % self.config.ocr_every_n == 0
            if due and reads < self.config.max_tracks_per_frame:
                reads += 1
                self._read_attempts += 1
                read = (self.reader.read(frame, detection.bbox, board=True)
                        if getattr(self.detector, "board_mode", False) else self.reader.read(frame, detection.bbox))
                self._cache[track_id] = read
                with self._state_lock:
                    route = self.state.vote(track_id, read.match.route, read.confidence, read.match.method,
                                            captured_at, k=self.config.vote_k, n=self.config.vote_n)
                    if route:
                        self.state.confirmed[track_id]["bbox"] = detection.bbox
                        self.state.confirmed[track_id]["direction"] = bbox_direction(detection.bbox, frame.shape[1])
                for attempt in read.attempts:
                    LOGGER.info("OCR frame=%s track=%s %s/%s text=%r -> %s (%s) %.1fms", frame_id,
                                track_id, attempt["crop"], attempt["variant"], attempt["text"],
                                attempt["route"], attempt["method"], attempt["latency_ms"])
            read = self._cache.get(track_id)
            with self._state_lock:
                confirmed = self.state.confirmed.get(track_id)
                fresh = bool(confirmed and self.wall_clock()-confirmed["timestamp"] < self.config.recent_seconds)
                route = confirmed["route"] if fresh else None
                if route:
                    confirmed["visible"] = True
                    confirmed["last_seen"] = captured_at
                target = self.state.target_bus.route if self.state.target_bus else None
                direction = bbox_direction(detection.bbox, frame.shape[1])
                if route and target == route and (not self.config.approaching_only or approaching):
                    text = f"Bus {route} detected {direction}."
                    event = BusEvent(route, confirmed["confidence"], detection.bbox, captured_at)
                    self.scheduler.submit(event, text_override=text)
                    self.state.last_announcement = text
                elif route and target and self.config.verbose:
                    self._say(f"Bus {route}, not yours.", identity=f"other_bus:{route}")
                box = {"bbox": list(detection.bbox), "conf": detection.conf, "cls": detection.cls,
                       "track_id": track_id, "approaching": approaching,
                       "matched_route": read.match.route if read else None,
                       "confirmed_route": route, "direction": direction,
                       "ocr_text": [line.text for line in read.lines] if read else [],
                       "ocr_lines": [{**asdict(line), "box_frame": frame_box} for line, frame_box in
                                     zip(read.lines, getattr(read, "frame_boxes", None) or [None]*len(read.lines))]
                                    if read else [],
                       "ocr_box_space": "preprocessed_crop", "ocr_crop": read.crop if read else None,
                       "ocr_variant": read.variant if read else None,
                       "ocr_attempts": read.attempts if read else []}
                boxes.append(box)
                all_text += box["ocr_text"]
        status, status_text = detection_status(boxes, self._cache)
        return {"frame_id": frame_id, "frame_width": frame.shape[1], "frame_height": frame.shape[0], "boxes": boxes, "ocr_text": all_text,
                "status": status, "status_text": status_text,
                "matched_route": next((box["confirmed_route"] for box in boxes if box["confirmed_route"]), None),
                "captured_at": captured_at, "processed_at": self.wall_clock(),
                "latency_ms": (time.perf_counter()-began)*1000, "error": None}

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                frame, frame_id, captured_at = self._pending
                self._pending = None
                self._processing = True
            try:
                result = self._process(frame, frame_id, captured_at)
            except Exception as error:
                LOGGER.exception("Frame %s failed", frame_id)
                result = {"frame_id": frame_id, "boxes": [], "ocr_text": [], "matched_route": None,
                          "captured_at": captured_at, "processed_at": self.wall_clock(), "error": str(error),
                          "status": "no_bus_in_view", "status_text": None}
            with self._condition:
                self._overlay = result
                self._processed += 1
                self._processing = False
                self._condition.notify_all()

    def close(self):
        self.scheduler.stop()
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            raise RuntimeError("OCR backend did not finish shutdown within five seconds")
        self.scheduler.is_relevant = self._previous_relevance
        if self._owns_scheduler:
            self.scheduler.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
