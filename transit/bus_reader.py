"""Read route boards from Person 1's Track observations, without loading YOLO."""
from __future__ import annotations

import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass

from common.events import Track
from transit.board_crop import candidates, variants
from transit.gtfs_loader import load_route_metadata, load_route_stops
from transit.ocr import create_ocr_backend
from transit.route_matcher import extract_route_candidates, match_route


@dataclass(frozen=True)
class CurrentBus:
    route_number: str
    route_name: str
    confidence: float
    ocr_raw: str
    seen_at: float
    track_id: int


class BusReader:
    def __init__(self, *, metadata=None, ocr=None, every_n=5, threshold=.70,
                 ttl=30., include_trucks=False, vote_k=2, vote_n=3):
        self.metadata = metadata if metadata is not None else load_route_metadata()
        self.ocr = ocr if ocr is not None else create_ocr_backend()
        self.every_n, self.threshold, self.ttl = every_n, threshold, ttl
        self.include_trucks, self.vote_k = include_trucks, vote_k
        self.votes = {}
        self.vote_n = vote_n
        self.last_frame = {}
        self.current_bus = None
        self.last_read = {}
        self.ocr_ms = 0.

    def current(self, now=None):
        now = time.time() if now is None else now
        bus = self.current_bus
        return asdict(bus) if bus and 0 <= now - bus.seen_at < self.ttl else None

    def _read(self, frame, track, *, include_frame=False):
        started = time.perf_counter()
        best = {"detected": True, "ocr_raw": "", "normalized": None, "route_number": None,
                "route_name": "", "confidence": 0., "bus_box": track.bbox_xyxy,
                "board_box": None, "track_id": track.track_id, "ocr_ms": 0., "route_stops": []}
        for region, crop, box in candidates(frame, track.bbox_xyxy, include_frame=include_frame):
            for variant, image, scale in variants(crop):
                lines = self.ocr.read(image)
                records = [{"text": line.text, "box": line.box} for line in lines]
                def line_height(line):
                    ys = [point[1] for point in line.box]
                    return max(ys) - min(ys) if ys else 0.
                route_heights = [line_height(line) for line in lines
                                 if match_route(line.text, self.metadata).route]
                tallest_route = max(route_heights, default=0.)
                raw_candidates = []
                for line in lines:
                    value = line.text.strip()
                    if re.fullmatch(r"[A-Z]{0,2}\d{1,3}[A-Z]{0,2}TO", value, re.I):
                        raw_candidates.append(value[:-2])
                        continue
                    raw_candidates.append(value)
                for raw in raw_candidates + extract_route_candidates(records):
                    if re.fullmatch(r"[A-Z]{0,2}\d{1,3}[A-Z]{0,2}TO", raw, re.I):
                        continue
                    # A wholly numeric board absent from GTFS may still be a
                    # real service. Do not invent a letter suffix from its last digit.
                    if re.fullmatch(r"\s*\d{1,3}\s*", raw) and raw.strip() not in self.metadata:
                        if not best["ocr_raw"] or not re.search(r"\d", best["ocr_raw"]):
                            best["ocr_raw"] = raw.strip()
                            best["board_box"] = box
                        if region in {"top_front", "windshield"} and any(line.text.strip() == raw.strip() and line.conf >= .7 for line in lines):
                            return self._finalize(best, started)
                        continue
                    match = match_route(raw, self.metadata)
                    if not match.route:
                        if not best["ocr_raw"] and re.search(r"\d", raw):
                            best["ocr_raw"] = raw
                        continue
                    compact = re.sub(r"[^A-Z0-9]", "", raw.upper())
                    support = [line for line in lines if compact and
                               (compact in re.sub(r"[^A-Z0-9]", "", line.text.upper()) or
                                re.sub(r"[^A-Z0-9]", "", line.text.upper()) in compact)]
                    ocr_conf = min((line.conf for line in support), default=min((line.conf for line in lines), default=0.))
                    weight = {"exact": 1., "confusion_fix": .9, "fuzzy": .8}.get(match.method, 0.)
                    region_weight = .8 if region in {"bus", "image"} else 1.
                    height = max((line_height(line) for line in support), default=0.)
                    height_weight = .7 + .3 * min(1., height / tallest_route) if tallest_route else 1.
                    confidence = ocr_conf * weight * region_weight * height_weight
                    if confidence > best["confidence"]:
                        best.update(ocr_raw=raw, normalized=match.route, route_number=match.route,
                                    route_name=self.metadata[match.route].get("route_name", ""),
                                    confidence=confidence, board_box=box, region=region, variant=variant)
                if best["confidence"] >= self.threshold:
                    break
            if best["confidence"] >= self.threshold:
                break
        return self._finalize(best, started)

    def _finalize(self, best, started):
        # A tentative GTFS match must never appear as an identified route.
        if best["confidence"] < self.threshold:
            best.update(normalized=None, route_number=None, route_name="", route_stops=[])
        elif best["route_number"]:
            best["route_stops"] = load_route_stops(best["route_number"])
        best["ocr_ms"] = (time.perf_counter() - started) * 1000
        self.ocr_ms = best["ocr_ms"]
        return best

    def update(self, frame, tracks: list[Track], *, frame_index=0, now=None):
        now = time.time() if now is None else now
        reports = []
        for track in tracks:
            if track.class_name != "bus" and not (self.include_trucks and track.class_name == "truck"):
                continue
            if frame_index - self.last_frame.get(track.track_id, -self.every_n) < self.every_n:
                continue
            if self.current_bus and self.current_bus.track_id == track.track_id and self.current(now):
                continue
            self.last_frame[track.track_id] = frame_index
            read = self._read(frame, track)
            self.last_read[track.track_id] = read
            reports.append(read)
            votes = self.votes.setdefault(track.track_id, deque(maxlen=self.vote_n))
            votes.append(read["route_number"] if read["confidence"] >= self.threshold else None)
            route = read["route_number"]
            if route and Counter(votes)[route] >= self.vote_k:
                self._commit(read, now)
        return reports

    def read_image(self, frame, tracks: list[Track], *, now=None):
        now = time.time() if now is None else now
        # Each uploaded photo replaces the previous sighting. A failed read
        # must not leave a different bus available to the speech command.
        self.current_bus = None
        reports = [self._read(frame, track, include_frame=True) for track in tracks
                   if track.class_name == "bus" or self.include_trucks and track.class_name == "truck"]
        best = max(reports, key=lambda report: report["confidence"], default=None)
        if best and best["route_number"] and best["confidence"] >= self.threshold:
            self._commit(best, now)
        return best or {"detected": False, "ocr_raw": "", "normalized": None,
                        "route_number": None, "route_name": "", "confidence": 0.,
                        "bus_box": None, "board_box": None, "ocr_ms": 0., "route_stops": []}

    def _commit(self, read, now):
        if self.current_bus and self.current_bus.seen_at > now:
            return
        self.current_bus = CurrentBus(read["route_number"], read["route_name"],
                                      read["confidence"], read["ocr_raw"], now, read["track_id"])
