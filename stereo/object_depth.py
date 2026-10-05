"""Robust object depth from a box interior and bounded per-track history."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from common.events import Track


@dataclass(frozen=True, slots=True)
class DepthObservation:
    raw_depth_m: float | None = None
    smoothed_depth_m: float | None = None
    depth_confidence: float = 0.0
    valid_depth_ratio: float = 0.0
    depth_mad_m: float | None = None
    temporal_variance_m2: float | None = None


class ObjectDepthExtractor:
    def __init__(
        self, roi_x_min: float = 0.2, roi_x_max: float = 0.8,
        roi_y_min: float = 0.4, roi_y_max: float = 0.9,
        trim_percent: float = 5.0, min_valid_pixels: int = 30,
        min_valid_ratio: float = 0.10, full_confidence_ratio: float = 0.65,
        relative_mad_scale: float = 0.12,
    ) -> None:
        if not (0 <= roi_x_min < roi_x_max <= 1 and 0 <= roi_y_min < roi_y_max <= 1):
            raise ValueError("invalid object depth ROI fractions")
        if not (0 <= trim_percent < 50 and min_valid_pixels > 0 and 0 < min_valid_ratio <= full_confidence_ratio <= 1):
            raise ValueError("invalid object depth validity thresholds")
        self.x0, self.x1, self.y0, self.y1 = roi_x_min, roi_x_max, roi_y_min, roi_y_max
        self.trim_percent = trim_percent
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_ratio = min_valid_ratio
        self.full_confidence_ratio = full_confidence_ratio
        self.relative_mad_scale = relative_mad_scale

    def extract(self, track: Track, depth_m: np.ndarray) -> DepthObservation:
        height, width = depth_m.shape
        x1, y1, x2, y2 = track.bbox_xyxy
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height or x2 <= x1 or y2 <= y1:
            return DepthObservation()
        bw, bh = x2 - x1, y2 - y1
        xa = max(0, int(np.floor(x1 + self.x0 * bw)))
        xb = min(width, int(np.ceil(x1 + self.x1 * bw)))
        ya = max(0, int(np.floor(y1 + self.y0 * bh)))
        yb = min(height, int(np.ceil(y1 + self.y1 * bh)))
        if xb <= xa or yb <= ya:
            return DepthObservation()
        patch = depth_m[ya:yb, xa:xb]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        ratio = float(valid.size / patch.size)
        if valid.size < self.min_valid_pixels or ratio < self.min_valid_ratio:
            return DepthObservation(valid_depth_ratio=ratio)
        if self.trim_percent and valid.size > 10:
            low, high = np.percentile(valid, [self.trim_percent, 100 - self.trim_percent])
            valid = valid[(valid >= low) & (valid <= high)]
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        ratio_confidence = min(1.0, ratio / self.full_confidence_ratio)
        dispersion_confidence = 1 / (1 + mad / max(median * self.relative_mad_scale, 1e-3))
        confidence = float(ratio_confidence * dispersion_confidence)
        return DepthObservation(median, None, confidence, ratio, mad)


class DepthHistoryStore:
    def __init__(
        self, max_samples: int = 7, stale_timeout_s: float = 2.0,
        max_tracks: int = 256, jump_threshold_m: float = 3.0,
        jump_fraction: float = 0.6, ema_alpha: float = 0.55,
    ) -> None:
        if max_samples < 2 or stale_timeout_s <= 0 or max_tracks < 1 or jump_threshold_m <= 0:
            raise ValueError("invalid depth history settings")
        if not 0 < jump_fraction < 1 or not 0 < ema_alpha <= 1:
            raise ValueError("invalid depth smoothing settings")
        self.max_samples = max_samples
        self.stale_timeout_s = stale_timeout_s
        self.max_tracks = max_tracks
        self.jump_threshold_m = jump_threshold_m
        self.jump_fraction = jump_fraction
        self.ema_alpha = ema_alpha
        self.history: dict[int, deque[float]] = {}
        self.last_seen: dict[int, float] = {}
        self.pending_jump: dict[int, float] = {}

    def prune(self, timestamp: float) -> None:
        for track_id, last in list(self.last_seen.items()):
            if timestamp - last > self.stale_timeout_s:
                self.history.pop(track_id, None)
                self.last_seen.pop(track_id, None)
                self.pending_jump.pop(track_id, None)
        while len(self.history) > self.max_tracks:
            oldest = min(self.last_seen, key=self.last_seen.get)
            self.history.pop(oldest, None)
            self.last_seen.pop(oldest, None)
            self.pending_jump.pop(oldest, None)

    def update(self, track_id: int, observation: DepthObservation, timestamp: float) -> DepthObservation:
        self.prune(timestamp)
        raw = observation.raw_depth_m
        if raw is None or not np.isfinite(raw):
            return observation  # Never carry a stale depth through a failed stereo measurement.
        past = self.history.get(track_id)
        if past:
            reference = float(np.median(list(past)[-3:]))
            limit = max(self.jump_threshold_m, self.jump_fraction * reference)
            if abs(raw - reference) > limit:
                previous_candidate = self.pending_jump.get(track_id)
                if previous_candidate is None or abs(raw - previous_candidate) > limit / 2:
                    self.pending_jump[track_id] = raw
                    return observation  # Two consistent large jumps are needed.
                past.clear()
        self.pending_jump.pop(track_id, None)
        if past is None:
            past = deque(maxlen=self.max_samples)
            self.history[track_id] = past
        past.append(raw)
        self.last_seen[track_id] = timestamp
        recent_median = float(np.median(list(past)[-3:]))
        smoothed = self.ema_alpha * raw + (1 - self.ema_alpha) * recent_median
        variance = float(np.var(past)) if len(past) > 1 else 0.0
        return DepthObservation(
            raw, smoothed, observation.depth_confidence,
            observation.valid_depth_ratio, observation.depth_mad_m, variance,
        )
