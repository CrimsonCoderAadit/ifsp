"""Bounded, per-ByteTrack-ID stereo observations."""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class StereoTrackObservation:
    timestamp: float
    frame_index: int
    track_id: int
    class_name: str
    raw_depth_m: float | None
    smoothed_depth_m: float | None
    depth_confidence: float
    raw_disparity_px: float | None
    smoothed_disparity_px: float | None
    disparity_confidence: float
    valid_pixel_ratio: float
    bbox_xyxy: tuple[float, float, float, float]
    path_relevance: float


class StereoTrackHistoryStore:
    def __init__(self, max_observations: int = 12, stale_timeout_s: float = 2.0,
                 max_tracks: int = 256, disparity_ema_alpha: float = 0.65) -> None:
        if max_observations < 3 or stale_timeout_s <= 0 or max_tracks < 1 or not 0 < disparity_ema_alpha <= 1:
            raise ValueError("invalid stereo history settings")
        self.max_observations = max_observations
        self.stale_timeout_s = stale_timeout_s
        self.max_tracks = max_tracks
        self.disparity_ema_alpha = disparity_ema_alpha
        self.histories: dict[int, deque[StereoTrackObservation]] = {}

    def prune(self, timestamp: float) -> None:
        for track_id, history in list(self.histories.items()):
            if timestamp - history[-1].timestamp > self.stale_timeout_s:
                del self.histories[track_id]
        while len(self.histories) > self.max_tracks:
            oldest = min(self.histories, key=lambda key: self.histories[key][-1].timestamp)
            del self.histories[oldest]

    def update(self, observation: StereoTrackObservation) -> deque[StereoTrackObservation]:
        self.prune(observation.timestamp)
        history = self.histories.get(observation.track_id)
        if history and observation.class_name != history[-1].class_name:
            history = None  # A reused ID with a different class must not inherit depth.
        if history and observation.timestamp <= history[-1].timestamp:
            raise ValueError("stereo track timestamps must increase")
        if history is None:
            history = deque(maxlen=self.max_observations)
            self.histories[observation.track_id] = history
        history.append(observation)
        self.prune(observation.timestamp)
        return history

    def smooth_disparity(self, track_id: int, raw: float | None) -> float | None:
        if raw is None or not math.isfinite(raw) or raw <= 0:
            return None
        history = self.histories.get(track_id)
        previous = history[-1].smoothed_disparity_px if history else None
        return raw if previous is None else self.disparity_ema_alpha * raw + (1 - self.disparity_ema_alpha) * previous
