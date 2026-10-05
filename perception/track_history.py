"""Bounded observations and lightweight EMA state for each ByteTrack ID."""

from collections import deque
from dataclasses import dataclass

from common.events import Track


@dataclass(frozen=True, slots=True)
class HistoryObservation:
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_width: float
    bbox_height: float
    bbox_area: float
    bbox_center: tuple[float, float]
    bbox_bottom_center: tuple[float, float]
    confidence: float
    smoothed_bbox_area: float
    smoothed_bbox_center: tuple[float, float]


class TrackHistory:
    def __init__(self, track_id: int, max_observations: int, area_alpha: float, center_alpha: float) -> None:
        self.track_id = track_id
        self.observations: deque[HistoryObservation] = deque(maxlen=max_observations)
        self.area_alpha = area_alpha
        self.center_alpha = center_alpha
        self.smoothed_approach_rate: float | None = None

    @property
    def last_seen(self) -> float:
        return self.observations[-1].timestamp

    def add(self, track: Track, timestamp: float) -> HistoryObservation:
        if self.observations and timestamp <= self.last_seen:
            raise ValueError("track timestamps must increase")
        if self.observations:
            previous = self.observations[-1]
            area = self.area_alpha * track.bbox_area + (1 - self.area_alpha) * previous.smoothed_bbox_area
            center = tuple(
                self.center_alpha * current + (1 - self.center_alpha) * old
                for current, old in zip(track.bbox_center, previous.smoothed_bbox_center)
            )
        else:
            area = track.bbox_area
            center = track.bbox_center
        observation = HistoryObservation(
            timestamp=timestamp,
            bbox_xyxy=track.bbox_xyxy,
            bbox_width=track.bbox_width,
            bbox_height=track.bbox_height,
            bbox_area=track.bbox_area,
            bbox_center=track.bbox_center,
            bbox_bottom_center=track.bbox_bottom_center,
            confidence=track.confidence,
            smoothed_bbox_area=area,
            smoothed_bbox_center=center,
        )
        self.observations.append(observation)
        return observation

    def smooth_rate(self, rate: float, alpha: float) -> float:
        self.smoothed_approach_rate = (
            rate if self.smoothed_approach_rate is None
            else alpha * rate + (1 - alpha) * self.smoothed_approach_rate
        )
        return self.smoothed_approach_rate


class TrackHistoryStore:
    def __init__(
        self,
        max_observations: int = 15,
        stale_timeout_s: float = 2.0,
        max_tracks: int = 256,
        area_ema_alpha: float = 0.4,
        center_ema_alpha: float = 0.45,
    ) -> None:
        if max_observations < 2 or stale_timeout_s <= 0 or max_tracks < 1:
            raise ValueError("invalid history limits")
        if not (0 < area_ema_alpha <= 1 and 0 < center_ema_alpha <= 1):
            raise ValueError("EMA alpha must be in (0, 1]")
        self.max_observations = max_observations
        self.stale_timeout_s = stale_timeout_s
        self.max_tracks = max_tracks
        self.area_ema_alpha = area_ema_alpha
        self.center_ema_alpha = center_ema_alpha
        self.histories: dict[int, TrackHistory] = {}

    def prune(self, timestamp: float) -> None:
        for track_id, history in list(self.histories.items()):
            if timestamp - history.last_seen > self.stale_timeout_s:
                del self.histories[track_id]

    def update(self, track: Track, timestamp: float) -> tuple[TrackHistory, HistoryObservation]:
        history = self.histories.get(track.track_id)
        if history is None:
            if len(self.histories) >= self.max_tracks:
                oldest_id = min(self.histories, key=lambda key: self.histories[key].last_seen)
                del self.histories[oldest_id]
            history = TrackHistory(track.track_id, self.max_observations, self.area_ema_alpha, self.center_ema_alpha)
            self.histories[track.track_id] = history
        return history, history.add(track, timestamp)
