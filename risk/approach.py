"""Apparent-size growth; no metric distance or velocity is inferred."""

import math
from dataclasses import dataclass

from perception.track_history import TrackHistory


@dataclass(frozen=True, slots=True)
class ApproachResult:
    approaching: bool
    approach_rate: float
    approach_confidence: float
    horizontal_shift_rate: float


class ApproachEstimator:
    def __init__(
        self,
        window_observations: int = 5,
        min_positive_intervals: int = 3,
        min_growth_rate: float = 0.08,
        rate_ema_alpha: float = 0.55,
        full_confidence_rate: float = 0.8,
    ) -> None:
        if window_observations < 3 or not (1 <= min_positive_intervals < window_observations):
            raise ValueError("invalid approach window")
        if min_growth_rate < 0 or full_confidence_rate <= 0 or not 0 < rate_ema_alpha <= 1:
            raise ValueError("invalid approach smoothing")
        self.window_observations = window_observations
        self.min_positive_intervals = min_positive_intervals
        self.min_growth_rate = min_growth_rate
        self.rate_ema_alpha = rate_ema_alpha
        self.full_confidence_rate = full_confidence_rate

    def estimate(self, history: TrackHistory, frame_width: int) -> ApproachResult:
        observations = list(history.observations)[-self.window_observations:]
        if len(observations) < self.window_observations:
            return ApproachResult(False, 0.0, 0.0, 0.0)
        first, last = observations[0], observations[-1]
        dt = last.timestamp - first.timestamp
        if dt <= 0 or first.smoothed_bbox_area <= 0 or last.smoothed_bbox_area <= 0:
            return ApproachResult(False, 0.0, 0.0, 0.0)

        interval_rates = [
            math.log(current.smoothed_bbox_area / previous.smoothed_bbox_area)
            / (current.timestamp - previous.timestamp)
            for previous, current in zip(observations, observations[1:])
        ]
        raw_rate = math.log(last.smoothed_bbox_area / first.smoothed_bbox_area) / dt
        rate = history.smooth_rate(raw_rate, self.rate_ema_alpha)
        positive = sum(value > self.min_growth_rate for value in interval_rates)
        approaching = rate > self.min_growth_rate and positive >= self.min_positive_intervals
        confidence = (
            (positive / len(interval_rates)) * min(1.0, max(0.0, rate) / self.full_confidence_rate)
            if approaching else 0.0
        )
        horizontal_shift = (
            (last.smoothed_bbox_center[0] - first.smoothed_bbox_center[0]) / (frame_width * dt)
            if frame_width > 0 else 0.0
        )
        return ApproachResult(approaching, rate, confidence, horizontal_shift)
